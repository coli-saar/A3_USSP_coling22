"""
This is an implementation of an unsupervised script parser by domain adaptation.

Zhai Fangzhou
"""
from typing import Dict, List, Any
import os
import math
import itertools

import torch
import torch.nn
from transformers import XLNetModel

from allennlp.data import Vocabulary
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset, DatasetReader
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.models.model import Model
from allennlp.data.batch import Batch
from allennlp.nn import util as allennlp_util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.predictors import Predictor

import dill
import numpy as np
# from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
# from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cdist as scipy_cdist

from global_constants import CONST
from aau_arks_allennlp_utils import cluster_distances, cosine_positional_encoding
from data import StoryInstanceReader
from misc import acquire_induced_metrics

from ssp_model import SequenceLabelingScriptParser as SSP_model_class
from SSP_optimization_script import ExecutionSettings as SSP_config
from SSP_optimization_script import ExecutionSettingsTaclmcrep as TaclSSPConfig
from model_checkpoints import tacl_mc_representation_checkpoint, acl_mc_representation_checkpoint


# noinspection PyCallingNonCallable
class UnsupervisedScriptParser(Model):
    def _forward_unimplemented(self, *what_the_hell_is_this_function: Any) -> None:
        pass

    def __init__(self, hypers, vocab, configurations, index_mappings):
        """
        :param hypers:
                - lr;
            the loss is now gamma_1 * similarity + gamma_2 * difference, this yields
                - gamma_2
        """
        super(UnsupervisedScriptParser, self).__init__(vocab)
        self.vocab = vocab
        self.configurations = configurations
        self.hypers = hypers
        self.index_mappings = index_mappings

        self.labels = 0
        self.coref_chain_idx = None

        self.coref_modifier = dill.load(open('coref_modifier', 'rb'))

        if self.configurations.use_pretrained_ssp_model:
            self.representation = self.load_ssp_model()
            self.representation.requires_grad_(False)
            self.representation.tagger.requires_grad_(True)
        else:
            self.representation = XLNetModel.from_pretrained(self.configurations.pretrained_model_name,
                                                             mem_len=1024)
            self.representation.requires_grad_(False)
            self.representation.layer[-1].requires_grad_(configurations.XLNet_requires_grad)

        self.to(configurations.device)

    def forward(self,
                story: Dict[str, dict],
                event_indices: List[List[int]],
                event_labels: Dict[str, torch.Tensor],
                participant_indices: List[List[int]],
                participant_labels: Dict[str, torch.Tensor],
                # the batch sampler will ensure that instances in each batch belong to a same scenario
                merged_indices: List[List[int]],
                merged_labels: Dict[str, torch.Tensor],
                scenario: Dict[str, Dict[str, torch.Tensor]],
                story_id: List[int],
                coreference_chain_idx: List[int],
                dep_marks: List[List[int]]) -> Dict[str, torch.Tensor]:
        # self.representation(**{'story': story, 'scenario': scenario, 'label_indices': merged_indices})
        """
            acquire representations to accommondate clustering. forward() does NOT cluster.
            returns:
                'loss': the regularization loss.
                'representation': Dict[str, Tensor] the representations of the candidates, condensed in one tensor.
        """
        self.coref_chain_idx = coreference_chain_idx
        mean_intra_coref_distance = 0

        batch_size = story['words']['token_ids'].size()[0]

        output_dict = dict()
        output_dict['mean_intra_cluster_dist'] = 0.
        output_dict['mean_inter_cluster_dist'] = 0.

        '''------- acquire representations from pretrained XLNET or SSP-------
        for SSP, this will shift event / participant / merged indices to the squeezed ones'''
        index_lengths = [max(inds) for inds in merged_indices]
        if self.configurations.use_pretrained_ssp_model:
            predictions = self.representation(story, scenario, merged_indices)
            squeezed_event_indices, squeezed_participant_indices = [], []
            for inst_idx in range(batch_size):
                inst_squeezed_event_indices, inst_squeezed_participant_indices = [], []
                for i, index in enumerate(merged_indices[inst_idx]):
                    if index in event_indices[inst_idx]:
                        inst_squeezed_event_indices.append(i)
                    else:
                        inst_squeezed_participant_indices.append(i)
                squeezed_event_indices.append(inst_squeezed_event_indices)
                squeezed_participant_indices.append(inst_squeezed_participant_indices)
            story_representations = predictions['classification_features']
            event_indices, participant_indices = squeezed_event_indices, squeezed_participant_indices
            merged_indices = [sorted(event_indices[i] + participant_indices[i]) for i in range(len(event_indices))]
        else:
            story_representations = self.representation(story['words']['token_ids'])[0]

        if self.configurations.use_cosine_positional_encoding:
            """
            add cosine positional encodings.
            for pretrained ssp representations, story_representations only collect candidate representations, thus the 
            max length of each story is taken to be max of merged_indices; we use candidate index / max merged index as 
            the proportion
            """
            mean_norm = torch.mean(torch.norm(story_representations, dim=-1))
            # shape: b * maxlen
            source_mask = allennlp_util.get_text_field_mask(story).to(self.configurations.device)
            # shape: b * maxlen
            proportions = source_mask.new_zeros(story_representations.size()[:2], dtype=torch.float)
            batch_size, _ = source_mask.size()
            for ind in range(batch_size):
                for j in merged_indices[ind]:
                    proportions[ind][j] = j / index_lengths[ind]
            # scale the positional encoding to match the magnitude of the representations
            cosine_encoding = cosine_positional_encoding(proportions) * self.hypers.lambda_cosine * mean_norm
            story_representations = torch.cat([story_representations, cosine_encoding], dim=-1)

        def _acquire_type_representations(_story_representations: torch.Tensor,
                                          indices: List[List[int]],
                                          labels,
                                          type_flag=0):
            """
            acquire event or participant representations and arrange them as clusters.
            this basically involves a few complex index-select steps.

            type_flag:
                -1: coref
                0: event
                1: participant
            """
            # select representations from story
            _labels_1d = 0
            _representations = 0
            for _i in range(batch_size):
                # shape: ind_len * dim
                index_tensor = torch.tensor(indices[_i], device=_story_representations.device)
                assert torch.max(index_tensor) < _story_representations.size()[1]
                candidate_representations_inst = _story_representations[_i].index_select(dim=0, index=index_tensor)

                if type(_representations) is int:
                    _representations = candidate_representations_inst
                else:
                    _representations = \
                        torch.cat(tensors=[_representations, candidate_representations_inst], dim=0)

                if type_flag == -1:
                    plus = np.array(labels[_i])
                else:
                    plus = labels['scr_labels']['tokens'][_i].detach().cpu().numpy()[:len(indices[_i])]

                if type(_labels_1d) == int:
                    _labels_1d = plus
                else:
                    _labels_1d = np.concatenate((_labels_1d, plus), axis=0)
            if type_flag == -1:
                _labels_1d = np.array([label for label in list(_labels_1d) if label != -1])
            # arrange representations into clusters
            label_set = list(set(_labels_1d))
            by_cluster_representation = dict()
            for label in label_set:
                _index = [i for i in range(len(_labels_1d)) if _labels_1d[i] == label]
                if len(_index) <= 1:
                    continue
                t_index = torch.tensor(_index, dtype=torch.long)
                t_index = t_index.to(device=_representations.device)
                label_representations = _representations.index_select(dim=0, index=t_index)
                by_cluster_representation[label] = label_representations

            # note: the later two is bugged for type_flag < 0, but we will never encounter that
            return by_cluster_representation, _labels_1d, _representations

        ''' event representations'''
        output_dict['event_representations_dict'], output_dict['event_labels_1d'], \
            output_dict['event_representations_tensor'] = \
            _acquire_type_representations(
                _story_representations=story_representations, indices=event_indices, labels=event_labels,
                type_flag=0
            )
        ''' ------------------  events  -------------------- '''
        if not self.configurations.do_not_evaluate:
            mean_event_intRA_cluster_dist, mean_event_intER_cluster_dist = \
                cluster_distances(representations=output_dict['event_representations_dict'],
                                  algorithm=self.configurations.optim_dist_metric,
                                  lb_coef=self.hypers.lb_coef,
                                  ub_coef=self.hypers.ub_coef)
            assert mean_event_intER_cluster_dist <= 1.415 and mean_event_intRA_cluster_dist <= 1.415
            output_dict['mean_intra_cluster_dist'] += mean_event_intRA_cluster_dist
            output_dict['mean_inter_cluster_dist'] += self.hypers.gamma_inter_e * mean_event_intER_cluster_dist
        ''' ------------------  participants  -------------------- '''
        output_dict['participant_representations_dict'], output_dict['participant_labels_1d'], \
            output_dict['participant_representations_tensor'] = \
            _acquire_type_representations(
                _story_representations=story_representations, indices=participant_indices, labels=participant_labels,
                type_flag=1
            )

        if not self.configurations.do_not_evaluate:
            ''' ------------------  participants  -------------------- '''
            mean_participant_intRA_cluster_dist, mean_participant_intER_cluster_dist = \
                cluster_distances(representations=output_dict['participant_representations_dict'],
                                  algorithm=self.configurations.optim_dist_metric,
                                  lb_coef=self.hypers.lb_coef,
                                  ub_coef=self.hypers.ub_coef)
            ''' ------------------  participants  -------------------- '''
            # assert mean_participant_intER_cluster_dist <= 1.415 and mean_participant_intRA_cluster_dist <= 1.415
            # if mean_participant_intRA_cluster_dist > 1.415 or mean_participant_intER_cluster_dist > 1.415:
            #     raise ArithmeticError(' distance is too large. sth is wrong. ')

            output_dict['mean_intra_cluster_dist'] += self.hypers.gamma_intra_p * mean_participant_intRA_cluster_dist
            output_dict['mean_inter_cluster_dist'] += self.hypers.gamma_inter_p * mean_participant_intER_cluster_dist

            ''' regularization '''
            if self.configurations.use_coref_regularizer:
                coref_representation_clusters, _, _ = _acquire_type_representations(
                    _story_representations=story_representations, indices=participant_indices,
                    labels=coreference_chain_idx, type_flag=-1
                )
                mean_intra_coref_distance, _ = \
                    cluster_distances(representations=coref_representation_clusters,
                                      algorithm=self.configurations.optim_dist_metric,
                                      lb_coef=self.hypers.lb_coef,
                                      ub_coef=self.hypers.ub_coef)

            dep_regularizer = 0
            if self.configurations.use_dep_regularizer:
                all_dep_marks = list()
                for marks in dep_marks:
                    for sub_mark in marks:
                        all_dep_marks += sub_mark
                depedency_cluster_indices = set(all_dep_marks)
                depedency_cluster_indices.discard(-1)
                for dep_index in depedency_cluster_indices:
                    tmp_labels = list()
                    for dep_mark in dep_marks:
                        tmp_labels.append([1 if dep_index in mark else -1 for mark in dep_mark])

                    dep_rep_clusters, _, _ = _acquire_type_representations(
                        _story_representations=story_representations, indices=merged_indices,
                        labels=tmp_labels, type_flag=-1
                    )
                    if dep_index >= 1000:
                        ''' participant '''
                        coef = self.hypers.gamma_intra_p
                    else:
                        ''' events '''
                        coef = 1.
                    mean_intra_dep_cluster_distance, _ = \
                        cluster_distances(representations=dep_rep_clusters,
                                          algorithm=self.configurations.optim_dist_metric,
                                          lb_coef=self.hypers.lb_coef,
                                          ub_coef=self.hypers.ub_coef)
                    dep_regularizer += mean_intra_dep_cluster_distance * coef * self.hypers.lambda_same_dep

            # this is a ***MINIMIZATION*** objective
            output_dict['loss'] = \
                output_dict['mean_intra_cluster_dist'] - output_dict['mean_inter_cluster_dist']
            if self.configurations.use_coref_regularizer:
                output_dict['loss']\
                    += self.hypers.gamma_intra_p * mean_intra_coref_distance * self.hypers.lambda_same_coref
            if self.configurations.use_dep_regularizer:
                output_dict['loss'] += dep_regularizer

        return output_dict

    def evaluate(self, output_dict: dict, scenario: str, gt_output_dict: dict):
        """
        evaluate aris. takes the output of forward().
        """
        metrics = dict()
        cluster_indices = dict()
        if 'e' in self.configurations.clustering_mode:
            cluster_indices['e'] = self.evaluate_type(representations=output_dict['event_representations_tensor'],
                                                      labels_1d=gt_output_dict['event_labels_1d'],
                                                      name='event', metrics=metrics, scenario=scenario)

        if 'p' in self.configurations.clustering_mode:
            cluster_indices['p'] = self.evaluate_type(representations=output_dict['participant_representations_tensor'],
                                                      labels_1d=gt_output_dict['participant_labels_1d'],
                                                      name='participant', metrics=metrics, scenario=scenario)
        return metrics, cluster_indices

    def evaluate_type(self, representations, labels_1d, name: str, metrics, scenario: str):
        """
        perform clustering and log results in the specified 'metrics' parameter. returns the clustering results.
        'name' refers to one of {'e', 'p'}.
        """
        representations_np = representations.detach().cpu().numpy()
        _representations_norm = normalize(representations_np, axis=1)
        cluster_algorithm = self.configurations.clustering_algorithms['agglw']

        def _cluster(_representations_norm):
            _dist = scipy_cdist(_representations_norm,
                                _representations_norm) if self.configurations.linkage != 'ward' else 0

            working_linkage = self.configurations.linkage if 'participant' in name else 'ward'

            ''' coreference distance modifier for participants '''
            if self.configurations.complex_coref_extension > 0 and not self.training and working_linkage != 'ward' and 'participant' in name:
                coref_idx_1d = list(itertools.chain(*self.coref_chain_idx))
                n_coref_chains = max(coref_idx_1d)
                batch_size, representation_dim = _representations_norm.shape
                coref_representation = np.zeros(shape=[batch_size, n_coref_chains], dtype='complex128')
                for i in range(batch_size):
                    ''' != -1 : in a coreference chain '''
                    if coref_idx_1d[i] != -1:
                        coref_representation[i][coref_idx_1d[i] - 1] = 1j
                ''' make sure the modifier is non-negative '''
                coref_distance = \
                    np.matmul(coref_representation, np.transpose(coref_representation)).real \
                    * self.hypers.lambda_inf_coref \
                    + self.hypers.lambda_inf_coref
                _dist += coref_distance

            _thres = self.configurations.event_dist_thres if 'event' in name \
                else self.configurations.participant_dist_thres

            if self.configurations.cluster_with_dist_thres:
                _instance = cluster_algorithm(
                    n_clusters=None,
                    distance_threshold=_thres,
                    linkage=working_linkage,
                    affinity=self.configurations.affinity[working_linkage])
            else:
                _instance = cluster_algorithm(
                    n_clusters=self.configurations.n_clusters,
                    linkage=working_linkage,
                    affinity=self.configurations.affinity[working_linkage])
            if working_linkage == 'ward':
                _predictions = _instance.fit_predict(_representations_norm)
            else:
                _predictions = _instance.fit_predict(_dist)

            if self.configurations.cluster_with_dist_thres:
                _n_predicted_clusters = len(set(list(_predictions)))
                if ('event' in name and not (10 < _n_predicted_clusters < 25)) or \
                        ('participant' in name and not (15 < _n_predicted_clusters < 25)):
                    _instance = cluster_algorithm(
                        n_clusters=20, linkage=working_linkage,
                        affinity=self.configurations.affinity[working_linkage])
                    if working_linkage == 'ward':
                        _predictions = _instance.fit_predict(_representations_norm)
                    else:
                        _predictions = _instance.fit_predict(_dist)

            return _predictions

        cls_prediction_cosine = _cluster(_representations_norm)

        if not self.configurations.do_not_evaluate:
            ''' n_gt_clusters is set to maximum to be compartible with the label indices '''
            induced_metrics = acquire_induced_metrics(
                predictions=cls_prediction_cosine, gt_labels=labels_1d,
                n_gt_clusters=self.vocab.get_vocab_size('scr_labels'),
                index_p2g=self.index_mappings[scenario][name] if 'GT' not in self.configurations.regularity else None,
                vocab=self.vocab, scenario=scenario, coref_modifier=self.coref_modifier if 'part' in name else None)
            cls_prediction_cosine = [cls_prediction_cosine]
            for metric in induced_metrics:
                if 'prediction_in' not in metric:
                    metrics[f'{name}_{metric}'] = induced_metrics[metric]
                else:
                    cls_prediction_cosine.append(induced_metrics[metric])
        return cls_prediction_cosine

    @classmethod
    def from_checkpoint(cls, check_point_config):
        combination_s = dill.load(open(check_point_config['combination_file'], 'rb'))
        combination = combination_s[check_point_config['index']]
        vocabulary = Vocabulary.from_files(check_point_config['vocab_folder'])
        model = cls(
            hypers=combination, vocab=vocabulary, configurations=check_point_config['configurations'],
            index_mappings=None)
        model.load_state_dict(torch.load(open(check_point_config['model_path'], 'rb'), map_location='cpu'))
        return model

    def inference(self, data):
        """
        acquire representations as np array. NB clustering is NOT performed here.
        The function is ok without batch sampler, as we only need the representations
        """
        # print('-------- inference --------', end='\n')
        with torch.no_grad():
            self.eval()
            cuda_device = self._get_prediction_device()
            dataset = Batch(AllennlpDataset(data).instances)
            dataset.index_instances(self.vocab)
            model_input = allennlp_util.move_to_device(dataset.as_tensor_dict(), cuda_device)
            predictions = self.make_output_human_readable(self(**model_input))

            return predictions

    def inference_scenario(self, scenario: str, dataset_reader: DatasetReader):
        data_path = os.path.join(self.configurations.data_folder, scenario)
        data = dataset_reader.read(data_path)
        inf_data = AllennlpDataset([inst for inst in data if inst.fields['scenario'].tokens[0].text == scenario])
        return self.inference(data=inf_data)

    def load_ssp_model(self):
        if self.configurations.evaluation_on == 'mc':
            check_point_config = tacl_mc_representation_checkpoint
            config = TaclSSPConfig
        else:
            if 'tacl' in self.configurations.regularity:
                # fixme: this is now acl
                _key = f'acl_{self.configurations.evaluation_on}'
            else:
                _key = f'tacl_{self.configurations.evaluation_on}'
            checkpoint_folder = CONST.ssp_checkpoint_folder[_key]
            ssp_model_folder = os.path.join(checkpoint_folder, self.configurations.validation_scenario)
            ssp_model_index = CONST.ssp_model_indices[self.configurations.validation_scenario]
            SSP_config.device = self.configurations.device
            check_point_config = dict()
            check_point_config['combination_file'] = os.path.join(ssp_model_folder, f'hyper_{ssp_model_index}')
            check_point_config['index'] = ssp_model_index
            check_point_config['vocab_folder'] = \
                os.path.join(ssp_model_folder, f'vocab_ep_{self.configurations.validation_scenario}_{ssp_model_index}')
            check_point_config['model_path'] = os.path.join(ssp_model_folder, 'best.th')
            config = SSP_config

        ssp_model = SSP_model_class.from_checkpoint(
            check_point_config=check_point_config, configurations=config, preceeds=None)
        return ssp_model

    def ussp(self, data: AllennlpDataset, gt_data=None, export_folder=None):
        """
        for stories belonging to a same scenario, acquire clustering results, and export detailed output, including
            1. each story, with cluster indices
            2. each cluster, with context

        this function is called at the end of each optimization trial to generate the results.

        data: stories belonging to a same scenario
        """
        if gt_data is None:
            gt_data = data
        metrics, cluster_indices = self.validate(data, gt_data)
        if export_folder is not None:
            self.export_results(data, cluster_indices, export_folder, coref_modifier=self.coref_modifier)
        return metrics

    def export_as_inscript_format(self, data: dict, data_folder, export_folder=None):
        for scenario, scenario_data in data.items():
            data_lines = open(os.path.join(data_folder, scenario), 'r').readlines()
            irregularity = \
                [i for i in range(len(data_lines))
                 if '#' in data_lines[i] and 'irr' not in data_lines[i]]
            global_token_index = 0
            _, cluster_indices = self.validate(scenario_data)
            event_indices_as_array = cluster_indices['e'][1]
            event_predictions_as_array = [self.vocab.get_token_from_index(int(ind), 'scr_labels') for ind in event_indices_as_array]
            index_of_event_indices = 0
            with open(os.path.join(export_folder, scenario), 'w') as ex_out:
                for instance in scenario_data:
                    token_ind = 0
                    while '<sep>' not in instance.fields['story'].tokens[token_ind].text:
                        token_ind += 1
                    while token_ind < len(instance.fields['story']) - 1:
                        token_ind += 1
                        token = instance.fields['story'].tokens[token_ind]
                        line = [token.text]
                        if token_ind in instance.fields['event_indices'].metadata:
                            line.append(event_predictions_as_array[index_of_event_indices])
                            index_of_event_indices += 1
                        elif global_token_index in irregularity:
                            line.append('irregular_event')
                        ex_out.write('\t'.join(line) + '\n')
                        global_token_index += 1

    def export_results(self, data: AllennlpDataset, cluster_indices, export_folder, coref_modifier=None):
        """
        export results.
        cluster_indices: dict[str, np_array]
            str in {'e', 'p'}
        """
        ''' per story '''
        story_file_path = os.path.join(export_folder, 'agglw_per_story')
        with open(story_file_path, 'w') as story_out:
            predicted_event_labels_1d = cluster_indices['e'][0]
            if 'p' in self.configurations.clustering_mode:
                predicted_part_labels_1d = cluster_indices['p'][0]

            global_event_count = 0
            global_participant_count = 0
            for instance in data.instances:
                story_id = instance.fields['story_id'].metadata
                story_out.write(f'-----------story {story_id} algorithm agglw_cosine-----------\n')
                story_out.write('--events--------\n')
                for i in range(len(instance.fields['event_indices'].metadata)):
                    ind = instance.fields['event_indices'].metadata[i]
                    label = instance.fields['event_labels'].tokens[i].text
                    story_out.write(StoryInstanceReader.context(instance, ind) + '\t' + 'predicted: '
                                    + str(predicted_event_labels_1d[global_event_count]) + '\t' +
                                    'ground_truth: ' + str(label) + '\n')
                    global_event_count += 1
                if 'p' in self.configurations.clustering_mode:
                    story_out.write('--participants--------\n')
                    for i in range(len(instance.fields['participant_indices'].metadata)):
                        ind = instance.fields['participant_indices'].metadata[i]
                        label = instance.fields['participant_labels'].tokens[i].text
                        story_out.write(StoryInstanceReader.context(instance, ind) + '\t' + 'predicted: '
                                        + str(predicted_part_labels_1d[global_participant_count]) + '\t' +
                                        'ground_truth: ' + str(label) + '\n')
                        global_participant_count += 1

        ''' per cluster '''
        event_candidates_by_cluster = dict()
        participant_candidates_by_cluster = dict()
        cluster_file_path = os.path.join(export_folder, 'agglw_per_cluster_')
        predicted_event_labels_1d = cluster_indices['e'][0]
        predicted_part_labels_1d = 0
        if 'p' in self.configurations.clustering_mode:
            predicted_part_labels_1d = cluster_indices['p'][0]

        global_event_candidate_id = 0
        for instance in data.instances:
            for i in range(len(instance.fields['event_indices'].metadata)):
                ind = instance.fields['event_indices'].metadata[i]
                label = instance.fields['event_labels'].tokens[i].text
                predicted_label = predicted_event_labels_1d[global_event_candidate_id]
                global_event_candidate_id += 1
                _context = StoryInstanceReader.context(instance, ind)
                if predicted_label not in event_candidates_by_cluster:
                    event_candidates_by_cluster[predicted_label] = list()
                event_candidates_by_cluster[predicted_label].append(
                    {
                        'context': _context,
                        'label': label,
                        'predicted_label': predicted_label
                    }
                )
        if 'p' in self.configurations.clustering_mode:
            global_participant_candidate_id = 0
            for instance in data.instances:
                for i in range(len(instance.fields['participant_indices'].metadata)):
                    ind = instance.fields['participant_indices'].metadata[i]
                    label = instance.fields['participant_labels'].tokens[i].text
                    predicted_label = predicted_part_labels_1d[global_participant_candidate_id]
                    global_participant_candidate_id += 1
                    _context = StoryInstanceReader.context(instance, ind)
                    if predicted_label not in participant_candidates_by_cluster:
                        participant_candidates_by_cluster[predicted_label] = list()
                    participant_candidates_by_cluster[predicted_label].append(
                        {
                            'context': _context,
                            'label': label,
                            'predicted_label': predicted_label
                        }
                    )

        with open(cluster_file_path + 'event', 'w') as cluster_out:
            for cluster_ind in event_candidates_by_cluster:
                cluster_out.write(f'============= cluster {cluster_ind} ===============\n')
                for candidate in event_candidates_by_cluster[cluster_ind]:
                    cluster_out.write(
                        candidate['context'] + '\t' +
                        'label: ' + candidate['label'] + '\t' +
                        'predicted_label: ' + str(candidate['predicted_label']) + '\n'
                    )

        with open(cluster_file_path + 'participant', 'w') as cluster_out:
            for cluster_ind in participant_candidates_by_cluster:
                cluster_out.write(f'============= cluster {cluster_ind} ===============\n')
                for candidate in participant_candidates_by_cluster[cluster_ind]:
                    cluster_out.write(
                        candidate['context'] + '\t' +
                        'label: ' + candidate['label'] + '\t' +
                        'predicted_label: ' + str(candidate['predicted_label']) + '\n'
                    )
            if coref_modifier:
                cluster_out.write(f'============= cluster -1 ===============\n')
                scenario = data[0].fields['scenario'].tokens[0].text
                for part, clist in coref_modifier[scenario].items():
                    for context in clist:
                        cluster_out.write(
                            context + '\t' +
                            'label: ' + part + '\t' +
                            'predicted_label: prot \n'
                        )

    def text_representation(self, text):
        self.representation.configurations.select_index = False
        output = self.representation(**{'story': {'words': {'token_ids': text}}, 'scenario': text, 'label_indices': text})
        return output['classification_features']

    def validate(self, validation_data: AllennlpDataset, gt_validation_data: AllennlpDataset = None):
        """
        performs validation on a specific scenario.
        evaluates validation metrics
        """
        if gt_validation_data is None:
            gt_validation_data = validation_data
        scenario = validation_data[0].fields['scenario'].tokens[0].text
        output_dict = self.inference(data=validation_data)
        gt_output_dict = self.inference(data=gt_validation_data)
        metrics, cluster_indices = self.evaluate(
            output_dict=output_dict, scenario=scenario, gt_output_dict=gt_output_dict)

        return metrics, cluster_indices


class SequenceLabellingModel(Model):
    def _forward_unimplemented(self, *ohhellothere: Any) -> None:
        pass

    def __init__(self, feature_dim, hypers, vocab, text_embeddings, text_embedding_type):
        super().__init__(vocab)
        self.vocab = vocab
        self.hypers = hypers
        self.text_embeddings = text_embeddings
        self.text_embeddings.requires_grad_(False)
        self.text_embedding_type = text_embedding_type
        self.accuracy = CategoricalAccuracy()
        self.classifier = torch.nn.Linear(
            in_features=feature_dim, out_features=self.vocab.get_vocab_size(namespace='seq_labels'))

    def forward(self, story, seq_labels) -> Dict[str, torch.Tensor]:
        invalid_index = self.vocab.get_token_index(token='X', namespace='story')
        if self.text_embedding_type == 'xlnet':
            text_representations = self.text_embeddings(story['story']['token_ids'])[0]
        else:
            text_representations = self.text_embeddings.text_representation(story['story']['token_ids'])

        logits = self.classifier(text_representations)
        predictions = torch.softmax(logits, dim=-1)

        xlnet_tokenizer_mask = story['story']['token_ids'] != invalid_index
        source_mask = allennlp_util.get_text_field_mask(story)
        final_mask = xlnet_tokenizer_mask * source_mask

        self.accuracy(predictions=predictions, gold_labels=seq_labels['seq_labels']['tokens'], mask=final_mask)
        loss = allennlp_util.sequence_cross_entropy_with_logits(
            logits=logits, targets=seq_labels['seq_labels']['tokens'], weights=final_mask
        )

        return{'loss': loss}

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

    def test(self, test_data, dataset_reader, batch_size=32):
        """
        acquire test metrics
        :param test_data:
        :param dataset_reader:
        :param batch_size:
        :return:
        """
        predictor = Predictor(self, dataset_reader)
        predictions = list()
        num_batches = int(np.ceil(len(test_data) / batch_size))
        for batch_index in range(num_batches):
            batch_instances = [test_data[i] for i in range(batch_size * batch_index, batch_size * (batch_index + 1))
                               if i < len(test_data)]
            batch_predictions = predictor.predict_batch_instance(batch_instances)
            predictions.extend(batch_predictions)
        metrics = self.get_metrics(reset=True)
        test_metrics = {'test_' + key: metrics[key] for key in metrics}
        return test_metrics


class TextClassificationModel(Model):
    def _forward_unimplemented(self, *ohhellothere: Any) -> None:
        pass

    def __init__(self, vocab, hypers, text_embeddings):
        super().__init__(vocab)
        self.vocab = vocab
        self.hypers = hypers
        self.text_embeddings = text_embeddings
        self.text_embeddings.requires_grad_(False)
        self.accuracy = CategoricalAccuracy()
        self.classifier = torch.nn.Linear(
            in_features=1024, out_features=2)

    def forward(self, story, label) -> Dict[str, torch.Tensor]:
        text_representations = self.text_embeddings.text_representation(story['story']['token_ids'])

        logits = self.classifier(text_representations)[:, 0, :]
        predictions = torch.softmax(logits, dim=-1)

        source_mask = allennlp_util.get_text_field_mask(story)[:, 0]

        self.accuracy(predictions=predictions, gold_labels=label, mask=source_mask)
        loss = allennlp_util.sequence_cross_entropy_with_logits(
            logits=logits, targets=label, weights=source_mask
        )
        return{'loss': loss}

    def get_metrics(self, reset=False):
        return {'accuracy': self.accuracy.get_metric(reset)}

    def test(self, test_data, dataset_reader, batch_size=32):
        """
        acquire test metrics
        :param test_data:
        :param dataset_reader:
        :param batch_size:
        :return:
        """
        # reset metrics is not necessary as allenNLP resets metrics in trainer.train()
        predictor = Predictor(self, dataset_reader)
        predictions = list()
        num_batches = int(np.ceil(len(test_data) / batch_size))
        for batch_index in range(num_batches):
            batch_instances = [test_data[i] for i in range(batch_size * batch_index, batch_size * (batch_index + 1))
                               if i < len(test_data)]
            batch_predictions = predictor.predict_batch_instance(batch_instances)
            predictions.extend(batch_predictions)
            print('.', end='')
        metrics = self.get_metrics(reset=True)
        test_metrics = {'test_' + key: metrics[key] for key in metrics}
        return test_metrics

#
    # def acquire_representations(self,
    #                             story: Dict[str, Dict[str, torch.Tensor]],
    #                             event_indices: List[List[int]],
    #                             event_labels: Dict[str, torch.Tensor],
    #                             participant_indices: List[List[int]],
    #                             participant_labels: Dict[str, torch.Tensor],
    #                             merged_indices: List[List[int]],
    #                             merged_labels: Dict[str, torch.Tensor],
    #                             scenario: Dict[str, Dict[str, torch.Tensor]],
    #                             scenario_phrase: Dict[str, Dict[str, torch.Tensor]],
    #                             story_id: List[int],
    #                             dep_marks
    #                             ):
    #     """
    #     acquire initial representations from XLNet. Used for training from scratch.
    #     """
    #     # XLNet model returns 4 tensors, [0] being the encoded sequence,
    #     #   see: https://huggingface.co/transformers/model_doc/xlnet.html?highlight=xlnetmodel#transformers.XLNetModel
    #     # somehow 'story'['words'] gets keys 'token_ids', 'mask' and 'type_ids'. yet other text fields like scenario
    #     # get only 'tokens'. This should be the outcome of XLNet's token indexer.
    #     # shape: batch * maxlen * dim
    #     story_representations_raw = self.representation(story['words']['token_ids'])[0]
    #     batch_size = story_representations_raw.size()[0]
    #
    #     def _acquire_type_representations_aq(story_representations: torch.Tensor,
    #                                          indices: List[List[int]], labels):
    #         """
    #         get sequence of event / participant representations from the story representation
    #         this basically involves a few complex index-select steps.
    #         """
    #         # select representations from story
    #         _labels_1d = 0
    #         _representations = 0
    #         for i in range(batch_size):
    #             # these two steps perform a batched index_select, which selects candidate representations from the
    #             # representations of the whole story
    #             # shape: ind_len * dim
    #             candidate_representations_inst = story_representations[i].index_select(
    #                 dim=0,
    #                 index=torch.tensor(indices[i], device=self.configurations.device))
    #
    #             if type(_representations) is int:
    #                 _representations = candidate_representations_inst
    #             else:
    #                 _representations = \
    #                     torch.cat(tensors=[_representations, candidate_representations_inst], dim=0)
    #             if type(_labels_1d) == int:
    #                 _labels_1d = \
    #                     labels['scr_labels']['tokens'][i].detach().cpu().numpy()[:len(indices[i])]
    #             else:
    #                 candidate_labels = \
    #                     labels['scr_labels']['tokens'][i].detach().cpu().numpy()[:len(indices[i])]
    #                 _labels_1d = np.concatenate((_labels_1d, candidate_labels), axis=0)
    #         _label_tokens_1d = list()
    #         for index in range(np.shape(_labels_1d)[0]):
    #             _label_tokens_1d.append(self.vocab.get_token_from_index(index=int(_labels_1d[index]),
    #                                                                     namespace='scr_labels'))
    #
    #         return _label_tokens_1d, _representations
    #
    #     event_metadata, initial_event_representations = \
    #         _acquire_type_representations_aq(
    #             story_representations=story_representations_raw,
    #             indices=event_indices,
    #             labels=event_labels)
    #
    #     participant_metadata, initial_participant_representations = \
    #         _acquire_type_representations_aq(
    #             story_representations=story_representations_raw,
    #             indices=participant_indices,
    #             labels=participant_labels)
    #
    #     return {'event_meta': event_metadata,
    #             'event_initial': initial_event_representations,
    #             'participant_meta': participant_metadata,
    #             'participant_initial': initial_participant_representations}

    # @staticmethod
    # def advanced_feature_select(features: torch.Tensor, valid_indices: List[List[int]], max_seq_len: int):
    #     """
    #     features: b*l*d
    #     valid_indices:
    #     """
    #     dense_feature_shape = features.size()[0], max_seq_len, features.size()[-1]
    #     _batch_size, _, feature_dim = dense_feature_shape
    #     dense_source_mask = features.new_zeros([_batch_size, max_seq_len])
    #     feature_selected = features.new_zeros(size=dense_feature_shape, dtype=torch.float)
    #     for i in range(_batch_size):
    #         feature_selected[i, :len(valid_indices[i]), :] = features[i]. \
    #             index_select(dim=0, index=feature_selected.new_tensor(valid_indices[i], dtype=torch.long))
    #         dense_source_mask[i, :len(valid_indices[i])] += 1
    #     dense_source_mask.bool()
    #     return feature_selected, dense_source_mask
    #
    # def _acquire_seq_features(self, merged_indices, event_indices, story_representations):
    #     if self.configurations.clustering_mode == 'ep':
    #         indices = merged_indices
    #     else:
    #         indices = event_indices
    #     max_seq_len = max([len(ind) for ind in indices])
    #     return UnsupervisedScriptParser.advanced_feature_select(story_representations, indices, max_seq_len)
    #
    #     if self.configurations.use_pretrained_ssp_model:
    #         # scenario_str = self.vocab.get_token_from_index(int(scenario['scenarios']['tokens'][0][0].detach().cpu()), 'scenarios')
    #         # n_candidates = sum([len(mi) for mi in merged_indices])
    #         # if scenario_str in CONST.ins_gt_merged_index_count:
    #         #     ''' inscript scenario '''
    #         #     if n_candidates == CONST.ins_gt_merged_index_count[scenario_str]:
    #         #         data_instances = self.gt_data[scenario_str]
    #         #     else:
    #         #         ''' inscript pseudo '''
    #         #         data_instances = self.pseudo_data[scenario_str]
    #         # else:
    #         #     ''' mcscript val scenario '''
    #         #     if scenario_str in CONST.mc1_gt_merged_index_count:
    #         #         data_instances = self.mcval_data_by_scenario[scenario_str]
    #         #     else:
    #         #         ''' mctest scenario '''
    #         #         if n_candidates == CONST.mc2_gt_merged_index_count[scenario_str]:
    #         #             data_instances = self.mctest_gt_by_scenario[scenario_str]
    #         #         else:
    #         #             data_instances = self.mctest_pd_by_scenario[scenario_str]
    #         #
    #         # # candidate_representations, event_labels_tensor.to(cuda_device), participant_labels_tensor.to(cuda_device), squeezed_event_indices, squeezed_participant_indices \
    #         # rep = self.acquire_ssp_representations(ssp_model=self.representation, data_instances=data_instances)
    #         rep = self.representation(**{'story': story, 'scenario': scenario, 'label_indices': merged_indices})
    #         story_representations, event_labels, participant_labels, event_indices, participant_indices = rep
    #         merged_indices = [sorted(event_indices[i] + participant_indices[i]) for i in range(len(event_indices))]
    #     else:
    #         story_representations = self.representation(story['words']['token_ids'])[0]
