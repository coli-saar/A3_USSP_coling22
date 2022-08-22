import itertools

import torch
import torch.nn.functional
import os
import time
import shutil
# import dill
import sys
import collections

import numpy as np

from allennlp.data.fields import TextField
from allennlp.data.tokenizers.pretrained_transformer_tokenizer import PretrainedTransformerTokenizer
from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.dataset_readers.dataset_reader import AllennlpDataset
from allennlp.data import Vocabulary
from allennlp.nn import util as allennlp_util

from torch import optim

from aau_random_hyper_search_optimizer import RandomSearchMetaOptimizer
from misc import PrintColors  # , EasyPlot
from data import StoryInstanceReader, ClusterInstanceSampler
from aau_arks_allennlp_utils import supply_token_indices  # , average_intra_cluster_distances
from global_constants import CONST
from model import UnsupervisedScriptParser


class ScriptRepresentationLearningMetaOptimizer(RandomSearchMetaOptimizer):
    """

    """
    def __init__(self, configurations, model: type(UnsupervisedScriptParser)):
        domains = configurations.param_domains
        parameters = {
            'batch_size': {'domain': domains['batch_size'], 'sample_criterion': '2e', 'type': 'int'},
            'lr': {'domain': domains['lr'], 'sample_criterion': '10e', 'type': 'float'},
            'l2': {'domain': domains['l2'], 'sample_criterion': '10e', 'type': 'float'},
            'clip': {'domain': domains['clip'], 'sample_criterion': '10e', 'type': 'float'},
            'representation_dim': {'domain': domains['representation_dim'], 'sample_criterion': '2e', 'type': 'int'},
            'gamma_inter_e': {'domain': domains['gamma_inter_e'], 'sample_criterion': '10e', 'type': 'float'},
            'gamma_intra_p': {'domain': domains['gamma_intra_p'], 'sample_criterion': '10e', 'type': 'float'},
            'gamma_inter_p': {'domain': domains['gamma_inter_p'], 'sample_criterion': '10e', 'type': 'float'},
            'lb_coef': {'domain': domains['lb_coef'], 'sample_criterion': 'u', 'type': 'float'},
            'ub_coef': {'domain': domains['ub_coef'], 'sample_criterion': 'u', 'type': 'float'},

            'lambda_cosine': {'domain': domains['lambda_cosine'], 'sample_criterion': '10e', 'type': 'float'},
            'lambda_same_coref': {'domain': domains['lambda_same_coref'], 'sample_criterion': '10e', 'type': 'float'},
            'lambda_same_dep': {'domain': domains['lambda_same_dep'], 'sample_criterion': '10e', 'type': 'float'},

            'lambda_inf_coref': {'domain': domains['lambda_inf_coref'], 'sample_criterion': '10e', 'type': 'float'},
            # 'lambda_relate_to_same_participant': {'domain': domains['lambda_relate_to_same_participant'], 'sample_criterion': '10e', 'type': 'float'},
        }
        super().__init__(parameters=parameters,
                         metric_names=[
                             # 'test2_event_acc', 'test2_event_micro_f1', 'test2_event_macro_f1',
                             # 'test2_participant_acc', 'test2_participant_micro_f1', 'test2_participant_macro_f1',
                             # 'test2_pred_event_acc', 'test2_pred_event_micro_f1', 'test2_pred_event_macro_f1',
                             # 'test2_pred_participant_acc', 'test2_pred_participant_micro_f1', 'test2_pred_participant_macro_f1',
                             'train_event_acc', 'val_event_acc', 'test_event_acc',
                             'train_event_macro_f1', 'val_event_macro_f1', 'test_event_macro_f1',
                             'train_event_micro_f1', 'val_event_micro_f1', 'test_event_micro_f1',
                             'train_participant_acc', 'val_participant_acc', 'test_participant_acc',
                             'train_participant_macro_f1', 'val_participant_macro_f1',
                             'test_participant_macro_f1',
                             'train_participant_micro_f1', 'val_participant_micro_f1',
                             'test_participant_micro_f1',
                             'train_mean_intra_cluster_dist', 'train_mean_inter_cluster_dist',
                             'time_consumed(hrs)', 'best_epoch',
                             ],
                         num_trials=configurations.num_trials,
                         tag=configurations.tag)
        self.configurations = configurations
        self.model_class = model

        # Dict[str (metric names), dict / list (metrics or by scenario metrics)]
        self.metrics = dict()

    def train_epoch(self, model, train_dataset, optimizer):
        """
        performs an epoch of training.
        """
        model.train()
        train_data_loader = PyTorchDataLoader(
            dataset=train_dataset,
            batch_sampler=ClusterInstanceSampler(data_source=train_dataset, batch_size=model.hypers.batch_size))

        losses = dict()

        # mini batch training
        n_batches = len(train_data_loader)
        for batch_index, batch in enumerate(train_data_loader):
            # this util function moves a Tensor dict to a cuda device
            # this allenNLP data loader uses the allenNLP collate_fn so we get a dictionary that contains all
            # fields of the batch
            batch = allennlp_util.move_to_device(batch, cuda_device=self.configurations.device)

            ''' main training '''
            optimizer.zero_grad()

            ''' values in output_dict are torch autograds '''
            output_dict = model.forward(**batch)
            if 'mean_intra_cluster_dist' not in losses:
                losses['mean_intra_cluster_dist'] = output_dict['mean_intra_cluster_dist']
            else:
                losses['mean_intra_cluster_dist'] += output_dict['mean_intra_cluster_dist']

            if 'mean_inter_cluster_dist' not in losses:
                losses['mean_inter_cluster_dist'] = output_dict['mean_inter_cluster_dist']
            else:
                losses['mean_inter_cluster_dist'] += output_dict['mean_inter_cluster_dist']

            output_dict['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), model.hypers.clip)
            optimizer.step()

            if batch_index % max(n_batches // 5, 1) == 0:
                print('.', end='')
                sys.stdout.flush()
        print('', end='\n')
        for key in losses:
            losses[key] /= n_batches

        return losses

    def train(self, args_hpo, index):
        """
        trains the model, and return the metrics to the meta optimizer.
        :param args_hpo:
        :param index:
        :return:
        """
        def _get_evaluation_metrics(_model, _inf_data=None, _gt_data=None):
            if type(_inf_data) == dict:
                sce_metric_list = list()
                for _scenario in _inf_data:
                    if 'GT' in self.configurations.regularity:
                        sce_metrics, _ = model.validate(validation_data=AllennlpDataset(_inf_data[_scenario]))
                    else:
                        sce_metrics, _ = model.validate(validation_data=AllennlpDataset(_inf_data[_scenario]),
                                                        gt_validation_data=AllennlpDataset(_gt_data[_scenario]))
                    sce_metric_list.append(sce_metrics)
                avg_metrics = {_key: sum([sce_metrics[_key] for sce_metrics in sce_metric_list]) / len(sce_metric_list) for _key in sce_metric_list[0] if 'detail' not in _key}
                return avg_metrics
            else:
                if 'GT' in self.configurations.regularity:
                    return model.validate(_inf_data)[0]
                else:
                    return model.validate(_inf_data, _gt_data)[0]

        def _maybe_batch_ussp(_model, _inf_data, _gt_data=None):
            if _gt_data is None:
                _gt_data = _inf_data
            if type(_inf_data) == dict:
                for _scenario in _inf_data:
                    model.ussp(data=AllennlpDataset(_inf_data[_scenario]), export_folder=trial_dir,
                               gt_data=AllennlpDataset(_gt_data[_scenario]))
            else:
                model.ussp(data=_inf_data, export_folder=trial_dir, gt_data=_gt_data)

        def _terminate(_index):
            os.rename(
                os.path.join(trial_dir, f'model_{_index}'),
                os.path.join(trial_dir, 'best'))
            metrics['time_consumed(hrs)'] = round((time.time() - starting_time) / 3600, 4)
            metrics['best_epoch'] = _index
            for file in os.listdir(trial_dir):
                if 'model' in file:
                    os.remove(os.path.join(trial_dir, file))

            _best_model = self.model_class(hypers=model.hypers,
                                           vocab=model.vocab,
                                           configurations=model.configurations,
                                           index_mappings=index_mappings,)

            _best_model.load_state_dict(torch.load(open(os.path.join(trial_dir, 'best'), 'rb')))

            ''' test best model '''
            _maybe_batch_ussp(_best_model, _inf_data=test_dataset, _gt_data=gt_test_dataset)
            t_test_metrics = _get_evaluation_metrics(_model=_best_model, _inf_data=test_dataset, _gt_data=gt_test_dataset)
            _best_model.export_as_inscript_format(
                data=test_dataset,
                export_folder=trial_dir,
                data_folder=CONST.mcscript_test_dir)

            test_metricspp = dict()
            for _key in t_test_metrics:
                test_metricspp['test_' + _key] = t_test_metrics[_key]
            return test_metricspp

        starting_time = time.time()
        PrintColors.prYellow(f'\n===== optimization attempt {index} . hypers: {args_hpo} =====')
        PrintColors.prGreen('----- in mode {} -----'.format(self.configurations.execution_mode))
        ''' ============ LOAD DATA ================================================================================ '''
        pretrained_tokenizer = PretrainedTransformerTokenizer(self.configurations.pretrained_model_name)
        pretrained_token_indexer = PretrainedTransformerIndexer(self.configurations.pretrained_model_name)
        gt_dataset_reader = StoryInstanceReader(
                word_indexer={'words': pretrained_token_indexer},
                include_scenario_phrase=self.configurations.scenario_phrase)
        mcval_reader = StoryInstanceReader(
            word_indexer={'words': pretrained_token_indexer},
            include_scenario_phrase=self.configurations.scenario_phrase)
        mcval_gt_data = mcval_reader.read(CONST.mcscript_val_1_dir)

        mctest_reader = StoryInstanceReader(
            word_indexer={'words': pretrained_token_indexer},
            include_scenario_phrase=self.configurations.scenario_phrase)
        mctest_gt_data = mctest_reader.read(CONST.mcscript_test_dir)

        if 'tacl' in self.configurations.regularity:
            mc1_marks_path = CONST.regularity_path['tacl']['mc1']
            mc2_marks_path = CONST.regularity_path['tacl']['mc2']
        else:
            mc1_marks_path = CONST.regularity_path['acl']['mc1']
            mc2_marks_path = CONST.regularity_path['acl']['mc2']

        mctest2_reader = StoryInstanceReader(
            word_indexer={'words': pretrained_token_indexer},
            include_scenario_phrase=self.configurations.scenario_phrase,
            regularity_prediction_folder=mc2_marks_path)
        mctest_pred_data = mctest2_reader.read(CONST.mcscript_test_dir)
        mc_val_pseudo_reader = StoryInstanceReader(
            word_indexer={'words': pretrained_token_indexer},
            include_scenario_phrase=self.configurations.scenario_phrase,
            regularity_prediction_folder=mc1_marks_path)
        mcval_pred_data = mc_val_pseudo_reader.read(CONST.mcscript_val_1_dir)

        mcval_gt_data_dict = {sce: [ins for ins in mcval_gt_data if ins.fields['scenario'].tokens[0].text == sce] for sce in CONST.mcval_scenarios}
        # mcval_pred_data_dict = {sce: [ins for ins in mcval_pred_data if ins.fields['scenario'].tokens[0].text == sce] for sce in CONST.mcval_scenarios}
        mctest_gt_data_dict = {sce: [ins for ins in mctest_gt_data if ins.fields['scenario'].tokens[0].text == sce] for sce in CONST.mctest_scenarios}
        mctest_pred_data_dict = {sce: [ins for ins in mctest_pred_data if ins.fields['scenario'].tokens[0].text == sce] for sce in CONST.mctest_scenarios}
        ins_gt_data = gt_dataset_reader.read(CONST.ins_original_dir)

        ins_train_dataset, ins_val_dataset, ins_test_dataset = gt_dataset_reader.split(
            data_instances=ins_gt_data,
            validation_scenario=self.configurations.validation_scenario,
            test_scenario=self.configurations.test_scenario
        )

        if 'tacl' not in self.configurations.regularity:
            ins_pseudo_marks_folder = CONST.regularity_path['acl']['ins']
        else:
            ins_pseudo_marks_folder = CONST.regularity_path['tacl']['ins']
        pseudo_dataset_reader = StoryInstanceReader(
            word_indexer={'words': pretrained_token_indexer},
            include_scenario_phrase=self.configurations.scenario_phrase,
            regularity_prediction_folder=ins_pseudo_marks_folder)
        ins_pseudo_data = pseudo_dataset_reader.read(CONST.ins_original_dir)
        ins_train_pred_dataset, ins_val_pred_dataset, ins_test_pred_dataset = pseudo_dataset_reader.split(
            data_instances=ins_pseudo_data,
            validation_scenario=self.configurations.validation_scenario,
            test_scenario=self.configurations.test_scenario
        )
        ''' acquire dicts to align candidates '''

        def _get_type_tables(_pseudo_instances, _gt_instances, _name='e'):
            tmp_p2g, tmp_g2p = dict(), dict()
            p_c2i, g_c2i = dict(), dict()
            if 'e' in _name:
                field_name = 'event_indices'
            else:
                field_name = 'participant_indices'
            gt_pointer = 0
            for instance in _gt_instances:
                for _ind in instance.fields[field_name]:
                    g_c2i[StoryInstanceReader.context(_instance=instance, index=_ind)] = gt_pointer
                    gt_pointer += 1
            pseudo_pointer = 0
            for instance in _pseudo_instances:
                for _ind in instance.fields[field_name]:
                    context = StoryInstanceReader.context(_instance=instance, index=_ind)
                    if context in g_c2i:
                        tmp_p2g[pseudo_pointer] = g_c2i[context]
                    pseudo_pointer += 1

            # tmp_g2p = {tmp_p2g[k]: k for k in tmp_p2g}
            return tmp_p2g

        def _get_index_mappings():
            _index_mappings = dict()
            _, _, _ = gt_dataset_reader.split(
                data_instances=ins_gt_data,
                validation_scenario=self.configurations.validation_scenario,
                test_scenario=self.configurations.test_scenario
            )
            for _scenario in CONST.scenario_s:
                pseudo_instances = pseudo_dataset_reader.data[_scenario]
                gt_instances = gt_dataset_reader.data[_scenario]
                _index_mappings[_scenario] = {'event': _get_type_tables(_pseudo_instances=pseudo_instances,
                                                                        _gt_instances=gt_instances,
                                                                        _name='e')}
                _index_mappings[_scenario]['participant'] = _get_type_tables(_pseudo_instances=pseudo_instances,
                                                                             _gt_instances=gt_instances,
                                                                             _name='p')
            ''' this generates a trivial identity mapping for the validation set'''
            for _scenario in mcval_gt_data_dict:
                pseudo_instances = mcval_gt_data_dict[_scenario]
                gt_instances = mcval_gt_data_dict[_scenario]
                _index_mappings[_scenario] = {'event': _get_type_tables(_pseudo_instances=pseudo_instances,
                                                                        _gt_instances=gt_instances,
                                                                        _name='e')}
                _index_mappings[_scenario]['participant'] = _get_type_tables(_pseudo_instances=pseudo_instances,
                                                                             _gt_instances=gt_instances,
                                                                             _name='p')
            for _scenario in mctest_gt_data_dict:
                pseudo_instances = mctest_pred_data_dict[_scenario]
                gt_instances = mctest_gt_data_dict[_scenario]
                _index_mappings[_scenario] = {'event': _get_type_tables(_pseudo_instances=pseudo_instances,
                                                                        _gt_instances=gt_instances,
                                                                        _name='e')}
                _index_mappings[_scenario]['participant'] = _get_type_tables(_pseudo_instances=pseudo_instances,
                                                                             _gt_instances=gt_instances,
                                                                             _name='p')
            return _index_mappings

        all_datasets = [ins_train_dataset, ins_test_pred_dataset, ins_val_dataset, ins_val_pred_dataset, ins_test_dataset, ins_test_pred_dataset,
                        mcval_gt_data, mcval_pred_data, mctest_gt_data, mctest_pred_data, ins_gt_data, ins_pseudo_data]
        all_data_instances = list()
        for dataset in all_datasets:
            all_data_instances += dataset

        for dataset in all_datasets:
            supply_token_indices(dataset, 'story', pretrained_tokenizer)

        vocabulary = Vocabulary.from_instances(all_data_instances)

        for dataset in all_datasets:
            dataset.index_with(vocabulary)
        # for instance in all_data_instances:
        #     for field in instance.fields.values():
        #         if type(field) == TextField:
        #             field.index(vocabulary)

        index_mappings = 0
        if self.configurations.evaluation_on == 'mc':
            if 'GT' in self.configurations.regularity:
                train_dataset = ins_gt_data
                test_dataset = mctest_gt_data_dict
            else:
                train_dataset = ins_pseudo_data
                test_dataset = mctest_pred_data_dict
                index_mappings = _get_index_mappings()
            gt_test_dataset = mctest_gt_data_dict
            gt_val_dataset = mcval_gt_data_dict
            val_dataset = mcval_gt_data_dict
        else:
            if 'GT' in self.configurations.regularity:
                train_dataset = ins_train_dataset
                val_dataset = ins_val_dataset
                test_dataset = ins_test_dataset
            else:
                train_dataset = ins_train_pred_dataset
                val_dataset = ins_val_pred_dataset
                test_dataset = ins_test_pred_dataset
                index_mappings = _get_index_mappings()
            gt_test_dataset = ins_test_dataset
            gt_val_dataset = ins_val_dataset

        ''' prepare serialization dir '''
        scenario_dir = os.path.join(*[CONST.serialization_dir, self.configurations.validation_scenario])
        trial_dir = os.path.join(scenario_dir, str(index))
        if not os.path.exists(scenario_dir):
            os.mkdir(scenario_dir)
        if os.path.exists(trial_dir):
            shutil.rmtree(trial_dir)
        os.mkdir(trial_dir)

        serialized_model_indices = list()

        ''' ============ TRAINING =========================================================================== '''
        model = self.model_class(args_hpo, vocabulary,
                                 configurations=self.configurations,
                                 index_mappings=index_mappings
                                 if 'GT' not in self.configurations.regularity else None)

        optimizer = optim.Adam(params=model.parameters(), lr=args_hpo.lr, weight_decay=args_hpo.l2)
        # track the metric that triggers early stopping
        main_metrics = collections.deque(maxlen=self.configurations.patience)
        # metric return to the meta-optimizer for concise logging
        return_metrics = collections.deque(maxlen=self.configurations.patience)
        print(f'--------- train epoch -1 ---------', end='\n')
        ''' validation, etc. '''
        # val_metrics = _get_evaluation_metrics(model, _inf_data=val_dataset, _gt_data=gt_val_dataset)
        _maybe_batch_ussp(model, _inf_data=val_dataset, _gt_data=gt_val_dataset)
        # print(str({k: val_metrics[k] for k in val_metrics if 'detail' not in k}))

        if self.configurations.max_epochs == -1:
            test_metrics = _get_evaluation_metrics(model, _inf_data=test_dataset, _gt_data=gt_test_dataset)
            _test_metrics = {f'val_{key}': test_metrics[key] for key in test_metrics}
            for name in self.metric_names:
                if name not in _test_metrics:
                    _test_metrics[name] = 9999.
            print(str({k: _test_metrics[k] for k in _test_metrics if 'detail' not in k}))
            return _test_metrics

        for epoch_number in range(self.configurations.max_epochs):
            print(f'\n--------- train epoch {epoch_number} ---------', end='\n')

            ''' train '''
            losses_np = self.train_epoch(model, train_dataset, optimizer=optimizer)
            print('train_losses: ' + str(losses_np))

            ''' validation. validates the model on both train and val sets '''
            print(f' ---- validation epoch {epoch_number} ---- ', end='\n')
            val_metrics = _get_evaluation_metrics(_model=model, _inf_data=val_dataset, _gt_data=gt_val_dataset)
            print('val_metrics: ' + str({k: val_metrics[k] for k in val_metrics if 'detail' not in k}))

            _tmp_train_metrics = list()
            for scenario in self.configurations.train_scenarios:
                gt_scenario_data = AllennlpDataset(gt_dataset_reader.data[scenario])
                if 'GT' not in self.configurations.regularity:
                    pseudo_scenario_data = AllennlpDataset(pseudo_dataset_reader.data[scenario])
                    scenario_metrics, _ = model.validate(
                        validation_data=pseudo_scenario_data, gt_validation_data=gt_scenario_data)
                else:
                    scenario_metrics, _ = model.validate(gt_scenario_data)
                _tmp_train_metrics.append(scenario_metrics)

            rich_return_metrics = dict()
            for key in val_metrics:
                rich_return_metrics['val_' + key] = val_metrics[key]
                rich_return_metrics['train_' + key] = \
                    sum([train_metric[key] for train_metric in _tmp_train_metrics if 'detail' not in key]) \
                    / len(self.configurations.train_scenarios)
            for key in losses_np:
                rich_return_metrics['train_' + key] = float(losses_np[key].detach().cpu().numpy())

            ''' early stopping '''
            # we use mean F1 of the best-performing clustering algorithm as the metric to track
            if 'p' in self.configurations.clustering_mode:
                main_metrics.append((val_metrics['event_macro_f1'] + val_metrics['participant_macro_f1'] +
                                     val_metrics['event_micro_f1'] + val_metrics['participant_micro_f1'])/4)
            else:
                main_metrics.append((val_metrics['event_macro_f1'] + val_metrics['event_micro_f1'])/2)
            return_metrics.append(rich_return_metrics)

            ''' serialization '''
            if len(serialized_model_indices) >= self.configurations.patience:
                os.remove(os.path.join(trial_dir, f'model_{serialized_model_indices[0]}'))
                serialized_model_indices.pop(0)
            torch.save(model.state_dict(), os.path.join(trial_dir, f'model_{epoch_number}'))
            serialized_model_indices.append(epoch_number)

            # trigger early stopping if main_metric have not increased in [patience] epochs
            if all([main_metrics[0] == max(main_metrics), len(main_metrics) == self.configurations.patience]):
                print(f'\n== early stopping triggered with mean F1 {max(main_metrics)} at epoch {epoch_number} ==')
                metrics = return_metrics[0]
                metrics.update(_terminate(epoch_number - self.configurations.patience + 1))
                ''' label best model '''
                return metrics

        # max_epoch reached
        print(' == max epoch reached, optimization terminated. ==')
        max_index = int(np.argmax(main_metrics))
        metrics = return_metrics[max_index]
        metrics.update(_terminate(max(self.configurations.max_epochs - self.configurations.patience + max_index, 0)))

        return metrics

# ''' ================================ ssp representation acquisition ============================== '''
        # mcscript_reader = StoryInstanceReader(
        #         word_indexer={'words': PretrainedTransformerIndexer(CONST.pretrained_model_name)},
        #         regularity_prediction_folder=os.path.join('.', 'pseudo_ext'))
        # ''' .read returns list of instances '''
        # mcscript_data = mcscript_reader.read(file_path=os.path.join('.', 'mcscript2_wordpiece_coref'))
        # if len(mcscript_reader.data) == 0:
        #     for scenario in os.listdir(os.path.join('.', 'mcscript2_wordpiece_coref')):
        #         mcscript_reader.data[scenario] = \
        #             [inst for inst in mcscript_data.instances if inst.fields['scenario'].tokens[0].text == scenario]
        # for mcscript_scenario in mcscript_reader.data:
        #     dill.dump(model.acquire_ssp_representations(
        #         scenario='haircut',
        #         data_instances=mcscript_reader.data[mcscript_scenario]),
        #         open(f'/local/fangzhou/ssp_pretrained_tensors/mcscript_pred_np/{mcscript_scenario}', 'wb'))
        #     print(mcscript_scenario)
        # ''' ============================================================================================== '''
