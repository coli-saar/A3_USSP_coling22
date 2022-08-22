"""
utils to facilitate AllenNLP / pytprch based projects
this file collects stuff written for allenNLP >= v1.0
"""
from typing import Optional, Union, Tuple, Dict, List, Any

import copy
import math
import numpy as np

import torch
import torch.nn.functional as F
from allennlp.training.metrics import Metric, F1Measure


def cluster_distances(representations: Dict[str, torch.Tensor], algorithm: str, lb_coef=0., ub_coef=0.):
    """
    evaluate distance score within of a batch of tensors.

    algorithms:
        l2: naive l2 distance
        cosine: l2 distance of normalized vectors. this equals to lambda x lambda y 2-2cos(x,y)
    params:
        representations: Dict[str, torch.FloatTensor (num_tensors * dim)]
    returns:
        intra_cluster_dist, inter_cluster_dist
    """

    def _bounded_intra_sum(_label_vectors):
        """
        compute the bounded average of intra cluster distances. i.e. distances less than
            lb_coef * average_distance
        will not be taken into account.

        returns: sum_distances, n_valid_distances
        """
        _distances = torch.cdist(_label_vectors, _label_vectors, p=2)
        _n_vectors = _label_vectors.size()[0]
        _lower_bound = torch.mean(_distances) * lb_coef
        _mask = _distances >= _lower_bound
        n_valid_pairs = torch.sum(_mask.to(torch.int))
        bounded_sum = torch.sum(_distances * _mask)
        return bounded_sum, n_valid_pairs

    def _bounded_inter_sum(_label_vectors, _diff_vectors):
        """
        compute the bounded average of inter cluster distances. i.e. distances more than
            max_distance - lu_coef * (max_distance - average_distance)
        will not be taken into account.

        returns: sum_distances, n_valid_distances
        """
        _distances = torch.cdist(_label_vectors, _diff_vectors, p=2)
        _n_label_vectors = _label_vectors.size()[0]
        _n_diff_vectors = _diff_vectors.size()[0]
        _max_distance, _mean_distance = torch.max(_distances), torch.mean(_distances)
        _upper_bound = _max_distance - ub_coef * (_max_distance - _mean_distance)
        _mask = _distances <= _upper_bound
        n_valid_pairs = torch.sum(_mask.to(torch.int))
        bounded_sum = torch.sum(_distances * _mask)
        return bounded_sum, n_valid_pairs

    # todo: continue here. test before proceeding

    if len(representations) == 0:
        return 0, 0

    distance_inter, distance_intra = 0, 0
    intra_count, inter_count = 0, 0
    for label in representations:
        # shape: n_vectors * dim
        label_vectors = representations[label]
        diff_vectors = torch.cat([representations[diff_label] for diff_label in representations if diff_label != label]) \
            if len(representations) > 1 else 0
        if algorithm == 'cosine':
            label_vectors = torch.nn.functional.normalize(label_vectors, dim=-1)
            if len(representations) > 1:
                diff_vectors = torch.nn.functional.normalize(diff_vectors, dim=-1)
        # n_vector_of_label = label_vectors.size()[0]
        # n_vectors_diff_label = diff_vectors.size()[0]

        label_vectors = label_vectors.unsqueeze(0)
        sum_dist_intra, n_dist_intra = _bounded_intra_sum(label_vectors)
        distance_intra += sum_dist_intra
        intra_count += n_dist_intra
        if len(representations) > 1:
            diff_vectors = diff_vectors.unsqueeze(0)
            sum_dist_inter, n_dist_inter = _bounded_inter_sum(_label_vectors=label_vectors,
                                                              _diff_vectors=diff_vectors)
            distance_inter += sum_dist_inter
            inter_count += n_dist_inter

    dist_inter = 0 if len(representations) <= 1 else distance_inter / inter_count

    return distance_intra / intra_count, dist_inter


def cosine_positional_encoding(proportion):
    """
    concatenated after vectors to mark positional differences wrt cosine distance
    """
    angle = proportion.unsqueeze(-1) * (math.pi / 2)
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)


def model_fusion(models: list, weights: list):
    """
    evaluate the weighted average of a list of models. light-weight ensemble.
    """
    if weights is None:
        weights = [1./len(models)] * len(models)

    weights = [w/sum(weights) for w in weights]

    output_model = copy.deepcopy(models[0])
    for key in output_model.state_dict():
        output_model[key] = sum([models[i].state_dict()[key] * weights[i] for i in range(len(models))])

    return output_model

def pairwise_cosine_similarity(x: torch.Tensor, y: torch.Tensor):
    """
    evaluates pair wise cosine similarities between each pair in x X y
    x: n1 * d
    y: n2 * d
    """
    xn = F.normalize(x, dim=-1)
    yn = F.normalize(y, dim=-1)
    return torch.matmul(xn, torch.transpose(yn, 0, 1))


def supply_token_indices(instances, text_field_name: str, pretrained_tokenizer):
    """
    attach text_id s to text_field tokens to patch the behavior of allenNLP's pretrained transformer token indexers
    :param instances:
    :param text_field_name:
    :param pretrained_tokenizer:
    :return:
    """
    for instance in instances:
        for token in instance.fields[text_field_name]:
            token.text_id = pretrained_tokenizer.tokenizer.convert_tokens_to_ids(token.text)
            token.type_id = 0


def to_categorical(y: torch.Tensor, num_classes) -> torch.Tensor:
    if y.dim() == 1:
        return torch.eye(num_classes, dtype=torch.long, device=y.device)[y]
    elif y.dim() == 2:
        b, le = y.size()
        r = y.new_zeros(size=[b, le, num_classes])
        for i in range(b):
            r[i] = to_categorical(y[i], num_classes)
        return r
    else:
        raise IndexError('This function only processes input with dimensions 1 or 2.')


class AverageF1(Metric):
    """
    track F1 by classes to allow evaluation of micro (by instance averaged) and macro (by class averaged) F1.
    """
    def __init__(self, n_clusters, valid_classes: List = None):
        """
        :param n_clusters:
        :param valid_classes: the classes whose F1 should be tracked. If unspecified, all classes are tracked.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.instance_counts = dict()
        self.by_class_F1 = dict()
        self.valid_classes = valid_classes or list(range(self.n_clusters))
        for class_label in self.valid_classes:
            self.by_class_F1[class_label] = F1Measure(positive_label=class_label)

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]):
        predictions, gold_labels, mask = predictions.detach(), gold_labels.detach(), mask.detach()
        for class_label in self.valid_classes:
            count = int(torch.sum((gold_labels == class_label) * mask, dim=list(range(gold_labels.dim())))
                        .detach().cpu())
            if class_label not in self.instance_counts:
                self.instance_counts[class_label] = 0
            self.instance_counts[class_label] += count
            self.by_class_F1[class_label](predictions, gold_labels, mask)

    def reset(self) -> None:
        self.instance_counts = dict()
        self.by_class_F1 = dict()
        for class_label in self.valid_classes:
            self.by_class_F1[class_label] = F1Measure(positive_label=class_label)

    def get_metric(self, reset: bool) -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        macro_F1, micro_F1 = self.get_customize_metric(valid_classes=self.valid_classes)
        if reset:
            self.reset()
        return macro_F1, micro_F1

    def get_customize_metric(self, valid_classes: List):
        cumulated_micro_F1 = 0.
        cumulated_macro_F1 = 0.
        n_valid_classes = 0.
        for class_label in valid_classes:
            if self.instance_counts[class_label] > 0:
                p, r, f1 = self.by_class_F1[class_label].get_metric(reset=False)
                cumulated_micro_F1 += f1 * self.instance_counts[class_label]
                cumulated_macro_F1 += f1
                n_valid_classes += 1
        n_valid_instances = sum([self.instance_counts[class_label] for class_label in valid_classes])
        micro_F1 = \
            cumulated_micro_F1 / n_valid_instances if n_valid_instances > 0. else 0.0
        macro_F1 = cumulated_macro_F1 / n_valid_classes if n_valid_classes > 0. else 0.0
        return macro_F1, micro_F1


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    def _forward_unimplemented(self, *_input: Any) -> None:
        pass
