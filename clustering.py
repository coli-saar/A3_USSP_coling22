"""
generate clusters and evaluates them. wrapped in a customized allennlp metric

Zhai Fangzhou, 2020.8.18
"""
from typing import Union, Tuple, Dict, List, Optional

import numpy as np

import torch
from allennlp.training.metrics import Metric
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


class Clustering:
    metrics = {'ARI': adjusted_rand_score}

    def __init__(self, data, gold_labels, n_clusters=10):
        self.data = data.detach().cpu().numpy()
        self.predictions = dict()
        self.gold_labels = gold_labels.detach().cpu().numpy()
        self.n_clusters = n_clusters
        self.clustering_results = dict()

    def _cluster(self, algorithm):
        """ cluster with the specified algorithm """
        if algorithm == 'kmeans':
            estimator = KMeans(n_clusters=self.n_clusters)
            # data: n_samples * n_features
            self.predictions['kmeans_{}'.format(self.n_clusters)] = estimator.fit_predict(self.data)

    def evaluate_metric(self, algorithm: str, metric: str, n_clusters: int = 10):
        if '{}_{}'.format(algorithm, n_clusters) in self.predictions:
            return Clustering.metrics[metric](self.predictions['{}_{}'.format(algorithm, n_clusters)],
                                              self.gold_labels)


class ARI(Metric):
    def __init__(self, n_total_inst, tag, n_clusters=10):
        self.n_total_inst = n_total_inst
        self.n_clusters = n_clusters
        self.predictions = 0
        self.gold_labels = 0
        self.tag = tag

    def __call__(self, predictions: torch.Tensor, gold_labels: torch.Tensor, mask: Optional[torch.BoolTensor]):
        """
        We do not want ARI computed in each batch, as batch size needs to be less than 32. So here's how it works:
            1. __call__() only records the vectors and gold labels
            2. get_metric() performs clustering on epoch end. Otherwise, it returns 0. to save time. We identify
                the epoch ending batch with reset
        """
        np_predictions = predictions.detach().cpu().numpy()
        np_gold_labels = gold_labels.detach().cpu().numpy().reshape([-1, ])
        if type(self.predictions) is int:
            self.predictions = np_predictions
        else:
            self.predictions = np.concatenate((self.predictions, np_predictions), axis=0)
        if type(self.gold_labels) is int:
            self.gold_labels = np_gold_labels
        else:
            self.gold_labels = np.concatenate((self.gold_labels, np_gold_labels), axis=0)

    def get_metric(self, reset: bool = False, force_evaluate: bool = False) \
            -> Union[float, Tuple[float, ...], Dict[str, float], Dict[str, List[float]]]:
        if reset:  # np.shape(self.gold_labels)[0] == self.n_total_inst or force_evaluate:
            # epoch ending batch, evaluate
            cls_predictions = KMeans(n_clusters=self.n_clusters).fit_predict(self.predictions)
            ari = adjusted_rand_score(cls_predictions, self.gold_labels)
            result = ari
            self.reset()
        else:
            # non-epoch ending batch, skip
            result = 0

        return result

    def reset(self) -> None:
        self.predictions = 0
        self.gold_labels = 0
