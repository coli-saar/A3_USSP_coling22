"""
misc utils to make a deep NLP life easier.
"""
import collections
import os
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.optimize


class EasyPlot:
    """
    a collection of functions that plots data and outputs them as .png files.
    """
    @staticmethod
    def plot_heatmap(tensor, y_names, x_names, out_name: str, title: str, show=False):
        """
        credits to Iza.
        :param tensor: numpy ndarray or torch tensor of shape n×m, where n can equal m, but not nesessarily
        :param y_names: list of str : names of tickmarks on y axis; should be in a desired order; the names
        will go from top to bottom along the y axis
        :param x_names: list of str : names of tickmarks on x axis; should be in a desired order; the names
        will go from top to bottom along the x axis
        :param out_name: name of the saved plot
        :param title:
        :param show:
        :return:
        """

        # convert to a numpy array and round the values to 2 decimals to be able to fit the cells
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.numpy()
        Cnp = np.around(tensor, decimals=2)

        # set up the plot: size and tick marks
        fig, ax = plt.subplots(figsize=(14, 14))  # in inches, ~*2 to get cm
        im = ax.imshow(tensor)
        ax.set_xticks(np.arange(len(x_names)))
        ax.set_yticks(np.arange(len(y_names)))

        # tick labels
        ax.set_xticklabels(x_names)
        ax.set_yticklabels(y_names)
        # tick labels: position and rotation for columns
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # iteratively insert the cell values into the plot; in the middle and in white
        for i in range(len(y_names)):
            for j in range(len(x_names)):
                ax.text(j, i, Cnp[i, j], ha="center", va="center", color="w")

        # add the title to the plot
        ax.set_title(title)
        # add a colorbar
        plt.colorbar(im)
        fig.tight_layout()

        if show:
            plt.show()

        # save the plot as .png, but other formats are available (e.g. .svg or .jpg)
        plt.savefig(out_name)

    @staticmethod
    def plot_line_graph(x, y, x_name, y_name, out_name: str, title: str = '', mode=None, show=False):
        plt.plot(x, y, mode) if mode else plt.plot(x, y)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        if len(title) > 0:
            plt.title(title)
        plt.savefig(f'{out_name}_.png', bbox_inches='tight')
        if show:
            plt.show()

    @staticmethod
    def plot_scatter_graph(x, y, x_name, y_name, out_name: str, title: str = '', color=None, size=None, show=False):
        plt.scatter(x=x, y=y, c=color, s=size)
        plt.xlabel(x_name)
        plt.ylabel(y_name)
        if len(title) > 0:
            plt.title(title)
        plt.savefig(f'{out_name}_.png', bbox_inches='tight')
        if show:
            plt.show()


class PrintColors:
    """
        add some color to your terminal!
    """

    @staticmethod
    def prRed(prt): print("\033[91m {}\033[00m".format(prt))

    @staticmethod
    def prGreen(prt): print("\033[92m {}\033[00m".format(prt))

    @staticmethod
    def prYellow(prt): print("\033[93m {}\033[00m".format(prt))

    @staticmethod
    def prLightPurple(prt): print("\033[94m {}\033[00m".format(prt))

    @staticmethod
    def prPurple(prt): print("\033[95m {}\033[00m".format(prt))

    @staticmethod
    def prCyan(prt): print("\033[96m {}\033[00m".format(prt))

    @staticmethod
    def prLightGray(prt): print("\033[97m {}\033[00m".format(prt))

    @staticmethod
    def prBlack(prt): print("\033[98m {}\033[00m".format(prt))


class QualityOfPythonLife:
    @staticmethod
    def batch_index_select(features, indices):
        # shape: batch, batch, dim
        selected_features_diag = features.index_select(
            dim=1,
            index=torch.tensor(indices, device=features.device))
        # shape: batch, dim
        selected_features = selected_features_diag.diagonal(dim1=0, dim2=1).transpose(0, 1)
        return selected_features

    @staticmethod
    def last_index_of(list_obj, element):
        """ poor python, why do I need to implement this """
        return len(list_obj) - list_obj[::-1].index(element) - 1

    @staticmethod
    def name_of(var):
        return f'{var=}'.split('=')[0]

    @staticmethod
    def preserve(f: float, n: int = 2):
        """ preserves n valid digits of float number f"""
        return float(('{:.' + str(n) + 'g}').format(f))

    @staticmethod
    def to_categorical(y, num_classes):
        """
        projects a number to one-hot encoding
        """
        return np.eye(num_classes, dtype='uint8')[y]


class Struct(dict):
    """
    extend a dictionary so we can access its keys as attributes. improves quality of life with auto-completion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            setattr(self, key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        setattr(self, key, value)


class TextTokenizer:
    """
    a text tokenizer that would come handy with pytorch. Probably not needed for allenNLP though.
    """

    def __init__(self, text, vocabulary_size=-1):
        """
        :param text: a corpus as a list of tokens
        """
        assert text
        self._encoder = dict()
        if vocabulary_size == -1:
            token_counter = collections.Counter(text).most_common(len(text))
        else:
            token_counter = collections.Counter(text).most_common(vocabulary_size)
        for type_, count in token_counter:
            self._encoder[type_] = len(self._encoder)
        self._decoder = dict(zip(self._encoder.values(), self._encoder.keys()))

    def encode(self, text):
        if type(text) == str:
            return torch.tensor(self._encoder[text])
        else:
            return torch.tensor([self._encoder[text[i]] for i in range(len(text))])

    def decode(self, index):
        if (type(index) == torch.Tensor and len(index.size()) == 0) or \
                (type(index) == np.ndarray and len(index.shape) == 0) or type(index) == int:
            return self._decoder[index]
        else:
            return [self._decoder[index[i]] for i in range(len(index))]

    def append_type(self, new_type):
        if new_type in self._encoder:
            PrintColors.prRed('warning: type {} already exists.'.format(new_type))
        else:
            index = len(self._encoder)
            self._encoder[new_type] = index
            self._decoder[index] = new_type

    def append_type_s(self, new_type_s: list):
        for new_type in new_type_s:
            self.append_type(new_type)

    @property
    def vocabulary_size(self):
        return len(self._encoder)


""" generic tools """


def cluster_result_to_acc(cluster_counts, n_gt_clusters):
    """
    cluster_counts: dict[cluster_index:int, dict[label:int, n_instances:int]]
        the keys go from 0 to n; the labels might be a random set.
    """
    c = np.zeros(shape=[len(cluster_counts), n_gt_clusters])
    cluster_indices = sorted(list(cluster_counts.keys()))
    gt_label_set = list()
    for key in cluster_counts:
        gt_label_set += list(cluster_counts[key].keys())
        gt_label_set = list(set(gt_label_set))
    for i, cluster_index in enumerate(cluster_indices):
        for j, label in enumerate(gt_label_set):
            if label not in cluster_counts[cluster_index]:
                c[i, j] = 0
            else:
                c[i, j] = cluster_counts[cluster_index][label]
    row, col = scipy.optimize.linear_sum_assignment(-c)
    maximum_correct = c[row, col].sum()
    cluster_sizes = [sum(list(cluster_counts[key].values())) for key in cluster_counts]
    total_cases = sum(cluster_sizes)
    acc = maximum_correct / total_cases

    ''' map predicted labels to the gt label set according to the best assignment. the assigned cluster indices is 
    always 0 to #indices - 1. the best assignment is row, col, with forall i: row[i] -> col[i] '''
    if n_gt_clusters >= len(cluster_indices):
        ''' every predicted label is assigned '''
        mapped_labels = np.zeros(shape=(n_gt_clusters,)) - 9999
        for index in range(col.shape[0]):
            if col[index] < len(gt_label_set):
                mapped_labels[row[index]] = gt_label_set[col[index]]
        ''' assign unmatched labels arbitrarily for completeness '''
        for i in range(n_gt_clusters):
            if mapped_labels[i] < 0:
                mapped_labels[i] = random.choice(gt_label_set)
    else:
        ''' every gt label is assigned to '''
        mapped_labels = np.zeros(shape=(len(cluster_indices),))
        for index in range(col.shape[0]):
            mapped_labels[cluster_indices[row[index]]] = gt_label_set[col[index]]

    return acc, col, mapped_labels


def cluster_file_to_metrics(path=os.path.join('.', 'aggl_per_cluster'), n_clusters=-1):
    """
    evaluate upperbound and lower bound for potential accuracy, given cluster results.
    """
    # key = cluster index; value = dict [label, #instances in cluster]
    cluster_counts = dict()
    with open(path, 'r') as cin:
        cluster_dict = None
        for line in cin:
            if '=' in line:
                # cluster starts
                splited_line = line.split()
                cluster_index = int(splited_line[2])
                cluster_dict = dict()
                cluster_counts[cluster_index] = cluster_dict
            else:
                # item line
                splited_line = line.split('\t')
                label = splited_line[1][7:]
                if label not in cluster_dict:
                    cluster_dict[label] = 0
                cluster_dict[label] += 1
    return cluster_result_to_acc(cluster_counts, n_clusters)[0]


def acquire_induced_metrics(predictions, gt_labels, n_gt_clusters, index_p2g):
    # cluster_counts: dict[cluster_index:int, dict[label:int, n_instances:int]]
    cluster_counts = dict()
    for _index in range(np.shape(predictions)[0]):
        prediction = int(predictions[_index])
        if index_p2g is not None:
            if _index in index_p2g:
                label = int(gt_labels[index_p2g[_index]])
            else:
                continue
        else:
            ''' assume trivial identical index_p2g '''
            label = int(gt_labels[_index])
        if prediction not in cluster_counts:
            cluster_counts[prediction] = dict()
        if label not in cluster_counts[prediction]:
            cluster_counts[prediction][label] = 0
        cluster_counts[prediction][label] += 1
    induced_acc, _, mapped_to_lables = cluster_result_to_acc(cluster_counts, n_gt_clusters)
    predictions_in_labels = [mapped_to_lables[int(predictions[i])] for i in range(predictions.shape[0])]
    macro_f1, micro_f1, detail_f1 = F1(np.array(predictions_in_labels), gt_labels, index_p2g)
    return{
        'acc': induced_acc,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'detail_f1': detail_f1
    }


def F1(predictions, gt_labels, index_mappings_p2g=None):
    # key=cluster index, value=dict{p: r: f: tp: tn: fn: fp:}
    by_cluster_metrics = dict()
    if index_mappings_p2g is None:
        index_mappings_p2g = {i: i for i in range(predictions.shape[0])}
    for i in range(predictions.shape[0]):
        if i not in index_mappings_p2g:
            prediction = int(predictions[i])
            if prediction not in by_cluster_metrics:
                by_cluster_metrics[prediction] = {
                    'truePositive': 0.,
                    'predicted': 1.,
                    'existed': 0.,
                }
            else:
                by_cluster_metrics[prediction]['predicted'] += 1
        else:
            prediction = int(predictions[i])
            gt = int(gt_labels[index_mappings_p2g[i]])
            for cluster_index in [gt, prediction]:
                if cluster_index not in by_cluster_metrics:
                    by_cluster_metrics[cluster_index] = {
                        'truePositive': 0.,
                        'predicted': 0.,
                        'existed': 0.,
                    }
            by_cluster_metrics[gt]['existed'] += 1
            by_cluster_metrics[prediction]['predicted'] += 1
            if gt == prediction:
                by_cluster_metrics[gt]['truePositive'] += 1

    for cluster_index in by_cluster_metrics:
        cluster_metrics = by_cluster_metrics[cluster_index]
        if cluster_metrics['predicted'] > 0.:
            cluster_metrics['precision'] = cluster_metrics['truePositive'] / cluster_metrics['predicted']
        else:
            cluster_metrics['precision'] = 0
        if cluster_metrics['existed'] > 0.:
            cluster_metrics['recall'] = cluster_metrics['truePositive'] / cluster_metrics['existed']
        else:
            cluster_metrics['recall'] = 0.
        if cluster_metrics['precision'] + cluster_metrics['recall'] > 0.:
            cluster_metrics['f1'] = 2 * cluster_metrics['precision'] * cluster_metrics['recall'] / \
                                    (cluster_metrics['precision'] + cluster_metrics['recall'])
        else:
            cluster_metrics['f1'] = 0.

    macro_f1 = sum([by_cluster_metrics[i]['f1'] for i in by_cluster_metrics]) / len(by_cluster_metrics)
    micro_f1 = sum([by_cluster_metrics[i]['f1'] * by_cluster_metrics[i]['existed'] for i in by_cluster_metrics]) \
        / sum([(by_cluster_metrics[i]['existed']) for i in by_cluster_metrics])
    return macro_f1, micro_f1, by_cluster_metrics


def float_in_double_evaluate_float_out(function):
    """
    convert the potentially float tensors into double tensors for computation, then convert them back to float. Also
    convert all double tensors in the output to float tensors. Used as a decorator to couple with numerical stability
    issues.

    note: this function packs solutions to many traps, DO NOT MODIFY

    the output type must be Dict[str, torch.Tensor] or torch.Tensor. Otherwise, it remains intact
    """
    def wrapped_module(*args, **kwargs):
        float_tensors = list()
        for arg in list(args) + list(kwargs.values()):
            if type(arg) == torch.Tensor:
                if arg.dtype == torch.float:
                    float_tensors.append(arg)
        dargs = list()
        dkwargs = dict()
        for arg in list(args):
            if arg in float_tensors:
                dargs.append(arg.to(torch.double))
            else:
                dargs.append(arg)
        for key, kwarg in kwargs.items():
            try:
                s = kwarg in float_tensors
                if s:
                    dkwargs[key] = kwarg.to(torch.double)
                else:
                    dkwargs[key] = kwarg
            except RuntimeError:
                dkwargs[key] = kwarg
        output = function(*dargs, **dkwargs)
        if type(output) == torch.Tensor:
            if output.dtype == torch.double:
                output = output.to(torch.float)
        elif type(output) == dict:
            for key, tensor in output.items():
                if tensor.dtype == torch.double:
                    output[key] = tensor.to(torch.float)
        return output
    return wrapped_module


def fleiss_kappa(table):
    """
    fleiss kappa, see:
    http://en.wikipedia.org/wiki/Fleiss%27_kappa
    :param table: array-like, 2D
        t[i,j]: # rates assigning category j to subject i
    :return:
    """
    table = 1.0 * np.asarray(table)
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()

    # marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.))

    # annotation agreement
    p_mean = p_rat.mean()

    # random agreement
    p_e = (p_cat * p_cat).sum()

    kappa = (p_mean - p_e) / (1 - p_e)
    return kappa


def latex_table_of_csv(csv_file: str, output_file: str = 'latex_out', spliter='\t', alignment: str = ''):
    """
    generate latex codes that compiles a table from a csv file
    :param csv_file:
    :param output_file:
    :param spliter:
    :param alignment:
    :return:
    """
    splited_line_s = list()
    with open(csv_file, 'r') as insteam:
        for line in insteam:
            splited_line_s.append(line.split(spliter))
    _alignment = ''.join(['c']*len(splited_line_s[0])) if alignment == '' else alignment
    output = '\\begin{table}[htbp]\n\\begin{center}\n\\begin{tabular}' \
             + '{' + _alignment + '}' + '\\hline\n'
    for splited_line in splited_line_s:
        output += ' & '.join(splited_line) + '\\\\\n'
    output += '\\hline\n\\end{tabular}\n\\\\\\noindent *:say something sweetie\n' + \
              '\\caption{moin}\\label{yo}\n\\end{center}\n\\end{table}\n'
    output = output.replace('_', '\\_')
    with open(os.path.join('.', output_file), 'w') as outsteam:
        outsteam.write(output)
    print(output)
    return output


def pad_sequence(sequence, target_length: int, padding_token: str):
    return list(sequence) + [padding_token] * (target_length - len(sequence))


if __name__ == '__main__':
    u = cluster_file_to_metrics()
    pass
