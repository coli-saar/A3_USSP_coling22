import dill
import os
import datetime
import random

import numpy as np

from misc import Struct, PrintColors


class RandomSearchMetaOptimizer:
    """
    to perform random hyper-parameter search
    usage:
        inherit the class and override self.train()
        call self.search to perform the search and log results
    """

    def __init__(self, parameters: dict, num_trials: int, tag: str, metric_names: list):
        """
        generates parameter combinations for random parameter search, stored in self.combinations as dictionaries
        all sampling parameters are dictionaries formed as dict{hyper_name: value}

        :param parameters:
            the parameters involved in hyper parameter search. a dict of dicts. The first hierarchy of keys are the
            parameter names; the second hierachy should include the following:
            'domain': the range of the parameter. should be a tuple.
                Note: for exponential sampling, the upperbound of the domain does not get sampled.
            'sample_criterion':
                'u': uniform over the domain
                '2e': the parameter's logrithm wrt 2 is sampled uniformly as an INTEGER
                '10e': the parameter's logrithm is sampled uniformly as a FLOAT
                'categorical': choose one from a given discret domain
            'type':
                'int': floor and returns an integer
                'float': float, direct copy
                'str': string
        :param metric_names:
            the name of the metrics returned by .train() that should be logged. if allennlp is used, the metric names
            are usually expected from trainer.train(), i.e. trainer.metrics.
        :param num_trials:
        :param tag:
        """
        self.hyper_combs = [Struct() for _ in range(num_trials)]
        self.num_trials = num_trials
        self.tag = tag
        self.log_path = f"logs_{self.tag}_{datetime.date.today()}.csv"
        self.combs_path = "hyper_combs_{}".format(self.tag)
        self.parameters = parameters
        self.hyper_names = [hyper for hyper in self.parameters]
        self.metric_names = metric_names

        ''' generate parameters '''
        for i, combination in enumerate(self.hyper_combs):
            for hyper in self.parameters:
                # return if only a single quantity is given
                if not isinstance(self.parameters[hyper]['domain'], list):
                    assert isinstance(self.parameters[hyper]['domain'], int) \
                           or isinstance(self.parameters[hyper]['domain'], float)
                    combination[hyper] = self.parameters[hyper]['domain']
                    continue
                elif self.parameters[hyper]['sample_criterion'] == 'categorical':
                    combination[hyper] = random.choice(self.parameters[hyper]['domain'])
                    continue
                else:
                    min_value, max_value = self.parameters[hyper]['domain']
                    assert min_value <= max_value
                rnd_ready = None
                if self.parameters[hyper]['sample_criterion'] == '2e':
                    assert min_value > 0
                    min_exp, max_exp = np.log2(min_value), np.log2(max_value)
                    rnd = np.random.uniform() * (max_exp - min_exp) + min_exp
                    rnd_ready = np.power(2., np.floor(rnd))
                elif self.parameters[hyper]['sample_criterion'] == '10e':
                    assert min_value > 0
                    min_exp, max_exp = np.log10(min_value), np.log10(max_value)
                    rnd = np.random.uniform() * (max_exp - min_exp) + min_exp
                    rnd_ready = np.power(10., rnd)
                elif self.parameters[hyper]['sample_criterion'] == 'u':
                    rnd_ready = np.random.uniform() * (max_value - min_value) + min_value

                if self.parameters[hyper]['type'] == 'int':
                    combination[hyper] = int(rnd_ready)
                elif self.parameters[hyper]['type'] == 'float':
                    combination[hyper] = rnd_ready

        ''' initialize log if applicable '''
        if not os.path.exists(self.log_path):
            header = 'index' + ',' + ','.join(self.hyper_names) + ',' + ','.join(self.metric_names)
            with open(self.log_path, 'a') as log:
                log.write(header + '\n')

        ''' save combinations '''
        dill.dump(self.hyper_combs, open(self.combs_path, 'wb'))

    def _hyper_comb_summary(self, hyper_comb):
        items = list()
        for name in self.hyper_names:
            if type(hyper_comb[name]) != str:
                items.append('{:.3g}'.format(hyper_comb[name]))
            else:
                items.append(hyper_comb[name])
        return ','.join(items)

    def _perform_search(self, hyper_comb, execution_idx):
        print('------ Random Hyper Search Round {} ------'.format(execution_idx))
        metrics = self.train(hyper_comb, execution_idx)
        log_line = str(execution_idx) + ',' + self._hyper_comb_summary(hyper_comb) + ',' + \
            ','.join(['{:.3g}'.format(metrics[name]) if type(metrics[name]) is float else str(metrics[name])
                      for name in self.metric_names])
        with open(self.log_path, 'a') as log_out:
            log_out.write(log_line + '\n')

    def search(self, test_mode=True):
        """
        main entrance, execute to perform the optimization and log the parameters.
        :param test_mode:
        :return:
        """
        PrintColors.prRed(f'======Performing Random Hyper Search for execution {self.tag}======')
        for execution_idx, hyper_comb in enumerate(self.hyper_combs):
            # self._perform_search(hyper_comb, execution_idx)
            if test_mode:
                self._perform_search(hyper_comb, execution_idx)
            else:
                try:
                    self._perform_search(hyper_comb, execution_idx)
                except RuntimeError as rte:
                    PrintColors.prPurple(rte)
                    continue

    def train(self, combination, index):
        """
        override to execute one round of random hyper-parameter search, and returns the metrics that needs to be logged.

        :param combination: hyper parameter combination
        :param index
        :return: metrics for evaluation as a dictionary
        """
        raise NotImplementedError

    @staticmethod
    def best_lines(indices: list, folder: str, separator=',', k=3, q=1, extra='', output_file_prefix='summary', line_index_range=None):
        """
        pick the best lines from a set of execution results.
        """
        best_lines = list()
        summary_path = os.path.join('.', f'{output_file_prefix}.csv')
        if os.path.exists(summary_path):
            os.remove(summary_path)
        files = sorted([f for f in list(os.listdir(folder)) if '.csv' in f and output_file_prefix not in f and extra in f])
        results = {0: list(), 1: list(), 2: list(), 3: list()}
        for file_idx, file in enumerate(files):
            idx_range = line_index_range if line_index_range else list(range(len(files)))
            full_path = os.path.join(folder, file)
            max_quantity = -1
            index = -1
            lines = list(open(full_path, 'r'))  # [:21]
            if file_idx == 0:
                best_lines.append(lines[0])
            for line_ind, line in enumerate(lines):
                if line_ind == 0 or line_ind not in idx_range:
                    continue
                splited_line = line.split(separator)
                quantity = sum([float(splited_line[i]) for i in indices])
                if quantity > max_quantity:
                    max_quantity = quantity
                    index = line_ind
            best_lines.append(lines[index])

            lines = sorted(lines[1:], key=lambda x: - sum([float(x.split(separator)[ind]) for ind in indices]))
            results[0].extend([float(line.split(separator)[indices[0] + q]) for line in lines[:k]])
            results[1].extend([float(line.split(separator)[indices[0] + 3 + q]) for line in lines[:k]])
            results[2].extend([float(line.split(separator)[indices[1] + q]) for line in lines[:k]])
            results[3].extend([float(line.split(separator)[indices[1] + 3 + q]) for line in lines[:k]])

        avgs = [sum([float(best_lines[i].split(separator)[j]) for i in range(len(best_lines)) if i > 0]) / len(files) for j in range(len(best_lines[0].split(separator)))]
        best_lines.append('\n')
        best_lines.append(separator.join([str(avg) for avg in avgs]))

        with open(summary_path, 'w') as fout:
            for line in best_lines:
                fout.write(line)

        def mean_std():
            ss = list()
            for qi in range(4):
                q = results[qi]
                var = 0
                for i0 in range(0, len(q), k):
                    var += np.var(q[i0: i0+k]) / 10
                ss.append([np.mean(q), np.sqrt(var)])
            st = ''
            for am in ss:
                st += format(float(am[0]) * 100, '.2f') +' \pm ' + format(float(am[1]) * 100, '.1f')
            return ss

        print('+')

        # print(f'{folder}:  {mean_std()}')


if __name__ == '__main__':
    # #  RandomSearchMetaOptimizer.best_lines(indices=[15, 24], folder='.')  # os.path.join('.', 'merged'))
    # s = {# '/local/fangzhou/exe/_5_pPcd_pred': [19, 28], '/local/fangzhou/exe/_5_ppcd_pred_3': [18, 27],
    #      # ppcd_pred
    #     # '/local/fangzhou/unsupervised_script_parsing/merged': [18, 27, 0],
    #     # '/local/fangzhou/exe/r_3_ppcd_pred_3': [18, 27],
    #     # '/local/fangzhou/unsupervised_script_parsing/exp_logs_fin/ppcd_pred': [19, 28],
    #     # '/local/fangzhou/unsupervised_script_parsing/exp_logs_fin/ppcd_pred_r2': [16, 25],
    #
    #     # '/local/fangzhou/unsupervised_script_parsing/exp_logs_fin/pPc_pred':[18,   27],
    #     # '/local/fangzhou/unsupervised_script_parsing/exp_logs_fin/ppc_pred_r2': [15, 24],
    #     '/local/fangzhou/exe/_1_ppc_pred_3': [17, 26, 1],
    #     '/local/fangzhou/exe/pp_pred': [22, 31, 1],
    #     '/local/fangzhou/unsupervised_script_parsing/exp_logs_fin/p_pred': [18, 27, 1],
    #
    #
    #
    #     '/local/fangzhou/unsupervised_script_parsing/exp_logs_fin/p_gt': [18, 27, 1],
    #
    #     '/local/fangzhou/unsupervised_script_parsing/exp_logs_fin/pP_gt': [18, 27, 1],
    #
    #     '/local/fangzhou/unsupervised_script_parsing/exp_logs_fin/ppc_gt_r2': [15, 24, 1],
    #
    #     '/local/fangzhou/unsupervised_script_parsing/exp_logs_fin/ppcd_gt_r2': [16, 25, 0],
    #
    # }
    #
    # indv = [20, 29, 0]  #  [19, 28, 1]
    # ind_reg_id = [27, 28, 0]
    #
    # ssm = {
    #     '/local/fangzhou/exe/1_full_cosine_pred': [19, 28, 0],
    #     '/local/fangzhou/exe/3_full_cosine_u_pred': [19, 28, 0],
    #     '/local/fangzhou/exe/5_full_cosine_u_pred_average': [19, 28, 0],
    #     '/local/fangzhou/exe/6_ppcdc_u_pred_average_corefinf': [19, 28, 0],
    #     '/local/fangzhou/exe/2_full_cosine_gt': [19, 28, 0],
    #     '/local/fangzhou/exe/4_full_cosine_u_gt': [19, 28, 0],
    #
    # }
    #
    # ss = {
    #
    #     '/local/fangzhou/exe/1_pp_d_pred': indv,
    #     '/local/fangzhou/exe/5_pp_d_coref_pred': indv,
    #     '/local/fangzhou/exe/2_pp_d_cosine_pred': indv,
    #     '/local/fangzhou/exe/7_pp_d_cosine_coref_pred': indv,
    #
    #     '/local/fangzhou/exe/4_pp_d_gt': indv,
    #     '/local/fangzhou/exe/6_pp_d_coref_gt': indv,
    #     '/local/fangzhou/exe/3_pp_d_cosine_gt': indv,
    #     '/local/fangzhou/exe/0_pp_d_cosine_coref_gt': indv,
    #
    # }
    #
    #
    # reg_id = {
    #     '/local/fangzhou/exe/0_reg_id_r1': ind_reg_id,
    #     '/local/fangzhou/exe/1_reg_id_rfull': ind_reg_id,
    #     '/local/fangzhou/exe/2_reg_id_rrandom': ind_reg_id,
    #     '/local/fangzhou/exe/3_reg_id_noda': ind_reg_id,
    # }
    #
    #
    #
    te = [20, 29]

    RandomSearchMetaOptimizer.best_lines(
            indices=te,
            folder='/local/fangzhou/mc_script_fin/logs2/ins/ev_pred/coref_cond',
            k=1,
            q=0,
            output_file_prefix='evpredCC',
        )

    line_ind_range={
        'coref_cond': [1, 2, 3],
        'coref': [4, 5, 6],
        'cond': [7, 8, 9]
    }
    for folder in os.listdir('/local/fangzhou/mc_script_fin/logs'):
        if 'mc' not in folder:
            for model in os.listdir(f'/local/fangzhou/mc_script_fin/logs/{folder}'):
                RandomSearchMetaOptimizer.best_lines(
                    indices=te,
                    folder=f'/local/fangzhou/mc_script_fin/logs/{folder}/{model}',
                    k=1,
                    q=0,
                    output_file_prefix=f'{folder}_{model}'
                )
            print(f'ins_{folder}')
        else:
            for model in line_ind_range:
                RandomSearchMetaOptimizer.best_lines(
                    indices=te,
                    folder=f'/local/fangzhou/mc_script_fin/logs/{folder}',
                    k=1,
                    q=0,
                    output_file_prefix=f'{folder}_{model}',
                    line_index_range=line_ind_range[model]
                )
            print(f'{folder}')

    # for folder in dic:
    #     RandomSearchMetaOptimizer.best_lines(indices=dic[folder][:2], folder=folder, k=5, q=dic[folder][-1], extra='reg_id')  #  os.path.join('.', 'merged'))

