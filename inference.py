"""
perform inference and analysis of the results
"""
import enum
import os


from allennlp.data.token_indexers.pretrained_transformer_indexer import PretrainedTransformerIndexer

from data import InScriptInstanceReader
from model import UnsupervisedScriptParser
from global_constants import CONST

execution_mode = 'optimization'
inf_data_dir = os.path.join('.', 'data_test') if execution_mode == 'optimization' else os.path.join('.', 'toy')

''' =============== checkpoint details ================= '''
'''-- test checkpoint ------------------------------------'''
test_folder = os.path.join(*['..', 'USSP_checkpoints', 'checkpoint_test'])


class TestCheckpointExecutionSettings:

    class ExecutionMode(enum.Enum):
        OPTIMIZATION = 'optimization'
        TEST = 'test'

    ' == high-level settings ============================='
    execution_mode = ExecutionMode.TEST
    clustering_mode = ['events', 'participants']
    tag = 'test'

    ' == parameters ======================================'
    validation_scenario = 'grocery'
    data_folder = os.path.join('.', 'clean_data_xlnet_000011111') \
        if execution_mode == ExecutionMode.OPTIMIZATION else os.path.join('.', 'toy')

    # train_data_dir = os.path.join('.', 'data_train') if execution_mode == ExecutionMode.OPTIMIZATION \
    #     else os.path.join('.', 'toy')
    # val_data_dir = os.path.join('.', 'data_train') if execution_mode == ExecutionMode.OPTIMIZATION \
    #     else os.path.join('.', 'toy')

    encoder_hidden_size = 768
    device = 5
    num_trials = 1
    patience = 5
    max_epochs = 1

    param_domains = {'batch_size': [16, 31] if execution_mode == ExecutionMode.OPTIMIZATION else 17,
                     'lr': [1e-4, 2e-3],
                     'l2': [1e-4, 5e-2],
                     'clip': [1, 10],
                     'gamma_inter': [1e-2, 1e2],
                     'gamma_intro': [1e-2, 1e2]}


check_point_test = {
    'combination_file': os.path.join(test_folder, 'hyper_combs_test'),
    'index': 0,
    'vocab_folder': os.path.join(test_folder, 'vocab_test_0'),
    'model_path': os.path.join(test_folder, 'best.th'),
    'configurations': TestCheckpointExecutionSettings
}


if __name__ == '__main__':
    checkpoint = check_point_test
    configurations = check_point_test['configurations']

    dataset_reader = InScriptInstanceReader(
        mode=configurations.clustering_mode,
        word_indexer={'words': PretrainedTransformerIndexer(CONST.pretrained_model_name)})
    # data = dataset_reader.read(configurations.data_folder)
    model = UnsupervisedScriptParser.from_checkpoint(check_point_test, 0)
    model.inference_scenario(scenario='grocery', dataset_reader=dataset_reader)
