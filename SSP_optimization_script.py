"""
configurations different in each execution.
No, I'm NOT a fan of config files.
"""
import enum
import os


class ExecutionSettings:

    class ExecutionMode(enum.Enum):
        OPTIMIZATION = 'optimization'
        TEST = 'test'

    execution_mode = ExecutionMode.OPTIMIZATION
    validation_scenario = 'bath'
    test_scenario = 'bicycle'

    select_index = True

    ''' one of {'none', 'lstm', 'crf', 'lstm-crf', 'transformer'}, specifies the tagging layer.'''
    tagger_type = 'lstm'

    ''' subset of {inscript, descript, backtranslation}. the list should match the data '''
    corpora = ['inscript']

    ''' a non-empty subset of {'events', 'participants'}, indicating which instances will be loaded. '''''
    clustering_mode = ['events', 'participants']

    ''' one of 'normal', 'regular_only', 'regular_identification', 'regular_only_for_ussp'
    check the doc of InScriptSequenceLabeling Reader '''
    loading_mode = 'regular_only_for_ussp'
    train_with_BT_instances = True
    predict_BT_instances = True

    tag = '_val:' + validation_scenario

    train_data_dir = os.path.join('.', 'data_train') if execution_mode == ExecutionMode.OPTIMIZATION \
        else os.path.join('.', 'toy')
    val_data_dir = os.path.join('.', 'data_val') if execution_mode == ExecutionMode.OPTIMIZATION \
        else os.path.join('.', 'toy')
    test_data_dir = os.path.join('.', 'data_test') if execution_mode == ExecutionMode.OPTIMIZATION \
        else os.path.join('.', 'toy')

    pretrained_model_name = 'xlnet-base-cased'
    encoder_hidden_size = 768

    freeze = True
    device = 5
    num_trials = 10 if execution_mode == ExecutionMode.OPTIMIZATION else 2
    patience = 5 if execution_mode == ExecutionMode.OPTIMIZATION else 3
    max_epochs = 80 if execution_mode == ExecutionMode.OPTIMIZATION else 1

    param_domains = {'batch_size': [4, 15] if freeze is False else 32,
                     'lr': [8e-5, 1e-3] if freeze is True else [5e-6, 1e-4],
                     'l2': [3e-5, 7e-4],
                     'clip': 10,
                     'dropout': [0.1, 0.2] if execution_mode == ExecutionMode.OPTIMIZATION else [0., 0.],
                     'tagger_dim': 512,  # [256, 1023],
                     'corpus_embedding_dim': 1024  # [256, 2047]}
                     }
    if execution_mode == ExecutionMode.TEST:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class ExecutionSettingsTaclmcrep:

    class ExecutionMode(enum.Enum):
        OPTIMIZATION = 'optimization'
        TEST = 'test'

    execution_mode = ExecutionMode.OPTIMIZATION
    validation_scenario = 'bath'
    test_scenario = 'bicycle'

    select_index = True

    ''' one of {'none', 'lstm', 'crf', 'lstm-crf', 'transformer'}, specifies the tagging layer.'''
    tagger_type = 'lstm'

    ''' subset of {inscript, descript, backtranslation}. the list should match the data '''
    corpora = ['inscript']

    ''' a non-empty subset of {'events', 'participants'}, indicating which instances will be loaded. '''''
    clustering_mode = ['events', 'participants']

    ''' one of 'normal', 'regular_only', 'regular_identification', 'regular_only_for_ussp'
    check the doc of InScriptSequenceLabeling Reader '''
    loading_mode = 'regular_only_for_ussp'
    ''' reg_id_mc, reg_id_ins, ussp_mc, ussp_ins, normal_mc, normal_ins '''
    data_split = 'ussp_mc'

    train_with_BT_instances = True
    predict_BT_instances = False
    condition_on_scenario = False
    ''' method for span representation. 
    'average' averages over the entire span; 'w_average' averages the first and last tokens;
     'concat' concatenates first and the last tokens; 'alex' concats the logits with span embeddings
     'cosine' uses cosine similarity to modify the output distribution. it is NOT parameterized
     'bi-cosine' refines the cosine similarity to a parameterized bilinear function vAb
     '''
    pooling = 'cosine'

    tag = f'{data_split},prtgnst:{False},pooling:{pooling},conditioning:{condition_on_scenario}_val_scenario:{validation_scenario}'
    # + validation_scenario

    train_data_dir = os.path.join('.', 'inscript_train+val+train_bt') if execution_mode == ExecutionMode.OPTIMIZATION \
        else os.path.join('.', 'toy')
    val_data_dir = os.path.join('.', 'inscript_val') if execution_mode == ExecutionMode.OPTIMIZATION \
        else os.path.join('.', 'toy')
    test_data_dir = os.path.join('.', 'inscript_val') if execution_mode == ExecutionMode.OPTIMIZATION \
        else os.path.join('.', 'toy')
    full_data_dir = os.path.join('.', 'data_inscript')

    pretrained_model_name = 'xlnet-large-cased'
    encoder_hidden_size = 1024

    freeze = True
    device = 2
    num_trials = 20 if execution_mode == ExecutionMode.OPTIMIZATION else 2
    patience = 5 if execution_mode == ExecutionMode.OPTIMIZATION else 3
    max_epochs = 80 if execution_mode == ExecutionMode.OPTIMIZATION else 1

    param_domains = {'batch_size': [4, 15] if freeze is False else 64,
                     'lr': [3e-5, 3e-3] if freeze is True else [5e-6, 1e-4],  # [8e-5, 1e-3]
                     'l2': [3e-5, 7e-4],
                     'clip': 10,
                     'dropout': [0., 0.] if execution_mode == ExecutionMode.OPTIMIZATION else [0., 0.],
                     'tagger_dim': 512,  # [256, 1023],
                     'corpus_embedding_dim': 1024,  # [256, 2047]}
                     # conditioner rank shall not exceed tagger dim
                     'conditioner_rank': [4, 512],
                     'alex_coef': [1e-2, 2.],
                     }
    if execution_mode == ExecutionMode.TEST:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

