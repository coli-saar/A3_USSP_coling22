"""
configurations different in each execution.
No, I'm NOT a fan of config files.

"""
import enum
import os


from sklearn.cluster import AgglomerativeClustering
from model import UnsupervisedScriptParser

from global_constants import CONST

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class ExecutionSettings:
    class ExecutionMode(enum.Enum):
        TRIAL = 'trial'
        CROSS_VALIDATION = 'cross_validation'
        TEST = 'test'
        INFERENCE = 'inference'

    ' == high-level settings ============================='
    execution_mode = ExecutionMode.CROSS_VALIDATION
    clustering_mode = 'ep'
    unannotated_inference = False

    ''' one of 'cosine', 'l2', 'norm_l2'. for 'cosine', 
    the similarity measure will be converted to a distance measure.'''
    optim_dist_metric = 'cosine'

    ''' if true, load pre-trained ssp model to initialize the representations; otherwise, load XLNet as representation '''
    use_pretrained_ssp_model = True

    ''' if True, participant candidates in the same coref chain will be brought closer to each other; those in different
     coref chains of the same story will be regularized to be further from each other '''
    use_coref_regularizer = True
    ''' if True, events that are related to the same participants will be brought closer to each other; similar to participants that relates to the same events '''
    use_dep_regularizer = True
    ''' if True, read regularity labels from the corpus; otherwise, use predictions from a pre-trained classifier '''
    load_GT_regularity = False

    '''================================================================='''
    ''' if True, load data with back-translation augmentation '''
    use_BT_in_training = False
    XLNet_requires_grad = True

    if use_BT_in_training:
        data_folder = os.path.join('.', 'data_inscript+bt')
        pseudo_labels_folder = os.path.join('.', 'pseudo_labels_ins+bt')
    else:
        data_folder = os.path.join('.', 'data_inscript')
        pseudo_labels_folder = os.path.join('.', 'pseudo_labels_ins')

    event_dist_thres = 2.60
    participant_dist_thres = 0.14

    clustering_algorithms = {
        'agglw': AgglomerativeClustering
    }

    ' == optimization parameters ======================================'
    validation_scenario = 'grocery'
    tag = ('pretrainssp_' if use_pretrained_ssp_model else 'XLNet_') \
          + ('coref_' if use_coref_regularizer else 'none_') \
          + ('dep_' if use_dep_regularizer else 'none_') + validation_scenario
    test_scenario = 'haircut'
    train_scenarios = list()
    for scenario in CONST.scenario_s:
        if scenario not in [test_scenario, validation_scenario]:
            train_scenarios.append(scenario)

    encoder_hidden_size = 768
    device = 3
    num_trials = 10
    patience = 4
    max_epochs = 20

    param_domains = {
        # this batch size (> 150) makes sure that each batch contains a whole scenario
        'batch_size': 256,
        'lr': [1e-5, 1e-3],
        'l2': [1e-5, 1e-2],
        'clip': 999,
        'representation_dim': [128, 511],
        'gamma_inter_e': [1e-2, 2.],
        'gamma_intra_p': [1e-3, 1.],
        'gamma_inter_p': [1e-3, 1.],
        'lb_coef': [8e-2, 1],
        'ub_coef': [2e-1, 1],

        'lambda_same_coref': [3e-3, 1.],
        'lambda_same_dep': [3e-3, 1.],
        # 'lambda_relate_to_same_participant': [1e-3, 10],
    }

    ''' if True, use distance threshold as the stopping criterion for clustering '''
    cluster_with_dist_thres = True


if __name__ == '__main__':
    # fixme: change if not on tony-2
    ussp_model_path = '/local/fangzhou/ussp_pretrained_models/5'
    ussp_index = 5
    ussp_check_point = {'model_path': os.path.join(ussp_model_path, 'best'),
                        'index': ussp_index,
                        'combination_file': os.path.join(ussp_model_path, 'hyper_combs_pretrainssp_coref_dep_bicycle'),
                        'vocab_folder': os.path.join(ussp_model_path, 'vocab'),
                        'configurations': ExecutionSettings}

    ussp_model = UnsupervisedScriptParser.from_checkpoint(check_point_config=ussp_check_point)
    ussp_model.text_representation()
