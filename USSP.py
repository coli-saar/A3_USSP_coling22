"""
configurations different in each execution.
No, I'm NOT a fan of config files.

"""
import enum
import os
import datetime
#  import copy

from sklearn.cluster import AgglomerativeClustering  # , SpectralClustering

from optimization import ScriptRepresentationLearningMetaOptimizer
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
    execution_mode = ExecutionMode.TRIAL
    clustering_mode = 'ep'
    do_not_evaluate = False

    pretrained_model_name = 'xlnet-base-cased'

    ''' ins or mc. (1) for ins there are 10 different models to use for representation;  
    (2) validation set will be set accordingly
        in 'mc': train on entire InScript, val on mc val, test on mc test
        in 'ins': train on 8 ins scenarios, val on a 9th, test on the 10th '''
    evaluation_on = 'mc'

    scenario_phrase = True
    ''' if True, the scenario name will be added to the beinning of each story, separated by <sep> '''
    encoder_hidden_size = CONST.pretrained_model_dim[pretrained_model_name]
    ''' if true, load pre-trained ssp model to initialize the representations; otherwise, load XLNet as representation '''
    use_pretrained_ssp_model = True
    ''' if True, events that are related to the same participants will be brought closer to each other; similar to participants that relates to the same events '''
    use_dep_regularizer = True
    ''' if True, read regularity labels from the corpus; otherwise, use predictions from a pre-trained classifier '''
    ''' one of acl_GT, acl_pred, tacl_GT, tacl_pred '''
    regularity = 'tacl_pred'
    ''' if > 0, representations are extended with coreference marks to reduce pairwise distances in [INFERENCE] time '''
    complex_coref_extension = -1

    ''' if True, participant candidates in the same coref chain will be brought closer to each other; those in different
     coref chains of the same story will be regularized to be further from each other '''
    use_coref_regularizer = True

    '''================================================================='''
    ''' if True, load data with back-translation augmentation '''
    use_BT_in_training = False

    device = 7
    num_trials = 5
    patience = 5
    max_epochs = 1

    ''' one of 'cosine', 'l2', 'norm_l2'. for 'cosine', 
    the similarity measure will be converted to a distance measure.'''
    optim_dist_metric = 'cosine'

    linkage = 'ward'
    affinity = {'ward': 'euclidean', 'average': 'precomputed'}
    ''' if True, add cosine encoding as the last two dimension of the representations '''
    use_cosine_positional_encoding = False

    ''' DO NOT MOIDIFY '''
    XLNet_requires_grad = True

    if use_BT_in_training:
        data_folder = os.path.join('.', 'data_inscript+bt')
        pseudo_labels_folder = os.path.join('.', 'pseudo_labels_ins+bt')
    else:
        data_folder = os.path.join('.', 'data_inscript')
        pseudo_labels_folder = os.path.join('.', 'pseudo_labels_ins')

    event_dist_thres = 2.60
    participant_dist_thres = 0.14
    n_clusters = 10
    ''' ward, single, complete, average. only for participants. '''

    clustering_algorithms = {
        'agglw': AgglomerativeClustering
    }

    ' == optimization parameters ======================================'
    validation_scenario = 'grocery'
    tag = ('pretrainssp_' if use_pretrained_ssp_model else 'XLNet_') \
        + ('coref_' if use_coref_regularizer else 'none_') \
        + ('dep_' if use_dep_regularizer else 'none_') + datetime.datetime.now().strftime("%H:%M:%S")
    test_scenario = 'haircut'
    train_scenarios = list()
    for scenario in CONST.scenario_s:
        if scenario not in [test_scenario, validation_scenario]:
            train_scenarios.append(scenario)

    pretrained_model_dim = CONST.pretrained_model_dim[pretrained_model_name]

    param_domains = {
        # this batch size (> 150) makes sure that each batch contains a whole scenario
        'batch_size': 256,
        'lr': 2e-5,
        'l2': 3e-3,
        'clip': 999,
        'representation_dim': [128, 511],
        'gamma_inter_e': .4,
        'gamma_intra_p': .4,
        'gamma_inter_p': .6,
        'lb_coef': .2,
        'ub_coef': .7,

        'lambda_same_coref': .2,
        'lambda_same_dep': .1,

        'lambda_cosine': 999999,

        'lambda_inf_coref': .1,
    }

    ''' if True, use distance threshold as the stopping criterion for clustering '''
    cluster_with_dist_thres = True


if __name__ == '__main__':
    meta_optimizer = ScriptRepresentationLearningMetaOptimizer(configurations=ExecutionSettings,
                                                               model=UnsupervisedScriptParser)
    meta_optimizer.search(test_mode=True)
