import torch
from torch import autograd
import pickle
import wandb
import random

from nn_models import lbp_message_passing_network, GIN_Network_withEdgeFeatures
from ising_model.pytorch_dataset import build_factorgraph_from_SpinGlassModel
from ising_model.spin_glass_model import SpinGlassModel
from factor_graph import FactorGraphData
# from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader as DataLoader_pytorchGeometric
from factor_graph import DataLoader_custom as DataLoader_pytorchGeometric

from ising_model.pytorch_geometric_data import spinGlass_to_torchGeometric


import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import parameters
from parameters import ROOT_DIR, alpha, alpha2, SHARE_WEIGHTS, BETHE_MLP, NUM_MLPS
import cProfile

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_map_flag', action='store_true', default=False)
parser.add_argument('--attractive_flag', action='store_false', default=True)
parser.add_argument('--updates', type=str, default="SEQRND")
parser.add_argument('--damping', type=float, default=0.)
parser.add_argument('--maxiter', type=int, default=None)
args = parser.parse_args()
print(args)
MODEL_MAP_FLAG = args.model_map_flag
ATTRACTIVE_FIELD = args.attractive_flag
UPDATES = args.updates
DAMPING = args.damping
MAXITER = args.maxiter


MODE = "train" #run "test" or "train" mode

#####Testing parameters
TEST_TRAINED_MODEL = True #test a pretrained model if True.  Test untrained model if False (e.g. LBP)
# EXPERIMENT_NAME = 'trained_mixedField_10layer_2MLPs_finalBetheMLP/' #used for saving results when MODE='test'
# EXPERIMENT_NAME = 'trained_attrField_10layer_2MLPs_noFinalBetheMLP/' #used for saving results when MODE='test'
EXPERIMENT_NAME = 'trained_MAP_attrField_10layer_2MLPs_noFinalBetheMLP/' #used for saving results when MODE='test'


# 10 layer models
# BPNN_trained_model_path = './wandb/run-20200209_071429-l8jike8k/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True
# BPNN_trained_model_path = './wandb/run-20200211_233743-tpiv47ws/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True, 2MLPs per layer, weight sharing across layers
# BPNN_trained_model_path = './wandb/run-20200219_090032-11077pcu/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True, 2MLPs per layer [wandb results](https://app.wandb.ai/jdkuck/learnBP_spinGlass/runs/11077pcu)



# GNN_trained_model_path = './wandb/run-20200209_091247-wz2g3fjd/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=True
# GNN_trained_model_path = './wandb/run-20200219_051810-bp7hke44/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=True, final MLP takes concatenation of all layers summed node features



# BPNN_trained_model_path = './wandb/run-20200209_201644-cj5b13c2/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False
# BPNN_trained_model_path = './wandb/run-20200211_222445-7ky0ix4y/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False, 2MLPs per layer
# BPNN_trained_model_path = './wandb/run-20200211_234428-ylbhlu1o/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False, 2MLPs per layer, weight sharing across layers
# BPNN_trained_model_path = './wandb/run-20200213_092753-4jdedu1x/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False, 2MLPs per layer, final Bethe MLP

# GNN_trained_model_path = './wandb/run-20200209_203009-o8owzdjv/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=False
# GNN_trained_model_path = './wandb/run-20200213_225352-eqnnbg3v/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=False, final MLP gets summed features from all layers, width 4


#15 layer models
# BPNN_trained_model_path = './wandb/run-20200211_083434-fimwr6fw/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True
# GNN_trained_model_path = './wandb/run-20200211_090711-tb0e4alc/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=True

#30 layer models
# BPNN_trained_model_path = './wandb/run-20200211_093808-qrddyif3/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True
# GNN_trained_model_path = './wandb/run-20200211_093445-xbcslpve/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=True
# BPNN_trained_model_path = './wandb/run-20200212_055535-s8qnrxjq/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False, 2MLPs per layer, no weight sharing across layers


# BPNN_trained_model_path = './wandb/run-20200219_020545-j2ef9bvp/model.pt'

USE_WANDB = True
# os.environ['WANDB_MODE'] = 'dryrun' #don't save to the cloud with this option
##########################
####### Training PARAMETERS #######
MAX_FACTOR_STATE_DIMENSIONS = 2
MSG_PASSING_ITERS = 10 #the number of iterations of message passing, we have this many layers with their own learnable parameters

EPSILON = 0 #set factor states with potential 0 to EPSILON for numerical stability

# MODEL_NAME = "debugCUDA_spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)
MODEL_NAME = "MAP_spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)

TRAINED_MODELS_DIR = ROOT_DIR + "trained_models_map/" #trained models are stored here



##########################################################################################################
# N_MIN = 8
# N_MAX = 11
# F_MAX = 5.0
# C_MAX = 5.0
N_MIN_TRAIN = 10
N_MAX_TRAIN = 10
F_MAX_TRAIN = .1
C_MAX_TRAIN = 5.0
# F_MAX = 1
# C_MAX = 10.0
ATTRACTIVE_FIELD_TRAIN = ATTRACTIVE_FIELD

N_MIN_VAL = 10
N_MAX_VAL = 10
F_MAX_VAL = .1
C_MAX_VAL = 5.0
ATTRACTIVE_FIELD_VAL = ATTRACTIVE_FIELD
# ATTRACTIVE_FIELD_TEST = True

REGENERATE_DATA = False
DATA_DIR = "./data/spin_glass_map/"


TRAINING_DATA_SIZE = 50
VAL_DATA_SIZE = 50#100
TEST_DATA_SIZE = 200

TRAIN_BATCH_SIZE=50
VAL_BATCH_SIZE=50

EPOCH_COUNT = 2
PRINT_FREQUENCY = 10
VAL_FREQUENCY = 10
SAVE_FREQUENCY = 100

TEST_DATSET = 'val' #can test and plot results for 'train', 'val', or 'test' datasets

##### Optimizer parameters #####
STEP_SIZE=300
LR_DECAY=.5
if ATTRACTIVE_FIELD_TRAIN == True:
    #works well for training on attractive field
        LEARNING_RATE = 0.001
#         LEARNING_RATE = 0.001 #testing
#     LEARNING_RATE = 0.00005 #10layer with Bethe_mlp
else:
    #think this works for mixed fields
#         LEARNING_RATE = 0.005 #10layer
#         LEARNING_RATE = 0.001 #30layer trial
    LEARNING_RATE = 0.0005 #10layer with Bethe_mlp
#     LEARNING_RATE = 0.0000005 #c_max = .5


##########################
if USE_WANDB:
    wandb.init(project="BP_map_marginal_spinGlass_new")
    wandb.config.epochs = EPOCH_COUNT
    wandb.config.MODEL_MAP_FLAG = MODEL_MAP_FLAG
    wandb.config.UPDATES = UPDATES
    wandb.config.DAMPING = DAMPING
    wandb.config.MAXITER = MAXITER
    wandb.config.ATTRACTIVE_FIELD = ATTRACTIVE_FIELD




def get_dataset(dataset_type):
    '''
    Store/load a list of SpinGlassModels
    When using, convert to BPNN or GNN form with either
    build_factorgraph_from_SpinGlassModel(pytorch_geometric=True) for BPNN or spinGlass_to_torchGeometric() for GNN
    '''
    assert(dataset_type in ['train', 'val', 'test'])
    if dataset_type == 'train':
        datasize = TRAINING_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_TRAIN
        N_MIN = N_MIN_TRAIN
        N_MAX = N_MAX_TRAIN
        F_MAX = F_MAX_TRAIN
        C_MAX = C_MAX_TRAIN
    elif dataset_type == 'val':
        datasize = VAL_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_VAL
        N_MIN = N_MIN_VAL
        N_MAX = N_MAX_VAL
        F_MAX = F_MAX_VAL
        C_MAX = C_MAX_VAL
    else:
        datasize = TEST_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_TEST

    dataset_file = DATA_DIR + dataset_type + '%d_%d_%d_%.2f_%.2f_attField=%s.pkl' % (datasize, N_MIN, N_MAX, F_MAX, C_MAX, ATTRACTIVE_FIELD)
    if REGENERATE_DATA or (not os.path.exists(dataset_file)):
        print("REGENERATING DATA!!")
        spin_glass_models_list = [SpinGlassModel(N=random.randint(N_MIN, N_MAX),\
                                                f=np.random.uniform(low=0, high=F_MAX),\
                                                c=np.random.uniform(low=0, high=C_MAX),\
                                                attractive_field=ATTRACTIVE_FIELD) for i in range(datasize)]
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(dataset_file, 'wb') as f:
            pickle.dump(spin_glass_models_list, f)
    else:
        with open(dataset_file, 'rb') as f:
            spin_glass_models_list = pickle.load(f)
    return spin_glass_models_list

device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# lbp_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS,\
                                      # msg_passing_iters=MSG_PASSING_ITERS, device=None,
                                      # map_flag=MODEL_MAP_FLAG,)

def test_loss_func(x, y, sg_model):
    '''
    x, y of shape [N, 1] representing the difference of log marginals

    returning
    - mse_diff_log_loss, l1_diff_log_loss
    - mse_prob_loss, l1_prob_loss, cross_entropy_prob_loss, (where prob denotes
    the pseudo probabilities calculated from marginals)
    - var_state_accuracy, graph_state_accuracy, (where state denotes the
    maximum states calculated from marginals)
    - mse_logscore_state_loss, l1_logscore_state_loss, (where logscore
    represents the logScore of the maximum state calculated from marginals)
    '''
    diff_log = torch.abs(x-y).reshape(-1)
    mse_diff_log_loss = (diff_log**2).tolist()
    l1_diff_log_loss = diff_log.tolist()

    prob_x = torch.exp(x)/(torch.exp(x)+1)
    prob_x = torch.cat([prob_x, 1-prob_x], dim=-1)
    prob_y = torch.exp(y)/(torch.exp(y)+1)
    prob_y = torch.cat([prob_y, 1-prob_y], dim=-1)
    diff_prob = torch.abs(prob_x[:,0]-prob_y[:,0]).reshape(-1)
    mse_prob_loss = (diff_prob**2).tolist()
    l1_prob_loss = diff_prob.tolist()
    cross_entropy_prob_loss = (-torch.sum(prob_y*torch.log(prob_x+1e-99), dim=1)).tolist()

    state_x = (x<=0).float()
    state_y = (y<=0).float()
    accuracy = (state_x==state_y).float().reshape(-1)
    var_state_accuracy = accuracy.tolist()
    graph_state_accuracy = [torch.prod(accuracy).item()]

    score_x = sg_model.logScore(tuple(state_x.reshape(-1).tolist()))
    score_y = sg_model.logScore(tuple(state_y.reshape(-1).tolist()))
    diff_score = abs(score_x-score_y)
    mse_logscore_state_loss = [diff_score**2]
    l1_logscore_state_loss = [diff_score]

    return(
        mse_diff_log_loss, l1_diff_log_loss,
        mse_prob_loss, l1_prob_loss, cross_entropy_prob_loss,
        var_state_accuracy, graph_state_accuracy,
        mse_logscore_state_loss, l1_logscore_state_loss,
    )


# lbp_net = lbp_net.to(device)

# lbp_net.double()
def train():
    # if USE_WANDB:
        # wandb.watch(lbp_net)

    # lbp_net.train()
    # Initialize optimizer
    # optimizer = torch.optim.Adam(lbp_net.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs


    spin_glass_models_list_train = get_dataset(dataset_type='train')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    # sg_models_fg_from_train = [build_factorgraph_from_SpinGlassModel(sg_model, map_flag=DATA_MAP_FLAG) for sg_model in spin_glass_models_list_train]
    # train_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from_train, batch_size=TRAIN_BATCH_SIZE)


    spin_glass_models_list_val = get_dataset(dataset_type='val')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    # sg_models_fg_from_val = [build_factorgraph_from_SpinGlassModel(sg_model, map_flag=DATA_MAP_FLAG) for sg_model in spin_glass_models_list_val]
    # val_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from_val, batch_size=VAL_BATCH_SIZE)
#     val_data_loader_pytorchGeometric_batchSize50 = DataLoader_pytorchGeometric(sg_models_fg_from_val, batch_size=VAL_BATCH_SIZE)

#     with autograd.detect_anomaly():
    for e in range(EPOCH_COUNT):
        mse_diff_log_losses, l1_diff_log_losses = [], []
        mse_prob_losses, l1_prob_losses, cross_entropy_prob_losses = [], [], []
        var_state_accuracies, graph_state_accuracies = [], []
        mse_logscore_state_losses, l1_logscore_state_losses = [], []

        for spin_glass_problem in spin_glass_models_list_train:
            exact_ln_partition_function = spin_glass_problem.marginal_junction_tree_libdai(
                map_flag=True, classification_flag=False,)
            estimated_ln_partition_function = spin_glass_problem.marginal_loopyBP_libdai(
                map_flag=MODEL_MAP_FLAG, updates=UPDATES,
                damping=DAMPING, classification_flag=False,
                maxiter=MAXITER,
            )

            mse_diff_log_loss, l1_diff_log_loss, \
                mse_prob_loss, l1_prob_loss, cross_entropy_prob_loss,\
                var_state_accuracy, graph_state_accuracy,\
                mse_logscore_state_loss, l1_logscore_state_loss\
                = test_loss_func(torch.tensor(estimated_ln_partition_function),
                                 torch.tensor(exact_ln_partition_function),
                                 spin_glass_problem)
            mse_diff_log_losses += mse_diff_log_loss
            l1_diff_log_losses += l1_diff_log_loss
            mse_prob_losses += mse_prob_loss
            l1_prob_losses += l1_prob_loss
            cross_entropy_prob_losses += cross_entropy_prob_loss
            var_state_accuracies += var_state_accuracy
            graph_state_accuracies += graph_state_accuracy
            mse_logscore_state_losses += mse_logscore_state_loss
            l1_logscore_state_losses += l1_logscore_state_loss

        print("root mean squared training error =", np.mean(var_state_accuracies))
        if USE_WANDB:
            wandb.log({
                "RMSE_DiffLog_training": np.sqrt(np.mean(mse_diff_log_losses)),
                "L1_DiffLog_training": np.mean(l1_diff_log_losses),
                'RMSE_Prob_training': np.sqrt(np.mean(mse_prob_losses)),
                'L1_Prob_training': np.mean(l1_prob_losses),
                'CrossEntropy_Prob_training': np.mean(cross_entropy_prob_losses),
                'ACC_VarState_training': np.mean(var_state_accuracies),
                'ACC_GraphState_training': np.mean(graph_state_accuracies),
                'RMSE_LogScore_training': np.sqrt(np.mean(mse_logscore_state_losses)),
                'L1_LogScore_training': np.mean(l1_logscore_state_losses),
            })

        mse_diff_log_losses, l1_diff_log_losses = [], []
        mse_prob_losses, l1_prob_losses, cross_entropy_prob_losses = [], [], []
        var_state_accuracies, graph_state_accuracies = [], []
        mse_logscore_state_losses, l1_logscore_state_losses = [], []
        for spin_glass_problem in spin_glass_models_list_val:
            exact_ln_partition_function = spin_glass_problem.marginal_junction_tree_libdai(
                map_flag=True, classification_flag=False,)
            estimated_ln_partition_function = spin_glass_problem.marginal_loopyBP_libdai(
                map_flag=MODEL_MAP_FLAG, updates=UPDATES,
                damping=DAMPING, classification_flag=False,
                maxiter=MAXITER,
            )

            mse_diff_log_loss, l1_diff_log_loss, \
                mse_prob_loss, l1_prob_loss, cross_entropy_prob_loss,\
                var_state_accuracy, graph_state_accuracy,\
                mse_logscore_state_loss, l1_logscore_state_loss\
                = test_loss_func(torch.tensor(estimated_ln_partition_function),
                                 torch.tensor(exact_ln_partition_function),
                                 spin_glass_problem)
            mse_diff_log_losses += mse_diff_log_loss
            l1_diff_log_losses += l1_diff_log_loss
            mse_prob_losses += mse_prob_loss
            l1_prob_losses += l1_prob_loss
            cross_entropy_prob_losses += cross_entropy_prob_loss
            var_state_accuracies += var_state_accuracy
            graph_state_accuracies += graph_state_accuracy
            mse_logscore_state_losses += mse_logscore_state_loss
            l1_logscore_state_losses += l1_logscore_state_loss

        print("root mean squared validation error =", np.mean(var_state_accuracies))
        print()
        if USE_WANDB:
            wandb.log({
                "RMSE_DiffLog_val": np.sqrt(np.mean(mse_diff_log_losses)),
                "L1_DiffLog_val": np.mean(l1_diff_log_losses),
                'RMSE_Prob_val': np.sqrt(np.mean(mse_prob_losses)),
                'L1_Prob_val': np.mean(l1_prob_losses),
                'CrossEntropy_Prob_val': np.mean(cross_entropy_prob_losses),
                'ACC_VarState_val': np.mean(var_state_accuracies),
                'ACC_GraphState_val': np.mean(graph_state_accuracies),
                'RMSE_LogScore_val': np.sqrt(np.mean(mse_logscore_state_losses)),
                'L1_LogScore_val': np.mean(l1_logscore_state_losses),
            })


def create_ising_model_figure(results_directory=ROOT_DIR, skip_our_model=False):
    if TEST_TRAINED_MODEL:
        lbp_net.load_state_dict(torch.load(BPNN_trained_model_path))
#         lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))

        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))


    lbp_net.eval()

    #data loader for BPNN
    spin_glass_models_list = get_dataset(dataset_type=TEST_DATSET)
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sg_models_fg_from = [build_factorgraph_from_SpinGlassModel(sg_model) for sg_model in spin_glass_models_list]
    data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from, batch_size=1, shuffle=False)

    #data loader for GNN
    val_data_list_GNN = [spinGlass_to_torchGeometric(sg_problem) for sg_problem in spin_glass_models_list]
    val_loader_GNN = DataLoader_pytorchGeometric(val_data_list_GNN, batch_size=1, shuffle=False)

#     gnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = GIN_Network_withEdgeFeatures(msg_passing_iters=MSG_PASSING_ITERS).to(device)
    gnn_model.load_state_dict(torch.load(GNN_trained_model_path))
    gnn_model.eval()

    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    BPNN_estimated_counts = []

    GNN_estimated_counts = []

#     LBPlibdai_5iters_estimated_counts = []
    LBPlibdai_10iters_estimated_counts = []
    LBPlibdai_100iters_estimated_counts = []
    LBPlibdai_1000iters_estimated_counts = []
    LBPlibdai_1000Seqiters_estimated_counts = []
    LBPlibdai_5kiters_estimated_counts = []

    meanFieldlibdai_5kiters_estimated_counts = []

#     lbp_losses_5iters = []
    lbp_losses_10iters = []
    lbp_losses_100iters = []
    lbp_losses_1000iters = []
    lbp_losses_1000Seqiters = []
    lbp_losses_5kiters = []

    mean_field_losses_5kiters = []

    losses = []
    GNN_losses = []

    lbp_losses = []
    mrftool_lbp_losses = []
    for idx, (spin_glass_problem, gnn_data) in enumerate(zip(data_loader_pytorchGeometric, val_loader_GNN)): #pytorch geometric form
        print("problem:", idx)
        # spin_glass_problem.compute_bethe_free_energy()
        sg_problem_SGM = spin_glass_models_list[idx]
        exact_ln_partition_function = spin_glass_problem.ln_Z
        if not skip_our_model:
            #run BPNN
#             spin_glass_problem = spin_glass_problem.to(device)
#             exact_ln_partition_function = exact_ln_partition_function.to(device)
            estimated_ln_partition_function = lbp_net(spin_glass_problem)
            BPNN_estimated_counts.append(estimated_ln_partition_function.item()-exact_ln_partition_function)
            loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
            losses.append(loss.item())

            #run GNN FIX ME
            assert(np.isclose(exact_ln_partition_function.item(), gnn_data.ln_Z.item())), (exact_ln_partition_function.item(), gnn_data.ln_Z.item())
            gnn_data = gnn_data.to(device)
            gnn_pred_ln_Z = gnn_model(x=gnn_data.x, edge_index=gnn_data.edge_index, edge_attr=gnn_data.edge_attr, batch=gnn_data.batch)
            gnn_pred_ln_Z = gnn_pred_ln_Z.squeeze()
            GNN_estimated_counts.append(gnn_pred_ln_Z.item()-exact_ln_partition_function)
            gnn_loss = loss_func(gnn_pred_ln_Z, exact_ln_partition_function.float().squeeze())
            GNN_losses.append(gnn_loss.item())

#         libdai_lbp_Z_5 = sg_problem_SGM.loopyBP_libdai(maxiter=5)
        libdai_lbp_Z_10 = sg_problem_SGM.loopyBP_libdai(maxiter=10, updates="PARALL", damping=".5")
        libdai_lbp_Z_100 = sg_problem_SGM.loopyBP_libdai(maxiter=100, updates="PARALL", damping=".5")
        libdai_lbp_Z_1000 = sg_problem_SGM.loopyBP_libdai(maxiter=1000, updates="PARALL", damping=".5")
        libdai_lbp_Z_1000Seq = sg_problem_SGM.loopyBP_libdai(maxiter=1000, updates="SEQRND", damping=None)
#         libdai_lbp_Z_5k = sg_problem_SGM.loopyBP_libdai(maxiter=5000)

#         LBPlibdai_5iters_estimated_counts.append(libdai_lbp_Z_5-exact_ln_partition_function)
        LBPlibdai_10iters_estimated_counts.append(libdai_lbp_Z_10-exact_ln_partition_function)
        LBPlibdai_100iters_estimated_counts.append(libdai_lbp_Z_100-exact_ln_partition_function)
        LBPlibdai_1000iters_estimated_counts.append(libdai_lbp_Z_1000-exact_ln_partition_function)
        LBPlibdai_1000Seqiters_estimated_counts.append(libdai_lbp_Z_1000Seq-exact_ln_partition_function)
#         LBPlibdai_5kiters_estimated_counts.append(libdai_lbp_Z_5k-exact_ln_partition_function)

        libdai_meanField_Z_5k = sg_problem_SGM.mean_field_libdai(maxiter=100000)
        meanFieldlibdai_5kiters_estimated_counts.append(libdai_meanField_Z_5k-exact_ln_partition_function)



        exact_solution_counts.append(exact_ln_partition_function)


#         lbp_losses_5iters.append(loss_func(torch.tensor(libdai_lbp_Z_5), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_10iters.append(loss_func(torch.tensor(libdai_lbp_Z_10), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_100iters.append(loss_func(torch.tensor(libdai_lbp_Z_100), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_1000iters.append(loss_func(torch.tensor(libdai_lbp_Z_1000), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_1000Seqiters.append(loss_func(torch.tensor(libdai_lbp_Z_1000Seq), exact_ln_partition_function.float().squeeze()).item())
#         lbp_losses_5kiters.append(loss_func(torch.tensor(libdai_lbp_Z_5k), exact_ln_partition_function.float().squeeze()).item())
        mean_field_losses_5kiters.append(loss_func(torch.tensor(libdai_meanField_Z_5k), exact_ln_partition_function.float().squeeze()).item())

        if not skip_our_model:
            print("GNN estimated_ln_partition_function:", estimated_ln_partition_function)
        print("exact_ln_partition_function:", exact_ln_partition_function)
        print()

    print("LBP libdai MSE:", np.sqrt(np.mean(lbp_losses_5kiters)))
    print("GNN MSE:", np.sqrt(np.mean(losses)))


    losses.sort()
    mrftool_lbp_losses.sort()
    lbp_losses.sort()


    if not skip_our_model:
        plt.plot(exact_solution_counts, BPNN_estimated_counts, 'x', label='%d layer BPNN, RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.mean(losses))))
        plt.plot(exact_solution_counts, GNN_estimated_counts, 'x', label='%d layer GNN, RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.mean(GNN_losses))))


#     plt.plot(exact_solution_counts, LBPlibdai_5iters_estimated_counts, '+', label='LBP 5 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_5iters))))

    plt.plot(exact_solution_counts, LBPlibdai_10iters_estimated_counts, '+', label='LBP 10 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_10iters))))

    plt.plot(exact_solution_counts, LBPlibdai_100iters_estimated_counts, '+', label='LBP 100 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_100iters))))

    plt.plot(exact_solution_counts, LBPlibdai_1000iters_estimated_counts, '+', label='LBP 1000 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_1000iters))))

    plt.plot(exact_solution_counts, LBPlibdai_1000Seqiters_estimated_counts, '+', label='LBP 1000 seq iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_1000Seqiters))))

    plt.plot(exact_solution_counts, meanFieldlibdai_5kiters_estimated_counts, '1', label='Mean Field, RMSE=%.2f' % (np.sqrt(np.mean(mean_field_losses_5kiters))))

    plt.plot([min(exact_solution_counts), max(exact_solution_counts)], [0, 0], '-', c='g', label='Exact')

    # plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='Ground Truth ln(Set Size)')
    plt.xlabel('ln(Z)', fontsize=14)
    plt.ylabel('ln(Estimate) - ln(Z)', fontsize=14)
    plt.yscale('symlog')
    plt.title('Exact Partition Function vs. Estimates', fontsize=20)
    # plt.legend(fontsize=8, loc=2, prop={'size': 6})
#     plt.legend(fontsize=12, prop={'size': 12})
    # Put a legend below current axis
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.12),
          fancybox=True, ncol=2, fontsize=12, prop={'size': 12})

    #make the font bigger
    matplotlib.rcParams.update({'font.size': 10})

    plt.grid(True)
    # Shrink current axis's height by 10% on the bottom
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])
    #fig.savefig('/Users/jkuck/Downloads/temp.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    if not os.path.exists(results_directory + 'plots/'):
        os.makedirs(results_directory + 'plots/')

    if TEST_DATSET == 'train':
        datasize = TRAINING_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_TRAIN
        N_MIN = N_MIN_TRAIN
        N_MAX = N_MAX_TRAIN
        F_MAX = F_MAX_TRAIN
        C_MAX = C_MAX_TRAIN
    elif TEST_DATSET == 'val':
        datasize = VAL_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_VAL
        N_MIN = N_MIN_VAL
        N_MAX = N_MAX_VAL
        F_MAX = F_MAX_VAL
        C_MAX = C_MAX_VAL
    else:
        assert(False), ("invalid TEST_DATASET")

        f5_c5_N10_attFldT
    # plot_name = 'trained=%s_%s_%diters_%d_%d_%.2f_%.2f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, MSG_PASSING_ITERS, N_MIN, N_MAX, F_MAX, C_MAX)
#     plot_name = 'trained=%s_dataset=%s%d_c%f_f%f_N%d%d_att=%s_%diters_alpha%f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, len(data_loader_pytorchGeometric), C_MAX, F_MAX, N_MIN, N_MAX, ATTRACTIVE_FIELD, MSG_PASSING_ITERS, parameters.alpha)
    plot_name = 'f%.2f_c%.2f_N%d-%d_attFld%s.png' % (F_MAX, C_MAX, N_MIN, N_MAX, ATTRACTIVE_FIELD)
    plt.savefig(results_directory + 'plots/' + plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    matplotlib.pyplot.clf()
    # plt.show()

    return np.sqrt(np.mean(losses)), np.sqrt(np.mean(GNN_losses)), np.sqrt(np.mean(lbp_losses_10iters)), np.sqrt(np.mean(lbp_losses_100iters)),\
           np.sqrt(np.mean(lbp_losses_1000iters)), np.sqrt(np.mean(lbp_losses_1000Seqiters)), np.sqrt(np.mean(mean_field_losses_5kiters))



def test(skip_our_model=False):
    if TEST_TRAINED_MODEL:
#         lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))
        lbp_net.load_state_dict(torch.load(BPNN_trained_model_path))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))


    lbp_net.eval()

    spin_glass_models_list = get_dataset(dataset_type=TEST_DATSET)
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sg_models_fg_from = [build_factorgraph_from_SpinGlassModel(sg_model) for sg_model in spin_glass_models_list]
    data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from, batch_size=1)


    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    BPNN_estimated_counts = []
    LBPlibdai_estimated_counts = []
    LBPmrftools_estimated_counts = []
    losses = []
    lbp_losses = []
    mrftool_lbp_losses = []
    for idx, spin_glass_problem in enumerate(data_loader_pytorchGeometric): #pytorch geometric form
        # spin_glass_problem.compute_bethe_free_energy()
        sg_problem_SGM = spin_glass_models_list[idx]
        exact_ln_partition_function = spin_glass_problem.ln_Z
        if not skip_our_model:
#             spin_glass_problem = spin_glass_problem.to(device)
#             exact_ln_partition_function = exact_ln_partition_function.to(device)
            estimated_ln_partition_function = lbp_net(spin_glass_problem)
            BPNN_estimated_counts.append(estimated_ln_partition_function.item()-exact_ln_partition_function)
            loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
            losses.append(loss.item())

        libdai_lbp_Z_recompute = sg_problem_SGM.loopyBP_libdai()
        mrftools_lbp_Z_recompute = sg_problem_SGM.loopyBP_mrftools()
        LBPlibdai_estimated_counts.append(libdai_lbp_Z_recompute-exact_ln_partition_function)
        LBPmrftools_estimated_counts.append(mrftools_lbp_Z_recompute-exact_ln_partition_function)
#         LBPlibdai_estimated_counts.append(libdai_lbp_Z_est)
#         LBPmrftools_estimated_counts.append(mrftools_lbp_Z_estimate)

        exact_solution_counts.append(exact_ln_partition_function)

#         print("libdai_lbp_Z_recompute:", libdai_lbp_Z_recompute)
#         print("libdai_lbp_Z_est:", libdai_lbp_Z_est)
        libdai_lbp_loss = loss_func(torch.tensor(libdai_lbp_Z_recompute), exact_ln_partition_function.float().squeeze())
        lbp_losses.append(libdai_lbp_loss.item())

        mrftools_lbp_loss = loss_func(torch.tensor(mrftools_lbp_Z_recompute), exact_ln_partition_function.float().squeeze())
        mrftool_lbp_losses.append(mrftools_lbp_loss.item())

        print("libdai lbp estimated_ln_partition_function:", libdai_lbp_Z_recompute)
        print("mrf tools lbp estimated_ln_partition_function:", mrftools_lbp_Z_recompute)
        if not skip_our_model:
            print("GNN estimated_ln_partition_function:", estimated_ln_partition_function)
        print("exact_ln_partition_function:", exact_ln_partition_function)
        print()

    print("LBP libdai MSE:", np.sqrt(np.mean(lbp_losses)))
    print("LBP mrftools MSE:", np.sqrt(np.mean(mrftool_lbp_losses)))
    print("GNN MSE:", np.sqrt(np.mean(losses)))


    losses.sort()
    mrftool_lbp_losses.sort()
    lbp_losses.sort()

    if not skip_our_model:
        plt.plot(exact_solution_counts, BPNN_estimated_counts, 'x', c='g', label='GNN estimate, %d iters, RMSE=%.2f, 10 lrgst removed RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.mean(losses)), np.sqrt(np.mean(losses[:-10]))))
    plt.plot(exact_solution_counts, LBPmrftools_estimated_counts, '+', c='r', label='LBP mrftools, %d iters, RMSE=%.2f, 10 lrgst removed RMSE=%.2f' % (parameters.MRFTOOLS_LBP_ITERS, np.sqrt(np.mean(mrftool_lbp_losses)), np.sqrt(np.mean(mrftool_lbp_losses[:-10]))))
    plt.plot(exact_solution_counts, LBPlibdai_estimated_counts, 'x', c='b', label='LBP libdai, %d iters, RMSE=%.2f, 10 lrgst removed RMSE=%.2f' % (parameters.LIBDAI_LBP_ITERS, np.sqrt(np.mean(lbp_losses)), np.sqrt(np.mean(lbp_losses[:-10]))))
    plt.plot([min(exact_solution_counts), max(exact_solution_counts)], [0, 0], '-', c='g', label='Exact')

    # plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='Ground Truth ln(Set Size)')
    plt.xlabel('ln(Exact Model Count)', fontsize=14)
    plt.ylabel('ln(Estimated Model Count) - ln(Exact Model Count)', fontsize=14)
    plt.title('Exact Model Count vs. Estimates', fontsize=20)
    # plt.legend(fontsize=8, loc=2, prop={'size': 6})
    plt.legend(fontsize=12, prop={'size': 8})
    #make the font bigger
    matplotlib.rcParams.update({'font.size': 10})

    plt.grid(True)
    # Shrink current axis's height by 10% on the bottom
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])
    #fig.savefig('/Users/jkuck/Downloads/temp.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

    if not os.path.exists(ROOT_DIR + 'plots/'):
        os.makedirs(ROOT_DIR + 'plots/')

    # plot_name = 'trained=%s_%s_%diters_%d_%d_%.2f_%.2f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, MSG_PASSING_ITERS, N_MIN, N_MAX, F_MAX, C_MAX)
    plot_name = 'trained=%s_dataset=%s%d_%diters_alpha%f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, len(data_loader_pytorchGeometric), MSG_PASSING_ITERS, parameters.alpha)
    plt.savefig(ROOT_DIR + 'plots/' + plot_name)
    # plt.show()



def create_many_ising_model_figures(results_dir=ROOT_DIR + '/data/experiments/' + EXPERIMENT_NAME, exp_file='ising_model_OOD.pkl'):
    all_results = {}
    for attractive_field in [True, False]:
        for n in [10, 14]:
            for f_max in [.1, .2, 1.0]:
                for c_max in [5.0, 10.0, 50.0]:
                    global N_MIN_VAL
                    global N_MAX_VAL
                    global F_MAX_VAL
                    global C_MAX_VAL
                    global ATTRACTIVE_FIELD_VAL
                    N_MIN_VAL = n
                    N_MAX_VAL = n
                    F_MAX_VAL = f_max
                    C_MAX_VAL = c_max
                    ATTRACTIVE_FIELD_VAL = attractive_field
                    BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf = create_ising_model_figure(results_directory=results_dir)
                    all_results[(attractive_field, n, f_max, c_max)] = (BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf)

                    if not os.path.exists(results_dir):
                        os.makedirs(results_dir)
                    with open(results_dir + exp_file, 'wb') as f:
                        pickle.dump(all_results, f)

if __name__ == "__main__":
    if MODE == "train":
        train()
#         cProfile.run("train()")

    elif MODE == "test":
#         test()
#         create_ising_model_figure()
        create_many_ising_model_figures()