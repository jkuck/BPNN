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
# parser.add_argument('--data_map_flag', action='store_true', default=False)
# parser.add_argument('--model_map_flag', action='store_true', default=False)
parser.add_argument('--F_MAX', type=float, default=.1,)
parser.add_argument('--C_MAX', type=float, default=5.)
parser.add_argument('--attractive_field_flag', action='store_true', default=False)
args = parser.parse_args()
print(args)
# DATA_MAP_FLAG = args.data_map_flag
# MODEL_MAP_FLAG = args.model_map_flag
DATA_MAP_FLAG = True
MODEL_MAP_FLAG = True
# UPDATES = args.updates
# DAMPING = args.damping
F_MAX = args.F_MAX
C_MAX = args.C_MAX
ATTRACTIVE_FIELD = args.attractive_field_flag

metric_names = [
    f'graph-{stat}-{m}'
    for stat in ['logscore', 'score', 'prob', 'ratio0', 'ratio_same']
    for m in ['mean', 'std', 'max', 'min', 'median']
] + [
    f'graph-diff-{stat}-{dist}-{m}'
    for stat in ['logscore', 'score', 'prob', 'ratio0', 'ratio_same']
    for dist in (['relative', 'mse', 'rmse',] if stat != 'prob' and 'ratio' not in stat else [
        'relative', 'mse', 'rmse',
        'cross_entropy', 'reverse_cross_entropy',
        'kl_diverge', 'reverse_kl_diverge',
    ])
    for m in ['mean', 'std', 'max', 'min', 'median']
] + [
    f'node-diff-{stat}-{dist}-{m}'
    for stat in ['logscore', 'score', 'prob', 'ratio0', 'ratio_same']
    for dist in (['relative', 'mse', 'rmse',] if stat != 'prob' and 'ratio' not in stat else [
        'relative', 'mse', 'rmse',
        'cross_entropy', 'reverse_cross_entropy',
        'kl_diverge', 'reverse_kl_diverge',
    ])
    for m in ['mean', 'std', 'max', 'min', 'median']
]


MODE = "train" #run "test" or "train" mode

#####Testing parameters
TEST_TRAINED_MODEL = True #test a pretrained model if True.  Test untrained model if False (e.g. LBP)
# EXPERIMENT_NAME = 'trained_mixedField_10layer_2MLPs_finalBetheMLP/' #used for saving results when MODE='test'
# EXPERIMENT_NAME = 'trained_attrField_10layer_2MLPs_noFinalBetheMLP/' #used for saving results when MODE='test'
EXPERIMENT_NAME = 'trained_MAP_attrField_10layer_2MLPs_noFinalBetheMLP/' #used for saving results when MODE='test'

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
F_MAX_TRAIN = F_MAX
C_MAX_TRAIN = C_MAX
# F_MAX = 1
# C_MAX = 10.0
ATTRACTIVE_FIELD_TRAIN = ATTRACTIVE_FIELD

N_MIN_VAL = 10
N_MAX_VAL = 10
F_MAX_VAL = F_MAX
C_MAX_VAL = C_MAX
ATTRACTIVE_FIELD_VAL = ATTRACTIVE_FIELD
# ATTRACTIVE_FIELD_TEST = True

REGENERATE_DATA = False
DATA_DIR = "./data/spin_glass_map/"


TRAINING_DATA_SIZE = 100000
VAL_DATA_SIZE = 1000#100
TEST_DATA_SIZE = 1000
# TRAINING_DATA_SIZE = 51
# VAL_DATA_SIZE = 51#100
# TEST_DATA_SIZE = 201

TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=32

EPOCH_COUNT = 1
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
    wandb.init(project="map_spinGlass_new")
    wandb.config.epochs = EPOCH_COUNT
    wandb.config.DATA_MAP_FLAG = DATA_MAP_FLAG
    wandb.config.MODEL_MAP_FLAG = MODEL_MAP_FLAG
    wandb.config.F_MAX = F_MAX
    wandb.config.C_MAX = C_MAX
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

# lbp_net = lbp_net.to(device)

def loss_func(max_states, sp_problem, metric_names):
    max_states_ratio0 = np.array([[np.mean(np.array(s)==0) for s in ms] for ms in max_states])
    max_states_ratio_same = np.maximum(max_states_ratio0,1-max_states_ratio0)
    max_logscores = np.array([[sp_problem.logScore(s) for s in ms] for ms in max_states])
    max_scores = np.exp(max_logscores)
    max_probs = np.exp(max_logscores-sp_problem.logZ)

    graph_map_state = sp_problem.map_state
    graph_map_state_ratio0 = np.mean(np.array(graph_map_state)==0)
    graph_map_state_ratio_same = np.maximum(graph_map_state_ratio0, 1-graph_map_state_ratio0)
    graph_map_logscore = sp_problem.map_logscore
    graph_map_score = np.exp(graph_map_logscore)
    graph_map_prob = np.exp(graph_map_logscore-sp_problem.logZ)

    # finding second map states
    tmp_logscores = max_logscores*(1-(max_logscores==graph_map_score))
    tmp_index = np.unravel_index(np.argmax(tmp_logscores), tmp_logscores.shape)
    graph_2nd_map_state = max_states[tmp_index[0]][tmp_index[1]]
    graph_2nd_map_state_ratio0 = np.mean(np.array(graph_2nd_map_state)==0)
    graph_2nd_map_state_ratio_same = np.maximum(graph_2nd_map_state_ratio0, 1-graph_2nd_map_state_ratio0)
    graph_2nd_map_logscore = max_logscores[tmp_index]
    assert(graph_2nd_map_logscore < graph_map_logscore)
    graph_2nd_map_score = np.exp(graph_2nd_map_logscore)
    graph_2nd_map_prob = np.exp(graph_2nd_map_logscore-sp_problem.logZ)

    values = []
    for mn in metric_names:
        value = None
        if 'graph' in mn and 'diff' not in mn:
            if 'prob' in mn:
                value = graph_map_prob
            elif 'logscore' in mn:
                value = graph_map_logscore
            elif 'score' in mn:
                value = graph_map_score
            elif 'ratio0' in mn:
                value = graph_map_state_ratio0
            elif 'ratio_same' in mn:
                value = graph_map_state_ratio_same
            else:
                raise ValueError("Wrong Metric Name: ", mn)
        elif 'graph-diff' in mn:
            if 'logscore' in mn:
                value = diff_func(graph_map_logscore, graph_2nd_map_logscore, mn)
            elif 'score' in mn:
                value = diff_func(graph_map_score, graph_2nd_map_score, mn)
            elif 'prob' in mn:
                value = diff_func(graph_map_prob, graph_2nd_map_prob, mn)
            elif 'ratio0' in mn:
                value = diff_func(graph_map_state_ratio0, graph_2nd_map_state_ratio0, mn)
            elif 'ratio_same' in mn:
                value = diff_func(graph_map_state_ratio_same, graph_2nd_map_state_ratio_same, mn)
            else:
                raise ValueError("Wrong Metric Name: ", mn)
        elif 'node-diff' in mn:
            if 'logscore' in mn:
                value = [diff_func(ss[s], ss[1-s], mn) for ss,s in zip(max_logscores, graph_map_state)]
            elif 'score' in mn:
                value = [diff_func(ss[s], ss[1-s], mn) for ss,s in zip(max_scores, graph_map_state)]
            elif 'prob' in mn:
                value = [diff_func(ss[s], ss[1-s], mn) for ss,s in zip(max_probs, graph_map_state)]
            elif 'ratio0' in mn:
                value = [diff_func(ss[s], ss[1-s], mn) for ss,s in zip(max_states_ratio0, graph_map_state)]
            elif 'ratio_same' in mn:
                value = [diff_func(ss[s], ss[1-s], mn) for ss,s in zip(max_states_ratio_same, graph_map_state)]
            else:
                raise ValueError("Wrong Metric Name: ", mn)
        else:
            raise ValueError("Wrong Metric Name: ", mn)
        values.append(value)
    return values


def diff_func(v1, v2, metric_name):
    mn = metric_name
    y, p = max(v1, v2), min(v1, v2)
    if 'relative' in mn:
        return(np.mean(np.abs((y-p)/y)))
    elif 'rmse' in mn:
        return np.sqrt(np.mean((y-p)**2))
    elif 'mse' in mn:
        return(np.mean((y-p)**2))
    elif 'reverse_cross_entropy' in mn:
        return(np.mean(-p*np.log(y)-(1-p)*np.log(1-y)))
    elif 'cross_entropy' in mn:
        return(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))
    elif 'reverse_kl_diverge' in mn:
        return(np.mean(p*np.log(p)-p*np.log(y)+(1-p)*np.log(1-p)-(1-p)*np.log(1-y)))
    elif 'kl_diverge' in mn:
        return(np.mean(y*np.log(y)-y*np.log(p)+(1-y)*np.log(1-y)-(1-y)*np.log(1-p)))
    else:
        raise ValueError("Wrong metric name: ", mn)

def aggregate_loss_func(losses, metric_names):
    losses = list(zip(*losses)) #transpose
    agg_losses = []
    for mn, l in zip(metric_names, losses):
        al = None
        if isinstance(l[0], list) :
            l = [lll for ll in l for lll in ll]
        if 'mean' in mn:
            al = np.mean(l)
        elif 'std' in mn:
            al = np.std(l)
        elif 'max' in mn:
            al = np.max(l)
        elif 'min' in mn:
            al = np.min(l)
        elif 'median' in mn:
            al = np.median(l)
        else:
            raise ValueError("Wrong Metric Name: ", mn)
        agg_losses.append(al)
    return agg_losses

# lbp_net.double()
def train():
    # if USE_WANDB:
        # wandb.watch(lbp_net)

    # lbp_net.train()
    # Initialize optimizer
    # optimizer = torch.optim.Adam(lbp_net.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs
    # loss_func = torch.nn.MSELoss()


    spin_glass_models_list_val = get_dataset(dataset_type='val')


#     with autograd.detect_anomaly():
    for e in range(EPOCH_COUNT):
        epoch_loss, losses = 0, []
        for spin_glass_problem in spin_glass_models_list_val:
            map_states = spin_glass_problem.map_marginal_junction_tree_libdai()
            loss = loss_func(
                map_states,
                spin_glass_problem,
                metric_names,
            )
            losses.append(loss)

        aggregated_losses = aggregate_loss_func(losses, metric_names)
        for mn, l in zip(metric_names, aggregated_losses):
            print(mn, ' = ', l)

        print()
        if USE_WANDB:
            train_logs = {mn:l for mn, l in zip(metric_names, aggregated_losses)}
            wandb.log(train_logs)
            # wandb.log({"RMSE_val": np.sqrt(np.mean(val_losses)), "RMSE_training": np.sqrt(np.mean(losses))})


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
