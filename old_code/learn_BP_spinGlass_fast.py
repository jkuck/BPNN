import torch
from torch import autograd
import pickle
import wandb
import sys

from nn_models import lbp_message_passing_network
from ising_model.pytorch_dataset import SpinGlassDataset, build_factorgraph_from_SpinGlassModel
from ising_model.spin_glass_model import SpinGlassModel
from factor_graph import FactorGraph, FactorGraphData
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader, Data
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import parameters
from parameters import ROOT_DIR
import random
import cProfile 
import time

##########################
##### Run me on Atlas
# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate

MODE = "train" #run "test" or "train" mode
TEST_TRAINED_MODEL = True #test a pretrained model if True.  Test untrained model if False (e.g. LBP)

##########################
####### PARAMETERS #######
MAX_FACTOR_STATE_DIMENSIONS = 2
MSG_PASSING_ITERS = 5 #the number of iterations of message passing, we have this many layers with their own learnable parameters

EPSILON = 0 #set factor states with potential 0 to EPSILON for numerical stability

# MODEL_NAME = "debugCUDA_spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)
MODEL_NAME = "spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)

TRAINED_MODELS_DIR = ROOT_DIR + "trained_models/" #trained models are stored here



##########################################################################################################
# N_MIN = 8
# N_MAX = 11
# F_MAX = 5.0
# C_MAX = 5.0
N_MIN = 10
N_MAX = 10
F_MAX = .1
C_MAX = 5.0
# F_MAX = 1
# C_MAX = 10.0
ATTRACTIVE_FIELD_TRAIN = True

N_MIN_VAL = 10
N_MAX_VAL = 10
F_MAX_VAL = .1
C_MAX_VAL = 5.0
ATTRACTIVE_FIELD_VAL = True

REGENERATE_DATA = False
DATA_DIR = "/atlas/u/jkuck/learn_BP/data/spin_glass/"


TRAINING_DATA_SIZE = 10
VAL_DATA_SIZE = 50#100
TEST_DATA_SIZE = 200


EPOCH_COUNT = 10000
PRINT_FREQUENCY = 1
VAL_FREQUENCY = 10
SAVE_FREQUENCY = 1

TEST_DATSET = 'test' #can test and plot results for 'train', 'val', or 'test' datasets
##########################

def spinGlass_to_torchGeometric(sg_model):
    '''
    Convert a spin glass model represented as a SpinGlassModel object
    to pytorch geometric Data
    Inputs:
    - sg_model (SpinGlassModel): representation of a spin glass model
    
    Outputs:
    - sg_model_torchGeom (torch_geometric.data.Data): representation of a spin glass model
        as pytorch geometric Data
    '''
    fg = build_factorgraph_from_SpinGlassModel(sg_model)
    sg_model_torchGeom = Data()
    sg_model_torchGeom.factor_potentials = fg.factor_potentials
    sg_model_torchGeom.facToVar_edge_idx = fg.factorToVar_edge_index
#     sg_model_torchGeom.factorToVar_edge_index = torch.tensor([2])#fg.factor_potentials#fg.factorToVar_edge_index
    sg_model_torchGeom.factor_degrees = fg.factor_degrees
    sg_model_torchGeom.var_degrees = fg.var_degrees
    sg_model_torchGeom.numVars = fg.numVars
    sg_model_torchGeom.numFactors = fg.numFactors
    sg_model_torchGeom.edge_var_indices = fg.edge_var_indices
    sg_model_torchGeom.state_dimensions = fg.state_dimensions
    sg_model_torchGeom.factor_potential_masks = fg.factor_potential_masks
    sg_model_torchGeom.facStates_to_varIdx = fg.facStates_to_varIdx
    
#     print("sg_model_torchGeom.factorToVar_edge_index.type():", sg_model_torchGeom.factorToVar_edge_index.type())
#     print("sg_model_torchGeom.factorToVar_edge_index.shape:", sg_model_torchGeom.factorToVar_edge_index.shape)
    print("sg_model_torchGeom.factor_potentials.shape:", sg_model_torchGeom.factor_potentials.shape)
    print("sg_model_torchGeom.numVars.shape:", sg_model_torchGeom.numVars.shape)
    return sg_model_torchGeom

train_data_list = [build_factorgraph_from_SpinGlassModel(SpinGlassModel(N=random.randint(N_MIN, N_MAX),\
# train_data_list = [spinGlass_to_torchGeometric(SpinGlassModel(N=random.randint(N_MIN, N_MAX),\
                                                        f=np.random.uniform(low=0, high=F_MAX),\
                                                        c=np.random.uniform(low=0, high=C_MAX),\
                                                        attractive_field=ATTRACTIVE_FIELD_TRAIN)) for i in range(TRAINING_DATA_SIZE)]
train_data_loader = DataLoader(train_data_list, batch_size=1)

val_data_list = [build_factorgraph_from_SpinGlassModel(SpinGlassModel(N=random.randint(N_MIN_VAL, N_MAX_VAL),\
# val_data_list = [spinGlass_to_torchGeometric(SpinGlassModel(N=random.randint(N_MIN_VAL, N_MAX_VAL),\
                                                        f=np.random.uniform(low=0, high=F_MAX_VAL),\
                                                        c=np.random.uniform(low=0, high=C_MAX_VAL),\
                                                        attractive_field=ATTRACTIVE_FIELD_VAL)) for i in range(VAL_DATA_SIZE)]
val_data_loader = DataLoader(val_data_list, batch_size=1)
 
device = torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lbp_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS,\
                                      msg_passing_iters=MSG_PASSING_ITERS, device=None)

lbp_net = lbp_net.to(device)



def trace(frame, event, arg):
    print("%s, %s:%d" % (event, frame.f_code.co_filename, frame.f_lineno))
    return trace  

# lbp_net.double()
def train():
#     sys.settrace(trace)

    wandb.init(project="gnn_sat")
    wandb.watch(lbp_net)
    
    lbp_net.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.005)
#     optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.00005)
#     optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.002) #used for training on 50
#     optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.001)
#     optimizer = torch.optim.SGD(lbp_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) #multiply lr by gamma every step_size epochs    

    loss_func = torch.nn.MSELoss()


#     sg_data_train, spin_glass_problems_SGMs_train = get_dataset(dataset_type='train')
#     # sg_data_train = SpinGlassDataset(dataset_size=datasize, N_min=N_MIN, N_max=N_MAX, f_max=F_MAX, c_max=C_MAX)
#     train_data_loader = DataLoader(sg_data_train, batch_size=1)
#     sg_data_val, spin_glass_problems_SGMs_val = get_dataset(dataset_type='val')
#     val_data_loader = DataLoader(sg_data_val, batch_size=1)


    # with autograd.detect_anomaly():
    epoch_times = []
    for e in range(EPOCH_COUNT):
        time_a = time.time()
        
        epoch_loss = 0
        optimizer.zero_grad()
        losses = []
#         for t, (spin_glass_problem, exact_ln_partition_function, lbp_Z_est, mrftools_lbp_Z_estimate) in enumerate(train_data_loader):
        for spin_glass_problem in train_data_loader:
#             sleep(entered_loop)
#             spin_glass_problem = FactorGraph.init_from_dictionary(spin_glass_problem, squeeze_tensors=True)
            assert((spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS).all())
#             spin_glass_problem.to_device(device)
            estimated_ln_partition_function = lbp_net(spin_glass_problem)
            exact_ln_partition_function = spin_glass_problem.ln_Z

#             print("estimated_ln_partition_function:", estimated_ln_partition_function)
            # print("type(estimated_ln_partition_function):", type(estimated_ln_partition_function))
#             print("exact_ln_partition_function:", exact_ln_partition_function)
            # print("type(exact_ln_partition_function):", type(exact_ln_partition_function))
#             print(estimated_ln_partition_function.device, exact_ln_partition_function.device)
#             exact_ln_partition_function = exact_ln_partition_function.to(device)
            loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
            epoch_loss += loss
            # print("loss:", loss)
            # print()
            losses.append(loss.item())
            
        epoch_loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        scheduler.step()
        time_b = time.time()
        epoch_times.append(time_b - time_a)        
                    
        
        if e % PRINT_FREQUENCY == 0:
            print("root mean squared training error =", np.sqrt(np.mean(losses)))
     
        if e % VAL_FREQUENCY == 0:
            print("average time per epoch:", np.mean(epoch_times))
            epoch_times = []
#         if False:
            val_losses = []
#             for t, (spin_glass_problem, exact_ln_partition_function, lbp_Z_est, mrftools_lbp_Z_estimate) in enumerate(val_data_loader):
            for spin_glass_problem in val_data_loader:
                exact_ln_partition_function = spin_glass_problem.ln_Z            
            
                assert(spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)
#                 spin_glass_problem.to_device(device)
                estimated_ln_partition_function = lbp_net(spin_glass_problem) 
                loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
                # print("estimated_ln_partition_function:", estimated_ln_partition_function)

                # print("exact_ln_partition_function:", exact_ln_partition_function)

                val_losses.append(loss.item())
            print("root mean squared validation error =", np.sqrt(np.mean(val_losses)))
            print()
            wandb.log({"RMSE_val": np.sqrt(np.mean(val_losses)), "RMSE_training": np.sqrt(np.mean(losses))})        
        else:
            wandb.log({"RMSE_training": np.sqrt(np.mean(losses))})

        if e % SAVE_FREQUENCY == 0:
            if not os.path.exists(TRAINED_MODELS_DIR):
                os.makedirs(TRAINED_MODELS_DIR)
            torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)
            # Save model to wandb
            torch.save(lbp_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

    if not os.path.exists(TRAINED_MODELS_DIR):
        os.makedirs(TRAINED_MODELS_DIR)
    torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)

def create_ising_model_figure(skip_our_model=False):
    if TEST_TRAINED_MODEL:
        lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))


    lbp_net.eval()

    sg_data, spin_glass_problems_SGMs = get_dataset(dataset_type=TEST_DATSET)

    data_loader = DataLoader(sg_data, batch_size=1)
    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    GNN_estimated_counts = []
    LBPlibdai_5iters_estimated_counts = []
    LBPlibdai_10iters_estimated_counts = []
    LBPlibdai_30iters_estimated_counts = []
    LBPlibdai_50iters_estimated_counts = []
    LBPlibdai_500iters_estimated_counts = []
    LBPlibdai_5kiters_estimated_counts = []
    
    meanFieldlibdai_5kiters_estimated_counts = []
    
    lbp_losses_5iters = []
    lbp_losses_10iters = []
    lbp_losses_30iters = []
    lbp_losses_50iters = []
    lbp_losses_500iters = []
    lbp_losses_5kiters = []
    
    mean_field_losses_5kiters = []
    
    losses = []
    lbp_losses = []
    mrftool_lbp_losses = []
    for idx, (spin_glass_problem, exact_ln_partition_function, libdai_lbp_Z_est, mrftools_lbp_Z_estimate) in enumerate(data_loader):
        print("problem:", idx)
        # spin_glass_problem.compute_bethe_free_energy()     
        sg_problem_SGM = spin_glass_problems_SGMs[idx]
        if not skip_our_model:
            spin_glass_problem = FactorGraph.init_from_dictionary(spin_glass_problem, squeeze_tensors=True)
#             spin_glass_problem = spin_glass_problem.to(device)
#             exact_ln_partition_function = exact_ln_partition_function.to(device)
            estimated_ln_partition_function = lbp_net(spin_glass_problem)
            GNN_estimated_counts.append(estimated_ln_partition_function.item()-exact_ln_partition_function)
            loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
            losses.append(loss.item())
            
        libdai_lbp_Z_5 = sg_problem_SGM.loopyBP_libdai(maxiter=5)
        libdai_lbp_Z_10 = sg_problem_SGM.loopyBP_libdai(maxiter=10)
        libdai_lbp_Z_30 = sg_problem_SGM.loopyBP_libdai(maxiter=20)
        libdai_lbp_Z_50 = sg_problem_SGM.loopyBP_libdai(maxiter=50)
        libdai_lbp_Z_500 = sg_problem_SGM.loopyBP_libdai(maxiter=500)
#         libdai_lbp_Z_5k = sg_problem_SGM.loopyBP_libdai(maxiter=5000)

        LBPlibdai_5iters_estimated_counts.append(libdai_lbp_Z_5-exact_ln_partition_function)
        LBPlibdai_10iters_estimated_counts.append(libdai_lbp_Z_10-exact_ln_partition_function)
        LBPlibdai_30iters_estimated_counts.append(libdai_lbp_Z_30-exact_ln_partition_function)
        LBPlibdai_50iters_estimated_counts.append(libdai_lbp_Z_50-exact_ln_partition_function)
        LBPlibdai_500iters_estimated_counts.append(libdai_lbp_Z_500-exact_ln_partition_function)
#         LBPlibdai_5kiters_estimated_counts.append(libdai_lbp_Z_5k-exact_ln_partition_function)

        libdai_meanField_Z_5k = sg_problem_SGM.mean_field_libdai(maxiter=100000)
        meanFieldlibdai_5kiters_estimated_counts.append(libdai_meanField_Z_5k-exact_ln_partition_function)

        
        
        exact_solution_counts.append(exact_ln_partition_function)


        lbp_losses_5iters.append(loss_func(torch.tensor(libdai_lbp_Z_5), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_10iters.append(loss_func(torch.tensor(libdai_lbp_Z_10), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_30iters.append(loss_func(torch.tensor(libdai_lbp_Z_30), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_50iters.append(loss_func(torch.tensor(libdai_lbp_Z_50), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_500iters.append(loss_func(torch.tensor(libdai_lbp_Z_500), exact_ln_partition_function.float().squeeze()).item())
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
        plt.plot(exact_solution_counts, GNN_estimated_counts, 'x', label='%d layer BPNN, RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.mean(losses))))
        
    plt.plot(exact_solution_counts, LBPlibdai_5iters_estimated_counts, '+', label='LBP 5 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_5iters))))
    
    plt.plot(exact_solution_counts, LBPlibdai_10iters_estimated_counts, '+', label='LBP 10 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_10iters))))
    
    plt.plot(exact_solution_counts, LBPlibdai_30iters_estimated_counts, '+', label='LBP 20 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_30iters))))
    
    plt.plot(exact_solution_counts, LBPlibdai_50iters_estimated_counts, '+', label='LBP 50 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_50iters))))
    
    plt.plot(exact_solution_counts, LBPlibdai_500iters_estimated_counts, '+', label='LBP 500 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_500iters))))
    
#     plt.plot(exact_sol”ution_counts, LBPlibdai_5kiters_estimated_counts, '+', label='LBP 5k iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_5kiters))))
    
    plt.plot(exact_solution_counts, meanFieldlibdai_5kiters_estimated_counts, '1', label='Mean Field, RMSE=%.2f' % (np.sqrt(np.mean(mean_field_losses_5kiters))))
    
    plt.plot([min(exact_solution_counts), max(exact_solution_counts)], [0, 0], '-', c='g', label='Exact')

    # plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='Ground Truth ln(Set Size)') 
    plt.xlabel('ln(Z)', fontsize=14)
    plt.ylabel('ln(Estimate) - ln(Z)', fontsize=14)
    plt.title('Exact Partition Function vs. Estimates', fontsize=20)
    # plt.legend(fontsize=8, loc=2, prop={'size': 6})    
    plt.legend(fontsize=12, prop={'size': 12})    
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
    plot_name = 'trained=%s_dataset=%s%d_c%f_f%f_%diters_alpha%f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, len(data_loader), C_MAX, F_MAX, MSG_PASSING_ITERS, parameters.alpha)
    plt.savefig(ROOT_DIR + 'plots/' + plot_name)
    # plt.show()



def test(skip_our_model=False):
    if TEST_TRAINED_MODEL:
        lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))


    lbp_net.eval()

    sg_data, spin_glass_problems_SGMs = get_dataset(dataset_type=TEST_DATSET)

    data_loader = DataLoader(sg_data, batch_size=1)
    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    GNN_estimated_counts = []
    LBPlibdai_estimated_counts = []
    LBPmrftools_estimated_counts = []
    losses = []
    lbp_losses = []
    mrftool_lbp_losses = []
    for idx, (spin_glass_problem, exact_ln_partition_function, libdai_lbp_Z_est, mrftools_lbp_Z_estimate) in enumerate(data_loader):
        # spin_glass_problem.compute_bethe_free_energy()     
        sg_problem_SGM = spin_glass_problems_SGMs[idx]
        if not skip_our_model:
            spin_glass_problem = FactorGraph.init_from_dictionary(spin_glass_problem, squeeze_tensors=True)
#             spin_glass_problem = spin_glass_problem.to(device)
#             exact_ln_partition_function = exact_ln_partition_function.to(device)
            estimated_ln_partition_function = lbp_net(spin_glass_problem)
            GNN_estimated_counts.append(estimated_ln_partition_function.item()-exact_ln_partition_function)
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
        plt.plot(exact_solution_counts, GNN_estimated_counts, 'x', c='g', label='GNN estimate, %d iters, RMSE=%.2f, 10 lrgst removed RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.mean(losses)), np.sqrt(np.mean(losses[:-10]))))
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
    plot_name = 'trained=%s_dataset=%s%d_%diters_alpha%f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, len(data_loader), MSG_PASSING_ITERS, parameters.alpha)
    plt.savefig(ROOT_DIR + 'plots/' + plot_name)
    # plt.show()


if __name__ == "__main__":
    if MODE == "train":
        train()
#         cProfile.run("train()") 

    elif MODE == "test":
#         test()
        create_ising_model_figure()