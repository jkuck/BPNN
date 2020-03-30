import torch
from torch import autograd
import pickle
import wandb
import random

from nn_models import lbp_message_passing_network
from community_detection.sbm_data import StochasticBlockModel, build_factorgraph_from_sbm
from factor_graph import FactorGraphData
from torch_geometric.data import DataLoader as DataLoader_pytorchGeometric


import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import parameters
from parameters import ROOT_DIR, alpha, alpha2, SHARE_WEIGHTS, BETHE_MLP, NUM_MLPS

USE_WANDB = True
# os.environ['WANDB_MODE'] = 'dryrun' #don't save to the cloud with this option
MODE = "train" #run "test" or "train" mode

##########################
##### Run me on Atlas
# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate



#####Testing parameters
TEST_TRAINED_MODEL = True #test a pretrained model if True.  Test untrained model if False (e.g. LBP)
EXPERIMENT_NAME = 'trained_attrField_10layer_2MLPs_noFinalBetheMLP/' #used for saving results when MODE='test'
BPNN_trained_model_path = './wandb/run-20200219_090032-11077pcu/model.pt' #location of the trained BPNN model, SET ME!


###################################
####### Training PARAMETERS #######
MAX_FACTOR_STATE_DIMENSIONS = 2
MSG_PASSING_ITERS = 10 #the number of iterations of message passing, we have this many layers with their own learnable parameters

# MODEL_NAME = "debugCUDA_spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)
MODEL_NAME = "spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)



##########################################
####### Data Generation PARAMETERS #######
N_TRAIN = 500 #nodes in each graph
P_TRAIN = 10.0/N_TRAIN #probability of an edge between vertices in the same community
Q_TRAIN = 2.0/N_TRAIN #probability of an edge between vertices in different communities
C_TRAIN = 2 #number of communities
# could also store the probability that each node belongs to each community, but that's uniform for now


N_VAL = 500
P_VAL = 10.0/N_TRAIN #probability of an edge between vertices in the same community
Q_VAL = 2.0/N_TRAIN #probability of an edge between vertices in different communities
C_VAL = 2 #number of communities

REGENERATE_DATA = True
DATA_DIR = "./data/community_detection/SBM"


TRAINING_DATA_SIZE = 2
VAL_DATA_SIZE = 2#100
TEST_DATA_SIZE = 200


EPOCH_COUNT = 300
PRINT_FREQUENCY = 1
VAL_FREQUENCY = 10
SAVE_FREQUENCY = 1

TEST_DATSET = 'val' #can test and plot results for 'train', 'val', or 'test' datasets

##### Optimizer parameters #####
STEP_SIZE=300
LR_DECAY=.5
LEARNING_RATE = 0.0005


##########################
if USE_WANDB:
    wandb.init(project="learnBP_spinGlass")
    wandb.config.epochs = EPOCH_COUNT
    wandb.config.N_TRAIN = N_TRAIN
    wandb.config.P_TRAIN = P_TRAIN
    wandb.config.Q_TRAIN = Q_TRAIN
    wandb.config.C_TRAIN = C_TRAIN
    wandb.config.TRAINING_DATA_SIZE = TRAINING_DATA_SIZE
    wandb.config.alpha = alpha
    wandb.config.alpha2 = alpha2
    wandb.config.SHARE_WEIGHTS = SHARE_WEIGHTS
    wandb.config.BETHE_MLP = BETHE_MLP
    wandb.config.MSG_PASSING_ITERS = MSG_PASSING_ITERS
    wandb.config.STEP_SIZE = STEP_SIZE
    wandb.config.LR_DECAY = LR_DECAY
    wandb.config.LEARNING_RATE = LEARNING_RATE
    wandb.config.NUM_MLPS = NUM_MLPS





def get_dataset(dataset_type):
    '''
    Store/load a list of SBMs
    When using, convert to BPNN with build_factorgraph_from_sbm()
    '''
    assert(dataset_type in ['train', 'val', 'test'])
    if dataset_type == 'train':
        datasize = TRAINING_DATA_SIZE
        N = N_TRAIN
        P = P_TRAIN
        Q = Q_TRAIN
        C = C_TRAIN
    elif dataset_type == 'val':
        datasize = VAL_DATA_SIZE
        N = N_VAL
        P = P_VAL
        Q = Q_VAL
        C = C_VAL     
    else:
        datasize = TEST_DATA_SIZE
        
    dataset_file = DATA_DIR + dataset_type + '%d_%d_%d_%.2f_%.2f.pkl' % (datasize, N, C, P, Q)
    if REGENERATE_DATA or (not os.path.exists(dataset_file)):
        print("REGENERATING DATA!!")
        spin_glass_models_list = [StochasticBlockModel(N=N, P=P, Q=Q, C=C) for i in range(datasize)]
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
bpnn_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS,\
                                      msg_passing_iters=MSG_PASSING_ITERS, device=None)

bpnn_net = bpnn_net.to(device)

# bpnn_net.double()
def train():
    if USE_WANDB:
        
        wandb.watch(bpnn_net)
    
    bpnn_net.train()
    # Initialize optimizer
    optimizer = torch.optim.Adam(bpnn_net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs    
    loss_func = torch.nn.MSELoss()


    spin_glass_models_list_train = get_dataset(dataset_type='train')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sbm_models_fg_train = [build_factorgraph_from_sbm(sg_model) for sg_model in spin_glass_models_list_train]
    train_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg_train, batch_size=1)

    
    spin_glass_models_list_val = get_dataset(dataset_type='val')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sbm_models_fg_val = [build_factorgraph_from_sbm(sg_model) for sg_model in spin_glass_models_list_val]
    val_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg_val, batch_size=1)
    
    # with autograd.detect_anomaly():
    for e in range(EPOCH_COUNT):
        epoch_loss = 0
        optimizer.zero_grad()
        losses = []
        for spin_glass_problem in train_data_loader_pytorchGeometric: #pytorch geometric form
            # spin_glass_problem.facToVar_edge_idx = spin_glass_problem.edge_index #this is a hack for batching, fix me!

            exact_ln_partition_function = spin_glass_problem.ln_Z
            assert((spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS).all())
            
            estimated_ln_partition_function = bpnn_net(spin_glass_problem)

            loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
            epoch_loss += loss
            # print("loss:", loss)
            # print()
            losses.append(loss.item())
            
        epoch_loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        scheduler.step()
                    
        
        if e % PRINT_FREQUENCY == 0:
            print("root mean squared training error =", np.sqrt(np.mean(losses)))
     
        if e % VAL_FREQUENCY == 0:
            val_losses = []
            for spin_glass_problem in val_data_loader_pytorchGeometric: #pytorch geometric form
#                 spin_glass_problem = spin_glass_problem.to(device)
                exact_ln_partition_function = spin_glass_problem.ln_Z   
                assert(spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)

                estimated_ln_partition_function = bpnn_net(spin_glass_problem)            
                loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
                # print("estimated_ln_partition_function:", estimated_ln_partition_function)
                # print("exact_ln_partition_function:", exact_ln_partition_function)

                val_losses.append(loss.item())
            print("root mean squared validation error =", np.sqrt(np.mean(val_losses)))
            print()
            if USE_WANDB:
                wandb.log({"RMSE_val": np.sqrt(np.mean(val_losses)), "RMSE_training": np.sqrt(np.mean(losses))})        
        else:
            if USE_WANDB:
                wandb.log({"RMSE_training": np.sqrt(np.mean(losses))})

        if e % SAVE_FREQUENCY == 0:
            if USE_WANDB:
                # Save model to wandb
                torch.save(bpnn_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

    if USE_WANDB:
        # Save model to wandb
        torch.save(bpnn_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


def test(skip_our_model=False):
    if TEST_TRAINED_MODEL:
        bpnn_net.load_state_dict(torch.load(BPNN_trained_model_path))


    bpnn_net.eval()

    spin_glass_models_list = get_dataset(dataset_type=TEST_DATSET)
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sbm_models_fg = [build_factorgraph_from_sbm(sg_model) for sg_model in spin_glass_models_list]
    data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg, batch_size=1)
    
    
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
            estimated_ln_partition_function = bpnn_net(spin_glass_problem)
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
   


        
if __name__ == "__main__":
    if MODE == "train":
        train()
    elif MODE == "test":
        test()
