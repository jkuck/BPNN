import torch
from torch import autograd
import pickle

from nn_models import lbp_message_passing_network
from ising_model.pytorch_dataset import SpinGlassDataset
from factor_graph import FactorGraph
from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import parameters

##########################
##### Run me on Atlas
# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate

MODE = "train" #run "test" or "train" mode
##########################
####### PARAMETERS #######
MAX_FACTOR_STATE_DIMENSIONS = 2
MSG_PASSING_ITERS = 10 #the number of iterations of message passing, we have this many layers with their own learnable parameters

EPSILON = 0 #set factor states with potential 0 to EPSILON for numerical stability

MODEL_NAME = "spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)

ROOT_DIR = "/atlas/u/jkuck/learn_BP/" #file path to the directory cloned from github
TRAINED_MODELS_DIR = ROOT_DIR + "trained_models/" #trained models are stored here



##########################################################################################################
N_MIN = 8
N_MAX = 11
F_MAX = 5.0
C_MAX = 5.0

REGENERATE_DATA = False
DATA_DIR = "/atlas/u/jkuck/learn_BP/data/spin_glass/"


TRAINING_DATA_SIZE = 50
VAL_DATA_SIZE = 5#100
TEST_DATA_SIZE = 50


EPOCH_COUNT = 1000
PRINT_FREQUENCY = 1
SAVE_FREQUENCY = 1

TEST_DATSET = 'test' #can test and plot results for 'train', 'val', or 'test' datasets
TEST_TRAINED_MODEL = True #test a pretrained model if True.  Test untrained model if False (e.g. LBP)
##########################


def get_dataset(dataset_type):
    assert(dataset_type in ['train', 'val', 'test'])
    if dataset_type == 'train':
        datasize = TRAINING_DATA_SIZE
    elif dataset_type == 'val':
        datasize = VAL_DATA_SIZE
    else:
        datasize = TEST_DATA_SIZE
    dataset_file = DATA_DIR + dataset_type + '%d_%d_%d_%.2f_%.2f.pkl' % (datasize, N_MIN, N_MAX, F_MAX, C_MAX)
    if REGENERATE_DATA or (not os.path.exists(dataset_file)):
        print("REGENERATING DATA!!")
        sg_data = SpinGlassDataset(dataset_size=datasize, N_min=N_MIN, N_max=N_MAX, f_max=F_MAX, c_max=C_MAX)
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(dataset_file, 'wb') as f:
            pickle.dump(sg_data, f)            
    else:
        with open(dataset_file, 'rb') as f:
            sg_data = pickle.load(f)
    return sg_data

lbp_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS, msg_passing_iters=MSG_PASSING_ITERS)
# lbp_net.double()
def train():
    lbp_net.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.0005)
#     optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.002) #used for training on 50
#     optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.001)
#     optimizer = torch.optim.SGD(lbp_net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5) #multiply lr by gamma every step_size epochs    

    loss_func = torch.nn.MSELoss()


    sg_data_train = get_dataset(dataset_type='train')
    # sg_data_train = SpinGlassDataset(dataset_size=datasize, N_min=N_MIN, N_max=N_MAX, f_max=F_MAX, c_max=C_MAX)
    train_data_loader = DataLoader(sg_data_train, batch_size=1)
    sg_data_val = get_dataset(dataset_type='val')
    val_data_loader = DataLoader(sg_data_val, batch_size=1)


    # with autograd.detect_anomaly():
    losses = []
    for e in range(EPOCH_COUNT):
        epoch_loss = 0
        optimizer.zero_grad()
        for t, (spin_glass_problem, exact_ln_partition_function, lbp_Z_est, mrftools_lbp_Z_estimate) in enumerate(train_data_loader):

            spin_glass_problem = FactorGraph.init_from_dictionary(spin_glass_problem, squeeze_tensors=True)
            assert(spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)
            estimated_ln_partition_function = lbp_net(spin_glass_problem)

            # print("estimated_ln_partition_function:", estimated_ln_partition_function)
            # print("type(estimated_ln_partition_function):", type(estimated_ln_partition_function))
            # print("exact_ln_partition_function:", exact_ln_partition_function)
            # print("type(exact_ln_partition_function):", type(exact_ln_partition_function))
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
            losses = []
            val_losses = []
            for t, (spin_glass_problem, exact_ln_partition_function, lbp_Z_est, mrftools_lbp_Z_estimate) in enumerate(val_data_loader):
                spin_glass_problem = FactorGraph.init_from_dictionary(spin_glass_problem, squeeze_tensors=True)
                assert(spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)
                estimated_ln_partition_function = lbp_net(spin_glass_problem)                
                loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
                # print("estimated_ln_partition_function:", estimated_ln_partition_function)

                # print("exact_ln_partition_function:", exact_ln_partition_function)

                val_losses.append(loss.item())
            print("root mean squared validation error =", np.sqrt(np.mean(val_losses)))
            print()

        if e % SAVE_FREQUENCY == 0:
            if not os.path.exists(TRAINED_MODELS_DIR):
                os.makedirs(TRAINED_MODELS_DIR)
            torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)


    if not os.path.exists(TRAINED_MODELS_DIR):
        os.makedirs(TRAINED_MODELS_DIR)
    torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)

def test():
    if TEST_TRAINED_MODEL:
        lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))


    lbp_net.eval()

    sg_data = get_dataset(dataset_type=TEST_DATSET)

    data_loader = DataLoader(sg_data, batch_size=1)
    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    GNN_estimated_counts = []
    LBPlibdai_estimated_counts = []
    LBPmrftools_estimated_counts = []
    losses = []
    lbp_losses = []
    mrftool_lbp_losses = []
    for spin_glass_problem, exact_ln_partition_function, libdai_lbp_Z_est, mrftools_lbp_Z_estimate in data_loader:
        # spin_glass_problem.compute_bethe_free_energy()
        spin_glass_problem = FactorGraph.init_from_dictionary(spin_glass_problem, squeeze_tensors=True)
        estimated_ln_partition_function = lbp_net(spin_glass_problem)
        GNN_estimated_counts.append(estimated_ln_partition_function.item())
        LBPlibdai_estimated_counts.append(libdai_lbp_Z_est)
        LBPmrftools_estimated_counts.append(mrftools_lbp_Z_estimate)
        exact_solution_counts.append(exact_ln_partition_function)
        loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
        losses.append(loss.item())
        libdai_lbp_loss = loss_func(libdai_lbp_Z_est, exact_ln_partition_function.float().squeeze())
        lbp_losses.append(libdai_lbp_loss.item())

        mrftools_lbp_loss = loss_func(mrftools_lbp_Z_estimate, exact_ln_partition_function.float().squeeze())
        mrftool_lbp_losses.append(mrftools_lbp_loss.item())

        print("libdai lbp estimated_ln_partition_function:", libdai_lbp_Z_est)
        print("mrf tools lbp estimated_ln_partition_function:", mrftools_lbp_Z_estimate)
        print("GNN estimated_ln_partition_function:", estimated_ln_partition_function)
        print("exact_ln_partition_function:", exact_ln_partition_function)
        print()

    print("LBP libdai MSE:", np.sqrt(np.mean(lbp_losses)))
    print("LBP mrftools MSE:", np.sqrt(np.mean(mrftool_lbp_losses)))
    print("GNN MSE:", np.sqrt(np.mean(losses)))


    losses.sort()
    mrftool_lbp_losses.sort()
    lbp_losses.sort()

    plt.plot(exact_solution_counts, GNN_estimated_counts, 'x', c='g', label='GNN estimate, %d iters, RMSE=%.2f, 10 lrgst removed RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.mean(losses)), np.sqrt(np.mean(losses[:-10]))))
    plt.plot(exact_solution_counts, LBPmrftools_estimated_counts, '+', c='r', label='LBP mrftools, %d iters, RMSE=%.2f, 10 lrgst removed RMSE=%.2f' % (parameters.MRFTOOLS_LBP_ITERS, np.sqrt(np.mean(mrftool_lbp_losses)), np.sqrt(np.mean(mrftool_lbp_losses[:-10]))))
    plt.plot(exact_solution_counts, LBPlibdai_estimated_counts, 'x', c='b', label='LBP libdai, %d iters, RMSE=%.2f, 10 lrgst removed RMSE=%.2f' % (parameters.LIBDAI_LBP_ITERS, np.sqrt(np.mean(lbp_losses)), np.sqrt(np.mean(lbp_losses[:-10]))))
    plt.plot([min(exact_solution_counts), max(exact_solution_counts)], [min(exact_solution_counts), max(exact_solution_counts)], '-', c='g', label='Exact')

    # plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='Ground Truth ln(Set Size)') 
    plt.xlabel('ln(Exact Model Count)', fontsize=14)
    plt.ylabel('ln(Estimated Model Count)', fontsize=14)
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
    elif MODE == "test":
        test()