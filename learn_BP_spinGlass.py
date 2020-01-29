import torch
from torch import autograd
from nn_models import lbp_message_passing_network
from ising_model.pytorch_dataset import SpinGlassDataset
from factor_graph import FactorGraph
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

##########################
##### Run me on Atlas
# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate


##########################
####### PARAMETERS #######
MAX_FACTOR_STATE_DIMENSIONS = 2
MSG_PASSING_ITERS = 2 #the number of iterations of message passing, we have this many layers with their own learnable parameters

EPSILON = 0 #set factor states with potential 0 to EPSILON for numerical stability

MODEL_NAME = "spinGlass_%dlayer.pth" % MSG_PASSING_ITERS
ROOT_DIR = "/atlas/u/jkuck/learn_BP/" #file path to the directory cloned from github
TRAINED_MODELS_DIR = ROOT_DIR + "trained_models/" #trained models are stored here



##########################################################################################################
N_MIN = 8
N_MAX = 11
F_MAX = 5.0
C_MAX = 5.0

#contains CNF files for training/validation/test problems
TRAINING_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/training_spinGlass/"

VALIDATION_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/val_spinGlass/"

# TEST_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/test_SAT_problems/"
TEST_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/training_SAT_problems/"

TRAINING_DATA_SIZE = 10
VAL_DATA_SIZE = 10#100
TEST_DATA_SIZE = 100


EPOCH_COUNT = 40
PRINT_FREQUENCY = 1
SAVE_FREQUENCY = 10
##########################


lbp_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS, msg_passing_iters=MSG_PASSING_ITERS)
# lbp_net.double()
def train():
    lbp_net.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #multiply lr by gamma every step_size epochs    

    loss_func = torch.nn.MSELoss()

    sg_data_train = SpinGlassDataset(dataset_size=TRAINING_DATA_SIZE, N_min=N_MIN, N_max=N_MAX, f_max=F_MAX, c_max=C_MAX)
    # sleep(temp)
    train_data_loader = DataLoader(sg_data_train, batch_size=1)

    sg_data_val = SpinGlassDataset(dataset_size=VAL_DATA_SIZE, N_min=N_MIN, N_max=N_MAX, f_max=F_MAX, c_max=C_MAX)
    val_data_loader = DataLoader(sg_data_val, batch_size=1)


    # with autograd.detect_anomaly():
    losses = []
    for e in range(EPOCH_COUNT):
        for t, (spin_glass_problem, exact_ln_partition_function, lbp_Z_est, mrftools_lbp_Z_estimate) in enumerate(train_data_loader):
            optimizer.zero_grad()

            spin_glass_problem = FactorGraph.init_from_dictionary(spin_glass_problem, squeeze_tensors=True)
            assert(spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)
            estimated_ln_partition_function = lbp_net(spin_glass_problem)

            # print("estimated_ln_partition_function:", estimated_ln_partition_function)
            # print("type(estimated_ln_partition_function):", type(estimated_ln_partition_function))
            # print("exact_ln_partition_function:", exact_ln_partition_function)
            # print("type(exact_ln_partition_function):", type(exact_ln_partition_function))
            loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
            # print("loss:", loss)
            # print()
            losses.append(loss.item())
            loss.backward()
            # nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

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

        # scheduler.step()

    if not os.path.exists(TRAINED_MODELS_DIR):
        os.makedirs(TRAINED_MODELS_DIR)
    torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)

def test():
    lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))
    # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
    # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))


    lbp_net.eval()

    sg_data = SpinGlassDataset(dataset_size=TEST_DATA_SIZE, N_min=N_MIN, N_max=N_MAX, f_max=F_MAX, c_max=C_MAX)

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
    plt.plot(exact_solution_counts, LBPmrftools_estimated_counts, '+', c='r', label='LBP mrftools, RMSE=%.2f, 10 lrgst removed RMSE=%.2f' % (np.sqrt(np.mean(mrftool_lbp_losses)), np.sqrt(np.mean(mrftool_lbp_losses[:-10]))))
    plt.plot(exact_solution_counts, LBPlibdai_estimated_counts, 'x', c='b', label='LBP libdai, RMSE=%.2f, 10 lrgst removed RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses)), np.sqrt(np.mean(lbp_losses[:-10]))))
    plt.plot([min(exact_solution_counts), max(exact_solution_counts)], [min(exact_solution_counts), max(exact_solution_counts)], '-', c='g', label='Exact')

    # plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='Ground Truth ln(Set Size)') 
    plt.xlabel('ln(Exact Model Count)', fontsize=14)
    plt.ylabel('ln(Estimated Model Count)', fontsize=14)
    plt.title('Exact Model Count vs. Estimates', fontsize=20)
    plt.legend(fontsize=12)    
    #make the font bigger
    matplotlib.rcParams.update({'font.size': 10})        

    plt.grid(True)
    # Shrink current axis's height by 10% on the bottom
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])
    #fig.savefig('/Users/jkuck/Downloads/temp.png', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    plt.show()





if __name__ == "__main__":
    # train()
    test()