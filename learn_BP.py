import torch
from torch import autograd
from models import lbp_message_passing_network
from sat_data import SatProblems
from factor_graph import FactorGraph
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
##########################
####### PARAMETERS #######
MAX_FACTOR_STATE_DIMENSIONS = 5 #number of variables in the largest factor -> factor has 2^MAX_FACTOR_STATE_DIMENSIONS states
MSG_PASSING_ITERS = 2 #the number of iterations of message passing, we have this many layers with their own learnable parameters


MODEL_NAME = "simple_4layer.pth"
ROOT_DIR = "/atlas/u/jkuck/learn_BP/" #file path to the directory cloned from github
TRAINED_MODELS_DIR = ROOT_DIR + "trained_models/" #trained models are stored here

TRAINING_DATA_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/"
TRAINING_DATA_SIZE = 10
VALIDATION_DATA_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/"
VAL_DATA_SIZE = 2#100

EPOCH_COUNT = 40
PRINT_FREQUENCY = 1
SAVE_FREQUENCY = 10
##########################

lbp_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS, msg_passing_iters=MSG_PASSING_ITERS)

def train(dataset_size, data_dir):
    lbp_net.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.0001)
    loss_func = torch.nn.MSELoss()

    sat_data_train = SatProblems(counts_dir_name=data_dir + "SAT_problems_solved_counts",
               problems_dir_name=data_dir + "SAT_problems_solved",
               # dataset_size=100, begin_idx=50)
               dataset_size=dataset_size)
    train_data_loader = DataLoader(sat_data_train, batch_size=1)

    sat_data_val = SatProblems(counts_dir_name=data_dir + "SAT_problems_solved_counts",
               problems_dir_name=data_dir + "SAT_problems_solved",
               # dataset_size=50, begin_idx=0)
               dataset_size=VAL_DATA_SIZE, begin_idx=TRAINING_DATA_SIZE)
    val_data_loader = DataLoader(sat_data_val, batch_size=1)


    # with autograd.detect_anomaly():
    losses = []
    for e in range(EPOCH_COUNT):
        for t, (sat_problem, exact_ln_partition_function) in enumerate(train_data_loader):
            optimizer.zero_grad()

            sat_problem = FactorGraph.init_from_dictionary(sat_problem, squeeze_tensors=True)
            assert(sat_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)
            estimated_ln_partition_function = lbp_net(sat_problem)

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
            # scheduler.step()

        if e % PRINT_FREQUENCY == 0:
            print("root mean squared training error =", np.sqrt(np.sum(losses)))
            losses = []
            val_losses = []
            for t, (sat_problem, exact_ln_partition_function) in enumerate(val_data_loader):
                sat_problem = FactorGraph.init_from_dictionary(sat_problem, squeeze_tensors=True)
                assert(sat_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)
                estimated_ln_partition_function = lbp_net(sat_problem)                
                loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
                val_losses.append(loss.item())
            print("root mean squared validation error =", np.sqrt(np.sum(val_losses)))
            print()

        if e % SAVE_FREQUENCY == 0:
            if not os.path.exists(TRAINED_MODELS_DIR):
                os.makedirs(TRAINED_MODELS_DIR)
            torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)

    if not os.path.exists(TRAINED_MODELS_DIR):
        os.makedirs(TRAINED_MODELS_DIR)
    torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)

def test(dataset_size, data_dir):
    # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))
    # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))

    lbp_net.eval()

    sat_data = SatProblems(counts_dir_name=data_dir + "SAT_problems_solved_counts",
               problems_dir_name=data_dir + "SAT_problems_solved",
               dataset_size=50, begin_idx=0)
               # dataset_size=100, begin_idx=50)
               # dataset_size=dataset_size)

    data_loader = DataLoader(sat_data, batch_size=1)
    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    lbp_estimated_counts = []
    losses = []
    for sat_problem, exact_ln_partition_function in data_loader:
        # sat_problem.compute_bethe_free_energy()
        sat_problem = FactorGraph.init_from_dictionary(sat_problem, squeeze_tensors=True)
        estimated_ln_partition_function = lbp_net(sat_problem)
        lbp_estimated_counts.append(estimated_ln_partition_function)
        exact_solution_counts.append(exact_ln_partition_function)
        loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
        losses.append(loss.item())

        print("estimated_ln_partition_function:", estimated_ln_partition_function)
        print("exact_ln_partition_function:", exact_ln_partition_function)
        print()

    plt.plot(exact_solution_counts, lbp_estimated_counts, 'x', c='b', label='Negative Bethe Free Energy, %d iters, RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.sum(losses))))
    plt.plot([min(exact_solution_counts), max(exact_solution_counts)], [min(exact_solution_counts), max(exact_solution_counts)], '-', c='g', label='Exact Estimate')

    # plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='Ground Truth ln(Set Size)') 
    plt.xlabel('ln(Exact Model Count)', fontsize=14)
    plt.ylabel('ln(Estimated Model Count)', fontsize=14)
    plt.title('Exact Model Count vs. Bethe Estimate', fontsize=20)
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
    train(dataset_size=TRAINING_DATA_SIZE, data_dir=TRAINING_DATA_DIR)
    # test(dataset_size=TRAINING_DATA_SIZE, data_dir=TRAINING_DATA_DIR)