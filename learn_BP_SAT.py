import torch
from torch import autograd
from nn_models import lbp_message_passing_network
from sat_helpers.sat_data import SatProblems, get_SATproblems_list, parse_dimacs
from sat_helpers.libdai_utils_sat import run_loopyBP
from factor_graph import FactorGraph
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from data.SAT_train_test_split import ALL_TRAIN_PROBLEMS, ALL_TEST_PROBLEMS
import wandb
from parameters import ROOT_DIR, alpha, alpha2, SHARE_WEIGHTS, BETHE_MLP, NUM_MLPS
import random
import resource
import time
import json

##########################
##### Run me on Atlas
# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate


##########################
####### PARAMETERS #######
MAX_FACTOR_STATE_DIMENSIONS = 5 #number of variables in the largest factor -> factor has 2^MAX_FACTOR_STATE_DIMENSIONS states
MSG_PASSING_ITERS = 2 #the number of iterations of message passing, we have this many layers with their own learnable parameters

EPSILON = 0 #set factor states with potential 0 to EPSILON for numerical stability

MODEL_NAME = "diverseData_2layer.pth"
ROOT_DIR = "/atlas/u/jkuck/learn_BP/" #file path to the directory cloned from github
TRAINED_MODELS_DIR = ROOT_DIR + "trained_models/" #trained models are stored here


#unused now
# TRAINING_DATA_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/"
# TRAINING_DATA_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/training_SAT_problems/"
# VALIDATION_DATA_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/"

##########################################################################################################
#contains CNF files for training/validation/test problems
# # TRAINING_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/SAT_problems_solved/"
# TRAINING_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/training_SAT_problems/"

# # VALIDATION_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/SAT_problems_solved/"
# VALIDATION_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/training_SAT_problems/"

# # TEST_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/test_SAT_problems/"
# TEST_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/training_SAT_problems/"
SAT_PROBLEMS_DIR = "/atlas/u/jkuck/learn_BP/data/sat_problems_noIndSets"

TRAINING_DATA_SIZE = 1000
VAL_DATA_SIZE = 1000#100
TEST_DATA_SIZE = 1000

########## info by problem groups and categories ##########
# problem counts reflect total number of problems before train/test split
##### network grid_problems
#20 problems in category problems_75 with <= 5 variables in the largest clause, min time: 0.024 max time: 236.78 mean time: 52.03439999999999
#107 problems in category problems_90 with <= 5 variables in the largest clause, min time: 0.004 max time: 479.892 mean time: 18.39442293457944

##### Network/DQMR problems
#150 problems in category or_50_problems with <= 5 variables in the largest clause, min time: 0.0 max time: 52.224000000000004 mean time: 4.153307146666667
#121 problems in category or_60_problems with <= 5 variables in the largest clause, min time: 0.0 max time: 79.148 mean time: 4.328342016528927
#111 problems in category or_70_problems with <= 5 variables in the largest clause, min time: 0.0 max time: 52.19457800000001 mean time: 4.2739784144144135
#138 problems in category or_100_problems with <= 5 variables in the largest clause, min time: 0.0 max time: 57.18000000000001 mean time: 3.1975750362318798

##### bit-blasted versions of SMTLIB benchmarks [guess based on](https://scholarship.rice.edu/bitstream/handle/1911/96419/TR16-03.pdf?sequence=1&isAllowed=y)
#147 problems in category blasted_problems with <= 5 variables in the largest clause, min time: 0.0 max time: 1390.756 mean time: 33.70159707482993

##### [Plan Recognition problems](https://sites.google.com/site/marcthurley/sharpsat/benchmarks/collected-model-counts)
#2 problems in category log_problems with <= 5 variables in the largest clause, min time: 0.019999999999999997 max time: 16.764 mean time: 8.392
#4 problems in category tire_problems with <= 5 variables in the largest clause, min time: 0.027536999999999996 max time: 0.19199999999999998 mean time: 0.09579425
#1 problems in category problem_4step with <= 5 variables in the largest clause, min time: 0.0 max time: 0.0 mean time: 0.0

##### Representations of circuits with a subset ofoutputs randomly xor-ed, see cnf files for this info
#68 problems in category s_problems with <= 5 variables in the largest clause, min time: 0.0 max time: 101.812 mean time: 3.836794382352941

##### Unkown origin
#9 problems in category sk_problems with <= 5 variables in the largest clause, min time: 0.0 max time: 6.483999999999999 mean time: 0.779111111111111


# PROBLEM_CATEGORY_TRAIN = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems', 'problems_75', 'problems_90'] #['problems_75', 'problems_90']#'problems_75'#
# PROBLEM_CATEGORY_TRAIN = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 'problems_75', 'problems_90'] #['problems_75', 'problems_90']#'problems_75'#
# PROBLEM_CATEGORY_TRAIN = ['problems_75', 'problems_90', 'blasted_problems']

# PROBLEM_CATEGORY_TRAIN = ['problems_75']
PROBLEM_CATEGORY_TRAIN = ['problems_90']
# PROBLEM_CATEGORY_TRAIN = ['or_50_problems']
# PROBLEM_CATEGORY_TRAIN = ['or_60_problems']
# PROBLEM_CATEGORY_TRAIN = ['or_70_problems']
# PROBLEM_CATEGORY_TRAIN = ['or_100_problems']
# PROBLEM_CATEGORY_TRAIN = ['blasted_problems']
# PROBLEM_CATEGORY_TRAIN = ['s_problems']



# PROBLEM_CATEGORY_VAL =  ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems']#['problems_75', 'problems_90', 'blasted_problems', 's_problems']
# PROBLEM_CATEGORY_VAL = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 'problems_75', 'problems_90']
# PROBLEM_CATEGORY_VAL = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems', 'problems_75', 'problems_90'] #['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems']#PROBLEM_CATEGORY_TRAIN#
PROBLEM_CATEGORY_VAL = PROBLEM_CATEGORY_TRAIN

PROBLEM_CATEGORY_TEST = 'or_60_problems'

#contains .txt files for each sat probolem with solution counts
# SOLUTION_COUNTS_DIR = "/atlas/u/jkuck/learn_BP/data/sat_counts_uai/"
SOLUTION_COUNTS_DIR = "/atlas/u/jkuck/learn_BP/data/exact_SAT_counts_noIndSets/"
# SOLUTION_COUNTS_DIR = TRAINING_DATA_DIR + "SAT_problems_solved"


EPOCH_COUNT = 1000
PRINT_FREQUENCY = 1
SAVE_FREQUENCY = 10
VAL_FREQUENCY = 10
##########################
##### Optimizer parameters #####
STEP_SIZE=100
LR_DECAY=.5 
LEARNING_RATE = 0.0001 #10layer with Bethe_mlp
##########################

# os.environ['WANDB_MODE'] = 'dryrun' #don't save to the cloud with this option
# wandb.init(project="learn_BP_sat2")
wandb.init(project="test")
wandb.config.epochs = EPOCH_COUNT
wandb.config.train_val_split = "easyTrain_hardVal"#"random_shuffle"#'separate_categories'#
wandb.config.PROBLEM_CATEGORY_TRAIN = PROBLEM_CATEGORY_TRAIN
wandb.config.PROBLEM_CATEGORY_VAL = PROBLEM_CATEGORY_VAL
# wandb.config.TRAINING_DATA_SIZE = TRAINING_DATA_SIZE
wandb.config.alpha = alpha
wandb.config.alpha2 = alpha2
wandb.config.SHARE_WEIGHTS = SHARE_WEIGHTS
wandb.config.BETHE_MLP = BETHE_MLP
wandb.config.MSG_PASSING_ITERS = MSG_PASSING_ITERS
wandb.config.STEP_SIZE = STEP_SIZE
wandb.config.LR_DECAY = LR_DECAY
wandb.config.LEARNING_RATE = LEARNING_RATE
wandb.config.NUM_MLPS = NUM_MLPS


# tiny_set = ["10.sk_1_46", "27.sk_3_32"]
lbp_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS, msg_passing_iters=MSG_PASSING_ITERS)
# lbp_net.double()


if True:#train val from the same distribution
#     train_problems_helper = [benchmark['problem'] for benchmark in ALL_TRAIN_PROBLEMS[PROBLEM_CATEGORY_TRAIN]]
    train_problems_helper = []
    for cur_train_category in PROBLEM_CATEGORY_TRAIN:
        print("cur_train_category:", cur_train_category)
        print("PROBLEM_CATEGORY_TRAIN:", PROBLEM_CATEGORY_TRAIN)


        train_problems_helper += [benchmark['problem'] for benchmark in ALL_TRAIN_PROBLEMS[cur_train_category]]


    # val_problems_helper = [benchmark['problem'] for benchmark in ALL_TRAIN_PROBLEMS[PROBLEM_CATEGORY_VAL]]
    # train_problems = train_problems_helper[:TRAINING_DATA_SIZE]
    # val_problems = val_problems_helper[:VAL_DATA_SIZE]
    if wandb.config.train_val_split == "random_shuffle":
        print("shuffling data")
        random.shuffle(train_problems_helper)
    else:
        assert(wandb.config.train_val_split == "easyTrain_hardVal")
    train_problems = train_problems_helper[:len(train_problems_helper)*7//10]
    val_problems = train_problems_helper[len(train_problems_helper)*7//10:]
    wandb.config.TRAINING_DATA_SIZE = len(train_problems_helper)*7//10
    wandb.config.VAL_DATA_SIZE = len(train_problems_helper) - len(train_problems_helper)*7//10
elif False: #use multiple categories for validation and train, using all problems from the categories
    assert(wandb.config.train_val_split == "separate_categories")
    train_problems = []
    for cur_train_category in PROBLEM_CATEGORY_TRAIN:
        train_problems += [benchmark['problem'] for benchmark in ALL_TRAIN_PROBLEMS[cur_train_category]]

    val_problems = []
    for cur_val_category in PROBLEM_CATEGORY_VAL:
        val_problems += [benchmark['problem'] for benchmark in ALL_TRAIN_PROBLEMS[cur_val_category]]

    wandb.config.TRAINING_DATA_SIZE = len(train_problems)
    wandb.config.VAL_DATA_SIZE = len(val_problems)
def train():
    wandb.watch(lbp_net)
    
    lbp_net.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs    

    loss_func = torch.nn.MSELoss()

    #pytorch geometric
    training_SAT_list = get_SATproblems_list(problems_to_load=train_problems,
               counts_dir_name=SOLUTION_COUNTS_DIR,
               problems_dir_name=SAT_PROBLEMS_DIR,
               # dataset_size=100, begin_idx=50, epsilon=EPSILON)
               dataset_size=TRAINING_DATA_SIZE, epsilon=EPSILON,
               max_factor_dimensions=MAX_FACTOR_STATE_DIMENSIONS)
    train_data_loader = DataLoader(training_SAT_list, batch_size=1)
   
    val_SAT_list = get_SATproblems_list(problems_to_load=val_problems,
               counts_dir_name=SOLUTION_COUNTS_DIR,
               problems_dir_name=SAT_PROBLEMS_DIR,
               # dataset_size=50, begin_idx=0, epsilon=EPSILON)
               dataset_size=VAL_DATA_SIZE, begin_idx=0, epsilon=EPSILON,
               max_factor_dimensions=MAX_FACTOR_STATE_DIMENSIONS)
    val_data_loader = DataLoader(val_SAT_list, batch_size=1)

    # with autograd.detect_anomaly():
    
    for e in range(EPOCH_COUNT):
#         for t, (sat_problem, exact_ln_partition_function) in enumerate(train_data_loader):
        losses = []
        for sat_problem in train_data_loader:
            exact_ln_partition_function = sat_problem.ln_Z
            optimizer.zero_grad()

#             sat_problem = FactorGraph.init_from_dictionary(sat_problem, squeeze_tensors=True)
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

        if e % PRINT_FREQUENCY == 0:
            print("root mean squared training error =", np.sqrt(np.mean(losses)))
            
        if e % VAL_FREQUENCY == 0:
            val_losses = []
#             for t, (sat_problem, exact_ln_partition_function) in enumerate(val_data_loader):
            for sat_problem in val_data_loader:
                exact_ln_partition_function = sat_problem.ln_Z
#                 sat_problem = FactorGraph.init_from_dictionary(sat_problem, squeeze_tensors=True)
                assert(sat_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)
                estimated_ln_partition_function = lbp_net(sat_problem)                
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
            torch.save(lbp_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
            
        scheduler.step()

    if not os.path.exists(TRAINED_MODELS_DIR):
        os.makedirs(TRAINED_MODELS_DIR)
    torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)

def test():
#     lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))
    # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
    # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))
#     lbp_net.load_state_dict(torch.load('wandb/run-20200217_073858-skhvebeh/model.pt'))
#     lbp_net.load_state_dict(torch.load('wandb/run-20200217_071515-yix18urv/model.pt'))

#     BPNN_trained_model_path = './wandb/run-20200217_221927-i4etpbs7/model.pt' #3 layer on all training data except 's' (best)
#     BPNN_trained_model_path = './wandb/run-20200217_221935-41r0m2ou//model.pt' #2 layer on all training data except 's' (faster)

#     BPNN_trained_model_path = './wandb/run-20200217_223828-1ur43pn3/model.pt' #2 layer on all training data except 's' (faster)

    BPNN_trained_model_path = './wandb/run-20200217_104742-n4d8lxp1/model.pt' #2 layer on random sampling of 'or_50' data

    
    
    
     
    runtimes_dir = '/atlas/u/jkuck/learn_BP/data/SAT_BPNN_runtimes/'
    if not os.path.exists(runtimes_dir):
        os.makedirs(runtimes_dir)
    lbp_net.load_state_dict(torch.load(BPNN_trained_model_path))


#     PROBLEM_CATEGORY_TEST = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 'problems_75', 'problems_90']
    PROBLEM_CATEGORY_TEST = ['or_50_problems']


    test_problems = []
    for cur_train_category in PROBLEM_CATEGORY_TEST:
        print("cur_train_category:", cur_train_category)
#         print("PROBLEM_CATEGORY_TRAIN:", PROBLEM_CATEGORY_TRAIN)
        test_problems += [benchmark['problem'] for benchmark in ALL_TRAIN_PROBLEMS[cur_train_category]]   
        
#     test_problems = [benchmark['problem'] for benchmark in ALL_TRAIN_PROBLEMS[PROBLEM_CATEGORY_TEST]]
#     test_problems = [benchmark['problem'] for benchmark in ALL_TEST_PROBLEMS[PROBLEM_CATEGORY_TEST]]

#     PROBLEM_CATEGORY_TEST = 'or_100_problems'
#     test_problems = [benchmark['problem'] for benchmark in ALL_TRAIN_PROBLEMS[PROBLEM_CATEGORY_TEST]]


    
    lbp_net.eval()

#     sat_data = SatProblems(problems_to_load=problems_solved_over2,
#                counts_dir_name=SOLUTION_COUNTS_DIR,
#                problems_dir_name=TEST_PROBLEMS_DIR,
#                dataset_size=TEST_DATA_SIZE, begin_idx=0, epsilon=EPSILON)

#     data_loader = DataLoader(sat_data, batch_size=1)
    test_SAT_list = get_SATproblems_list(problems_to_load=test_problems,
               counts_dir_name=SOLUTION_COUNTS_DIR,
               problems_dir_name=SAT_PROBLEMS_DIR,
               # dataset_size=50, begin_idx=0, epsilon=EPSILON)
#                dataset_size=VAL_DATA_SIZE, begin_idx=0, epsilon=EPSILON,
               dataset_size=10000, begin_idx=0, epsilon=EPSILON,
               max_factor_dimensions=MAX_FACTOR_STATE_DIMENSIONS)
    test_data_loader = DataLoader(test_SAT_list, batch_size=1)
    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    squared_errors = []
    BPNN_estimated_counts = []
    losses = []
    problem_names = []
#     for sat_problem, exact_ln_partition_function in test_data_loader:
    runtimes = []
    for idx, sat_problem in enumerate(test_data_loader):
        problem_names.append(test_problems[idx])
        runLBP = False
        if runLBP:
            print("about to parse dimacs")
            n_vars, clauses, load_successful = parse_dimacs(filename = SAT_PROBLEMS_DIR + "/" + test_problems[idx] + '.cnf.gz.no_w.cnf')
            print("about to run_loopyBP")
            lbp_estimate = run_loopyBP(clauses, n_vars, maxiter=10, updates="SEQRND", damping='.5')
            print("LBP:", lbp_estimate)
        exact_ln_partition_function = sat_problem.ln_Z
        # sat_problem.compute_bethe_free_energy()
#         sat_problem = FactorGraph.init_from_dictionary(sat_problem, squeeze_tensors=True)
        t0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
        ta = time.time()
        print("about to run BPNN")
        estimated_ln_partition_function = lbp_net(sat_problem)
        print("done to run BPNN")        
        t1 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
        tb = time.time()
        
        print("timing good?:", t1-t0)
        print("timing bad?:", tb-ta)
        runtimes.append(tb-ta)

        
        
        if runLBP:
            print("sat problem:", test_problems[idx], "exact ln_Z:", exact_ln_partition_function, "estimate:", estimated_ln_partition_function, "LBP:", lbp_estimate)        
        BPNN_estimated_counts.append(estimated_ln_partition_function.item())
        exact_solution_counts.append(exact_ln_partition_function.item())
        loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
        losses.append(loss.item())
        print("squared error:", (estimated_ln_partition_function.item() - exact_ln_partition_function.float().squeeze().item())**2)
        print("loss:", (estimated_ln_partition_function.item() - exact_ln_partition_function.float().squeeze().item())**2)
        squared_errors.append((estimated_ln_partition_function.item() - exact_ln_partition_function.float().squeeze().item())**2)
        print("estimated_ln_partition_function:", estimated_ln_partition_function)
        print("exact_ln_partition_function:", exact_ln_partition_function)
        print()

#     runtimes_json_string = json.dumps(runtimes)
    results = {'squared_errors': squared_errors, 'runtimes': runtimes, 
               'BPNN_estimated_ln_counts': BPNN_estimated_counts, 
               'exact_ln_solution_counts': exact_solution_counts,
               'problem_names': problem_names}
#     with open(runtimes_dir + PROBLEM_CATEGORY_TEST + "_runtimes.json", 'w') as outfile:
#         json.dump(runtimes, outfile)

    with open(runtimes_dir + "or50_trainSet_runtimesAndErrors_3layer.json", 'w') as outfile:
        json.dump(results, outfile)
        
        
    plt.plot(exact_solution_counts, BPNN_estimated_counts, 'x', c='b', label='Negative Bethe Free Energy, %d iters, RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.mean(losses))))
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

#     plot_name = 'trained=%s_dataset=%s%d_%diters_alpha%f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, len(data_loader), MSG_PASSING_ITERS, parameters.alpha)
    plot_name = 'quick_plot.png'
    plt.savefig(ROOT_DIR + 'sat_plots/' + plot_name)    
#     plt.show()





if __name__ == "__main__":
#     train()
    test()