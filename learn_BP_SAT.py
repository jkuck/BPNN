import torch
from torch import autograd
from nn_models import lbp_message_passing_network, USE_OLD_CODE
from sat_helpers.libdai_utils_sat import run_loopyBP
# from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader

if USE_OLD_CODE:
# if False:
    from sat_helpers.sat_data_partialRefactor import SatProblems, get_SATproblems_list, parse_dimacs
    from factor_graph_partialRefactor import DataLoader_custom as DataLoader
    SQUEEZE_BELIEF_REPEATS = True
    
else:
    from sat_helpers.sat_data import SatProblems, get_SATproblems_list, get_SATproblems_list_parallel, parse_dimacs, SATDataset
    from factor_graph import DataLoader_custom as DataLoader
    SQUEEZE_BELIEF_REPEATS = False
    
    
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from data.SAT_train_test_split import ALL_TRAIN_PROBLEMS, ALL_TEST_PROBLEMS
import wandb
from parameters import ROOT_DIR, alpha2
import random
import resource
import time
import json
import argparse

SET_TRUE_POST_DEBUGGING = True
NEW_FAST_DATA_LOADING = True

def boolean_string(s):    
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
#number of variables in the largest factor -> factor has 2^args.max_factor_state_dimensions states
parser.add_argument('--max_factor_state_dimensions', type=int, default=5)
#the number of iterations of message passing, we have this many layers with their own learnable parameters
parser.add_argument('--msg_passing_iters', type=int, default=5)
#messages have var_cardinality states in standard belief propagation.  belief_repeats artificially
#increases this number so that messages have belief_repeats*var_cardinality states, analogous
#to increasing node feature dimensions in a standard graph neural network
parser.add_argument('--belief_repeats', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=50)

# 0.0001
# 0.0005
parser.add_argument('--learning_rate', type=float, default=0.0001)
# parser.add_argument('--learning_rate', type=float, default=0.0)

# parser.add_argument('--learning_rate', type=float, default=0.0005)
# parser.add_argument('--learning_rate', type=float, default=0.001)

#damping parameter
parser.add_argument('--alpha_damping_FtoV', type=float, default=.5)
parser.add_argument('--alpha_damping_VtoF', type=float, default=1.0) #this damping wasn't used in the old code


#if true, mlps operate in standard space rather than log space
parser.add_argument('--lne_mlp', type=boolean_string, default=False)

#original MLPs that operate on factor beliefs (problematic because they're not index invariant)
parser.add_argument('--use_MLP1', type=boolean_string, default=False)
parser.add_argument('--use_MLP2', type=boolean_string, default=False)

#new MLPs that operate on variable beliefs
parser.add_argument('--use_MLP3', type=boolean_string, default=True)
parser.add_argument('--use_MLP4', type=boolean_string, default=True)

#new MLP that hopefully preservers consistency
parser.add_argument('--use_MLP5', type=boolean_string, default=False)
parser.add_argument('--use_MLP6', type=boolean_string, default=False)
parser.add_argument('--use_MLP_EQUIVARIANT', type=boolean_string, default=False)

#new MLPs that operate on message differences, e.g. in place of damping
parser.add_argument('--USE_MLP_DAMPING_FtoV', type=boolean_string, default=False)
parser.add_argument('--USE_MLP_DAMPING_VtoF', type=boolean_string, default=False)

#if true, share the weights between layers in a BPNN
parser.add_argument('--SHARE_WEIGHTS', type=boolean_string, default=False)

#if true, subtract previously sent messages (to avoid 'double counting')
parser.add_argument('--subtract_prv_messages', type=boolean_string, default=True)

#if 'none' then use the standard bethe approximation with no learning
#otherwise, describes (potential) non linearities in the MLP
parser.add_argument('--bethe_mlp', type=str, default='linear',\
    choices=['shifted','standard','linear','none'])

#if True, use the old Bethe approximation that doesn't work with batches
#only valid for bethe_mlp='none'
parser.add_argument('--use_old_bethe', type=boolean_string, default=False)

#for reproducing random train/val split
#args.random_seed = 0 and 1 seem to produce very different results for s_problems
parser.add_argument('--random_seed', type=int, default=1)

parser.add_argument('--problem_category_train', type=str, default='group2',\
    choices=['problems_75','problems_90','or_50_problems','or_60_problems','or_70_problems',\
    'or_100_problems', 'blasted_problems','s_problems','group1','group2','group3','group4'])

parser.add_argument('--train_val_split', type=str, default='random_shuffle',\
    choices=["random_shuffle", "easyTrain_hardVal", "separate_categories"])


# parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
# parser.add_argument('--test_freq', type=int, default=200)
# parser.add_argument('--save_freq', type=int, default=1000)
args, _ = parser.parse_known_args()

print("args.problem_category_train:", args.problem_category_train)


##########################
##### Run me on Atlas
# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate


##########################
####### PARAMETERS #######

#the number of states a variable can take, e.g. 2 for binary variables
VAR_CARDINALITY = 2


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
VAL_DATA_SIZE = 1000 #100
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
# PROBLEM_CATEGORY_TRAIN = ['problems_90']
# PROBLEM_CATEGORY_TRAIN = ['or_50_problems']
# PROBLEM_CATEGORY_TRAIN = ['or_60_problems']
# PROBLEM_CATEGORY_TRAIN = ['or_70_problems']
# PROBLEM_CATEGORY_TRAIN = ['or_100_problems']
# PROBLEM_CATEGORY_TRAIN = ['blasted_problems']
# PROBLEM_CATEGORY_TRAIN = ['s_problems']

if args.problem_category_train == 'group1':
    PROBLEM_CATEGORY_TRAIN =  ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems']#['problems_75', 'problems_90', 'blasted_problems', 's_problems']
elif args.problem_category_train == 'group2':
    PROBLEM_CATEGORY_TRAIN = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 'problems_75', 'problems_90']
elif args.problem_category_train == 'group3':
    PROBLEM_CATEGORY_TRAIN =  ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems', 'problems_75', 'problems_90'] #['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems']#PROBLEM_CATEGORY_TRAIN#    
elif args.problem_category_train == 'group4':
    pass
    # PROBLEM_CATEGORY_VAL =                
else:   
    PROBLEM_CATEGORY_TRAIN = [args.problem_category_train]

# PROBLEM_CATEGORY_VAL =  ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems']#['problems_75', 'problems_90', 'blasted_problems', 's_problems']
# PROBLEM_CATEGORY_VAL = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 'problems_75', 'problems_90']
# PROBLEM_CATEGORY_VAL = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems', 'problems_75', 'problems_90'] #['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 's_problems']#PROBLEM_CATEGORY_TRAIN#
PROBLEM_CATEGORY_VAL = PROBLEM_CATEGORY_TRAIN

PROBLEM_CATEGORY_TEST = 'or_60_problems'

#contains .txt files for each sat probolem with solution counts
# SOLUTION_COUNTS_DIR = "/atlas/u/jkuck/learn_BP/data/sat_counts_uai/"
SOLUTION_COUNTS_DIR = "/atlas/u/jkuck/learn_BP/data/exact_SAT_counts_noIndSets/"
# SOLUTION_COUNTS_DIR = "/atlas/u/jkuck/learn_BP/data/dummy_debugging_counts/"
# SOLUTION_COUNTS_DIR = TRAINING_DATA_DIR + "SAT_problems_solved"


EPOCH_COUNT = 1000
PRINT_FREQUENCY = 10
SAVE_FREQUENCY = 100
VAL_FREQUENCY = 10
##########################
##### Optimizer parameters #####
STEP_SIZE=200
LR_DECAY=.5 
LEARNING_RATE = args.learning_rate
# LEARNING_RATE = 0.0001 #10layer with Bethe_mlp
# LEARNING_RATE = 0.0005
# LEARNING_RATE = 0.002


# LEARNING_RATE = 0.0005 #debugging
# LEARNING_RATE = 0.02 #debugging
# LEARNING_RATE = 0.00000001 #testing sgd

# LEARNING_RATE = 4*0.0001*args.batch_size #try to fix bad training for large batch size

#trying new
# LEARNING_RATE = 0.00000001 #10layer with Bethe_mlp

##########################

# USE_WANDB = False
# if USE_WANDB:
# os.environ['WANDB_MODE'] = 'dryrun' #don't save to the cloud with this option
# wandb.init(project="learn_BP_sat_reproduce6")
# wandb.init(project="learn_BP_sat_reproduceFromOldCode")
wandb.init(project="learn_BP_sat_compareParams_mlp4Debug_normalizeBeliefs")
# wandb.init(project="learn_BP_sat_debug")

# wandb.init(project="test")
wandb.config.epochs = EPOCH_COUNT
wandb.config.train_val_split = args.train_val_split #"random_shuffle"#"easyTrain_hardVal"#'separate_categories'#
wandb.config.PROBLEM_CATEGORY_TRAIN = args.problem_category_train
wandb.config.PROBLEM_CATEGORY_VAL = PROBLEM_CATEGORY_VAL
# wandb.config.TRAINING_DATA_SIZE = TRAINING_DATA_SIZE
wandb.config.alpha_damping_FtoV = args.alpha_damping_FtoV
wandb.config.alpha_damping_VtoF = args.alpha_damping_VtoF
wandb.config.alpha2 = alpha2
wandb.config.msg_passing_iters = args.msg_passing_iters
wandb.config.STEP_SIZE = STEP_SIZE
wandb.config.LR_DECAY = LR_DECAY
wandb.config.LEARNING_RATE = LEARNING_RATE
wandb.config.belief_repeats = args.belief_repeats
wandb.config.var_cardinality = VAR_CARDINALITY
wandb.config.max_factor_state_dimensions = args.max_factor_state_dimensions
wandb.config.random_seed = args.random_seed
wandb.config.training_batch_size = args.batch_size

wandb.config.BETHE_MLP = args.bethe_mlp
wandb.config.lne_mlp = args.lne_mlp
wandb.config.use_MLP1 = args.use_MLP1
wandb.config.use_MLP2 = args.use_MLP2
wandb.config.use_MLP3 = args.use_MLP3
wandb.config.use_MLP4 = args.use_MLP4
wandb.config.use_MLP5 = args.use_MLP5
wandb.config.use_MLP6 = args.use_MLP6
wandb.config.use_MLP_EQUIVARIANT = args.use_MLP_EQUIVARIANT
wandb.config.USE_MLP_DAMPING_FtoV = args.USE_MLP_DAMPING_FtoV
wandb.config.USE_MLP_DAMPING_VtoF = args.USE_MLP_DAMPING_VtoF
wandb.config.SHARE_WEIGHTS = args.SHARE_WEIGHTS
wandb.config.subtract_prv_messages = args.subtract_prv_messages

wandb.config.use_old_bethe = args.use_old_bethe

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
print("device:", device)

# tiny_set = ["10.sk_1_46", "27.sk_3_32"]
lbp_net = lbp_message_passing_network(max_factor_state_dimensions=args.max_factor_state_dimensions, msg_passing_iters=args.msg_passing_iters,\
    lne_mlp=args.lne_mlp, use_MLP1=args.use_MLP1, use_MLP2=args.use_MLP2, use_MLP3=args.use_MLP3, use_MLP4=args.use_MLP4,\
    use_MLP5=args.use_MLP5,  use_MLP6=args.use_MLP6, use_MLP_EQUIVARIANT=args.use_MLP_EQUIVARIANT,\
    USE_MLP_DAMPING_FtoV=args.USE_MLP_DAMPING_FtoV, USE_MLP_DAMPING_VtoF=args.USE_MLP_DAMPING_VtoF,\
    subtract_prv_messages=args.subtract_prv_messages, share_weights = args.SHARE_WEIGHTS, bethe_MLP=args.bethe_mlp,\
    belief_repeats=args.belief_repeats, var_cardinality=VAR_CARDINALITY, alpha_damping_FtoV=args.alpha_damping_FtoV,\
    alpha_damping_VtoF=args.alpha_damping_VtoF, use_old_bethe=args.use_old_bethe)
# lbp_net.double()
lbp_net = lbp_net.to(device)

# for name, param in lbp_net.named_parameters():
# # for param in lbp_net.parameters():
#     if param.device.type != 'cuda':
#         print('param {}, not on GPU!!!!!!!!!!!!!!'.format(name))
#     else:
#         print('param {}, has device type {}'.format(name, param.device.type))
# #         print(param)
        
# # sleep(debug)



if NEW_FAST_DATA_LOADING == False:
    if not args.train_val_split=='separate_categories':#train val from the same distribution
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
            random.seed(args.random_seed)
            random.shuffle(train_problems_helper)
        else:
            assert(wandb.config.train_val_split == "easyTrain_hardVal")
        train_problems = train_problems_helper[:len(train_problems_helper)*7//10]
        val_problems = train_problems_helper[len(train_problems_helper)*7//10:]
        wandb.config.TRAINING_DATA_SIZE = len(train_problems_helper)*7//10
        wandb.config.VAL_DATA_SIZE = len(train_problems_helper) - len(train_problems_helper)*7//10
    else: #use multiple categories for validation and train, using all problems from the categories
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
    print("HISLKDJFLSJFLK")
    wandb.watch(lbp_net)
    
    lbp_net.train()

    # Initialize optimizer
#     optimizer = torch.optim.SGD(lbp_net.parameters(), lr=LEARNING_RATE, momentum=0.7)        
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs    

    loss_func = torch.nn.MSELoss()


    if NEW_FAST_DATA_LOADING:
        if not args.train_val_split=='separate_categories':#train val from the same distribution
            train_problems_helper = []
            for cur_train_category in PROBLEM_CATEGORY_TRAIN:
                cur_category_problems = SATDataset(root="./data/SAT_pytorchGeom_proccesed/", dataset_type='train',\
                    problem_category=cur_train_category, belief_repeats=args.belief_repeats, epsilon=EPSILON,\
                    max_factor_dimensions=args.max_factor_state_dimensions)
                train_problems_helper.extend(cur_category_problems)
            if wandb.config.train_val_split == "random_shuffle":
                print("shuffling data")
                random.seed(args.random_seed)
                random.shuffle(train_problems_helper)
            else:
                assert(wandb.config.train_val_split == "easyTrain_hardVal")
            training_SAT_list = train_problems_helper[:len(train_problems_helper)*7//10]
            val_SAT_list = train_problems_helper[len(train_problems_helper)*7//10:]
            wandb.config.TRAINING_DATA_SIZE = len(train_problems_helper)*7//10
            wandb.config.VAL_DATA_SIZE = len(train_problems_helper) - len(train_problems_helper)*7//10
        else: #use multiple categories for validation and train, using all problems from the categories
            assert(wandb.config.train_val_split == "separate_categories")
            training_SAT_list = []
            for cur_train_category in PROBLEM_CATEGORY_TRAIN:
                cur_category_problems = SATDataset(root="./data/SAT_pytorchGeom_proccesed/", dataset_type='train',\
                    problem_category=cur_train_category, belief_repeats=args.belief_repeats, epsilon=EPSILON,\
                    max_factor_dimensions=args.max_factor_state_dimensions)
                training_SAT_list.extend(cur_category_problems)

            val_SAT_list = []
            for cur_val_category in PROBLEM_CATEGORY_VAL:
                cur_category_problems = SATDataset(root="./data/SAT_pytorchGeom_proccesed/", dataset_type='train',\
                    problem_category=cur_val_category, belief_repeats=args.belief_repeats, epsilon=EPSILON,\
                    max_factor_dimensions=args.max_factor_state_dimensions)
                val_SAT_list.extend(cur_category_problems)

            wandb.config.TRAINING_DATA_SIZE = len(training_SAT_list)
            wandb.config.VAL_DATA_SIZE = len(val_SAT_list)

        # training_SAT_list = training_SAT_list[:3]
        # val_SAT_list = val_SAT_list[:3]
        print("len(training_SAT_list) =", len(training_SAT_list))
        print("len(val_SAT_list) =", len(val_SAT_list))
        train_data_loader = DataLoader(training_SAT_list, batch_size=args.batch_size, shuffle=True)
        val_data_loader = DataLoader(val_SAT_list, batch_size=args.batch_size, shuffle=True)
        
    else:
        #pytorch geometric
        training_SAT_list = get_SATproblems_list(problems_to_load=train_problems,
                counts_dir_name=SOLUTION_COUNTS_DIR,
                problems_dir_name=SAT_PROBLEMS_DIR,
                # dataset_size=100, begin_idx=50, epsilon=EPSILON)
                dataset_size=TRAINING_DATA_SIZE, epsilon=EPSILON,
                max_factor_dimensions=args.max_factor_state_dimensions, belief_repeats=args.belief_repeats)
        train_data_loader = DataLoader(training_SAT_list, batch_size=args.batch_size, shuffle=True)
    
        val_SAT_list = get_SATproblems_list(problems_to_load=val_problems,
                counts_dir_name=SOLUTION_COUNTS_DIR,
                problems_dir_name=SAT_PROBLEMS_DIR,
                # dataset_size=50, begin_idx=0, epsilon=EPSILON)
                dataset_size=VAL_DATA_SIZE, begin_idx=0, epsilon=EPSILON,
                max_factor_dimensions=args.max_factor_state_dimensions, belief_repeats=args.belief_repeats)
        val_data_loader = DataLoader(val_SAT_list, batch_size=args.batch_size, shuffle=True)

    # with autograd.detect_anomaly():
    
    for e in range(EPOCH_COUNT):
#         for t, (sat_problem, exact_ln_partition_function) in enumerate(train_data_loader):
        loss_sum = 0
        training_problem_count_check = 0
        epoch_loss = 0
#         optimizer.zero_grad()
        for sat_problem in train_data_loader:
            optimizer.zero_grad()
            if SQUEEZE_BELIEF_REPEATS:
                sat_problem.prv_varToFactor_messages = sat_problem.prv_varToFactor_messages.squeeze()
                sat_problem.prv_factorToVar_messages = sat_problem.prv_factorToVar_messages.squeeze()
                sat_problem.prv_factor_beliefs = sat_problem.prv_factor_beliefs.squeeze()
                sat_problem.prv_var_beliefs = sat_problem.prv_var_beliefs.squeeze()   
                sat_problem.factor_potential_masks = sat_problem.factor_potential_masks.squeeze()   
                sat_problem.factor_potentials = sat_problem.factor_potentials.squeeze()   

                
#                 print("sat_problem.prv_varToFactor_messages.shape:", sat_problem.prv_varToFactor_messages.shape)
#                 print("sat_problem.prv_factorToVar_messages.shape:", sat_problem.prv_factorToVar_messages.shape)
#                 print("sat_problem.prv_factor_beliefs.shape:", sat_problem.prv_factor_beliefs.shape)
#                 print("sat_problem.prv_var_beliefs.shape:", sat_problem.prv_var_beliefs.shape)
                
            if SET_TRUE_POST_DEBUGGING:
                assert(sat_problem.num_vars == torch.sum(sat_problem.numVars))
            sat_problem.state_dimensions = sat_problem.state_dimensions[0] #hack for batching,
            if SET_TRUE_POST_DEBUGGING:
                sat_problem.var_cardinality = sat_problem.var_cardinality[0] #hack for batching,
                sat_problem.belief_repeats = sat_problem.belief_repeats[0] #hack for batching,

            sat_problem = sat_problem.to(device)
            exact_ln_partition_function = sat_problem.ln_Z

            assert(sat_problem.state_dimensions == args.max_factor_state_dimensions)
            estimated_ln_partition_function = lbp_net(sat_problem)
            # estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
            #     prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(sat_problem)
            # vTof_convergence_loss = loss_func(prv_varToFactor_messages, prv_prv_varToFactor_messages)
            # fTov_convergence_loss = loss_func(prv_factorToVar_messages, prv_prv_factorToVar_messages)

            # print("estimated_ln_partition_function:", estimated_ln_partition_function)
            # print("exact_ln_partition_function:", exact_ln_partition_function)            
#             time.sleep(.5)
#             
#             sleep(stop_early)
            # print("type(estimated_ln_partition_function):", type(estimated_ln_partition_function))
            # print("type(exact_ln_partition_function):", type(exact_ln_partition_function))
#             print("estimated_ln_partition_function.shape:", estimated_ln_partition_function.shape)
#             print("exact_ln_partition_function.shape:", exact_ln_partition_function.shape)
            
            loss = loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float().squeeze())
            # print("loss:", loss)
            # print()
            assert(estimated_ln_partition_function.numel() == exact_ln_partition_function.numel()), (estimated_ln_partition_function.numel(), exact_ln_partition_function.numel())
            loss_sum += loss.item()*estimated_ln_partition_function.numel()
            training_problem_count_check += estimated_ln_partition_function.numel()
#             epoch_loss += loss
            loss.backward()
#             # nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

#         epoch_loss = epoch_loss
#         epoch_loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), args.clip)
#         optimizer.step()

        assert(len(training_SAT_list) == training_problem_count_check)
        training_RMSE = np.sqrt(loss_sum/training_problem_count_check)
        if e % PRINT_FREQUENCY == 0:
            print("epoch:", e, "root mean squared training error =", training_RMSE)
            # print("vTof_convergence_loss =", vTof_convergence_loss)
            # print("fTov_convergence_loss =", fTov_convergence_loss)
            
        if e % VAL_FREQUENCY == 0:
#             print('-'*40, "check weights 1234", '-'*40)
#             for param in lbp_net.parameters():
#                 print(param.data)
            val_loss_sum = 0
            val_problem_count_check = 0
#             for t, (sat_problem, exact_ln_partition_function) in enumerate(val_data_loader):
            for sat_problem in val_data_loader:
                if SQUEEZE_BELIEF_REPEATS:
                    sat_problem.prv_varToFactor_messages = sat_problem.prv_varToFactor_messages.squeeze()
                    sat_problem.prv_factorToVar_messages = sat_problem.prv_factorToVar_messages.squeeze()
                    sat_problem.prv_factor_beliefs = sat_problem.prv_factor_beliefs.squeeze()
                    sat_problem.prv_var_beliefs = sat_problem.prv_var_beliefs.squeeze()   
                    sat_problem.factor_potential_masks = sat_problem.factor_potential_masks.squeeze()   
                    sat_problem.factor_potentials = sat_problem.factor_potentials.squeeze()   


#                     print("sat_problem.prv_varToFactor_messages.shape:", sat_problem.prv_varToFactor_messages.shape)
#                     print("sat_problem.prv_factorToVar_messages.shape:", sat_problem.prv_factorToVar_messages.shape)
#                     print("sat_problem.prv_factor_beliefs.shape:", sat_problem.prv_factor_beliefs.shape)
#                     print("sat_problem.prv_var_beliefs.shape:", sat_problem.prv_var_beliefs.shape)

                sat_problem.state_dimensions = sat_problem.state_dimensions[0] #hack for batching,
                if SET_TRUE_POST_DEBUGGING:
                    sat_problem.var_cardinality = sat_problem.var_cardinality[0] #hack for batching,
                    sat_problem.belief_repeats = sat_problem.belief_repeats[0] #hack for batching,
            
                sat_problem = sat_problem.to(device)
                exact_ln_partition_function = sat_problem.ln_Z
                assert(sat_problem.state_dimensions == args.max_factor_state_dimensions)
                estimated_ln_partition_function = lbp_net(sat_problem)   
                # estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                #     prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(sat_problem)                

                loss = loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float().squeeze())
#                 print("estimated_ln_partition_function:", estimated_ln_partition_function)
#                 print("exact_ln_partition_function:", exact_ln_partition_function)
#                 print("loss:", loss)
                
                assert(estimated_ln_partition_function.numel() == exact_ln_partition_function.numel()), (estimated_ln_partition_function.numel(), exact_ln_partition_function.numel())
                val_loss_sum += loss.item()*estimated_ln_partition_function.numel()
                val_problem_count_check += estimated_ln_partition_function.numel()

            assert(len(val_SAT_list) == val_problem_count_check)
            val_RMSE = np.sqrt(val_loss_sum/val_problem_count_check)

            print("root mean squared validation error =", val_RMSE)
            print()
            wandb.log({"RMSE_val": val_RMSE, "RMSE_training": training_RMSE})   
            
        if e%100 == 0:
            for name, param in lbp_net.named_parameters():
                if 'alpha' in name:
                    print("123 parameter name:", name)
                    print("123 parameter:", param)
            
        else:
            wandb.log({"RMSE_training": training_RMSE})
            
            
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
               max_factor_dimensions=args.max_factor_state_dimensions, belief_repeats=args.belief_repeats)
    test_data_loader = DataLoader(test_SAT_list, batch_size=1)
    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    squared_errors = []
    BPNN_estimated_counts = []
    loss_sum = 0
    test_problem_count_check = 0
    problem_names = []
#     for sat_problem, exact_ln_partition_function in test_data_loader:
    runtimes = []
    for idx, sat_problem in enumerate(test_data_loader):
        sat_problem = sat_problem.to(device)
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
        assert(estimated_ln_partition_function.numel() == exact_ln_partition_function.numel())
        loss_sum += loss.item()*estimated_ln_partition_function.numel()
        test_problem_count_check += estimated_ln_partition_function.numel()
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
        
    assert(len(test_SAT_list) == test_problem_count_check)
    test_RMSE = np.sqrt(loss_sum/test_problem_count_check)
    plt.plot(exact_solution_counts, BPNN_estimated_counts, 'x', c='b', label='Negative Bethe Free Energy, %d iters, RMSE=%.2f' % (args.msg_passing_iters, test_RMSE))
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

#     plot_name = 'trained=%s_dataset=%s%d_%diters_alpha%f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, len(data_loader), args.msg_passing_iters, parameters.alpha)
    plot_name = 'quick_plot.png'
    plt.savefig(ROOT_DIR + 'sat_plots/' + plot_name)    
#     plt.show()





if __name__ == "__main__":
    train()
#     test()
