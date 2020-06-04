import torch
from torch import autograd
from nn_models import lbp_message_passing_network, USE_OLD_CODE
# from sat_helpers.libdai_utils_sat import run_loopyBP
# from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader

if USE_OLD_CODE:
# if False:
    from sat_helpers.sat_data_partialRefactor import SatProblems, get_SATproblems_list, parse_dimacs
    from factor_graph_partialRefactor import DataLoader_custom as DataLoader
    SQUEEZE_BELIEF_REPEATS = True
    
else:
    from sat_helpers.sat_data import SatProblems, get_SATproblems_list, get_SATproblems_list_parallel, parse_dimacs, SATDataset, build_factorgraph_from_SATproblem
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

TINY_DEBUG = False
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

parser.add_argument('--batch_size', type=int, default=5)

# 0.0001
# 0.0005
# parser.add_argument('--learning_rate', type=float, default=0.0001)

# parser.add_argument('--learning_rate', type=float, default=0.00002)
# parser.add_argument('--learning_rate', type=float, default=0.0001)
# parser.add_argument('--learning_rate', type=float, default=0.0002)
# parser.add_argument('--learning_rate', type=float, default=0.0005)

parser.add_argument('--learning_rate', type=float, default=0.0001)
# parser.add_argument('--learning_rate', type=float, default=0.000002)
# parser.add_argument('--learning_rate', type=float, default=0.000008)
# parser.add_argument('--learning_rate', type=float, default=0.000)

#damping parameter
parser.add_argument('--alpha_damping_FtoV', type=float, default=.5)
parser.add_argument('--alpha_damping_VtoF', type=float, default=1.0)


#if true, mlps operate in standard space rather than log space
parser.add_argument('--lne_mlp', type=boolean_string, default=False)

#original MLPs that operate on factor beliefs (problematic because they're not index invariant)
parser.add_argument('--use_MLP1', type=boolean_string, default=False)
parser.add_argument('--use_MLP2', type=boolean_string, default=False)

#new MLPs that operate on variable beliefs
parser.add_argument('--use_MLP3', type=boolean_string, default=False)
parser.add_argument('--use_MLP4', type=boolean_string, default=False)

#new MLP that hopefully preservers consistency
parser.add_argument('--use_MLP5', type=boolean_string, default=False)
parser.add_argument('--use_MLP6', type=boolean_string, default=False)
parser.add_argument('--use_MLP_EQUIVARIANT', type=boolean_string, default=False)

#new MLPs that operate on message differences, e.g. in place of damping
parser.add_argument('--USE_MLP_DAMPING_FtoV', type=boolean_string, default=True)
parser.add_argument('--USE_MLP_DAMPING_VtoF', type=boolean_string, default=False)

#if true, share the weights between layers in a BPNN
parser.add_argument('--SHARE_WEIGHTS', type=boolean_string, default=False)

#if true, subtract previously sent messages (to avoid 'double counting')
parser.add_argument('--subtract_prv_messages', type=boolean_string, default=True)

#if 'none' then use the standard bethe approximation with no learning
#otherwise, describes (potential) non linearities in the MLP
parser.add_argument('--bethe_mlp', type=str, default='linear',\
    choices=['shifted','standard','linear','none'])

#if True, average over permutations of factor beliefs/potentials when computing Bethe approximation
parser.add_argument('--factor_graph_representation_invariant', type=boolean_string, default=True)



#if True, use the old Bethe approximation that doesn't work with batches
#only valid for bethe_mlp='none'
parser.add_argument('--use_old_bethe', type=boolean_string, default=False)

#for reproducing random train/val split
#args.random_seed = 0 and 1 seem to produce very different results for s_problems
parser.add_argument('--random_seed', type=int, default=1)

parser.add_argument('--problem_category_train', type=str, default='group3',\
    choices=['problems_75','problems_90','or_50_problems','or_60_problems','or_70_problems',\
    'or_100_problems', 'blasted_problems','s_problems','group1','group2','group3','group4'])
#if True load SAT problems where the variable names have been randomly swapped and the order of variables in clauses has been permuted randomly
PERMUTED_SAT_DATA = True

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

MODEL_NAME = "debug.pth"
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
SAVE_FREQUENCY = 1
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
# wandb.init(project="learn_BP_sat_debug")

# wandb.init(project="learn_BP_sat_compareParams_mlp4Debug_normalizeBeliefs")
# wandb.init(project="learn_BP_sat_dampingMLP_70to100iter")

# wandb.init(project="learn_BP_sat_12345")

# wandb.init(project="learn_BP_sat_debug1")
# wandb.init(project="learn_BP_sat_dampingMLPwithMsgInfo")
# wandb.init(project="learn_BP_sat_dampingMLPwithoutMsgInfo1")

# wandb.init(project="learn_BP_sat_BetheInvariant_allCategories")
# wandb.init(project="learn_BP_test")
# wandb.init(project="learn_BP_test")
wandb.init(project="learn_BP_SAT_debuggedInvariance1")
# wandb.init(project="learn_BP_sat_MLP34_CompareDoubleCount_andBethe")


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
wandb.config.factor_graph_representation_invariant = args.factor_graph_representation_invariant

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
    alpha_damping_VtoF=args.alpha_damping_VtoF, use_old_bethe=args.use_old_bethe, FACTOR_GRAPH_REPRESENTATION_INVARIANT=args.factor_graph_representation_invariant)
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
        wandb.config.TRAINING_DATA_SIZE = len(train_problems)
        wandb.config.VAL_DATA_SIZE = len(val_problems)
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
    # lbp_net.load_state_dict(torch.load('wandb/run-20200515_052334-erpuze4k/model.pt')) #'75'

    # lbp_net.load_state_dict(torch.load('wandb/run-20200515_053747-0p3hyls0/model.pt')) #'or_50'
    # lbp_net.load_state_dict(torch.load('wandb/run-20200527_215757-b7bn1nda/model.pt')) #'or_50'
    # lbp_net.load_state_dict(torch.load('wandb/run-20200527_223821-9xg7z54y/model.pt')) #'or_50'

    # BPNN_trained_model_path = './wandb/run-20200602_081049-x5l4qhs9/model.pt' #invariant BPNN trained on group2
    # lbp_net.load_state_dict(torch.load(BPNN_trained_model_path)) #'or_50'
    
    # lbp_net.linear1.weight = torch.nn.Parameter(torch.rand_like(lbp_net.linear1.weight))
    # lbp_net.linear1.bias = torch.nn.Parameter(torch.rand_like(lbp_net.linear1.bias))

    # lbp_net.linear1a.weight = torch.nn.Parameter(torch.rand_like(lbp_net.linear1a.weight))
    # lbp_net.linear1a.bias = torch.nn.Parameter(torch.rand_like(lbp_net.linear1a.bias))

    # lbp_net.linear1b.weight = torch.nn.Parameter(torch.rand_like(lbp_net.linear1b.weight))
    # lbp_net.linear1b.bias = torch.nn.Parameter(torch.rand_like(lbp_net.linear1b.bias))
    # lbp_net.linear2.weight = torch.nn.Parameter(torch.rand_like(lbp_net.linear2.weight))
    # lbp_net.linear2.bias = torch.nn.Parameter(torch.rand_like(lbp_net.linear2.bias))

    # print()
    # print("loaded trained model")
    # print()

    # 
    # print("LOAD")
    wandb.watch(lbp_net)
    
    lbp_net.train()

    # Initialize optimizer
#     optimizer = torch.optim.SGD(lbp_net.parameters(), lr=LEARNING_RATE, momentum=0.7)        
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs    

    loss_func = torch.nn.MSELoss()

    if TINY_DEBUG:
        if TINY_DEBUG:
            # clauses = [[1,-2], [2, 3]]
            # clauses = [[-2,1], [2, 3]]
            # clauses = [[-2, 3], [1,2]]
            # clauses = [[2, 3], [2,1]]
            print("HI 123456 :)")
            # clauses = [[3, -1], [2,3], [-2, 4], [1, 5]]
            # clauses = [[3, -1], [2,3], [4, -2], [1, 5]]

            # clauses = [[5, 3], [1, -5], [2,1], [4, -2]]

            #difference between these two!!
            # clauses = [[3, -1], [2,3], [-2, 4]]
            clauses = [[3, -1], [3,2], [4, -2]]

            # clauses = [[1, 2, 3, 4, -5], [-1, 3, 5, 6, 7], [4, 2, -3, 1, 8], [5, -1]]
            # clauses = [[5, 2, -1, 3, 4], [-5, 3, 1, 6, 7], [4, 2, -3, 5, 8], [1, -5]]

            # /atlas/u/jkuck/learn_BP/data/sat_problems_permuted/or-50-5-1-UC-10.cnf.gz.no_w.cnf
            # clauses = [[74, -31], [74, -29], [74, -21], [74, -77], [77, 31, 29, 21, -74], [51, -58], [51, -95], [-14, 51], [-68, 51], [-51, 58, 95, 68, 14], [70, -68], [-12, 70], [70, -77], [-84, 70], [68, 77, -70, 12, 84], [26, -25], [-88, 26], [26, -67], [26, -100], [-26, 100, 67, 25, 88], [99, -85], [99, -25], [99, -22], [99, -81], [85, 25, 22, 81, -99], [47, -8], [47, -3], [-55, 47], [47, -92], [8, 92, 55, -47, 3], [97, -9], [97, -79], [97, -38], [-25, 97], [25, 38, 79, -97, 9], [-89, 1], [1, -24], [-8, 1], [1, -42], [8, 42, 24, -1, 89], [-35, 73], [-68, 73], [73, -53], [73, -87], [35, 68, -73, 53, 87], [-55, 36], [36, -88], [-35, 36], [-95, 36], [95, -36, 88, 55, 35], [59, -67], [59, -29], [59, -35], [-21, 59], [35, 67, -59, 29, 21], [93, -12], [93, -95], [-23, 93], [-16, 93], [-93, 12, 23, 95, 16], [-94, 4], [-7, 4], [4, -55], [4, -29], [7, 29, 94, -4, 55], [-15, 96], [96, -79], [96, -22], [-85, 96], [-96, 15, 22, 85, 79], [57, -45], [-23, 57], [-3, 57], [-20, 57], [45, 3, 23, 20, -57], [-71, 91], [91, -81], [91, -67], [91, -8], [-91, 71, 81, 8, 67], [27, -9], [-84, 27], [-11, 27], [27, -15], [84, 15, -27, 11, 9], [41, -45], [-16, 41], [-60, 41], [41, -30], [30, 16, 45, 60, -41], [-45, 56], [56, -84], [-29, 56], [56, -81], [45, 29, 84, -56, 81], [90, -11], [-88, 90], [-45, 90], [-44, 90], [45, 88, 44, 11, -90], [76, -49], [76, -30], [76, -12], [76, -68], [49, -76, 12, 68, 30], [72, -67], [-85, 72], [-31, 72], [72, -35], [31, 35, 67, 85, -72], [-79, 61], [61, -12], [-92, 61], [61, -64], [64, 92, -61, 12, 79], [43, -79], [43, -84], [-3, 43], [-67, 43], [3, 79, 67, -43, 84], [-55, 10], [-85, 10], [-95, 10], [10, -77], [-10, 77, 85, 95, 55], [83, -87], [83, -23], [83, -60], [-42, 83], [60, -83, 87, 42, 23], [-64, 40], [40, -15], [40, -45], [-84, 40], [-40, 45, 15, 84, 64], [-38, 19], [-20, 19], [19, -100], [-95, 19], [38, 95, 20, 100, -19], [78, -60], [-87, 78], [78, -30], [78, -48], [-78, 48, 60, 30, 87], [-17, 69], [69, -81], [69, -29], [-53, 69], [53, -69, 17, 81, 29], [-11, 33], [33, -32], [-94, 33], [-9, 33], [94, 9, 11, 32, -33], [80, -42], [-12, 80], [80, -17], [80, -8], [17, 8, 12, 42, -80], [13, -53], [-17, 13], [13, -85], [13, -77], [-13, 77, 53, 17, 85], [86, -35], [-81, 86], [86, -53], [-58, 86], [58, -86, 53, 81, 35], [-14, 37], [37, -12], [-84, 37], [37, -87], [84, 14, 12, 87, -37], [18, -60], [18, -42], [-5, 18], [-58, 18], [5, -18, 58, 60, 42], [65, -58], [65, -25], [65, -81], [65, -95], [25, -65, 58, 95, 81], [28, -45], [-9, 28], [28, -25], [-64, 28], [-28, 45, 9, 64, 25], [-58, 75], [-49, 75], [75, -53], [75, -42], [42, 58, 53, 49, -75], [-2, 46], [46, -31], [46, -88], [-7, 46], [-46, 31, 88, 2, 7], [-29, 62], [62, -92], [62, -20], [62, -7], [20, 92, 7, 29, -62], [-7, 39], [39, -87], [-38, 39], [-35, 39], [-39, 38, 87, 35, 7], [-24, 52], [-42, 52], [52, -17], [52, -3], [17, 24, 42, -52, 3], [63, -95], [-17, 63], [-3, 63], [-94, 63], [94, 17, -63, 95, 3], [66, -60], [-11, 66], [-79, 66], [-23, 66], [23, 11, 60, 79, -66], [-77, 34], [-38, 34], [-23, 34], [34, -12], [38, 23, -34, 12, 77], [-21, 6], [6, -15], [6, -17], [-79, 6], [15, -6, 79, 21, 17], [54, -95], [-82, 54], [-14, 54], [54, -3], [-54, 82, 95, 14, 3], [98, -9], [98, -38], [-15, 98], [98, -29], [9, 15, 38, 29, -98], [-24, 50], [50, -20], [-7, 50], [50, -12], [24, 20, -50, 12, 7], [75], [-39], [-56]]
            # print()
            # print("first clause set")
            # print()
            # /atlas/u/jkuck/learn_BP/data/sat_problems_noIndSets/or-50-5-1-UC-10.cnf.gz.no_w.cnf
            # clauses = [[-26, 51], [-36, 51], [-44, 51], [-48, 51], [-51, 26, 36, 44, 48], [-49, 52], [-9, 52], [-6, 52], [-50, 52], [-52, 49, 9, 6, 50], [-50, 53], [-3, 53], [-48, 53], [-29, 53], [-53, 50, 3, 48, 29], [-23, 54], [-18, 54], [-20, 54], [-34, 54], [-54, 23, 18, 20, 34], [-41, 55], [-23, 55], [-46, 55], [-15, 55], [-55, 41, 23, 46, 15], [-4, 56], [-16, 56], [-10, 56], [-8, 56], [-56, 4, 16, 10, 8], [-2, 57], [-19, 57], [-30, 57], [-23, 57], [-57, 2, 19, 30, 23], [-38, 58], [-1, 58], [-4, 58], [-14, 58], [-58, 38, 1, 4, 14], [-37, 59], [-50, 59], [-13, 59], [-35, 59], [-59, 37, 50, 13, 35], [-10, 60], [-18, 60], [-37, 60], [-9, 60], [-60, 10, 18, 37, 9], [-20, 61], [-36, 61], [-37, 61], [-44, 61], [-61, 20, 36, 37, 44], [-3, 62], [-9, 62], [-27, 62], [-45, 62], [-62, 3, 9, 27, 45], [-31, 63], [-22, 63], [-10, 63], [-36, 63], [-63, 31, 22, 10, 36], [-40, 64], [-19, 64], [-46, 64], [-41, 64], [-64, 40, 19, 46, 41], [-39, 65], [-27, 65], [-16, 65], [-21, 65], [-65, 39, 27, 16, 21], [-42, 66], [-15, 66], [-20, 66], [-4, 66], [-66, 42, 15, 20, 4], [-2, 67], [-29, 67], [-24, 67], [-40, 67], [-67, 2, 29, 24, 40], [-39, 68], [-45, 68], [-25, 68], [-28, 68], [-68, 39, 45, 25, 28], [-39, 69], [-29, 69], [-36, 69], [-15, 69], [-69, 39, 29, 36, 15], [-24, 70], [-18, 70], [-39, 70], [-33, 70], [-70, 24, 18, 39, 33], [-5, 71], [-28, 71], [-3, 71], [-50, 71], [-71, 5, 28, 3, 50], [-20, 72], [-41, 72], [-26, 72], [-37, 72], [-72, 20, 41, 26, 37], [-19, 73], [-3, 73], [-8, 73], [-12, 73], [-73, 19, 3, 8, 12], [-19, 74], [-29, 74], [-16, 74], [-20, 74], [-74, 19, 29, 16, 20], [-10, 75], [-41, 75], [-9, 75], [-48, 75], [-75, 10, 41, 9, 48], [-35, 76], [-27, 76], [-25, 76], [-14, 76], [-76, 35, 27, 25, 14], [-12, 77], [-40, 77], [-39, 77], [-29, 77], [-77, 12, 40, 39, 29], [-30, 78], [-21, 78], [-34, 78], [-9, 78], [-78, 30, 21, 34, 9], [-25, 79], [-35, 79], [-28, 79], [-17, 79], [-79, 25, 35, 28, 17], [-11, 80], [-15, 80], [-36, 80], [-13, 80], [-80, 11, 15, 36, 13], [-24, 81], [-47, 81], [-31, 81], [-2, 81], [-81, 24, 47, 31, 2], [-14, 82], [-3, 82], [-11, 82], [-4, 82], [-82, 14, 3, 11, 4], [-13, 83], [-11, 83], [-41, 83], [-48, 83], [-83, 13, 11, 41, 48], [-37, 84], [-15, 84], [-13, 84], [-49, 84], [-84, 37, 15, 13, 49], [-6, 85], [-3, 85], [-29, 85], [-35, 85], [-85, 6, 3, 29, 35], [-25, 86], [-14, 86], [-43, 86], [-49, 86], [-86, 25, 14, 43, 49], [-49, 87], [-23, 87], [-15, 87], [-9, 87], [-87, 49, 23, 15, 9], [-39, 88], [-2, 88], [-23, 88], [-12, 88], [-88, 39, 2, 23, 12], [-49, 89], [-5, 89], [-13, 89], [-14, 89], [-89, 49, 5, 13, 14], [-7, 90], [-26, 90], [-18, 90], [-22, 90], [-90, 7, 26, 18, 22], [-36, 91], [-8, 91], [-21, 91], [-22, 91], [-91, 36, 8, 21, 22], [-22, 92], [-35, 92], [-30, 92], [-37, 92], [-92, 22, 35, 30, 37], [-1, 93], [-14, 93], [-11, 93], [-16, 93], [-93, 1, 14, 11, 16], [-9, 94], [-11, 94], [-16, 94], [-31, 94], [-94, 9, 11, 16, 31], [-25, 95], [-24, 95], [-19, 95], [-27, 95], [-95, 25, 24, 19, 27], [-48, 96], [-30, 96], [-27, 96], [-3, 96], [-96, 48, 30, 27, 3], [-44, 97], [-40, 97], [-11, 97], [-19, 97], [-97, 44, 40, 11, 19], [-9, 98], [-32, 98], [-6, 98], [-16, 98], [-98, 9, 32, 6, 16], [-2, 99], [-30, 99], [-40, 99], [-36, 99], [-99, 2, 30, 40, 36], [-1, 100], [-21, 100], [-22, 100], [-3, 100], [-100, 1, 21, 22, 3], [89], [-92], [-69]]
            # print("second clause set")

            factor_graph = build_factorgraph_from_SATproblem(clauses, epsilon=EPSILON, max_factor_dimensions=args.max_factor_state_dimensions, ln_Z=-1, belief_repeats=args.belief_repeats)
            training_SAT_list = [factor_graph]
        train_data_loader = DataLoader(training_SAT_list, batch_size=args.batch_size, shuffle=True)


    elif NEW_FAST_DATA_LOADING:
        if not args.train_val_split=='separate_categories':#train val from the same distribution
            train_problems_helper = []
            for cur_train_category in PROBLEM_CATEGORY_TRAIN:
                cur_category_problems = SATDataset(root="./data/SAT_pytorchGeom_proccesed/", dataset_type='train',\
                    problem_category=cur_train_category, belief_repeats=args.belief_repeats, epsilon=EPSILON,\
                    max_factor_dimensions=args.max_factor_state_dimensions, permuted=PERMUTED_SAT_DATA)
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
                    max_factor_dimensions=args.max_factor_state_dimensions, permuted=PERMUTED_SAT_DATA)
                training_SAT_list.extend(cur_category_problems)

            val_SAT_list = []
            for cur_val_category in PROBLEM_CATEGORY_VAL:
                cur_category_problems = SATDataset(root="./data/SAT_pytorchGeom_proccesed/", dataset_type='train',\
                    problem_category=cur_val_category, belief_repeats=args.belief_repeats, epsilon=EPSILON,\
                    max_factor_dimensions=args.max_factor_state_dimensions, permuted=PERMUTED_SAT_DATA)
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
        print("len(train_data_loader)", len(train_data_loader))
        # sleep(temp)
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
            # if args.SHARE_WEIGHTS:
            if True:
                estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                    prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(sat_problem)#, shared_weight_iteration=args.msg_passing_iters)

                max_vTOf = torch.max(torch.abs(prv_prv_varToFactor_messages - prv_varToFactor_messages))
                max_fTOv = torch.max(torch.abs(prv_prv_factorToVar_messages - prv_factorToVar_messages))
                print()
                print("max_vTOf:", max_vTOf)
                print("max_fTOv:", max_fTOv)
                                    
            else:
                estimated_ln_partition_function = lbp_net(sat_problem)

            # vTof_convergence_loss = loss_func(prv_varToFactor_messages, prv_prv_varToFactor_messages)
            # fTov_convergence_loss = loss_func(prv_factorToVar_messages, prv_prv_factorToVar_messages)

            print("estimated_ln_partition_function:", estimated_ln_partition_function)
            if TINY_DEBUG:
                print('exiting early, debugging')
                exit(0)
            print("exact_ln_partition_function:", exact_ln_partition_function)   
            print("torch.abs(exact_ln_partition_function-estimated_ln_partition_function):", torch.abs(exact_ln_partition_function.squeeze()-estimated_ln_partition_function.squeeze()))   

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

                # if args.SHARE_WEIGHTS:
                if True:
                    estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                        prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(sat_problem)#, shared_weight_iteration=args.msg_passing_iters)
                else:                
                    estimated_ln_partition_function = lbp_net(sat_problem)   

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

    # BPNN_trained_model_path = './wandb/run-20200217_104742-n4d8lxp1/model.pt' #2 layer on random sampling of 'or_50' data
    # BPNN_trained_model_path = './wandb/run-20200529_001251-szfxspl1/model.pt' 


    # BPNN_trained_model_path = './wandb/run-20200602_081049-x5l4qhs9/model.pt' #invariant (NOT, actually BUGGY!) BPNN trained on group2
    # BPNN_trained_model_path = './wandb/run-20200602_081049-grv1r22f/model.pt' #not invariant BPNN trained on group2


    BPNN_trained_model_path = './wandb/run-20200603_233046-tag9cucs/model.pt' #invariant  BPNN trained on or50
    # BPNN_trained_model_path = './wandb/run-20200604_001728-ll19cho7/model.pt' #invariant  BPNN trained on problems_75


    # BPNN_trained_model_path = './wandb/run-20200602_084712-uk1mn1z2/model.pt' #NOT invariant  BPNN trained on problems_75
    # BPNN_trained_model_path = './wandb/run-20200602_082928-3oqzpvm6/model.pt' #NOT invariant  BPNN trained on or50

     
    
    
     
    runtimes_dir = '/atlas/u/jkuck/learn_BP/data/SAT_BPNN_runtimes/'
    if not os.path.exists(runtimes_dir):
        os.makedirs(runtimes_dir)
    lbp_net.load_state_dict(torch.load(BPNN_trained_model_path))


    PROBLEM_CATEGORY_TEST = ['or_50_problems']
    # PROBLEM_CATEGORY_TEST = ['or_50_problems', 'or_60_problems', 'or_70_problems', 'or_100_problems', 'blasted_problems', 'problems_75', 'problems_90']
    


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


    test_SAT_list = []
    for cur_train_category in PROBLEM_CATEGORY_TEST:
        cur_category_problems = SATDataset(root="./data/SAT_pytorchGeom_proccesed/", dataset_type='train',\
            problem_category=cur_train_category, belief_repeats=args.belief_repeats, epsilon=EPSILON,\
            max_factor_dimensions=args.max_factor_state_dimensions, permuted=PERMUTED_SAT_DATA)
        test_SAT_list.extend(cur_category_problems)

    test_data_loader = DataLoader(test_SAT_list, batch_size=25, shuffle=False)
    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    squared_errors = []
    BPNN_estimated_counts = []
    loss_sum = 0
    test_problem_count_check = 0
    problem_names = []
#     for sat_problem, exact_ln_partition_function in test_data_loader:
    runtimes = []
    problem_idx = 0
    for repeat_idx in range(1): #the first time the network is run seems to be slow, initialization i assume, so discard initial run (or 2) to be safe
        for sat_problem in test_data_loader:
            sat_problem = sat_problem.to(device)
            sat_problem.state_dimensions = sat_problem.state_dimensions[0] #hack for batching,
            sat_problem.var_cardinality = sat_problem.var_cardinality[0] #hack for batching,
            sat_problem.belief_repeats = sat_problem.belief_repeats[0] #hack for batching,

            exact_ln_partition_function = sat_problem.ln_Z
            # sat_problem.compute_bethe_free_energy()
            t0 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
            ta = time.time()
            print("about to run BPNN")
            if True:
                estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                    prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(sat_problem)#, shared_weight_iteration=args.msg_passing_iters)
            else:
                estimated_ln_partition_function = lbp_net(sat_problem)
            print("done to run BPNN")        
            t1 = resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
            tb = time.time()
            
            print("timing good?:", t1-t0)
            print("timing bad?:", tb-ta)
            # sleep(time_Check)
            # if repeat_idx != 1:
            #     continue
            
            assert(estimated_ln_partition_function.numel() == exact_ln_partition_function.numel())
            for batch_idx in range(estimated_ln_partition_function.numel()):
                runtimes.append((tb-ta)/estimated_ln_partition_function.numel())

                problem_names.append(test_problems[problem_idx])
                runLBP = False
                if runLBP:
                    print("about to parse dimacs")
                    n_vars, clauses, load_successful = parse_dimacs(filename = SAT_PROBLEMS_DIR + "/" + test_problems[problem_idx] + '.cnf.gz.no_w.cnf')
                    # print("about to run_loopyBP")
                    # lbp_estimate = run_loopyBP(clauses, n_vars, maxiter=10, updates="SEQRND", damping='.5')
                    print("LBP:", lbp_estimate)
                    print("sat problem:", test_problems[problem_idx], "exact ln_Z:", exact_ln_partition_function[batch_idx], "estimate:", estimated_ln_partition_function[batch_idx], "LBP:", lbp_estimate)        

                BPNN_estimated_counts.append(estimated_ln_partition_function[batch_idx].item())
                exact_solution_counts.append(exact_ln_partition_function[batch_idx].item())
                loss = loss_func(estimated_ln_partition_function[batch_idx], exact_ln_partition_function[batch_idx].float().squeeze())
                # assert(estimated_ln_partition_function.numel() == exact_ln_partition_function.numel())
                loss_sum += loss.item()#*estimated_ln_partition_function.numel()
                test_problem_count_check += 1#estimated_ln_partition_function.numel()
                print("squared error:", (estimated_ln_partition_function[batch_idx].item() - exact_ln_partition_function[batch_idx].float().squeeze().item())**2)
                print("loss:", (estimated_ln_partition_function[batch_idx].item() - exact_ln_partition_function[batch_idx].float().squeeze().item())**2)
                squared_errors.append((estimated_ln_partition_function[batch_idx].item() - exact_ln_partition_function[batch_idx].float().squeeze().item())**2)
                print("estimated_ln_partition_function[batch_idx]:", estimated_ln_partition_function[batch_idx])
                print("exact_ln_partition_function:", exact_ln_partition_function[batch_idx])
                print()
                problem_idx += 1

    print("runtimes:", runtimes)
#     runtimes_json_string = json.dumps(runtimes)
    results = {'squared_errors': squared_errors, 'runtimes': runtimes, 
               'BPNN_estimated_ln_counts': BPNN_estimated_counts, 
               'exact_ln_solution_counts': exact_solution_counts,
               'problem_names': problem_names}

    print('BPNN RMSE =', np.sqrt(np.mean(results['squared_errors'])))

#     with open(runtimes_dir + PROBLEM_CATEGORY_TEST + "_runtimes.json", 'w') as outfile:
#         json.dump(runtimes, outfile)

    # with open(runtimes_dir + "or50_trainSet_runtimesAndErrors_5layer.json", 'w') as outfile:
    with open(runtimes_dir + "group2_trainSet_runtimesAndErrors_5layer.json", 'w') as outfile:
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
    # train()
    test()
