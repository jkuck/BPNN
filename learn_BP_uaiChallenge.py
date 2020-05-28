import torch
torch.set_printoptions(precision=5, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
from torch import autograd
from nn_models import lbp_message_passing_network, USE_OLD_CODE
from sat_helpers.libdai_utils_sat import run_loopyBP
# from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader
ROOT_DIR = "/atlas/u/jkuck/learn_BP/" #file path to the directory cloned from github
MULTIYEAR_UAI_DATASET=True
if USE_OLD_CODE:
# if False:
    from factor_graph_partialRefactor import DataLoader_custom as DataLoader
    SQUEEZE_BELIEF_REPEATS = True
    
else:
    if MULTIYEAR_UAI_DATASET:
        from UAI_inference_data.multi_year.data_loader import UAI_Dataset
        UAI_dataset_dir = ROOT_DIR + "UAI_inference_data/multi_year/data/"
    else:
        # from UAI_inference_data.2014.data_loader import UAI_Dataset
        UAI_dataset_dir = ROOT_DIR + "UAI_inference_data/2014/data/"
        
    from factor_graph import DataLoader_custom as DataLoader
    SQUEEZE_BELIEF_REPEATS = False
    
import pickle
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

def boolean_string(s):    
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
#number of variables in the largest factor -> factor has 2^args.max_factor_state_dimensions states
parser.add_argument('--max_factor_state_dimensions', type=int, default=2)
#the number of states a variable can take, e.g. 2 for binary variables
parser.add_argument('--var_cardinality', type=int, default=2)

#the number of iterations of message passing, we have this many layers with their own learnable parameters
parser.add_argument('--msg_passing_iters', type=int, default=20)
#messages have var_cardinality states in standard belief propagation.  belief_repeats artificially
#increases this number so that messages have belief_repeats*var_cardinality states, analogous
#to increasing node feature dimensions in a standard graph neural network
parser.add_argument('--belief_repeats', type=int, default=1)

parser.add_argument('--training_batch_size', type=int, default=30)
parser.add_argument('--val_batch_size', type=int, default=5)

# 0.0001
# 0.0005
# parser.add_argument('--learning_rate', type=float, default=0.00001)
# parser.add_argument('--learning_rate', type=float, default=0.000001)
# parser.add_argument('--learning_rate', type=float, default=0.0000001)

# parser.add_argument('--learning_rate', type=float, default=0.1)
# parser.add_argument('--learning_rate', type=float, default=0.005)

parser.add_argument('--learning_rate', type=float, default=0.0001)
# parser.add_argument('--learning_rate', type=float, default=0.000001)

# parser.add_argument('--learning_rate', type=float, default=0.0)


# parser.add_argument('--learning_rate', type=float, default=0.0005)
# parser.add_argument('--learning_rate', type=float, default=0.001)

#damping parameter
parser.add_argument('--alpha_damping_FtoV', type=float, default=.9)
parser.add_argument('--alpha_damping_VtoF', type=float, default=.9) #this damping wasn't used in the old code


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
parser.add_argument('--USE_MLP_DAMPING_VtoF', type=boolean_string, default=True)

#if true, share the weights between layers in a BPNN
parser.add_argument('--SHARE_WEIGHTS', type=boolean_string, default=False)

#if true, subtract previously sent messages (to avoid 'double counting')
parser.add_argument('--subtract_prv_messages', type=boolean_string, default=True)

#if 'none' then use the standard bethe approximation with no learning
#otherwise, describes (potential) non linearities in the MLP
parser.add_argument('--bethe_mlp', type=str, default='none',\
    choices=['shifted','standard','linear','none'])

#if True, use the old Bethe approximation that doesn't work with batches
#only valid for bethe_mlp='none'
parser.add_argument('--use_old_bethe', type=boolean_string, default=False)

#for reproducing random train/val split
#args.random_seed = 0 and 1 seem to produce very different results for s_problems
parser.add_argument('--random_seed', type=int, default=1)

var_card_equal_categories = []
HARD_SEGMENT_TRAINING_PROBLEMS_ONLY = False
EASY_VAL_ONLY = False
USE_DBN_AUGMENTED_TRAINING_DATA = True
parser.add_argument('--problem_category_train', type=str, default='DBN',\
    choices=['ObjDetect', 'Segment', 'Grids', 'DBN', 'Promedas', 'Grids'])
    # choices=['logistics', 'Segmentation', 'Grids', 'DBN', 'Promedus', '2bit', 'ungrouped',\
    #          'r_problems', 'ObjectDetection', 'c_problems', 'relational', 'prob0', 'group1',\
    #          'group2', 'group3', 'group4', 'group5'])


parser.add_argument('--train_val_split', type=str, default='random_shuffle',\
    choices=["random_shuffle", "easyTrain_hardVal", "separate_categories"])


# parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
# parser.add_argument('--test_freq', type=int, default=200)
# parser.add_argument('--save_freq', type=int, default=1000)
args, _ = parser.parse_known_args()

print("args.alpha_damping_VtoF:", args.alpha_damping_VtoF)


##########################
##### Run me on Atlas
# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate


##########################
####### PARAMETERS #######




MODEL_NAME = "debug.pth"



TRAINING_DATA_SIZE = 1000
VAL_DATA_SIZE = 1000 #100
TEST_DATA_SIZE = 1000

  
PROBLEM_CATEGORY_TRAIN = [args.problem_category_train]
PROBLEM_CATEGORY_VAL = PROBLEM_CATEGORY_TRAIN
# PROBLEM_CATEGORY_VAL = ['DBN']

PROBLEM_CATEGORY_TEST = None


EPOCH_COUNT = 1000
PRINT_FREQUENCY = 10
SAVE_FREQUENCY = 1
VAL_FREQUENCY = 10
##########################
##### Optimizer parameters #####
STEP_SIZE=200
LR_DECAY=.5 
LEARNING_RATE = args.learning_rate

##########################

# wandb.init(project="learn_BP_sat_compareParams_mlp4Debug_normalizeBeliefs")
wandb.init(project="learn_BP_uaiChallenge")


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
wandb.config.var_cardinality = args.var_cardinality
wandb.config.max_factor_state_dimensions = args.max_factor_state_dimensions
wandb.config.random_seed = args.random_seed
wandb.config.training_batch_size = args.training_batch_size

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
    belief_repeats=args.belief_repeats, var_cardinality=args.var_cardinality, alpha_damping_FtoV=args.alpha_damping_FtoV,\
    alpha_damping_VtoF=args.alpha_damping_VtoF, use_old_bethe=args.use_old_bethe)
# lbp_net.double()
lbp_net = lbp_net.to(device)


def train():
    # lbp_net.load_state_dict(torch.load('./wandb/run-20200519_040549-ku5kbt8f/model.pt'))
    # lbp_net.load_state_dict(torch.load('./wandb/run-20200523_170623-wi2im90h/model.pt')) #finds good fixed point for segment problem 0
    # lbp_net.load_state_dict(torch.load('./wandb/run-20200523_170930-htukn7ui/model.pt')) #finds good fixed point for segment problem 12
    # lbp_net.load_state_dict(torch.load('./wandb/run-20200523_173520-bpq0oroy/model.pt')) #finds good fixed point for segment problem 4

    # lbp_net.load_state_dict(torch.load('./wandb/run-20200523_193714-lk2a0lr4/model.pt'))
    # lbp_net.load_state_dict(torch.load('./wandb/run-20200526_072138-y739tcrt/model.pt'))


    wandb.watch(lbp_net)
    lbp_net.train()

    # Initialize optimizer
#     optimizer = torch.optim.SGD(lbp_net.parameters(), lr=LEARNING_RATE, momentum=0.7)        
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs    

    mse_loss_func = torch.nn.MSELoss()
    sumSquaredError_loss_func = torch.nn.MSELoss(reduction='sum')
    sumAbsoluteValueError_loss_func = torch.nn.L1Loss(reduction='sum')
    if not args.train_val_split=='separate_categories':#train val from the same distribution
        train_problems_helper = []
        for cur_train_category in PROBLEM_CATEGORY_TRAIN:
            cur_category_problems = UAI_Dataset(root=UAI_dataset_dir, dataset_type='train',\
            # cur_category_problems = UAI_Dataset(root=UAI_dataset_dir, dataset_type='debug_train',\
                problem_category=cur_train_category, belief_repeats=args.belief_repeats,\
                max_factor_dimension=args.max_factor_state_dimensions, var_cardinality=args.var_cardinality)
            train_problems_helper.extend(cur_category_problems)
        if wandb.config.train_val_split == "random_shuffle":
            print("shuffling data")
            random.seed(args.random_seed)
            random.shuffle(train_problems_helper)
        else:
            assert(wandb.config.train_val_split == "easyTrain_hardVal")
        training_list = train_problems_helper[:len(train_problems_helper)*7//10]
        if HARD_SEGMENT_TRAINING_PROBLEMS_ONLY: #only run on problems in the 'Segment' category that BP does particularly poorly on
            debug_training_list=[]
            for train_idx, train_problem in enumerate(training_list):
                # if train_idx in [4]:#,4,5,12]:
                if train_idx in [0,4,5,12]:
                    debug_training_list.append(train_problem)
            training_list = debug_training_list
        # training_list = train_problems_helper #for debug_train
        val_list = train_problems_helper[len(train_problems_helper)*7//10:]
        if EASY_VAL_ONLY:
            easy_val_list=[]
            for val_idx, val_problem in enumerate(val_list):
                # if val_idx in [4]:#,4,5,12]:
                if val_idx not in [6]:
                    easy_val_list.append(val_problem)
            val_list = easy_val_list

        wandb.config.TRAINING_DATA_SIZE = len(train_problems_helper)*7//10
        wandb.config.VAL_DATA_SIZE = len(train_problems_helper) - len(train_problems_helper)*7//10
    else: #use multiple categories for validation and train, using all problems from the categories
        assert(wandb.config.train_val_split == "separate_categories")
        training_list = []
        for cur_train_category in PROBLEM_CATEGORY_TRAIN:
            cur_category_problems = UAI_Dataset(root=UAI_dataset_dir, dataset_type='train',\
                problem_category=cur_train_category, belief_repeats=args.belief_repeats,\
                max_factor_dimension=args.max_factor_state_dimensions, var_cardinality=args.var_cardinality)
            training_list.extend(cur_category_problems)

        val_list = []
        for cur_val_category in PROBLEM_CATEGORY_VAL:
            print('----')
            print('----')
            print('----')
            print("TESTING EVAL2012 for validation!!!!!!!!!!!!!")
            print('----')
            print('----')
            print('----')

            # cur_category_problems = UAI_Dataset(root=UAI_dataset_dir, dataset_type='train',\
            cur_category_problems = UAI_Dataset(root=UAI_dataset_dir, dataset_type='eval2012',\
                problem_category=cur_val_category, belief_repeats=args.belief_repeats,\
                max_factor_dimension=args.max_factor_state_dimensions, var_cardinality=args.var_cardinality)
            val_list.extend(cur_category_problems)

        wandb.config.TRAINING_DATA_SIZE = len(training_list)
        wandb.config.VAL_DATA_SIZE = len(val_list)

    # training_list = training_list[:1]
    # val_list = val_list[:3]
    print("len(training_list) =", len(training_list))
    print("len(val_list) =", len(val_list))

    if USE_DBN_AUGMENTED_TRAINING_DATA:
        training_list = []
        # for seed in range(100):
        for seed in range(1000):
            file_name = "/atlas/u/jkuck/learn_BP/UAI_inference_data/multi_year/data/DBN/augmentation_data/random_factorGraph_%d.pkl" % seed
            with open(file_name, 'rb') as pickle_file:
                cur_factor_graph = pickle.load(pickle_file)
                # if cur_factor_graph.ln_Z > -np.inf:
                if cur_factor_graph.ln_Z > 100:
                    print("cur_factor_graph.ln_Z:", cur_factor_graph.ln_Z)
                    training_list.append(cur_factor_graph)


    train_data_loader = DataLoader(training_list, batch_size=args.training_batch_size, shuffle=True)
    val_data_loader = DataLoader(val_list, batch_size=args.val_batch_size, shuffle=True)
    # val_data_loader = DataLoader(val_list, batch_size=5, shuffle=False)

    # with autograd.detect_anomaly():
    
    for e in range(EPOCH_COUNT):
        print()
        print('epoch:', e)
#         for t, (factor_graph, exact_ln_partition_function) in enumerate(train_data_loader):
        loss_sum = 0
        training_problem_count_check = 0
        epoch_loss = 0

        epoch_est_loss = 0
        epoch_fTOv_loss = 0
        epoch_vTOf_loss = 0

#         optimizer.zero_grad()
        for factor_graph in train_data_loader:
            optimizer.zero_grad()
            if SQUEEZE_BELIEF_REPEATS:
                factor_graph.prv_varToFactor_messages = factor_graph.prv_varToFactor_messages.squeeze()
                factor_graph.prv_factorToVar_messages = factor_graph.prv_factorToVar_messages.squeeze()
                factor_graph.prv_factor_beliefs = factor_graph.prv_factor_beliefs.squeeze()
                factor_graph.prv_var_beliefs = factor_graph.prv_var_beliefs.squeeze()   
                factor_graph.factor_potential_masks = factor_graph.factor_potential_masks.squeeze()   
                factor_graph.factor_potentials = factor_graph.factor_potentials.squeeze()   

                
#                 print("factor_graph.prv_varToFactor_messages.shape:", factor_graph.prv_varToFactor_messages.shape)
#                 print("factor_graph.prv_factorToVar_messages.shape:", factor_graph.prv_factorToVar_messages.shape)
#                 print("factor_graph.prv_factor_beliefs.shape:", factor_graph.prv_factor_beliefs.shape)
#                 print("factor_graph.prv_var_beliefs.shape:", factor_graph.prv_var_beliefs.shape)
                
            if SET_TRUE_POST_DEBUGGING:
                assert(factor_graph.num_vars == torch.sum(factor_graph.numVars))
            factor_graph.state_dimensions = factor_graph.state_dimensions[0] #hack for batching,
            if SET_TRUE_POST_DEBUGGING:
                factor_graph.var_cardinality = factor_graph.var_cardinality[0] #hack for batching,
                factor_graph.belief_repeats = factor_graph.belief_repeats[0] #hack for batching,

            factor_graph = factor_graph.to(device)
            if USE_DBN_AUGMENTED_TRAINING_DATA:
                exact_ln_partition_function = factor_graph.ln_Z/np.log(10) #accidentally multipled ln_Z by ln(10) before saving
            else:
                exact_ln_partition_function = factor_graph.ln_Z
            

            assert(factor_graph.state_dimensions <= args.max_factor_state_dimensions), (factor_graph.state_dimensions, args.max_factor_state_dimensions)
            # if args.SHARE_WEIGHTS:
            if True:
                estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                    prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(factor_graph, shared_weight_iteration=args.msg_passing_iters)

                max_vTOf = torch.max(torch.abs(prv_prv_varToFactor_messages - prv_varToFactor_messages))
                max_fTOv = torch.max(torch.abs(prv_prv_factorToVar_messages - prv_factorToVar_messages))
                print("max_vTOf:", max_vTOf)
                print("max_fTOv:", max_fTOv)
                                    
            else:
                estimated_ln_partition_function = lbp_net(factor_graph)

            vTof_convergence_loss = sumAbsoluteValueError_loss_func(prv_varToFactor_messages, prv_prv_varToFactor_messages)
            fTov_convergence_loss = sumAbsoluteValueError_loss_func(prv_factorToVar_messages, prv_prv_factorToVar_messages)

            print("estimated_ln_partition_function:", estimated_ln_partition_function)
            print("exact_ln_partition_function:", exact_ln_partition_function)   
            print("torch.abs(exact_ln_partition_function - estimated_ln_partition_function):", torch.abs(exact_ln_partition_function.squeeze() - estimated_ln_partition_function.squeeze()))
            print()
            # if (torch.abs(exact_ln_partition_function - estimated_ln_partition_function) > 50).any():
            #     print('difference GREATER THAN 50!!!')
            #     print()
            #     print()
            #     print()
            #     print()

#             time.sleep(.5)
#             
#             sleep(stop_early)
            # print("type(estimated_ln_partition_function):", type(estimated_ln_partition_function))
            # print("type(exact_ln_partition_function):", type(exact_ln_partition_function))
#             print("estimated_ln_partition_function.shape:", estimated_ln_partition_function.shape)
#             print("exact_ln_partition_function.shape:", exact_ln_partition_function.shape)
            
            est_loss = mse_loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float().squeeze())
            epoch_est_loss += est_loss.item()*estimated_ln_partition_function.numel()
            epoch_fTOv_loss += fTov_convergence_loss.item()
            epoch_vTOf_loss += vTof_convergence_loss.item()

            COMPUTE_REGULARIZATION_LOSS = False
            if COMPUTE_REGULARIZATION_LOSS:
                regularization_loss = 0
                for param in lbp_net.parameters():
                    if len(param.shape) == 1:
                        regularization_loss += torch.sum(torch.abs(param))
                    elif len(param.shape) == 2:
                        regularization_loss += torch.sum(torch.abs(param-torch.eye(4, device='cuda')))
                    else:
                        assert(False), 'Error, unexpected parameter'
            # for name, param in lbp_net.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
            # sleep(temp)

            loss = est_loss #+ 100*regularization_loss#+ vTof_convergence_loss + fTov_convergence_loss
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

        assert(len(training_list) == training_problem_count_check)
        training_RMSE = np.sqrt(loss_sum/training_problem_count_check)

        epoch_est_loss = np.sqrt(epoch_est_loss/training_problem_count_check)
        epoch_fTOv_loss = np.sqrt(epoch_fTOv_loss/training_problem_count_check)
        epoch_vTOf_loss = np.sqrt(epoch_vTOf_loss/training_problem_count_check)
        if COMPUTE_REGULARIZATION_LOSS:
            wandb.log({"epoch_vTOf_loss": epoch_vTOf_loss, "epoch_fTOv_loss": epoch_fTOv_loss,
                        "estimation loss": epoch_est_loss, "regularization_loss": regularization_loss}, commit=False)   
        else:
            wandb.log({"epoch_vTOf_loss": epoch_vTOf_loss, "epoch_fTOv_loss": epoch_fTOv_loss,
                        "estimation loss": epoch_est_loss}, commit=False)   

        if e % PRINT_FREQUENCY == 0:
            print("epoch:", e, "root mean squared training error =", training_RMSE)

            print("epoch_vTOf_loss:", epoch_vTOf_loss)
            print("epoch_fTOv_loss:", epoch_fTOv_loss)
            print("estimation loss:", epoch_est_loss)



            # print("vTof_convergence_loss =", vTof_convergence_loss)
            # print("fTov_convergence_loss =", fTov_convergence_loss)
            
        if e % VAL_FREQUENCY == 0:
            print()
            print('-'*80)
            print("VALIDATION")
#             print('-'*40, "check weights 1234", '-'*40)
#             for param in lbp_net.parameters():
#                 print(param.data)
            val_loss_sum = 0
            val_problem_count_check = 0
#             for t, (factor_graph, exact_ln_partition_function) in enumerate(val_data_loader):
            for factor_graph in val_data_loader:
                if SQUEEZE_BELIEF_REPEATS:
                    factor_graph.prv_varToFactor_messages = factor_graph.prv_varToFactor_messages.squeeze()
                    factor_graph.prv_factorToVar_messages = factor_graph.prv_factorToVar_messages.squeeze()
                    factor_graph.prv_factor_beliefs = factor_graph.prv_factor_beliefs.squeeze()
                    factor_graph.prv_var_beliefs = factor_graph.prv_var_beliefs.squeeze()   
                    factor_graph.factor_potential_masks = factor_graph.factor_potential_masks.squeeze()   
                    factor_graph.factor_potentials = factor_graph.factor_potentials.squeeze()   


#                     print("factor_graph.prv_varToFactor_messages.shape:", factor_graph.prv_varToFactor_messages.shape)
#                     print("factor_graph.prv_factorToVar_messages.shape:", factor_graph.prv_factorToVar_messages.shape)
#                     print("factor_graph.prv_factor_beliefs.shape:", factor_graph.prv_factor_beliefs.shape)
#                     print("factor_graph.prv_var_beliefs.shape:", factor_graph.prv_var_beliefs.shape)

                factor_graph.state_dimensions = factor_graph.state_dimensions[0] #hack for batching,
                if SET_TRUE_POST_DEBUGGING:
                    factor_graph.var_cardinality = factor_graph.var_cardinality[0] #hack for batching,
                    factor_graph.belief_repeats = factor_graph.belief_repeats[0] #hack for batching,
            
                factor_graph = factor_graph.to(device)
                exact_ln_partition_function = factor_graph.ln_Z
                assert(factor_graph.state_dimensions <= args.max_factor_state_dimensions)

                # if args.SHARE_WEIGHTS:
                if True:
                    estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                        prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(factor_graph, shared_weight_iteration=args.msg_passing_iters)
                else:                
                    estimated_ln_partition_function = lbp_net(factor_graph)   

                loss = mse_loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float().squeeze())
                print("estimated_ln_partition_function:", estimated_ln_partition_function)
                print("exact_ln_partition_function:", exact_ln_partition_function)
                print("validation absolute value errors:", torch.abs(estimated_ln_partition_function.squeeze()-exact_ln_partition_function.squeeze()))
                print()
#                 print("loss:", loss)
                assert(estimated_ln_partition_function.numel() == exact_ln_partition_function.numel()), (estimated_ln_partition_function.numel(), exact_ln_partition_function.numel())
                val_loss_sum += loss.item()*estimated_ln_partition_function.numel()
                val_problem_count_check += estimated_ln_partition_function.numel()

            assert(len(val_list) == val_problem_count_check)
            val_RMSE = np.sqrt(val_loss_sum/val_problem_count_check)

            print("root mean squared validation error =", val_RMSE)
            print()
            # sleep(val_Check)

            wandb.log({"RMSE_val": val_RMSE, "RMSE_training": training_RMSE})   
            
        # if e%100 == 0:
        #     # for name, param in lbp_net.named_parameters():
        #     #     if 'alpha' in name:
        #     #         print("123 parameter name:", name)
        #     #         print("123 parameter:", param)
            
        else:
            wandb.log({"RMSE_training": training_RMSE})
                
            
        if e % SAVE_FREQUENCY == 0:
            torch.save(lbp_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
            
        scheduler.step()

    torch.save(lbp_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

def test():
    pass



if __name__ == "__main__":
        

    train()
#     test()
