import torch
from torch import autograd
from nn_models import lbp_message_passing_network
from sat_data import SatProblems
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
# TRAINING_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/SAT_problems_solved/"
TRAINING_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/training_SAT_problems/"

# VALIDATION_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/SAT_problems_solved/"
VALIDATION_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/training_SAT_problems/"

# TEST_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/test_SAT_problems/"
TEST_PROBLEMS_DIR = "/atlas/u/jkuck/GNN_sharpSAT/data/training_SAT_problems/"

TRAINING_DATA_SIZE = 100
VAL_DATA_SIZE = 50#100
TEST_DATA_SIZE = 1000

#contains .txt files for each sat probolem with solution counts
SOLUTION_COUNTS_DIR = "/atlas/u/jkuck/learn_BP/data/sat_counts_uai/"
# SOLUTION_COUNTS_DIR = TRAINING_DATA_DIR + "SAT_problems_solved"


EPOCH_COUNT = 40
PRINT_FREQUENCY = 1
SAVE_FREQUENCY = 10
##########################

#need to deal with independent sets?
#left out 54.sk_12_97.cnf.gz.no_w.cnf and 54.sk_12_97.cnf.gz.no_w.no_independent_set.cnf
TRAINING_SAT_PROBLEM_NAMES = ["01A-1","01B-1","01B-2","01B-3","01B-4","01B-5","02A-1","02A-2","02A-3","02B-1","02B-2","02B-3","02B-4","02B-5","03A-1","03A-2","03B-1","03B-2","03B-3","03B-4","04A-1","04A-2","04A-3","04B-1","04B-2","04B-3","04B-4","05A-1","05A-2","05B-1","05B-2","05B-3","06A-1","06A-2","06A-3","06A-4","06B-1","06B-2","06B-3","06B-4","07A-1","07A-2","07A-3","07A-4","07A-5","07B-1","07B-2","07B-3","07B-4","07B-5","07B-6","08A-1","08A-2","08A-3","08A-4","08B-1","08B-2","08B-3","08B-4","09A-1","09A-2","09A-3","09B-1","09B-2","09B-3","09B-4","09B-5","09B-6","107.sk_3_90","109.sk_4_36","10A-1","10A-2","10A-3","10A-4","10B-10","10B-11","10B-1","10B-2","10B-3","10B-4","10B-5","10B-6","10B-7","10B-8","10B-9","10.sk_1_46","110.sk_3_88","111.sk_2_36","11A-1","11A-2","11A-3","11A-4","11B-1","11B-2","11B-3","11B-4","11B-5","12A-1","12A-2","12A-3","12A-4","12B-1","12B-2","12B-3","12B-4","12B-5","12B-6","13A-1","13A-2","13A-3","13A-4","13B-1","13B-2","13B-3","13B-4","13B-5","14A-1","14A-2","14A-3","15A-1","15A-2","15A-3","15A-4","15B-1","15B-2","15B-3","15B-4","15B-5","17A-1","17A-2","17A-3","17A-4","17A-5","17A-6","17B-1","17B-2","17B-3","17B-4","17B-5","17.sk_3_45","18A-1","18A-2","18A-3","18A-4","19.sk_3_48","20.sk_1_51","27.sk_3_32","29.sk_3_45","30.sk_5_76","32.sk_4_38","35.sk_3_52","36.sk_3_77","4step","50-10-10-q","50-10-1-q","50-10-2-q","50-10-3-q","50-10-4-q","50-10-5-q","50-10-6-q","50-10-7-q","50-10-8-q","50-10-9-q","50-12-10-q","50-12-1-q","50-12-2-q","50-12-3-q","50-12-4-q","50-12-5-q","50-12-6-q","50-12-7-q","50-12-8-q","50-12-9-q","50-14-10-q","50-14-1-q","50-14-2-q","50-14-3-q","50-14-4-q","50-14-5-q","50-14-6-q","50-14-7-q","50-14-8-q","50-14-9-q","50-16-10-q","50-16-1-q","50-16-2-q","50-16-3-q","50-16-4-q","50-16-5-q","50-16-6-q","50-16-7-q","50-16-8-q","50-16-9-q","50-18-10-q","50-18-1-q","50-18-2-q","50-18-3-q","50-18-4-q","50-18-5-q","50-18-6-q","50-18-7-q","50-18-8-q","50-18-9-q","50-20-10-q","50-20-1-q","50-20-2-q","50-20-3-q","50-20-4-q","50-20-5-q","50-20-6-q","50-20-7-q","50-20-8-q","50-20-9-q","51.sk_4_38","53.sk_4_32","55.sk_3_46","56.sk_6_38","57.sk_4_64","5step","63.sk_3_64","70.sk_3_40","71.sk_3_65","75-10-10-q","75-10-1-q","75-10-2-q","75-10-3-q","75-10-4-q","75-10-5-q","75-10-6-q","75-10-7-q","75-10-8-q","75-10-9-q","75-12-10-q","75-12-1-q","75-12-2-q","75-12-3-q","75-12-4-q","75-12-5-q","75-12-6-q","75-12-7-q","75-12-8-q","75-12-9-q","75-14-10-q","75-14-1-q","75-14-2-q","75-14-3-q","75-14-4-q","75-14-5-q","75-14-6-q","75-14-7-q","75-14-8-q","75-14-9-q","75-15-10-q","75-15-1-q","75-15-2-q","75-15-3-q","75-15-4-q","75-15-5-q","75-15-6-q","75-15-7-q","75-15-8-q","75-15-9-q","75-16-10-q","75-16-1-q","75-16-2-q","75-16-3-q","75-16-4-q","75-16-5-q","75-16-6-q","75-16-7-q","75-16-8-q","75-16-9-q","75-17-10-q","75-17-1-q","75-17-2-q","75-17-3-q","75-17-4-q","75-17-5-q","75-17-6-q","75-17-7-q","75-17-8-q","75-17-9-q","75-18-10-q","75-18-1-q","75-18-2-q","75-18-3-q","75-18-4-q","75-18-5-q","75-18-6-q","75-18-7-q","75-18-8-q","75-18-9-q","75-19-10-q","75-19-1-q","75-19-2-q","75-19-3-q","75-19-4-q","75-19-5-q","75-19-6-q","75-19-7-q","75-19-8-q","75-19-9-q","75-20-10-q","75-20-1-q","75-20-2-q","75-20-3-q","75-20-4-q","75-20-5-q","75-20-6-q","75-20-7-q","75-20-8-q","75-20-9-q","75-21-10-q","75-21-1-q","75-21-2-q","75-21-3-q","75-21-4-q","75-21-5-q","75-21-6-q","75-21-7-q","75-21-8-q","75-21-9-q","75-22-10-q","75-22-1-q","75-22-2-q","75-22-3-q","75-22-4-q","75-22-5-q","75-22-6-q","75-22-7-q","75-22-8-q","75-22-9-q","75-23-10-q","75-23-1-q","75-23-2-q","75-23-3-q","75-23-4-q","75-23-5-q","75-23-6-q","75-23-7-q","75-23-8-q","75-23-9-q","75-24-10-q","75-24-1-q","75-24-2-q","75-24-3-q","75-24-4-q","75-24-5-q","75-24-6-q","75-24-7-q","75-24-8-q","75-24-9-q","75-25-10-q","75-25-1-q","75-25-2-q","75-25-3-q","75-25-4-q","75-25-5-q","75-25-6-q","75-25-7-q","75-25-8-q","75-25-9-q","75-26-10-q","75-26-1-q","75-26-2-q","75-26-3-q","75-26-4-q","75-26-5-q","75-26-6-q","75-26-7-q","75-26-8-q","75-26-9-q","77.sk_3_44","79.sk_4_40","7.sk_4_50","80.sk_2_48","81.sk_5_51","84.sk_4_77","90-10-10-q","90-10-1-q","90-10-2-q","90-10-3-q","90-10-4-q","90-10-5-q","90-10-6-q","90-10-7-q","90-10-8-q","90-10-9-q","90-12-10-q","90-12-1-q","90-12-2-q","90-12-3-q","90-12-4-q","90-12-5-q","90-12-6-q","90-12-7-q","90-12-8-q","90-12-9-q","90-14-10-q","90-14-1-q","90-14-2-q","90-14-3-q","90-14-4-q","90-14-5-q","90-14-6-q","90-14-7-q","90-14-8-q","90-14-9-q","90-15-10-q","90-15-1-q","90-15-2-q","90-15-3-q","90-15-4-q","90-15-5-q","90-15-6-q","90-15-7-q","90-15-8-q","90-15-9-q","90-16-10-q","90-16-1-q","90-16-2-q","90-16-3-q","90-16-4-q","90-16-5-q","90-16-6-q","90-16-7-q","90-16-8-q","90-16-9-q","90-17-10-q","90-17-1-q","90-17-2-q","90-17-3-q","90-17-4-q","90-17-5-q","90-17-6-q","90-17-7-q","90-17-8-q","90-17-9-q","90-18-10-q","90-18-1-q","90-18-2-q","90-18-3-q","90-18-4-q","90-18-5-q","90-18-6-q","90-18-7-q","90-18-8-q","90-18-9-q","90-19-10-q","90-19-1-q","90-19-2-q","90-19-3-q","90-19-4-q","90-19-5-q","90-19-6-q","90-19-7-q","90-19-8-q","90-19-9-q","90-20-10-q","90-20-1-q","90-20-2-q","90-20-3-q","90-20-4-q","90-20-5-q","90-20-6-q","90-20-7-q","90-20-8-q","90-20-9-q","90-21-10-q","90-21-1-q","90-21-2-q","90-21-3-q","90-21-4-q","90-21-5-q","90-21-6-q","90-21-7-q","90-21-8-q","90-21-9-q","90-22-10-q","90-22-1-q","90-22-2-q","90-22-3-q","90-22-4-q","90-22-5-q","90-22-6-q","90-22-7-q","90-22-8-q","90-22-9-q","90-23-10-q","90-23-1-q","90-23-2-q","90-23-3-q","90-23-4-q","90-23-5-q","90-23-6-q","90-23-7-q","90-23-8-q","90-23-9-q","90-24-10-q","90-24-1-q","90-24-2-q","90-24-3-q","90-24-4-q","90-24-5-q","90-24-6-q","90-24-7-q","90-24-8-q","90-24-9-q","90-25-10-q","90-25-1-q","90-25-2-q","90-25-3-q","90-25-4-q","90-25-5-q","90-25-6-q","90-25-7-q","90-25-8-q","90-25-9-q","90-26-10-q","90-26-1-q","90-26-2-q","90-26-3-q","90-26-4-q","90-26-5-q","90-26-6-q","90-26-7-q","90-26-8-q","90-26-9-q","90-30-10-q","90-30-1-q","90-30-2-q","90-30-3-q","90-30-4-q","90-30-5-q","90-30-6-q","90-30-7-q","90-30-8-q","90-30-9-q","90-34-10-q","90-34-1-q","90-34-2-q","90-34-3-q","90-34-4-q","90-34-5-q","90-34-6-q","90-34-7-q","90-34-8-q","90-34-9-q","90-38-10-q","90-38-1-q","90-38-2-q","90-38-3-q","90-38-4-q","90-38-5-q","90-38-6-q","90-38-7-q","90-38-8-q","90-38-9-q","90-42-10-q","90-42-1-q","90-42-2-q","90-42-3-q","90-42-4-q","90-42-5-q","90-42-6-q","90-42-7-q","90-42-8-q","90-42-9-q","90-46-10-q","90-46-1-q","90-46-2-q","90-46-3-q","90-46-4-q","90-46-5-q","90-46-6-q","90-46-7-q","90-46-8-q","90-46-9-q","90-50-10-q","90-50-1-q","90-50-2-q","90-50-3-q","90-50-4-q","90-50-5-q","90-50-6-q","90-50-7-q","90-50-8-q","90-50-9-q","ActivityService2.sk_10_27","ActivityService.sk_11_27"]

#all 159 training problems that we have solution counts for and have max variable degree <= 5 (there are 44 others with variable degree > 5)
TRAINING_SAT_PROBLEMS_WITH_SOLUTIONS = ["10.sk_1_46", "27.sk_3_32", "4step", "50-10-10-q", "50-10-5-q", "50-10-7-q", "50-10-8-q", "5step", "75-10-10-q", "75-10-1-q", "75-10-2-q", "75-10-4-q", "75-10-5-q", "75-10-6-q", "75-10-7-q", "75-10-8-q", "75-10-9-q", "75-12-10-q", "75-12-1-q", "75-12-2-q", "75-12-3-q", "75-12-4-q", "75-12-5-q", "75-12-6-q", "75-12-7-q", "75-12-8-q", "75-12-9-q", "75-14-10-q", "75-14-1-q", "75-14-2-q", "75-14-3-q", "75-14-4-q", "75-14-5-q", "75-14-6-q", "75-14-8-q", "75-15-1-q", "75-15-3-q", "75-15-4-q", "75-15-9-q", "90-10-10-q", "90-10-1-q", "90-10-3-q", "90-10-4-q", "90-10-5-q", "90-10-7-q", "90-10-9-q", "90-12-10-q", "90-12-1-q", "90-12-2-q", "90-12-3-q", "90-12-4-q", "90-12-5-q", "90-12-6-q", "90-12-7-q", "90-12-8-q", "90-14-10-q", "90-14-1-q", "90-14-2-q", "90-14-3-q", "90-14-5-q", "90-14-6-q", "90-14-7-q", "90-14-8-q", "90-14-9-q", "90-15-10-q", "90-15-1-q", "90-15-2-q", "90-15-3-q", "90-15-4-q", "90-15-5-q", "90-15-6-q", "90-15-7-q", "90-15-8-q", "90-16-10-q", "90-16-1-q", "90-16-2-q", "90-16-3-q", "90-16-4-q", "90-16-5-q", "90-16-6-q", "90-16-7-q", "90-16-9-q", "90-17-10-q", "90-17-1-q", "90-17-3-q", "90-17-4-q", "90-17-5-q", "90-17-6-q", "90-17-8-q", "90-17-9-q", "90-18-10-q", "90-18-1-q", "90-18-2-q", "90-18-3-q", "90-18-4-q", "90-18-5-q", "90-18-6-q", "90-18-7-q", "90-18-8-q", "90-18-9-q", "90-19-10-q", "90-19-1-q", "90-19-2-q", "90-19-4-q", "90-19-6-q", "90-19-7-q", "90-19-8-q", "90-20-10-q", "90-20-1-q", "90-20-2-q", "90-20-3-q", "90-20-4-q", "90-20-5-q", "90-20-6-q", "90-20-7-q", "90-20-8-q", "90-20-9-q", "90-21-10-q", "90-21-1-q", "90-21-2-q", "90-21-3-q", "90-21-5-q", "90-21-7-q", "90-21-9-q", "90-22-10-q", "90-22-1-q", "90-22-2-q", "90-22-3-q", "90-22-4-q", "90-22-5-q", "90-22-6-q", "90-22-7-q", "90-22-9-q", "90-23-10-q", "90-23-3-q", "90-23-4-q", "90-23-5-q", "90-23-6-q", "90-23-7-q", "90-23-8-q", "90-24-10-q", "90-24-2-q", "90-24-3-q", "90-24-4-q", "90-24-5-q", "90-24-7-q", "90-24-8-q", "90-24-9-q", "90-25-10-q", "90-25-1-q", "90-25-2-q", "90-25-3-q", "90-25-5-q", "90-25-6-q", "90-25-7-q", "90-25-8-q", "90-26-10-q", "90-26-4-q", "90-26-5-q"]

#the 120 trainining problems beginning with 90-
TRAINING_PROBLEMS_90 = ["90-10-10-q", "90-10-1-q", "90-10-3-q", "90-10-4-q", "90-10-5-q", "90-10-7-q", "90-10-9-q", "90-12-10-q", "90-12-1-q", "90-12-2-q", "90-12-3-q", "90-12-4-q", "90-12-5-q", "90-12-6-q", "90-12-7-q", "90-12-8-q", "90-14-10-q", "90-14-1-q", "90-14-2-q", "90-14-3-q", "90-14-5-q", "90-14-6-q", "90-14-7-q", "90-14-8-q", "90-14-9-q", "90-15-10-q", "90-15-1-q", "90-15-2-q", "90-15-3-q", "90-15-4-q", "90-15-5-q", "90-15-6-q", "90-15-7-q", "90-15-8-q", "90-16-10-q", "90-16-1-q", "90-16-2-q", "90-16-3-q", "90-16-4-q", "90-16-5-q", "90-16-6-q", "90-16-7-q", "90-16-9-q", "90-17-10-q", "90-17-1-q", "90-17-3-q", "90-17-4-q", "90-17-5-q", "90-17-6-q", "90-17-8-q", "90-17-9-q", "90-18-10-q", "90-18-1-q", "90-18-2-q", "90-18-3-q", "90-18-4-q", "90-18-5-q", "90-18-6-q", "90-18-7-q", "90-18-8-q", "90-18-9-q", "90-19-10-q", "90-19-1-q", "90-19-2-q", "90-19-4-q", "90-19-6-q", "90-19-7-q", "90-19-8-q", "90-20-10-q", "90-20-1-q", "90-20-2-q", "90-20-3-q", "90-20-4-q", "90-20-5-q", "90-20-6-q", "90-20-7-q", "90-20-8-q", "90-20-9-q", "90-21-10-q", "90-21-1-q", "90-21-2-q", "90-21-3-q", "90-21-5-q", "90-21-7-q", "90-21-9-q", "90-22-10-q", "90-22-1-q", "90-22-2-q", "90-22-3-q", "90-22-4-q", "90-22-5-q", "90-22-6-q", "90-22-7-q", "90-22-9-q", "90-23-10-q", "90-23-3-q", "90-23-4-q", "90-23-5-q", "90-23-6-q", "90-23-7-q", "90-23-8-q", "90-24-10-q", "90-24-2-q", "90-24-3-q", "90-24-4-q", "90-24-5-q", "90-24-7-q", "90-24-8-q", "90-24-9-q", "90-25-10-q", "90-25-1-q", "90-25-2-q", "90-25-3-q", "90-25-5-q", "90-25-6-q", "90-25-7-q", "90-25-8-q", "90-26-10-q", "90-26-4-q", "90-26-5-q"]

#the 39 trainining problems that don't begin with 90-
TRAINING_PROBLEMS_not90 = ["10.sk_1_46", "27.sk_3_32", "4step", "50-10-10-q", "50-10-5-q", "50-10-7-q", "50-10-8-q", "5step", "75-10-10-q", "75-10-1-q", "75-10-2-q", "75-10-4-q", "75-10-5-q", "75-10-6-q", "75-10-7-q", "75-10-8-q", "75-10-9-q", "75-12-10-q", "75-12-1-q", "75-12-2-q", "75-12-3-q", "75-12-4-q", "75-12-5-q", "75-12-6-q", "75-12-7-q", "75-12-8-q", "75-12-9-q", "75-14-10-q", "75-14-1-q", "75-14-2-q", "75-14-3-q", "75-14-4-q", "75-14-5-q", "75-14-6-q", "75-14-8-q", "75-15-1-q", "75-15-3-q", "75-15-4-q", "75-15-9-q"]

lbp_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS, msg_passing_iters=MSG_PASSING_ITERS)
# lbp_net.double()
def train():
    lbp_net.train()

    # Initialize optimizer
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) #multiply lr by gamma every step_size epochs    

    loss_func = torch.nn.MSELoss()

    sat_data_train = SatProblems(problems_to_load=TRAINING_PROBLEMS_not90,
               counts_dir_name=SOLUTION_COUNTS_DIR,
               problems_dir_name=TRAINING_PROBLEMS_DIR,
               # dataset_size=100, begin_idx=50, epsilon=EPSILON)
               dataset_size=TRAINING_DATA_SIZE, epsilon=EPSILON)
    # sleep(temp)
    train_data_loader = DataLoader(sat_data_train, batch_size=1)

    sat_data_val = SatProblems(problems_to_load=TRAINING_PROBLEMS_90,
               counts_dir_name=SOLUTION_COUNTS_DIR,
               problems_dir_name=VALIDATION_PROBLEMS_DIR,
               # dataset_size=50, begin_idx=0, epsilon=EPSILON)
               dataset_size=VAL_DATA_SIZE, begin_idx=0, epsilon=EPSILON)
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

        if e % PRINT_FREQUENCY == 0:
            print("root mean squared training error =", np.sqrt(np.sum(losses)))
            losses = []
            val_losses = []
            for t, (sat_problem, exact_ln_partition_function) in enumerate(val_data_loader):
                sat_problem = FactorGraph.init_from_dictionary(sat_problem, squeeze_tensors=True)
                assert(sat_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)
                estimated_ln_partition_function = lbp_net(sat_problem)                
                loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
                # print("estimated_ln_partition_function:", estimated_ln_partition_function)

                # print("exact_ln_partition_function:", exact_ln_partition_function)

                val_losses.append(loss.item())
            print("root mean squared validation error =", np.sqrt(np.sum(val_losses)))
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
    # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))
    # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
    # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))


    lbp_net.eval()

    sat_data = SatProblems(problems_to_load=TRAINING_PROBLEMS_90,
               counts_dir_name=SOLUTION_COUNTS_DIR,
               problems_dir_name=TEST_PROBLEMS_DIR,
               dataset_size=TEST_DATA_SIZE, begin_idx=0, epsilon=EPSILON)

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
    # train()
    test()