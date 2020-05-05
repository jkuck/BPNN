import torch
from torch import autograd
import pickle
import wandb
import random

from torch_geometric.data import InMemoryDataset

from nn_models import lbp_message_passing_network, GIN_Network_withEdgeFeatures

from ising_model.pytorch_dataset import build_factorgraph_from_SpinGlassModel
from ising_model.spin_glass_model import SpinGlassModel

#debugging, do we get expected behavior (exact ln(Z) and message convergence) when we run on trees?
# from tree_factor_graph.pytorch_dataset import build_tree_factorgraph_from_TreeSpinGlassModel as build_factorgraph_from_SpinGlassModel
# from tree_factor_graph.tree_spin_glass_model import TreeSpinGlassModel as SpinGlassModel

from factor_graph import FactorGraphData
# from torch.utils.data import DataLoader
# from torch_geometric.data import DataLoader as DataLoader_pytorchGeometric
from factor_graph import DataLoader_custom as DataLoader_pytorchGeometric

from ising_model.pytorch_geometric_data import spinGlass_to_torchGeometric


import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import parameters
from parameters import ROOT_DIR, alpha, alpha2
import cProfile 


import argparse

QUICK_TEST = False
BPNN_quick_test_model_path = './wandb/run-20200501_045947-7pobako9/model.pt'


def boolean_string(s):    
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
#the number of iterations of message passing, we have this many layers with their own learnable parameters
parser.add_argument('--msg_passing_iters', type=int, default=10)
#messages have var_cardinality states in standard belief propagation.  belief_repeats artificially
#increases this number so that messages have belief_repeats*var_cardinality states, analogous
#to increasing node feature dimensions in a standard graph neural network
parser.add_argument('--belief_repeats', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=10)

#works well for training on attractive field
# LEARNING_RATE = 0.0005
#works well for training on mixed field
# LEARNING_RATE = 0.005 #10layer        
# LEARNING_RATE = 0.001 #30layer trial 
if QUICK_TEST:
    parser.add_argument('--learning_rate', type=float, default=0.00)
else:
    # parser.add_argument('--learning_rate', type=float, default=0.0005)

    parser.add_argument('--learning_rate', type=float, default=0.0002)

    # parser.add_argument('--learning_rate', type=float, default=0.02)

#damping parameter
parser.add_argument('--alpha_damping_FtoV', type=float, default=1.0)
parser.add_argument('--alpha_damping_VtoF', type=float, default=1.0) #this damping wasn't used in the old code

#if true, mlps operate in standard space rather than log space
parser.add_argument('--lne_mlp', type=boolean_string, default=True)

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


#if true, share the weights between layers in a BPNN
parser.add_argument('--SHARE_WEIGHTS', type=boolean_string, default=True)

#if true, subtract previously sent messages (to avoid 'double counting')
parser.add_argument('--subtract_prv_messages', type=boolean_string, default=True)

#if 'none' then use the standard bethe approximation with no learning
#otherwise, describes (potential) non linearities in the MLP
parser.add_argument('--bethe_mlp', type=str, default='none',\
    choices=['shifted','standard','linear','none'])

#if true, run compute out of distribution validation losses
# if QUICK_TEST:
#     parser.add_argument('--val_ood', type=boolean_string, default=False)
# else:
parser.add_argument('--val_ood', type=boolean_string, default=True)

#if True, use the old Bethe approximation that doesn't work with batches
#only valid for bethe_mlp='none'
parser.add_argument('--use_old_bethe', type=boolean_string, default=False)

args, _ = parser.parse_known_args()


##########################
##### Run me on Atlas
# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate


MODE = "train"# "test"# #run "test" or "train" mode

#####Testing parameters
TEST_TRAINED_MODEL = True #test a pretrained model if True.  Test untrained model if False (e.g. LBP)
# EXPERIMENT_NAME = 'trained_mixedField_10layer_2MLPs_finalBetheMLP/' #used for saving results when MODE='test'
EXPERIMENT_NAME = 'trained_attrField_10layer_2MLPs_noFinalBetheMLP/' #used for saving results when MODE='test'


# 10 layer models
# BPNN_trained_model_path = './wandb/run-20200209_071429-l8jike8k/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True
# BPNN_trained_model_path = './wandb/run-20200211_233743-tpiv47ws/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True, 2MLPs per layer, weight sharing across layers
# BPNN_trained_model_path = './wandb/run-20200219_090032-11077pcu/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True, 2MLPs per layer [wandb results](https://app.wandb.ai/jdkuck/learnBP_spinGlass/runs/11077pcu)

BPNN_trained_model_path = './wandb/run-20200416_213644-2qrbkg30/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True, 2MLPs (MLP3 and MLP4 on variable beliefs) per layer, weight sharing https://app.wandb.ai/jdkuck/learnBP_spinGlass_debug/runs/2qrbkg30

BPNN_trained_model_path = './wandb/run-20200418_170915-2865vckk/model.pt'
BPNN_trained_model_path = './wandb/run-20200418_174008-fqpdm7z4/model.pt' #with "double counting", final bethe MLP has standard ReLUs
BPNN_trained_model_path = './wandb/run-20200418_174621-zgxqvayu/model.pt' #without "double counting", final bethe MLP has standard ReLUs
BPNN_trained_model_path = './wandb/run-20200418_184456-ctss7fhx/model.pt' #without "double counting", final bethe MLP has shifted ReLUs

BPNN_trained_model_path = './wandb/run-20200419_200101-7oyb5xij/model.pt' #without "double counting", final bethe MLP has shifted ReLUs



# GNN_trained_model_path = './wandb/run-20200209_091247-wz2g3fjd/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=True
GNN_trained_model_path = './wandb/run-20200219_051810-bp7hke44/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=True, final MLP takes concatenation of all layers summed node features


# 20 layer models
# BPNN_trained_model_path = './wandb/run-20200403_184715-x8p7a0o7/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True, 2MLPs per layer [wandb results](https://app.wandb.ai/jdkuck/learnBP_spinGlass_reproduce/runs/x8p7a0o7?workspace=user-)

# GNN_trained_model_path = './wandb/run-20200403_191628-s6oaxy9y/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=True, final MLP takes concatenation of all layers summed node features


# BPNN_trained_model_path = './wandb/run-20200209_201644-cj5b13c2/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False
# BPNN_trained_model_path = './wandb/run-20200211_222445-7ky0ix4y/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False, 2MLPs per layer
# BPNN_trained_model_path = './wandb/run-20200211_234428-ylbhlu1o/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False, 2MLPs per layer, weight sharing across layers
# BPNN_trained_model_path = './wandb/run-20200213_092753-4jdedu1x/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False, 2MLPs per layer, final Bethe MLP

# GNN_trained_model_path = './wandb/run-20200209_203009-o8owzdjv/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=False
# GNN_trained_model_path = './wandb/run-20200213_225352-eqnnbg3v/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=False, final MLP gets summed features from all layers, width 4


#15 layer models
# BPNN_trained_model_path = './wandb/run-20200211_083434-fimwr6fw/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True
# GNN_trained_model_path = './wandb/run-20200211_090711-tb0e4alc/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=True

#30 layer models
# BPNN_trained_model_path = './wandb/run-20200211_093808-qrddyif3/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=True
# GNN_trained_model_path = './wandb/run-20200211_093445-xbcslpve/model.pt' #location of the trained GNN model, trained with ATTRACTIVE_FIELD=True
# BPNN_trained_model_path = './wandb/run-20200212_055535-s8qnrxjq/model.pt' #location of the trained BPNN model, trained with ATTRACTIVE_FIELD=False, 2MLPs per layer, no weight sharing across layers


# BPNN_trained_model_path = './wandb/run-20200219_020545-j2ef9bvp/model.pt'

USE_WANDB = False
# os.environ['WANDB_MODE'] = 'dryrun' #don't save to the cloud with this option
##########################
####### Training PARAMETERS #######
MAX_FACTOR_STATE_DIMENSIONS = 2
VAR_CARDINALITY = 2

MSG_PASSING_ITERS = args.msg_passing_iters #the number of iterations of message passing, we have this many layers with their own learnable parameters


EPSILON = 0 #set factor states with potential 0 to EPSILON for numerical stability

# MODEL_NAME = "debugCUDA_spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)
MODEL_NAME = "spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)

TRAINED_MODELS_DIR = ROOT_DIR + "trained_models/" #trained models are stored here



##########################################################################################################
# N_MIN = 8
# N_MAX = 11
# F_MAX = 5.0
# C_MAX = 5.0
N_MIN_TRAIN = 10
N_MAX_TRAIN = 10
F_MAX_TRAIN = .1
C_MAX_TRAIN = 5
# F_MAX = 1
# C_MAX = 10.0
ATTRACTIVE_FIELD_TRAIN = True

N_MIN_VAL = 10
N_MAX_VAL = 10
F_MAX_VAL = .1
C_MAX_VAL = 5
ATTRACTIVE_FIELD_VAL = True
# ATTRACTIVE_FIELD_TEST = True

REGENERATE_DATA = False
# DATA_DIR = "/atlas/u/jkuck/learn_BP/data/spin_glass/"
DATA_DIR = "./data/spin_glass/"


TRAINING_DATA_SIZE = 50
VAL_DATA_SIZE = 50#100
TEST_DATA_SIZE = 200

TRAIN_BATCH_SIZE=args.batch_size
VAL_BATCH_SIZE=args.batch_size

EPOCH_COUNT = 1000
PRINT_FREQUENCY = 10
VAL_FREQUENCY = 10
SAVE_FREQUENCY = 100

TEST_DATSET = 'val' #can test and plot results for 'train', 'val', or 'test' datasets

##### Optimizer parameters #####
STEP_SIZE=300
LR_DECAY=.5
LEARNING_RATE = args.learning_rate

# if ATTRACTIVE_FIELD_TRAIN == True:
#     #works well for training on attractive field
#     LEARNING_RATE = 0.0005
# #     LEARNING_RATE = 0.00

# #         LEARNING_RATE = 0.0002  
# #         LEARNING_RATE = 0.00200001
# #         LEARNING_RATE = 0.00201

# #         LEARNING_RATE = 0.0005        
# #         LEARNING_RATE = 0.002        

# #         LEARNING_RATE = 4*TRAIN_BATCH_SIZE*0.0005        

# #         LEARNING_RATE = 0.00000005 #testing sgd     
# #         LEARNING_RATE = 0.0000002 #testing sgd     

# #     LEARNING_RATE = 0.00005 #10layer with Bethe_mlp
# else:
#     #think this works for mixed fields
#         LEARNING_RATE = 0.005 #10layer        
# #         LEARNING_RATE = 0.001 #30layer trial 

##########################
if USE_WANDB:
    wandb.init(project="learnBP_spinGlass_debug12")
    wandb.config.SHARE_WEIGHTS = args.SHARE_WEIGHTS
    wandb.config.BETHE_MLP = args.bethe_mlp
    wandb.config.lne_mlp = args.lne_mlp
    wandb.config.use_MLP1 = args.use_MLP1
    wandb.config.use_MLP2 = args.use_MLP2
    wandb.config.use_MLP3 = args.use_MLP3
    wandb.config.use_MLP4 = args.use_MLP4
    wandb.config.use_MLP5 = args.use_MLP5
    wandb.config.use_MLP6 = args.use_MLP6
    wandb.config.use_MLP_EQUIVARIANT = args.use_MLP_EQUIVARIANT
    
    wandb.config.subtract_prv_messages = args.subtract_prv_messages
    
    wandb.config.belief_repeats = args.belief_repeats
    wandb.config.MSG_PASSING_ITERS = MSG_PASSING_ITERS
    
    wandb.config.alpha = alpha
    wandb.config.alpha2 = alpha2    
    wandb.config.alpha_damping_FtoV = args.alpha_damping_FtoV
    wandb.config.alpha_damping_VtoF = args.alpha_damping_VtoF    
    wandb.config.epochs = EPOCH_COUNT
    wandb.config.N_MIN_TRAIN = N_MIN_TRAIN
    wandb.config.N_MAX_TRAIN = N_MAX_TRAIN
    wandb.config.F_MAX_TRAIN = F_MAX_TRAIN
    wandb.config.C_MAX_TRAIN = C_MAX_TRAIN
    wandb.config.ATTRACTIVE_FIELD_TRAIN = ATTRACTIVE_FIELD_TRAIN
    wandb.config.TRAINING_DATA_SIZE = TRAINING_DATA_SIZE


    wandb.config.STEP_SIZE = STEP_SIZE
    wandb.config.LR_DECAY = LR_DECAY
    wandb.config.LEARNING_RATE = LEARNING_RATE
    wandb.config.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
    wandb.config.VAL_BATCH_SIZE = VAL_BATCH_SIZE
    wandb.config.VAR_CARDINALITY = VAR_CARDINALITY

    wandb.config.use_old_bethe = args.use_old_bethe

    
def get_dataset(dataset_type, F_MAX=None, C_MAX=None):
    '''
    Store/load a list of SpinGlassModels
    When using, convert to BPNN or GNN form with either 
    build_factorgraph_from_SpinGlassModel(pytorch_geometric=True) for BPNN or spinGlass_to_torchGeometric() for GNN
    '''
    assert(dataset_type in ['train', 'val', 'test'])
    if dataset_type == 'train':
        datasize = TRAINING_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_TRAIN
        N_MIN = N_MIN_TRAIN
        N_MAX = N_MAX_TRAIN
        if F_MAX is None:
            F_MAX = F_MAX_TRAIN
        if C_MAX is None:                    
            C_MAX = C_MAX_TRAIN
    elif dataset_type == 'val':
        datasize = VAL_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_VAL
        N_MIN = N_MIN_VAL
        N_MAX = N_MAX_VAL
        if F_MAX is None:        
            F_MAX = F_MAX_VAL
        if C_MAX is None:        
            C_MAX = C_MAX_VAL        
    else:
        datasize = TEST_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_TEST
        
    dataset_file = DATA_DIR + dataset_type + '%d_%d_%d_%.2f_%.2f_attField=%s.pkl' % (datasize, N_MIN, N_MAX, F_MAX, C_MAX, ATTRACTIVE_FIELD)
    if REGENERATE_DATA or (not os.path.exists(dataset_file)):
        print("REGENERATING DATA!!")
        spin_glass_models_list = [SpinGlassModel(N=random.randint(N_MIN, N_MAX),\
                                                f=np.random.uniform(low=0, high=F_MAX),\
                                                c=np.random.uniform(low=0, high=C_MAX),\
                                                attractive_field=ATTRACTIVE_FIELD) for i in range(datasize)]
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(dataset_file, 'wb') as f:
            pickle.dump(spin_glass_models_list, f)            
    else:
        with open(dataset_file, 'rb') as f:
            spin_glass_models_list = pickle.load(f)
    return spin_glass_models_list
 

class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, dataset_type, datasize, N_MIN, N_MAX, F_MAX, C_MAX, ATTRACTIVE_FIELD, belief_repeats, transform=None, pre_transform=None):
        self.dataset_type = dataset_type
        self.datasize = datasize
        self.N_MIN = N_MIN
        self.N_MAX = N_MAX
        self.F_MAX = F_MAX
        self.C_MAX = C_MAX
        self.ATTRACTIVE_FIELD = ATTRACTIVE_FIELD
        self.belief_repeats = belief_repeats
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        print("HI, looking for raw files :)")
        # return ['some_file_1', 'some_file_2', ...]
        dataset_file = './' + self.root + '/' + self.dataset_type + '%d_%d_%d_%.2f_%.2f_attField=%s.pkl' % (self.datasize, self.N_MIN, self.N_MAX, self.F_MAX, self.C_MAX, self.ATTRACTIVE_FIELD)
        print("dataset_file:", dataset_file)
        return [dataset_file]

    @property
    def processed_file_names(self):
        processed_dataset_file = self.dataset_type + '%d_%d_%d_%.2f_%.2f_%d_attField=%s_pyTorchGeomProccesed.pt' % (self.datasize, self.N_MIN, self.N_MAX, self.F_MAX, self.C_MAX, self.belief_repeats, self.ATTRACTIVE_FIELD)
        return [processed_dataset_file]

    def download(self):
        pass
        # assert(False), "Error, need to generate new data!!"
        # Download to `self.raw_dir`.

    def process(self):
        # Read data into huge `Data` list.
        # data_list = [...]
        dataset_file = self.raw_file_names[0]
        with open(dataset_file, 'rb') as f:
            spin_glass_models_list = pickle.load(f)

        #convert from list of SpinGlassModels to factor graphs for use with BPNN
        sg_models_fg_list = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list]
        # data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg, batch_size=TRAIN_BATCH_SIZE)
        print("training list set built!")

        print("sg_models_fg_list[0]:", sg_models_fg_list[0])
        print("type(sg_models_fg_list[0]):", type(sg_models_fg_list[0]))
        
        print("sg_models_fg_list[0].__class__():", sg_models_fg_list[0].__class__())

        data, slices = self.collate(sg_models_fg_list)
        torch.save((data, slices), self.processed_paths[0])



# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)
lbp_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS, msg_passing_iters=MSG_PASSING_ITERS,\
    lne_mlp=args.lne_mlp, use_MLP1=args.use_MLP1, use_MLP2=args.use_MLP2, use_MLP3=args.use_MLP3, use_MLP4=args.use_MLP4,\
    use_MLP5=args.use_MLP5,  use_MLP6=args.use_MLP6, use_MLP_EQUIVARIANT=args.use_MLP_EQUIVARIANT,\
    subtract_prv_messages=args.subtract_prv_messages, share_weights = args.SHARE_WEIGHTS, bethe_MLP=args.bethe_mlp,\
    belief_repeats=args.belief_repeats, var_cardinality=VAR_CARDINALITY, alpha_damping_FtoV=args.alpha_damping_FtoV,\
    alpha_damping_VtoF=args.alpha_damping_VtoF, use_old_bethe=args.use_old_bethe)

lbp_net = lbp_net.to(device)

print('entering traing')
# lbp_net.double()
def train():
    if QUICK_TEST:
        pass
        # lbp_net.load_state_dict(torch.load(BPNN_quick_test_model_path))
    
    if USE_WANDB:
        
        wandb.watch(lbp_net)
    
    lbp_net.train()
    # Initialize optimizer
    optimizer = torch.optim.Adam(lbp_net.parameters(), lr=LEARNING_RATE)
#     optimizer = torch.optim.SGD(lbp_net.parameters(), lr=LEARNING_RATE, momentum=0.7)    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs    
    loss_func = torch.nn.MSELoss()


    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    print("about to call build_factorgraph_from_SpinGlassModel")
    # spin_glass_models_list_train = get_dataset(dataset_type='train')   
    # sg_models_fg_from_train = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list_train]
   

    sg_models_fg_from_train = MyOwnDataset(root=DATA_DIR, dataset_type='train', datasize=TRAINING_DATA_SIZE, N_MIN=N_MIN_TRAIN, N_MAX=N_MAX_TRAIN, F_MAX=F_MAX_TRAIN,\
                                           C_MAX=C_MAX_TRAIN, ATTRACTIVE_FIELD=ATTRACTIVE_FIELD_TRAIN, belief_repeats=args.belief_repeats, transform=None, pre_transform=None)
   
    train_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    print("training set built!")

    
    # spin_glass_models_list_val = get_dataset(dataset_type='val')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    # sg_models_fg_from_val = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list_val]

    sg_models_fg_from_val = MyOwnDataset(root=DATA_DIR, dataset_type='val', datasize=VAL_DATA_SIZE, N_MIN=N_MIN_VAL, N_MAX=N_MAX_VAL, F_MAX=F_MAX_VAL,\
                                           C_MAX=C_MAX_VAL, ATTRACTIVE_FIELD=ATTRACTIVE_FIELD_VAL, belief_repeats=args.belief_repeats, transform=None, pre_transform=None)
   
    val_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from_val, batch_size=VAL_BATCH_SIZE)
#     val_data_loader_pytorchGeometric_batchSize50 = DataLoader_pytorchGeometric(sg_models_fg_from_val, batch_size=VAL_BATCH_SIZE)
    print("val set built!")
    
    
    if args.val_ood:
        # spin_glass_models_list_valOOD1 = get_dataset(dataset_type='val', F_MAX=.2, C_MAX=5.0)
        #convert from list of SpinGlassModels to factor graphs for use with BPNN
        # sg_models_fg_from_valOOD1 = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list_valOOD1]
        sg_models_fg_from_valOOD1 = MyOwnDataset(root=DATA_DIR, dataset_type='val', datasize=VAL_DATA_SIZE, N_MIN=N_MIN_VAL, N_MAX=N_MAX_VAL, F_MAX=.2,\
                                           C_MAX=5.0, ATTRACTIVE_FIELD=ATTRACTIVE_FIELD_VAL, belief_repeats=args.belief_repeats, transform=None, pre_transform=None)
   
        valOOD1_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from_valOOD1, batch_size=VAL_BATCH_SIZE)

        # spin_glass_models_list_valOOD2 = get_dataset(dataset_type='val', F_MAX=.1, C_MAX=10.0)
        #convert from list of SpinGlassModels to factor graphs for use with BPNN
        # sg_models_fg_from_valOOD2 = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list_valOOD2]
        sg_models_fg_from_valOOD2 = MyOwnDataset(root=DATA_DIR, dataset_type='val', datasize=VAL_DATA_SIZE, N_MIN=N_MIN_VAL, N_MAX=N_MAX_VAL, F_MAX=.1,\
                                           C_MAX=10.0, ATTRACTIVE_FIELD=ATTRACTIVE_FIELD_VAL, belief_repeats=args.belief_repeats, transform=None, pre_transform=None)       
        valOOD2_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from_valOOD2, batch_size=VAL_BATCH_SIZE)
        
        # spin_glass_models_list_valOOD3 = get_dataset(dataset_type='val', F_MAX=.2, C_MAX=10.0)
        #convert from list of SpinGlassModels to factor graphs for use with BPNN
        # sg_models_fg_from_valOOD3 = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list_valOOD3]
        sg_models_fg_from_valOOD3 = MyOwnDataset(root=DATA_DIR, dataset_type='val', datasize=VAL_DATA_SIZE, N_MIN=N_MIN_VAL, N_MAX=N_MAX_VAL, F_MAX=.2,\
                                           C_MAX=10.0, ATTRACTIVE_FIELD=ATTRACTIVE_FIELD_VAL, belief_repeats=args.belief_repeats, transform=None, pre_transform=None)        
        valOOD3_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from_valOOD3, batch_size=VAL_BATCH_SIZE)

        # spin_glass_models_list_valOOD4 = get_dataset(dataset_type='val', F_MAX=1.0, C_MAX=50.0)
        #convert from list of SpinGlassModels to factor graphs for use with BPNN
        # sg_models_fg_from_valOOD4 = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list_valOOD4]
        sg_models_fg_from_valOOD4 = MyOwnDataset(root=DATA_DIR, dataset_type='val', datasize=VAL_DATA_SIZE, N_MIN=N_MIN_VAL, N_MAX=N_MAX_VAL, F_MAX=1.0,\
                                           C_MAX=50.0, ATTRACTIVE_FIELD=ATTRACTIVE_FIELD_VAL, belief_repeats=args.belief_repeats, transform=None, pre_transform=None)        
        
        valOOD4_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from_valOOD4, batch_size=VAL_BATCH_SIZE)
        
            
#     with autograd.detect_anomaly():
    for e in range(EPOCH_COUNT):
        epoch_loss = 0
        optimizer.zero_grad()
        losses = []
        count = 0
        for spin_glass_problem in train_data_loader_pytorchGeometric: #pytorch geometric form
            assert(spin_glass_problem.num_vars == torch.sum(spin_glass_problem.numVars))
            spin_glass_problem = spin_glass_problem.to(device)
    

            spin_glass_problem.state_dimensions = spin_glass_problem.state_dimensions[0] #hack for batching,
            spin_glass_problem.var_cardinality = spin_glass_problem.var_cardinality[0] #hack for batching,
            spin_glass_problem.belief_repeats = spin_glass_problem.belief_repeats[0] #hack for batching,
             
        
            exact_ln_partition_function = spin_glass_problem.ln_Z
            assert((spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS).all())

            # estimated_ln_partition_function = lbp_net(spin_glass_problem)
            estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(spin_glass_problem)

            # print("estimated_ln_partition_function.shape:", estimated_ln_partition_function.shape)
            # print("prv_prv_varToFactor_messages.shape:", prv_prv_varToFactor_messages.shape)
            # print("prv_prv_factorToVar_messages.shape:", prv_prv_factorToVar_messages.shape)
            # print("prv_varToFactor_messages.shape:", prv_varToFactor_messages.shape)
            # print("prv_factorToVar_messages.shape:", prv_factorToVar_messages.shape)
            vTof_convergence_loss = loss_func(prv_varToFactor_messages, prv_prv_varToFactor_messages)
            fTov_convergence_loss = loss_func(prv_factorToVar_messages, prv_prv_factorToVar_messages)

            max_diff_vTof_convergence = torch.max(torch.abs(prv_varToFactor_messages - prv_prv_varToFactor_messages))
            max_diff_fTov_convergence = torch.max(torch.abs(prv_factorToVar_messages - prv_prv_factorToVar_messages))

            # print("vTof_convergence_loss.shape:", vTof_convergence_loss.shape)
            # print("vTof_convergence_loss:", vTof_convergence_loss)
            # print("hi")

            # sleep(temp1)

#             print("type(estimated_ln_partition_function):", type(estimated_ln_partition_function))



#             loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
#             print("estimated_ln_partition_function.shape:", estimated_ln_partition_function.shape)
#             print("exact_ln_partition_function.shape:", exact_ln_partition_function.shape)    
            loss = loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float())
#             print("loss:", loss)
#             sleep(check_loss)


            if QUICK_TEST:
                print("estimated_ln_partition_function:", estimated_ln_partition_function)
                print("exact_ln_partition_function:", exact_ln_partition_function)
                # print("type(exact_ln_partition_function):", type(exact_ln_partition_function))
    #             print(estimated_ln_partition_function.device, exact_ln_partition_function.device)

                print("exact_ln_partition_function - estimated_ln_partition_function:", exact_ln_partition_function - estimated_ln_partition_function)

                message_count = 460#25 #10 # 10
                batch_size=50
                norm_per_isingmodel_vTOf = torch.norm((prv_prv_varToFactor_messages - prv_varToFactor_messages).view([batch_size, message_count*args.belief_repeats*2]), dim=1)
                norm_per_isingmodel_fTOv = torch.norm((prv_prv_factorToVar_messages - prv_factorToVar_messages).view([batch_size, message_count*args.belief_repeats*2]), dim=1)
                print("norm_per_isingmodel_vTOf:", norm_per_isingmodel_vTOf)
                print("norm_per_isingmodel_fTOv:", norm_per_isingmodel_fTOv)
                print("torch.max(prv_prv_factorToVar_messages - prv_factorToVar_messages):", torch.max(prv_prv_factorToVar_messages - prv_factorToVar_messages))
                print("torch.max(prv_prv_varToFactor_messages - prv_varToFactor_messages):", torch.max(prv_prv_varToFactor_messages - prv_varToFactor_messages))
                print("loss =", loss)
                print()
                sleep(temp)
            debug = False
            if debug:
                for idx, val in enumerate(estimated_ln_partition_function):
                    cur_loss = loss_func(val, exact_ln_partition_function.float()[idx])
#                     print("cur_loss between", val, "and", exact_ln_partition_function.float()[idx], "is:", cur_loss)
                    epoch_loss += cur_loss
            else:
                epoch_loss += loss #+ vTof_convergence_loss + fTov_convergence_loss
            # print("loss:", loss)
            # print()
            losses.append(loss.item())

#         sleep(debug_lasdjflkas)

        epoch_loss.backward()
        # nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        optimizer.step()
        scheduler.step()


        if e % PRINT_FREQUENCY == 0:
            print("epoch:", e, "root mean squared training error =", np.sqrt(np.mean(losses)))
            print("epoch loss =", epoch_loss)
            print("partition function loss =", loss)
            print("vTof_convergence_loss =", vTof_convergence_loss)
            print("fTov_convergence_loss =", fTov_convergence_loss)

            print("max_diff_vTof_convergence =", max_diff_vTof_convergence)
            print("max_diff_fTov_convergence =", max_diff_fTov_convergence)            
            print("lower bound check, torch.min(exact_ln_partition_function - estimated_ln_partition_function) =", torch.min(exact_ln_partition_function - estimated_ln_partition_function))

        if e % VAL_FREQUENCY == 0:
#             print('-'*40, "check weights 1234", '-'*40)
#             for param in lbp_net.parameters():
#                 print(param.data)            
            val_losses = []
            for spin_glass_problem in val_data_loader_pytorchGeometric: #pytorch geometric form
#                 print("spin_glass_problem.state_dimensions:", spin_glass_problem.state_dimensions)
#                 print("spin_glass_problem.factor_potentials:", spin_glass_problem.factor_potentials)
#                 print("spin_glass_problem.facStates_to_varIdx:", spin_glass_problem.facStates_to_varIdx)
#                 print("spin_glass_problem.facToVar_edge_idx:", spin_glass_problem.facToVar_edge_idx)
#                 print("spin_glass_problem.edge_index:", spin_glass_problem.edge_index)
#                 print("spin_glass_problem.factor_degrees:", spin_glass_problem.factor_degrees)
#                 print("spin_glass_problem.var_degrees:", spin_glass_problem.var_degrees)
#                 print("spin_glass_problem.numVars:", spin_glass_problem.numVars)
#                 print("spin_glass_problem.numFactors:", spin_glass_problem.numFactors)
#                 print("spin_glass_problem.edge_var_indices:", spin_glass_problem.edge_var_indices)
#                 print("spin_glass_problem.varToFactorMsg_scatter_indices:", spin_glass_problem.varToFactorMsg_scatter_indices)
#                 print("spin_glass_problem.factor_potential_masks:", spin_glass_problem.factor_potential_masks)
                
                
                
                
                spin_glass_problem = spin_glass_problem.to(device)
                spin_glass_problem.facToVar_edge_idx = spin_glass_problem.edge_index #hack for batching, see FactorGraphData in factor_graph.py


                exact_ln_partition_function = spin_glass_problem.ln_Z   
                assert((spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS).all()), (spin_glass_problem.state_dimensions, MAX_FACTOR_STATE_DIMENSIONS)
                spin_glass_problem.state_dimensions = spin_glass_problem.state_dimensions[0] #hack for batching,
                spin_glass_problem.var_cardinality = spin_glass_problem.var_cardinality[0] #hack for batching,
                spin_glass_problem.belief_repeats = spin_glass_problem.belief_repeats[0] #hack for batching,
                
                # estimated_ln_partition_function = lbp_net(spin_glass_problem)   
                estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                    prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(spin_glass_problem)

                loss = loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float())
#                 print("estimated_ln_partition_function:", estimated_ln_partition_function)
#                 print("exact_ln_partition_function:", exact_ln_partition_function)
#                 print("loss:", loss)
                
                val_losses.append(loss.item())

                

                
                
            print("root mean squared validation error =", np.sqrt(np.mean(val_losses)))
            if USE_WANDB:
                wandb.log({"RMSE_val": np.sqrt(np.mean(val_losses)), "RMSE_training": np.sqrt(np.mean(losses))}, commit=(not args.val_ood))       
                
                
                
            if args.val_ood:
                val_losses1 = []
                for spin_glass_problem in valOOD1_data_loader_pytorchGeometric: #pytorch geometric form
                    assert(spin_glass_problem.num_vars == torch.sum(spin_glass_problem.numVars))
                    spin_glass_problem = spin_glass_problem.to(device)
                    spin_glass_problem.state_dimensions = spin_glass_problem.state_dimensions[0] #hack for batching,
                    spin_glass_problem.var_cardinality = spin_glass_problem.var_cardinality[0] #hack for batching,
                    spin_glass_problem.belief_repeats = spin_glass_problem.belief_repeats[0] #hack for batching,

                    exact_ln_partition_function = spin_glass_problem.ln_Z
                    assert((spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS).all())

                    # estimated_ln_partition_function = lbp_net(spin_glass_problem)
                    estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                        prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(spin_glass_problem)

                    loss = loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float())
                    val_losses1.append(loss.item())

                    if USE_WANDB:
                        wandb.log({"RMSE_valOOD1": np.sqrt(np.mean(val_losses1))})       

                    print("root mean squared validation OOD1 error =", np.sqrt(np.mean(val_losses1)))

                    
                val_losses2 = []
                for spin_glass_problem in valOOD2_data_loader_pytorchGeometric: #pytorch geometric form
                    assert(spin_glass_problem.num_vars == torch.sum(spin_glass_problem.numVars))
                    spin_glass_problem = spin_glass_problem.to(device)
                    spin_glass_problem.state_dimensions = spin_glass_problem.state_dimensions[0] #hack for batching,
                    spin_glass_problem.var_cardinality = spin_glass_problem.var_cardinality[0] #hack for batching,
                    spin_glass_problem.belief_repeats = spin_glass_problem.belief_repeats[0] #hack for batching,

                    exact_ln_partition_function = spin_glass_problem.ln_Z
                    assert((spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS).all())

                    # estimated_ln_partition_function = lbp_net(spin_glass_problem)
                    estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                        prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(spin_glass_problem)

                    loss = loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float())
                    val_losses2.append(loss.item())

                    if USE_WANDB:
                        wandb.log({"RMSE_valOOD2": np.sqrt(np.mean(val_losses2))})       

                    print("root mean squared validation OOD2 error =", np.sqrt(np.mean(val_losses2)))
                
                val_losses3 = []
                for spin_glass_problem in valOOD3_data_loader_pytorchGeometric: #pytorch geometric form
                    assert(spin_glass_problem.num_vars == torch.sum(spin_glass_problem.numVars))
                    spin_glass_problem = spin_glass_problem.to(device)
                    spin_glass_problem.state_dimensions = spin_glass_problem.state_dimensions[0] #hack for batching,
                    spin_glass_problem.var_cardinality = spin_glass_problem.var_cardinality[0] #hack for batching,
                    spin_glass_problem.belief_repeats = spin_glass_problem.belief_repeats[0] #hack for batching,

                    exact_ln_partition_function = spin_glass_problem.ln_Z
                    assert((spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS).all())

                    # estimated_ln_partition_function = lbp_net(spin_glass_problem)
                    estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                        prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(spin_glass_problem)

                    loss = loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float())
                    val_losses3.append(loss.item())

                    if USE_WANDB:
                        wandb.log({"RMSE_valOOD3": np.sqrt(np.mean(val_losses3))})       

                    print("root mean squared validation OOD3 error =", np.sqrt(np.mean(val_losses3)))
                
                val_losses4 = []
                for spin_glass_problem in valOOD4_data_loader_pytorchGeometric: #pytorch geometric form
                    assert(spin_glass_problem.num_vars == torch.sum(spin_glass_problem.numVars))
                    spin_glass_problem = spin_glass_problem.to(device)
                    spin_glass_problem.state_dimensions = spin_glass_problem.state_dimensions[0] #hack for batching,
                    spin_glass_problem.var_cardinality = spin_glass_problem.var_cardinality[0] #hack for batching,
                    spin_glass_problem.belief_repeats = spin_glass_problem.belief_repeats[0] #hack for batching,

                    exact_ln_partition_function = spin_glass_problem.ln_Z
                    assert((spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS).all())

                    # estimated_ln_partition_function = lbp_net(spin_glass_problem)
                    estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages,\
                        prv_varToFactor_messages, prv_factorToVar_messages = lbp_net(spin_glass_problem)

                    loss = loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float())
                    val_losses4.append(loss.item())

                    if USE_WANDB:
                        wandb.log({"RMSE_valOOD4": np.sqrt(np.mean(val_losses4))}, commit=True)       

                    print("root mean squared validation OOD4 error =", np.sqrt(np.mean(val_losses4)))
            print("----------123456----------")
            print()            
                
        else:
            if USE_WANDB:
                wandb.log({"RMSE_training": np.sqrt(np.mean(losses))})

        if e % SAVE_FREQUENCY == 0:
            if not os.path.exists(TRAINED_MODELS_DIR):
                os.makedirs(TRAINED_MODELS_DIR)
            torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)
            if USE_WANDB:
                # Save model to wandb
                torch.save(lbp_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

    if not os.path.exists(TRAINED_MODELS_DIR):
        os.makedirs(TRAINED_MODELS_DIR)
    torch.save(lbp_net.state_dict(), TRAINED_MODELS_DIR + MODEL_NAME)


def create_ising_model_figure(results_directory=ROOT_DIR, skip_our_model=False):
    if TEST_TRAINED_MODEL:
        lbp_net.load_state_dict(torch.load(BPNN_trained_model_path))
#         lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))

        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))


    lbp_net.eval()
    
    #data loader for BPNN
    spin_glass_models_list = get_dataset(dataset_type=TEST_DATSET)
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sg_models_fg_from = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list]
    data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from, batch_size=1, shuffle=False)

    #data loader for GNN
    val_data_list_GNN = [spinGlass_to_torchGeometric(sg_problem) for sg_problem in spin_glass_models_list]
    val_loader_GNN = DataLoader_pytorchGeometric(val_data_list_GNN, batch_size=1, shuffle=False)
    
#     gnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_model = GIN_Network_withEdgeFeatures(msg_passing_iters=MSG_PASSING_ITERS).to(device)
    gnn_model.load_state_dict(torch.load(GNN_trained_model_path))
    gnn_model.eval()  
    
    loss_func = torch.nn.MSELoss()

    exact_solution_counts = []
    BPNN_estimated_counts = []

    GNN_estimated_counts = []

#     LBPlibdai_5iters_estimated_counts = []
    LBPlibdai_10iters_estimated_counts = []
    LBPlibdai_100iters_estimated_counts = []
    LBPlibdai_1000iters_estimated_counts = []
    LBPlibdai_1000Seqiters_estimated_counts = []
    LBPlibdai_5kiters_estimated_counts = []
    
    meanFieldlibdai_5kiters_estimated_counts = []
    
#     lbp_losses_5iters = []
    lbp_losses_10iters = []
    lbp_losses_100iters = []
    lbp_losses_1000iters = []
    lbp_losses_1000Seqiters = []
    lbp_losses_5kiters = []
    
    mean_field_losses_5kiters = []
    
    losses = []
    GNN_losses = []
    
    lbp_losses = []
    mrftool_lbp_losses = []
    for idx, (spin_glass_problem, gnn_data) in enumerate(zip(data_loader_pytorchGeometric, val_loader_GNN)): #pytorch geometric form
        print("problem:", idx)
        # spin_glass_problem.compute_bethe_free_energy()     
        sg_problem_SGM = spin_glass_models_list[idx]
        exact_ln_partition_function = spin_glass_problem.ln_Z
        if not skip_our_model:
            #run BPNN
#             spin_glass_problem = spin_glass_problem.to(device)
#             exact_ln_partition_function = exact_ln_partition_function.to(device)
            estimated_ln_partition_function = lbp_net(spin_glass_problem)
            BPNN_estimated_counts.append(estimated_ln_partition_function.item()-exact_ln_partition_function)
            loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
            losses.append(loss.item())
            
            #run GNN FIX ME
            assert(np.isclose(exact_ln_partition_function.item(), gnn_data.ln_Z.item())), (exact_ln_partition_function.item(), gnn_data.ln_Z.item())
            gnn_data = gnn_data.to(device)
            gnn_pred_ln_Z = gnn_model(x=gnn_data.x, edge_index=gnn_data.edge_index, edge_attr=gnn_data.edge_attr, batch=gnn_data.batch)
            gnn_pred_ln_Z = gnn_pred_ln_Z.squeeze()
            GNN_estimated_counts.append(gnn_pred_ln_Z.item()-exact_ln_partition_function)
            gnn_loss = loss_func(gnn_pred_ln_Z, exact_ln_partition_function.float().squeeze())
            GNN_losses.append(gnn_loss.item())
            
#         libdai_lbp_Z_5 = sg_problem_SGM.loopyBP_libdai(maxiter=5)
        libdai_lbp_Z_10 = sg_problem_SGM.loopyBP_libdai(maxiter=10, updates="PARALL", damping=".5")
        libdai_lbp_Z_100 = sg_problem_SGM.loopyBP_libdai(maxiter=100, updates="PARALL", damping=".5")
        libdai_lbp_Z_1000 = sg_problem_SGM.loopyBP_libdai(maxiter=1000, updates="PARALL", damping=".5")
        libdai_lbp_Z_1000Seq = sg_problem_SGM.loopyBP_libdai(maxiter=1000, updates="SEQRND", damping=None)
#         libdai_lbp_Z_5k = sg_problem_SGM.loopyBP_libdai(maxiter=5000)

#         LBPlibdai_5iters_estimated_counts.append(libdai_lbp_Z_5-exact_ln_partition_function)
        LBPlibdai_10iters_estimated_counts.append(libdai_lbp_Z_10-exact_ln_partition_function)
        LBPlibdai_100iters_estimated_counts.append(libdai_lbp_Z_100-exact_ln_partition_function)
        LBPlibdai_1000iters_estimated_counts.append(libdai_lbp_Z_1000-exact_ln_partition_function)
        LBPlibdai_1000Seqiters_estimated_counts.append(libdai_lbp_Z_1000Seq-exact_ln_partition_function)
#         LBPlibdai_5kiters_estimated_counts.append(libdai_lbp_Z_5k-exact_ln_partition_function)

        libdai_meanField_Z_5k = sg_problem_SGM.mean_field_libdai(maxiter=100000)
        meanFieldlibdai_5kiters_estimated_counts.append(libdai_meanField_Z_5k-exact_ln_partition_function)

        
        
        exact_solution_counts.append(exact_ln_partition_function)


#         lbp_losses_5iters.append(loss_func(torch.tensor(libdai_lbp_Z_5), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_10iters.append(loss_func(torch.tensor(libdai_lbp_Z_10), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_100iters.append(loss_func(torch.tensor(libdai_lbp_Z_100), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_1000iters.append(loss_func(torch.tensor(libdai_lbp_Z_1000), exact_ln_partition_function.float().squeeze()).item())
        lbp_losses_1000Seqiters.append(loss_func(torch.tensor(libdai_lbp_Z_1000Seq), exact_ln_partition_function.float().squeeze()).item())
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
        plt.plot(exact_solution_counts, BPNN_estimated_counts, 'x', label='%d layer BPNN, RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.mean(losses))))
        plt.plot(exact_solution_counts, GNN_estimated_counts, 'x', label='%d layer GNN, RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(np.mean(GNN_losses))))

        
#     plt.plot(exact_solution_counts, LBPlibdai_5iters_estimated_counts, '+', label='LBP 5 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_5iters))))
    
    plt.plot(exact_solution_counts, LBPlibdai_10iters_estimated_counts, '+', label='LBP 10 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_10iters))))
    
    plt.plot(exact_solution_counts, LBPlibdai_100iters_estimated_counts, '+', label='LBP 100 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_100iters))))
    
    plt.plot(exact_solution_counts, LBPlibdai_1000iters_estimated_counts, '+', label='LBP 1000 iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_1000iters))))
    
    plt.plot(exact_solution_counts, LBPlibdai_1000Seqiters_estimated_counts, '+', label='LBP 1000 seq iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_1000Seqiters))))
    
#     plt.plot(exact_sol”ution_counts, LBPlibdai_5kiters_estimated_counts, '+', label='LBP 5k iters, RMSE=%.2f' % (np.sqrt(np.mean(lbp_losses_5kiters))))
    
    plt.plot(exact_solution_counts, meanFieldlibdai_5kiters_estimated_counts, '1', label='Mean Field, RMSE=%.2f' % (np.sqrt(np.mean(mean_field_losses_5kiters))))
    
    plt.plot([min(exact_solution_counts), max(exact_solution_counts)], [0, 0], '-', c='g', label='Exact')

    # plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='Ground Truth ln(Set Size)') 
    plt.xlabel('ln(Z)', fontsize=14)
    plt.ylabel('ln(Estimate) - ln(Z)', fontsize=14)
    plt.yscale('symlog')
    plt.title('Exact Partition Function vs. Estimates', fontsize=20)
    # plt.legend(fontsize=8, loc=2, prop={'size': 6})    
#     plt.legend(fontsize=12, prop={'size': 12})    
    # Put a legend below current axis
    lgd = plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.12),
          fancybox=True, ncol=2, fontsize=12, prop={'size': 12})

    #make the font bigger
    matplotlib.rcParams.update({'font.size': 10})        

    plt.grid(True)
    # Shrink current axis's height by 10% on the bottom
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])
    #fig.savefig('/Users/jkuck/Downloads/temp.png', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    if not os.path.exists(results_directory + 'plots/'):
        os.makedirs(results_directory + 'plots/')

    if TEST_DATSET == 'train':
        datasize = TRAINING_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_TRAIN
        N_MIN = N_MIN_TRAIN
        N_MAX = N_MAX_TRAIN
        F_MAX = F_MAX_TRAIN
        C_MAX = C_MAX_TRAIN
    elif TEST_DATSET == 'val':
        datasize = VAL_DATA_SIZE
        ATTRACTIVE_FIELD = ATTRACTIVE_FIELD_VAL
        N_MIN = N_MIN_VAL
        N_MAX = N_MAX_VAL
        F_MAX = F_MAX_VAL
        C_MAX = C_MAX_VAL   
    else:
        assert(False), ("invalid TEST_DATASET")
        
        f5_c5_N10_attFldT
    # plot_name = 'trained=%s_%s_%diters_%d_%d_%.2f_%.2f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, MSG_PASSING_ITERS, N_MIN, N_MAX, F_MAX, C_MAX)
#     plot_name = 'trained=%s_dataset=%s%d_c%f_f%f_N%d%d_att=%s_%diters_alpha%f.png' % (TEST_TRAINED_MODEL, TEST_DATSET, len(data_loader_pytorchGeometric), C_MAX, F_MAX, N_MIN, N_MAX, ATTRACTIVE_FIELD, MSG_PASSING_ITERS, parameters.alpha)
    plot_name = 'f%.2f_c%.2f_N%d-%d_attFld%s.png' % (F_MAX, C_MAX, N_MIN, N_MAX, ATTRACTIVE_FIELD)
    plt.savefig(results_directory + 'plots/' + plot_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    matplotlib.pyplot.clf()
    # plt.show()
    
    return np.sqrt(np.mean(losses)), np.sqrt(np.mean(GNN_losses)), np.sqrt(np.mean(lbp_losses_10iters)), np.sqrt(np.mean(lbp_losses_100iters)),\
           np.sqrt(np.mean(lbp_losses_1000iters)), np.sqrt(np.mean(lbp_losses_1000Seqiters)), np.sqrt(np.mean(mean_field_losses_5kiters))



def test(skip_our_model=False):
    if TEST_TRAINED_MODEL:
#         lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + MODEL_NAME))
        lbp_net.load_state_dict(torch.load(BPNN_trained_model_path))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "simple_4layer_firstWorking.pth"))
        # lbp_net.load_state_dict(torch.load(TRAINED_MODELS_DIR + "trained39non90_2layer.pth"))


    lbp_net.eval()

    spin_glass_models_list = get_dataset(dataset_type=TEST_DATSET)
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sg_models_fg_from = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list]
    data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from, batch_size=1)
    
    
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
            estimated_ln_partition_function = lbp_net(spin_glass_problem)
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
   


def create_many_ising_model_figures(results_dir=ROOT_DIR + '/data/experiments/' + EXPERIMENT_NAME, exp_file='ising_model_OOD.pkl'):
    all_results = {}
    for attractive_field in [True, False]:
        for n in [10, 14]:
            for f_max in [.1, .2, 1.0]:
                for c_max in [5.0, 10.0, 50.0]:
                    global N_MIN_VAL
                    global N_MAX_VAL
                    global F_MAX_VAL
                    global C_MAX_VAL
                    global ATTRACTIVE_FIELD_VAL
                    N_MIN_VAL = n
                    N_MAX_VAL = n
                    F_MAX_VAL = f_max
                    C_MAX_VAL = c_max
                    ATTRACTIVE_FIELD_VAL = attractive_field
                    BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf = create_ising_model_figure(results_directory=results_dir)  
                    all_results[(attractive_field, n, f_max, c_max)] = (BPNN, GNN, lbp10, lbp100, lbp1k, lbp1kSeq, mf)
                    
                    if not os.path.exists(results_dir):
                        os.makedirs(results_dir)                    
                    with open(results_dir + exp_file, 'wb') as f:
                        pickle.dump(all_results, f)
    
    
def simple_val():
    lbp_net.load_state_dict(torch.load(BPNN_trained_model_path))
    
#     lbp_net.train()
    lbp_net.eval()    
    loss_func = torch.nn.MSELoss()

    spin_glass_models_list_val = get_dataset(dataset_type='val')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sg_models_fg_from_val = [build_factorgraph_from_SpinGlassModel(sg_model, belief_repeats=args.belief_repeats) for sg_model in spin_glass_models_list_val]
    val_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sg_models_fg_from_val, batch_size=VAL_BATCH_SIZE)
#     val_data_loader_pytorchGeometric_batchSize50 = DataLoader_pytorchGeometric(sg_models_fg_from_val, batch_size=VAL_BATCH_SIZE)
    
    epoch_loss = 0
    losses = []
    count = 0
    for spin_glass_problem in val_data_loader_pytorchGeometric: #pytorch geometric form
        assert(spin_glass_problem.num_vars == torch.sum(spin_glass_problem.numVars))
        spin_glass_problem = spin_glass_problem.to(device)
        spin_glass_problem.state_dimensions = spin_glass_problem.state_dimensions[0] #hack for batching,
        spin_glass_problem.var_cardinality = spin_glass_problem.var_cardinality[0] #hack for batching,
        spin_glass_problem.belief_repeats = spin_glass_problem.belief_repeats[0] #hack for batching,

        exact_ln_partition_function = spin_glass_problem.ln_Z
        assert((spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS).all())

        estimated_ln_partition_function = lbp_net(spin_glass_problem)

        loss = loss_func(estimated_ln_partition_function.squeeze(), exact_ln_partition_function.float())
        epoch_loss += loss
        losses.append(loss.item())

    print("root mean squared validation error =", np.sqrt(np.mean(losses)))
    
    
if __name__ == "__main__":
    if MODE == "train":
        train()
#         cProfile.run("train()") 
        
    elif MODE == "test":
        simple_val()        
#         test()
#         create_ising_model_figure()
#         create_many_ising_model_figures()    