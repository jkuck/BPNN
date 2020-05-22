import torch
from torch import autograd
import pickle
import wandb
import random
import itertools
import numpy as np
from nn_models_sbm import lbp_message_passing_network
from community_detection.sbm_libdai import runLBPLibdai, runJT, build_libdaiFactorGraph_from_SBM
from community_detection.sbm_data_shuvam import StochasticBlockModel, build_factorgraph_from_sbm
from factor_graph import FactorGraphData
from factor_graph import DataLoader_custom as DataLoader_pytorchGeometric
#from torch_geometric.data import DataLoader as DataLoader_pytorchGeometric
import time
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import parameters_sbm
from parameters_sbm_bethe import ROOT_DIR, alpha, alpha2, SHARE_WEIGHTS, FINAL_MLP, BETHE_MLP, NUM_MLPS, N, A_TRAIN, B_TRAIN, A_VAL, B_VAL, C, NUM_SAMPLES_TRAIN, NUM_SAMPLES_VAL, SMOOTHING, BELIEF_REPEATS, LEARN_BP_INIT, NUM_BP_LAYERS, PRE_BP_MLP, USE_MLP_1, USE_MLP_2, USE_MLP_3, USE_MLP_4, INITIALIZE_EXACT_BP, USE_MLP_DAMPING_FtoV, USE_MLP_DAMPING_VtoF

INITIAL_TEST = True
USE_WANDB = True
# os.environ['WANDB_MODE'] = 'dryrun' #don't save to the cloud with this option
MODE = "test" #run "test" or "train" mode

##########################
##### Run me on Atlas
# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate



#####Testing parameters
TEST_TRAINED_MODEL = False #test a pretrained model if True.  Test untrained model if False (e.g. LBP)
EXPERIMENT_NAME = 'SBM_BP_2class_snr3.1' #used for saving results when MODE='test'
BPNN_trained_model_path = '/atlas/u/shuvamc/community_detection/SBM/wandb/run-20200219_090032-11077pcu/model.pt' #location of the trained BPNN model, SET ME!


###################################
####### Training PARAMETERS #######
MAX_FACTOR_STATE_DIMENSIONS = 2
MSG_PASSING_ITERS = 30 #the number of iterations of message passing, we have this many layers with their own learnable parameters

# MODEL_NAME = "debugCUDA_spinGlass_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, parameters.alpha)
MODEL_NAME = "sbm_%dlayer_alpha=%f.pth" % (MSG_PASSING_ITERS, alpha)



##########################################
####### Data Generation PARAMETERS #######
N_TRAIN = N #nodes in each graph
P_TRAIN = A_TRAIN/N_TRAIN #probability of an edge between vertices in the same community
Q_TRAIN = B_TRAIN/N_TRAIN #probability of an edge between vertices in different communities
C_TRAIN = C #number of communities
# could also store the probability that each node belongs to each community, but that's uniform for now


N_VAL = N
P_VAL = A_VAL/N_VAL #probability of an edge between vertices in the same community
Q_VAL = B_VAL/N_VAL #probability of an edge between vertices in different communities
C_VAL = C #number of communities

REGENERATE_DATA = True
DATA_DIR = "/atlas/u/shuvamc/community_detection/SBM"


TRAINING_DATA_SIZE = NUM_SAMPLES_TRAIN
VAL_DATA_SIZE = NUM_SAMPLES_VAL
TEST_DATA_SIZE = 200


EPOCH_COUNT = 300
PRINT_FREQUENCY = 1
VAL_FREQUENCY = 10
SAVE_FREQUENCY = 1

TEST_DATSET = 'val' #can test and plot results for 'train', 'val', or 'test' datasets

##### Optimizer parameters #####
STEP_SIZE=50
LR_DECAY=1
LEARNING_RATE = 1e-5



##########################
if USE_WANDB:
    wandb.init(project="learn_partition_sbm", name="Experiments_bpnn_partition_dampingMLPs_mlp34")
    wandb.config.epochs = EPOCH_COUNT
    wandb.config.N_TRAIN = N_TRAIN
    wandb.config.P_TRAIN = P_TRAIN
    wandb.config.Q_TRAIN = Q_TRAIN
    wandb.config.C_TRAIN = C_TRAIN
    wandb.config.N_VAL = N_VAL
    wandb.config.P_VAL = P_VAL
    wandb.config.Q_VAL = Q_VAL
    wandb.config.C_VAL = C_VAL
    wandb.config.TRAINING_DATA_SIZE = NUM_SAMPLES_TRAIN * N_TRAIN * C_TRAIN
    wandb.config.VAL_DATA_SIZE = NUM_SAMPLES_VAL * N_VAL * C_VAL
    wandb.config.USE_MLP_1 = USE_MLP_1
    wandb.config.USE_MLP_2 = USE_MLP_2 
    wandb.config.USE_MLP_3 = USE_MLP_3
    wandb.config.USE_MLP_4 = USE_MLP_4
    wandb.config.INITIALIZE_EXACT_BP = INITIALIZE_EXACT_BP
    wandb.config.USE_MLP_DAMPING_FtoV = USE_MLP_DAMPING_FtoV
    wandb.config.USE_MLP_DAMPING_VtoF = USE_MLP_DAMPING_VtoF
    wandb.config.alpha = alpha
    wandb.config.alpha2 = alpha2
    wandb.config.SHARE_WEIGHTS = SHARE_WEIGHTS
    wandb.config.BETHE_MLP = BETHE_MLP
    wandb.config.MSG_PASSING_ITERS = MSG_PASSING_ITERS
    wandb.config.STEP_SIZE = STEP_SIZE
    wandb.config.LR_DECAY = LR_DECAY
    wandb.config.LEARNING_RATE = LEARNING_RATE
    wandb.config.LR_DECAY = LR_DECAY
    wandb.config.STEP_SIZE = STEP_SIZE
    wandb.config.BELIEF_REPEATS = BELIEF_REPEATS



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
        sbm_models_list = [StochasticBlockModel(N=N, P=P, Q=Q, C=C, community_probs = [.8, .2]) for i in range(datasize)]
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        with open(dataset_file, 'wb') as f:
            pickle.dump(sbm_models_list, f)            
    else:
        print("here")
        with open(dataset_file, 'rb') as f:
            sbm_models_list = pickle.load(f)
    return sbm_models_list


def getPermInvariantAcc(out, y, n, device):
    perms = list(itertools.permutations(np.arange(n)))
    acc = out.eq(y).sum()
    for p in perms[1:]:
        dct = {i: j for i,j in enumerate(p)}
        perm_lab = list(map(dct.get, y.cpu().numpy()))
        curr_acc = out.eq(torch.Tensor(perm_lab).long().to(device)).sum()
        acc = torch.max(acc, curr_acc)
    return acc.item()


def getPermInvariantLoss(out, y, n, crit, device):
    perms = list(itertools.permutations(np.arange(n)))
    loss = crit(out, y)
    for p in perms[1:]:
        dct = {i: j for i,j in enumerate(p)}
        perm_lab = list(map(dct.get, y.cpu().numpy()))
        curr_loss = crit(out, torch.Tensor(perm_lab).long().to(device))
        loss = torch.min(loss, curr_loss)
    return loss

class crossEntropySmoothing(torch.nn.Module):
    def __init__(self, smooth):
        super(crossEntropySmoothing, self).__init__()
        self.pos = smooth
    def forward(self, pred, target):
        N, dims = pred.shape
        neg = (1 - self.pos) / (dims - 1)
        new_labs = torch.full((N, dims), neg)
        pos_inds = torch.stack((torch.arange(N), target), dim = 1)
        new_labs[tuple(pos_inds.T)] = self.pos
        logsoftmax = torch.nn.LogSoftmax()
        return torch.mean(torch.sum(-new_labs * logsoftmax(pred), dim=1))
    
def train(model, data, optim, loss, scheduler, device, bp_data = None):
    #if USE_WANDB:        
    #    wandb.watch(bpnn_net)
    model.train()
    tot_loss = 0
    tot_acc = 0
    #optim.zero_grad()
    data = zip(data, bp_data) if bp_data is not None else data
    train_loss = 0
    for i, sbm_problem in enumerate(data):
        #sbm_problem.to(device)
        if bp_data is not None:
            sbm_problem, sbm_problem_bp = sbm_problem
            sbm_problem_bp.to(device)
        sbm_problem.to(device)
        #labels = sbm_problem.gt_variable_labels
        true_partition = sbm_problem.ln_Z
        #var_beliefs = model(sbm_problem, device) if bp_data is None else model(sbm_problem, device, sbm_problem_bp)
        estimated_partition, fv_diff, vf_diff = model(sbm_problem, device, perform_bethe = True)
        if USE_WANDB:
            wandb.log({"Final Train max ftoV diff": fv_diff, "Final Train max vtoF diff": vf_diff})
        if BETHE_MLP:
            estimated_partition = estimated_partition.squeeze(dim = 1)
        curr_loss = loss(estimated_partition, true_partition.float())
        train_loss += curr_loss
        #train_loss = getPermInvariantLoss(var_beliefs, labels, C_TRAIN, loss, device)
        #tot_acc +=  getPermInvariantAcc(var_beliefs.max(dim = 1)[1], labels, C_TRAIN, device)
        if i % 16 == 0:
            train_loss.backward()
            optim.step()
            train_loss = 0
            optim.zero_grad()
        #scheduler.step()
        
        #for p in model.parameters():
        #    print (p, p.grad)
        #    time.sleep(1)
        #time.sleep(2)
        tot_loss += curr_loss
    #for p in model.parameters():
    #    p.retain_grad()
    #tot_loss.backward()
    #for p in model.parameters():
    #    if torch.isnan(p.data).any():
    #     print(p)
    #    time.sleep(1)
    #print(tot_loss)
    #optim.step()
    #tot_acc = tot_acc / (TRAINING_DATA_SIZE * N_TRAIN)
    #train_overlap = (tot_acc - 1/C_TRAIN)/(1-1/C_TRAIN)
    return tot_loss.item() / len(data) #, train_overlap
        

def test(model, data, orig_data, device, bp_data = None, run_fc = True, initial = False):
    #model.eval()
    tot_correct = 0
    tot_loss = 0
    data = zip(data, bp_data) if bp_data is not None else data
    for i, sbm_model in enumerate(data):
        #sbm_model.to(device)
        if bp_data is not None:
            sbm_model, sbm_model_bp = sbm_model
            sbm_model_bp.to(device)
        sbm_model.to(device)
        if not initial:
            with torch.no_grad():
                #estimated_partition = model(sbm_model, device) if bp_data is None else model(sbm_model, device, sbm_model_bp)
                estimated_partition, fv_diff, vf_diff = model(sbm_model, device, perform_bethe = True)
                if USE_WANDB:
                    wandb.log({"Final Test max ftoV diff": fv_diff, "Final Test max vtoF diff": vf_diff})
                if BETHE_MLP:
                    estimated_partition = estimated_partition.squeeze(dim = 1)
        else:
            sbm_model_orig = orig_data[i]
            sbm_fg = build_libdaiFactorGraph_from_SBM(sbm_model)
            var_beliefs, _ = runLBPLibdai(sbm_fg, sbm_model_orig)
        true_partition = sbm_model.ln_Z
        tot_loss += (estimated_partition - true_partition.float()) ** 2
    #acc = tot_correct / (VAL_DATA_SIZE * N_VAL)
    #overlap = (acc - 1/C_VAL) / (1 - 1/C_VAL)
    return tot_loss.item() / len(data)
    #print("accuracy: " + str(acc))
    #print("overlap: " + str((acc - 1 / C_VAL) / (1-1/C_VAL)))
    
def constructFixedDataset(sbm_models_lst, init = False):
    fg_models = []
    init_mse = 0
    for sbm_model in sbm_models_lst:
        for i in range(N):
            for j in range(C): 
                sbm_fg = build_libdaiFactorGraph_from_SBM(sbm_model, fixed_var = i, fixed_val = j)     
                jt_ln_z = runJT(sbm_fg)
                if init:       
                    _, lbp_ln_z = runLBPLibdai(sbm_fg, sbm_model)
                    print(jt_ln_z, lbp_ln_z)
                    init_mse += (jt_ln_z - lbp_ln_z) ** 2
                bpnn_fg = build_factorgraph_from_sbm(sbm_model, C, BELIEF_REPEATS, fixed_var = i, fixed_val = j, logZ = jt_ln_z)
                fg_models.append(bpnn_fg)

    return fg_models, init_mse / len(fg_models)
                
if __name__ == "__main__":
    
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bpnn_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS, msg_passing_iters=MSG_PASSING_ITERS, device=device, share_weights = SHARE_WEIGHTS, bethe_MLP = BETHE_MLP, var_cardinality = C_TRAIN, belief_repeats = BELIEF_REPEATS, final_fc_layers = FINAL_MLP, learn_BP_init = LEARN_BP_INIT, num_BP_layers = NUM_BP_LAYERS, pre_BP_mlp = PRE_BP_MLP, use_mlp_1 = USE_MLP_1, use_mlp_2 = USE_MLP_2, use_mlp_3 = USE_MLP_3, use_mlp_4 = USE_MLP_4, init_exact_bp = INITIALIZE_EXACT_BP, mlp_damping_FtoV = USE_MLP_DAMPING_FtoV, mlp_damping_VtoF = USE_MLP_DAMPING_VtoF)
    bpnn_net = bpnn_net.to(device)
    optimizer = torch.optim.Adam(bpnn_net.parameters(), lr=LEARNING_RATE)
    #optimizer = torch.optim.SGD(bpnn_net.parameters(), lr = LEARNING_RATE, momentum = .9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs    
    #loss_func = torch.nn.CrossEntropyLoss() if SMOOTHING is None else crossEntropySmoothing(SMOOTHING)
    loss_func = torch.nn.MSELoss()

    sbm_models_list_train = get_dataset(dataset_type='train')
    #sbm_models_fg_train = [build_factorgraph_from_sbm(sbm_model, C_TRAIN, BELIEF_REPEATS) for sbm_model in sbm_models_list_train]
    sbm_models_fg_train, _ = constructFixedDataset(sbm_models_list_train)
    train_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg_train, batch_size=1, shuffle = not LEARN_BP_INIT)
    train_data_loader_pytorchGeometric_bp = None
    if LEARN_BP_INIT:
        sbm_models_fg_train_bp = [build_factorgraph_from_sbm(sbm_model, C_TRAIN, 1) for sbm_model in sbm_models_list_train]
        train_data_loader_pytorchGeometric_bp = DataLoader_pytorchGeometric(sbm_models_fg_train_bp, batch_size=1, shuffle = False)
   
    sbm_models_list_val = get_dataset(dataset_type='val')
    #sbm_models_fg_val = [build_factorgraph_from_sbm(sbm_model, C_TRAIN, BELIEF_REPEATS) for sbm_model in sbm_models_list_val]
    sbm_models_fg_val, init_test_mse = constructFixedDataset(sbm_models_list_val, init = True)
    val_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg_val, batch_size=1)
    val_data_loader_pytorchGeometric_bp = None
    if LEARN_BP_INIT:
        sbm_models_fg_val_bp = [build_factorgraph_from_sbm(sbm_model, C_TRAIN, 1) for sbm_model in sbm_models_list_val]
        val_data_loader_pytorchGeometric_bp = DataLoader_pytorchGeometric(sbm_models_fg_val_bp, batch_size=1, shuffle = False)

    if INITIAL_TEST: 
        print("Initial MSE: " + str(init_test_mse))
        if USE_WANDB:
            wandb.log({"Initial Test MSE": init_test_mse})
        #init_test_acc, init_test_overlap, init_test_loss = test(bpnn_net, val_data_loader_pytorchGeometric, sbm_models_list_val, device=torch.device('cpu'), run_fc = False, initial = True)
        #print("Initial BP Overlap: " + str(init_test_overlap))
        #if USE_WANDB:
        #    wandb.log({"Initial BP Overlap": init_test_overlap, "Initial Loss": init_test_loss})        
    for i in range(1, EPOCH_COUNT+1):
        train_loss = train(bpnn_net, train_data_loader_pytorchGeometric, optimizer, loss_func, scheduler, device, train_data_loader_pytorchGeometric_bp)
        test_loss = test(bpnn_net, val_data_loader_pytorchGeometric, sbm_models_list_val, device, val_data_loader_pytorchGeometric_bp)
        print('Epoch {:03d}, Train: {:.4f}, Test: {:.4f}'.format(i, train_loss, test_loss))
        if USE_WANDB:
            wandb.log({"Train Loss": train_loss, "Test Loss": test_loss})
            if i % 10 == 0:
                wandb.log({"Train Loss Smooth": train_loss, "Test Loss Smooth": test_loss})
        #scheduler.step()
        if LEARN_BP_INIT:
            perm = np.random.permutation(NUM_SAMPLES_TRAIN)
            sbm_models_fg_train = [sbm_models_fg_train[i] for i in perm]
            train_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg_train, batch_size = 1, shuffle = False)
            sbm_models_fg_train_bp = [sbm_models_fg_train_bp[i] for i in perm]
            train_data_loader_pytorchGeometric_bp = DataLoader_pytorchGeometric(sbm_models_fg_train_bp, batch_size = 1, shuffle = False)

