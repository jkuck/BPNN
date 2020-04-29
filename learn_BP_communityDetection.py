mport torch
from torch import autograd
import pickle
import wandb
import random
import itertools
from nn_models_sbm import lbp_message_passing_network
from community_detection.sbm_libdai import runLBPLibdai 
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
from parameters_sbm import ROOT_DIR, alpha, alpha2, SHARE_WEIGHTS, FINAL_MLP, BETHE_MLP, NUM_MLPS, N, A, B, C, NUM_SAMPLES_TRAIN, NUM_SAMPLES_VAL, SMOOTHING, BELIEF_REPEATS 

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
P_TRAIN = A/N_TRAIN #probability of an edge between vertices in the same community
Q_TRAIN = B/N_TRAIN #probability of an edge between vertices in different communities
C_TRAIN = C #number of communities
# could also store the probability that each node belongs to each community, but that's uniform for now


N_VAL = N
P_VAL = A/N_VAL #probability of an edge between vertices in the same community
Q_VAL = B/N_VAL #probability of an edge between vertices in different communities
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
LR_DECAY=.8
LEARNING_RATE = 1e-3



##########################
if USE_WANDB:
    wandb.init(project="learnBP_sbm", name="New_code_random_init")
    wandb.config.epochs = EPOCH_COUNT
    wandb.config.N_TRAIN = N_TRAIN
    wandb.config.P_TRAIN = P_TRAIN
    wandb.config.Q_TRAIN = Q_TRAIN
    wandb.config.C_TRAIN = C_TRAIN
    wandb.config.TRAINING_DATA_SIZE = NUM_SAMPLES_TRAIN
    wandb.config.VAL_DATA_SIZE = NUM_SAMPLES_VAL
    wandb.config.alpha = alpha
    wandb.config.alpha2 = alpha2
    wandb.config.SHARE_WEIGHTS = SHARE_WEIGHTS
    wandb.config.BETHE_MLP = BETHE_MLP
    wandb.config.MSG_PASSING_ITERS = MSG_PASSING_ITERS
    wandb.config.STEP_SIZE = STEP_SIZE
    wandb.config.LR_DECAY = LR_DECAY
    wandb.config.LEARNING_RATE = LEARNING_RATE
    wandb.config.NUM_MLPS = NUM_MLPS
    wandb.config.LR_DECAY = LR_DECAY
    wandb.config.STEP_SIZE = STEP_SIZE
    wandb.config.BELIEF_REPEATS = BELIEF_REPEATS
    wandb.config.FINAL_FC = FINAL_MLP



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
        sbm_models_list = [StochasticBlockModel(N=N, P=P, Q=Q, C=C) for i in range(datasize)]
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
    
def train(model, data, optim, loss, scheduler, device):
    #if USE_WANDB:        
    #    wandb.watch(bpnn_net)
    model.train()
    tot_loss = 0
    tot_acc = 0
    #optim.zero_grad()
    for sbm_problem in data:
        sbm_problem.to(device)
        labels = sbm_problem.gt_variable_labels
        optim.zero_grad()
        var_beliefs = model(sbm_problem, return_beliefs = True, run_fc = True)
        #print(var_beliefs)
        #time.sleep(1)
        #train_loss = loss(var_beliefs, labels)
        train_loss = getPermInvariantLoss(var_beliefs, labels, C_TRAIN, loss, device)
        tot_acc +=  getPermInvariantAcc(var_beliefs.max(dim = 1)[1], labels, C_TRAIN, device)
        #ce_loss = getPermInvariantLoss(var_beliefs, labels, C_TRAIN, torch.nn.CrossEntropyLoss(), device)
        #print(train_loss.item(), var_beliefs)
        train_loss.backward()
        optim.step()
        #scheduler.step()
        
        #for p in model.parameters():
        #    print (p, p.grad)
        #    time.sleep(1)
        #time.sleep(1000)
        tot_loss += train_loss
    #for p in model.parameters():
    #    p.retain_grad()
    #tot_loss.backward()
    #for p in model.parameters():
    #    if torch.isnan(p.data).any():
    #     print(p)
    #    time.sleep(1)
    #print(tot_loss)
    #optim.step()
    tot_acc = tot_acc / (TRAINING_DATA_SIZE * N_TRAIN)
    train_overlap = (tot_acc - 1/C_TRAIN)/(1-1/C_TRAIN)
    return tot_loss.item() / TRAINING_DATA_SIZE , train_overlap
        
    '''    
    bpnn_net.train()
    # Initialize optimizer
    optimizer = torch.optim.Adam(bpnn_net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs    
    loss_func = torch.nn.CrossEntropyLoss()


    sbm_models_list_train = get_dataset(dataset_type='train')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sbm_models_fg_train = [build_factorgraph_from_sbm(sbm_model) for sbm_model in sbm_models_list_train]
    train_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg_train, batch_size=1)

    
    spin_glass_models_list_val = get_dataset(dataset_type='val')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sbm_models_fg_val = [build_factorgraph_from_sbm(sg_model) for sg_model in spin_glass_models_list_val]
    val_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg_val, batch_size=1)
    
    # with autograd.detect_anomaly():
    for e in range(EPOCH_COUNT):
        epoch_loss = 0
        optimizer.zero_grad()
        losses = []
        for sbm_problem in train_data_loader_pytorchGeometric: #pytorch geometric form
            # spin_glass_problem.facToVar_edge_idx = spin_glass_problem.edge_index #this is a hack for batching, fix me!

            labels = sbm_problem.gt_variable_labels
            var_beliefs = bpnn_net(sbm_problem)

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
            
            print("test overlap = ", str(overlap))
        
        if e % VAL_FREQUENCY == 0:
            val_losses = []
            for spin_glass_problem in val_data_loader_pytorchGeometric: #pytorch geometric form
#                 spin_glass_problem = spin_glass_problem.to(device)
                exact_ln_partition_function = spin_glass_problem.ln_Z   
                assert(spin_glass_problem.state_dimensions == MAX_FACTOR_STATE_DIMENSIONS)

                estimated_ln_partition_function = bpnn_net(spin_glass_problem)            
                loss = loss_func(estimated_ln_partition_function, exact_ln_partition_function.float().squeeze())
                # print("estimated_ln_partition_function:", estimated_ln_partition_function)
                # print("exact_ln_partition_function:", exact_ln_partition_function)

                val_losses.append(loss.item())
            print("root mean squared validation error =", np.sqrt(np.mean(val_losses)))
            print()
            if USE_WANDB:
                wandb.log({"RMSE_val": np.sqrt(np.mean(val_losses)), "RMSE_training": np.sqrt(np.mean(losses))})        
        else:
            if USE_WANDB:
                wandb.log({"RMSE_training": np.sqrt(np.mean(losses))})

        if e % SAVE_FREQUENCY == 0:
            if USE_WANDB:
                # Save model to wandb
                torch.save(bpnn_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

    if USE_WANDB:
        # Save model to wandb
        torch.save(bpnn_net.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
    '''

def test(model, data, orig_data, device, run_fc = True, initial = False):
    if TEST_TRAINED_MODEL:
        bpnn_net.load_state_dict(torch.load(BPNN_trained_model_path))

    #model.eval()
    tot_correct = 0
    tot_loss = 0
    for i, sbm_model in enumerate(data):
        sbm_model.to(device)
        if not initial:
            with torch.no_grad():
                var_beliefs = model(sbm_model, return_beliefs = True, run_fc = run_fc)
        else:
            sbm_model_orig = orig_data[i]
            var_beliefs = runLBPLibdai(sbm_model_orig)
        labels = sbm_model.gt_variable_labels
        out_labs = var_beliefs.max(dim=1)[1]
        #print(var_beliefs_2-var_beliefs)
        #time.sleep(5)
        #print(torch.stack((var_beliefs_2[:,1], var_beliefs_2[:,0]), dim = 1) - var_beliefs)
        tot_correct += getPermInvariantAcc(out_labs, labels, C_VAL, device)
        tot_loss += getPermInvariantLoss(var_beliefs, labels, C_VAL, torch.nn.CrossEntropyLoss(), device).item()
    acc = tot_correct / (VAL_DATA_SIZE * N_VAL)
    overlap = (acc - 1/C_VAL) / (1 - 1/C_VAL)
    return acc, overlap, tot_loss / VAL_DATA_SIZE
    #print("accuracy: " + str(acc))
    #print("overlap: " + str((acc - 1 / C_VAL) / (1-1/C_VAL)))
    

    
'''    
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
            estimated_ln_partition_function = bpnn_net(spin_glass_problem)
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
   
'''

        
if __name__ == "__main__":
    
    #device = torch.device('cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bpnn_net = lbp_message_passing_network(max_factor_state_dimensions=MAX_FACTOR_STATE_DIMENSIONS, msg_passing_iters=MSG_PASSING_ITERS, device=device, share_weights = SHARE_WEIGHTS, bethe_MLP = BETHE_MLP, var_cardinality = C_TRAIN, belief_repeats = BELIEF_REPEATS)
    bpnn_net = bpnn_net.to(device)
    optimizer = torch.optim.Adam(bpnn_net.parameters(), lr=LEARNING_RATE)
    #optimizer = torch.optim.SGD(bpnn_net.parameters(), lr = LEARNING_RATE, momentum = .9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=LR_DECAY) #multiply lr by gamma every step_size epochs    
    loss_func = torch.nn.CrossEntropyLoss() if SMOOTHING is None else crossEntropySmoothing(SMOOTHING)


    sbm_models_list_train = get_dataset(dataset_type='train')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sbm_models_fg_train = [build_factorgraph_from_sbm(sbm_model, BELIEF_REPEATS) for sbm_model in sbm_models_list_train]
    train_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg_train, batch_size=1)

    
    sbm_models_list_val = get_dataset(dataset_type='val')
    #convert from list of SpinGlassModels to factor graphs for use with BPNN
    sbm_models_fg_val = [build_factorgraph_from_sbm(sbm_model, BELIEF_REPEATS) for sbm_model in sbm_models_list_val]
    val_data_loader_pytorchGeometric = DataLoader_pytorchGeometric(sbm_models_fg_val, batch_size=1)
    init_test_acc, init_test_overlap, init_test_loss = test(bpnn_net, val_data_loader_pytorchGeometric, sbm_models_list_val, device=torch.device('cpu'), run_fc = False, initial = True)
    print("Initial BP Overlap: " + str(init_test_overlap))
    #time.sleep(100000)
    if USE_WANDB:
        wandb.log({"Initial BP Overlap": init_test_overlap, "Initial Loss": init_test_loss})        
    for i in range(1, EPOCH_COUNT+1):
        train_loss, train_overlap = train(bpnn_net, train_data_loader_pytorchGeometric, optimizer, loss_func, scheduler, device)
        test_acc, test_overlap, test_loss = test(bpnn_net, val_data_loader_pytorchGeometric, sbm_models_list_val, device)
        print('Epoch {:03d}, Loss: {:.4f}, Train: {:.4f}, Test: {:.4f}'.format(i, train_loss, train_overlap, test_overlap))
        if USE_WANDB:
            wandb.log({"Train Loss": train_loss, "Train Overlap": train_overlap, "Test Overlap": test_overlap, "Test Loss": test_loss})
        scheduler.step()
    '''       
    if MODE == "train":
        train()
    elif MODE == "test":
        test()
    '''
