import os.path as osp
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os 
import wandb

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GINConv


from torch_geometric.data import Data, DataLoader
from ising_model.pytorch_geometric_data import spinGlass_to_torchGeometric
from ising_model.spin_glass_model import SpinGlassModel

from nn_models import GIN_Network_withEdgeFeatures

from parameters import ROOT_DIR
import pickle

MODE = 'train'
##########################################################################################################
MSG_PASSING_ITERS = 10

N_MIN_TRAIN = 10
N_MAX_TRAIN = 10
F_MAX_TRAIN = .1
C_MAX_TRAIN = 5.0
# F_MAX = 1
# C_MAX = 10.0
TRAIN_DATA_SIZE = 500
ATTRACTIVE_FIELD_TRAIN = False

# N_MIN_VAL = 10
# N_MAX_VAL = 10
# F_MAX_VAL = .1
# C_MAX_VAL = 5
N_MIN_VAL = 10
N_MAX_VAL = 10
F_MAX_VAL = .1
C_MAX_VAL = 5.0
VAL_DATA_SIZE = 50
ATTRACTIVE_FIELD_VAL = False

SAVE_FREQUENCY = 100
EPOCH_COUNT = 2500

######### FOR GETTING THE SAME DATA USED BY BPNN ##############
USE_BPNN_DATA=True
DATA_DIR = "/atlas/u/jkuck/learn_BP/data/spin_glass/"
def get_dataset(dataset_type='test'):
#     assert(dataset_type in ['train', 'val', 'test'])
    assert(dataset_type in ['train', 'val'])
    if dataset_type == 'train':
        dataset_file = DATA_DIR + dataset_type + '%d_%d_%d_%.2f_%.2f_attField=%s.pkl' % (TRAIN_DATA_SIZE, N_MIN_TRAIN, N_MAX_TRAIN, F_MAX_TRAIN, C_MAX_TRAIN, ATTRACTIVE_FIELD_TRAIN)    
    elif dataset_type == 'val':
        dataset_file = DATA_DIR + dataset_type + '%d_%d_%d_%.2f_%.2f_attField=%s.pkl' % (VAL_DATA_SIZE, N_MIN_VAL, N_MAX_VAL, F_MAX_VAL, C_MAX_VAL, ATTRACTIVE_FIELD_VAL)
    else:
        assert(False), "Invalid dataset type requested"
    print("dataset_file:", dataset_file)
    if (not os.path.exists(dataset_file)):
        assert(False), "test dataset missing!"         
    else:
        with open(dataset_file, 'rb') as f:
            spin_glass_models_list = pickle.load(f)
    return spin_glass_models_list


if USE_BPNN_DATA:
    spin_glass_models_list_train = get_dataset(dataset_type='train')
    train_data_list = [spinGlass_to_torchGeometric(sg_problem) for sg_problem in spin_glass_models_list_train]
    train_loader = DataLoader(train_data_list, batch_size=50)
    
    spin_glass_models_list_val = get_dataset(dataset_type='val')
    val_data_list = [spinGlass_to_torchGeometric(sg_problem) for sg_problem in spin_glass_models_list_val]
    val_loader = DataLoader(val_data_list, batch_size=50)
else:
    train_data_list = [spinGlass_to_torchGeometric(SpinGlassModel(N=random.randint(N_MIN, N_MAX),\
                                                            f=np.random.uniform(low=0, high=F_MAX),\
                                                            c=np.random.uniform(low=0, high=C_MAX),\
                                                            attractive_field=ATTRACTIVE_FIELD_TRAIN)) for i in range(TRAIN_DATA_SIZE)]
    train_loader = DataLoader(train_data_list, batch_size=50)

    val_data_list = [spinGlass_to_torchGeometric(SpinGlassModel(N=random.randint(N_MIN_VAL, N_MAX_VAL),\
                                                            f=np.random.uniform(low=0, high=F_MAX_VAL),\
                                                            c=np.random.uniform(low=0, high=C_MAX_VAL),\
                                                            attractive_field=ATTRACTIVE_FIELD_VAL)) for i in range(VAL_DATA_SIZE)]
    val_loader = DataLoader(val_data_list, batch_size=200)    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN_Network_withEdgeFeatures(msg_passing_iters=MSG_PASSING_ITERS).to(device)

#works well for attractive field
if ATTRACTIVE_FIELD_TRAIN:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) #works for width/hidden_size=4
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.2)    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    
loss_func = torch.nn.MSELoss(reduction='sum') #sum of squared errors

def train_epoch():
#     print("hi")

    
    model.train()
    total_loss = 0
    for data in train_loader:
#         print("data.edge_index.shape:", data.edge_index.shape)
#         print("data.unary_potentials.shape:", data.unary_potentials.shape)
#         print("data.ln_Z.shape:", data.ln_Z.shape)        
#         sleep(shape_check)
        
        data = data.to(device)
        optimizer.zero_grad()
        pred_ln_Z = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        pred_ln_Z = pred_ln_Z.squeeze()
        loss = loss_func(pred_ln_Z, data.ln_Z)
        loss.backward()
        total_loss += loss.item() # * data.num_graphs
#         print("predicted Z:", pred_ln_Z)
#         print("exact Z:", data.ln_Z)
#         print("total_loss:", total_loss)
#         print("loss.item():", loss.item())
        check_loss = torch.sum((pred_ln_Z - data.ln_Z)*(pred_ln_Z - data.ln_Z))
#         print("(pred_ln_Z - data.ln_Z):", (pred_ln_Z - data.ln_Z))
#         print("check loss shape, loss.shape:", loss.shape)
#         print("data.batch.size:", data.batch.size())
        optimizer.step()
    return total_loss / TRAIN_DATA_SIZE


def val_epoch(plot=False, val_mode=False):
    if val_mode:
        model.load_state_dict(torch.load('./wandb/run-20200209_091247-wz2g3fjd/model.pt'))
    model.eval()
    total_loss = 0
    for data in val_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred_ln_Z = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        pred_ln_Z = pred_ln_Z.squeeze()
        loss = loss_func(pred_ln_Z, data.ln_Z)
        total_loss += loss.item() # * data.num_graphs
        
        
    if plot == True:
        print("data.ln_Z.cpu().detach().numpy():", data.ln_Z.cpu().detach().numpy())
        print("pred_ln_Z.cpu().detach().numpy():", pred_ln_Z.cpu().detach().numpy())
        print("pred_ln_Z.cpu().detach().numpy() - data.ln_Z.cpu().detach().numpy():", pred_ln_Z.cpu().detach().numpy() - data.ln_Z.cpu().detach().numpy())
        plt.plot(data.ln_Z.cpu().detach().numpy(), pred_ln_Z.cpu().detach().numpy() - data.ln_Z.cpu().detach().numpy(), 'x', c='g', label='GNN estimate, %d iters, RMSE=%.2f' % (MSG_PASSING_ITERS, np.sqrt(total_loss/VAL_DATA_SIZE)))

        plt.xlabel('ln(Z)', fontsize=14)
        plt.ylabel('ln(Estimate) - ln(Z)', fontsize=14)
        plt.title('Exact Partition Function vs. Estimates', fontsize=20)
        # plt.legend(fontsize=8, loc=2, prop={'size': 6})    
        plt.legend(fontsize=12, prop={'size': 12})    
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
        plot_name = 'baselineGNN_%diters.png' % (MSG_PASSING_ITERS)
        plt.savefig(ROOT_DIR + 'plots/' + plot_name)
        matplotlib.pyplot.clf()
        # plt.show()        
    return total_loss / VAL_DATA_SIZE

def train():
    wandb.init(project="gnn_sat")
    wandb.watch(model)
    for epoch in range(1, EPOCH_COUNT):
    
    # for epoch in range(1, 501):
        loss = train_epoch()
    #     print('Epoch {:03d}, Loss: {:.4f}'.format(epoch, loss,))    
        if epoch % 1000 == 0:
            test_loss = val_epoch(plot=True)
        else:
            test_loss = val_epoch(plot=False)
        if epoch %100 == 0:
            print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
                epoch, np.sqrt(loss), np.sqrt(test_loss)))
        wandb.log({"RMSE_val": np.sqrt(test_loss), "RMSE_training": np.sqrt(loss)})        
    
        if epoch % SAVE_FREQUENCY == 0:
            # Save model to wandb
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))    
        scheduler.step()
        
if __name__ == "__main__":
    if MODE == "train":
        train()
    elif MODE == "val":
        val_loss = val_epoch(plot=True, val_mode=True)
        print('Val loss: {:.4f}'.format(np.sqrt(val_loss)))
    else:
        assert(False), ("Invalid MODE")