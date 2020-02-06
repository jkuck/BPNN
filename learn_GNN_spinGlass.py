import os.path as osp
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os 

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
##########################################################################################################
MSG_PASSING_ITERS = 5

N_MIN = 10
N_MAX = 10
F_MAX = .1
C_MAX = 5.0
# F_MAX = 1
# C_MAX = 10.0
TRAIN_DATA_SIZE = 50
ATTRACTIVE_FIELD_TRAIN = True

# N_MIN_VAL = 10
# N_MAX_VAL = 10
# F_MAX_VAL = .1
# C_MAX_VAL = 5
N_MIN_VAL = 8
N_MAX_VAL = 11
F_MAX_VAL = 5
C_MAX_VAL = 5
VAL_DATA_SIZE = 50
ATTRACTIVE_FIELD_VAL = False

train_data_list = [spinGlass_to_torchGeometric(SpinGlassModel(N=random.randint(N_MIN, N_MAX),\
                                                        f=np.random.uniform(low=0, high=F_MAX),\
                                                        c=np.random.uniform(low=0, high=C_MAX),\
                                                        attractive_field=ATTRACTIVE_FIELD_TRAIN)) for i in range(TRAIN_DATA_SIZE)]
train_loader = DataLoader(train_data_list, batch_size=50)


# DATA_DIR = "/atlas/u/jkuck/learn_BP/data/spin_glass/"
# REGENERATE_DATA = False
# N_MIN_BP = 10
# N_MAX_BP = 10
# F_MAX_BP = .1
# C_MAX_BP = 5.0
# DATA_SIZE = VAL_DATA_SIZE
# def get_dataset(dataset_type='test'):
#     assert(dataset_type in ['train', 'val', 'test'])
#     dataset_file = DATA_DIR + dataset_type + '%d_%d_%d_%.2f_%.2f.pkl' % (DATA_SIZE, N_MIN_BP, N_MAX_BP, F_MAX_BP, C_MAX_BP)
#     if REGENERATE_DATA or (not os.path.exists(dataset_file)):
#         assert(False), "test dataset missing!"
# #         print("REGENERATING DATA!!")
# #         sg_data = SpinGlassDataset(dataset_size=datasize, N_min=N_MIN, N_max=N_MAX, f_max=F_MAX, c_max=C_MAX)
# #         spin_glass_problems_SGMs = sg_data.generate_problems(return_sg_objects=True)
# #         if not os.path.exists(DATA_DIR):
# #             os.makedirs(DATA_DIR)
# #         with open(dataset_file, 'wb') as f:
# #             pickle.dump((sg_data, spin_glass_problems_SGMs), f)            
#     else:
#         with open(dataset_file, 'rb') as f:
#             (sg_data, spin_glass_problems_SGMs) = pickle.load(f)
#     return sg_data, spin_glass_problems_SGMs

# sg_data, spin_glass_problems_SGMs = get_dataset(dataset_type='test')
# val_data_list = [spinGlass_to_torchGeometric(sg_problem) for sg_problem in spin_glass_problems_SGMs]
# train_loader = DataLoader(val_data_list, batch_size=50)

USE_BPNN_DATA=False
######### FOR GETTING THE SAME TEST SET ##############
if USE_BPNN_DATA:
    DATA_DIR = "/atlas/u/jkuck/learn_BP/data/spin_glass/"
    REGENERATE_DATA = False
    N_MIN_BP = 10
    N_MAX_BP = 10
    F_MAX_BP = .1
    C_MAX_BP = 5.0
    DATA_SIZE = VAL_DATA_SIZE
    def get_dataset(dataset_type='test'):
        assert(dataset_type in ['train', 'val', 'test'])
        dataset_file = DATA_DIR + dataset_type + '%d_%d_%d_%.2f_%.2f.pkl' % (DATA_SIZE, N_MIN_BP, N_MAX_BP, F_MAX_BP, C_MAX_BP)
        if REGENERATE_DATA or (not os.path.exists(dataset_file)):
            assert(False), "test dataset missing!"
    #         print("REGENERATING DATA!!")
    #         sg_data = SpinGlassDataset(dataset_size=datasize, N_min=N_MIN, N_max=N_MAX, f_max=F_MAX, c_max=C_MAX)
    #         spin_glass_problems_SGMs = sg_data.generate_problems(return_sg_objects=True)
    #         if not os.path.exists(DATA_DIR):
    #             os.makedirs(DATA_DIR)
    #         with open(dataset_file, 'wb') as f:
    #             pickle.dump((sg_data, spin_glass_problems_SGMs), f)            
        else:
            with open(dataset_file, 'rb') as f:
                (sg_data, spin_glass_problems_SGMs) = pickle.load(f)
        return sg_data, spin_glass_problems_SGMs

    sg_data, spin_glass_problems_SGMs = get_dataset(dataset_type='test')
    val_data_list = [spinGlass_to_torchGeometric(sg_problem) for sg_problem in spin_glass_problems_SGMs]
    val_loader = DataLoader(val_data_list, batch_size=50)
else:
    val_data_list = [spinGlass_to_torchGeometric(SpinGlassModel(N=random.randint(N_MIN_VAL, N_MAX_VAL),\
                                                            f=np.random.uniform(low=0, high=F_MAX_VAL),\
                                                            c=np.random.uniform(low=0, high=C_MAX_VAL),\
                                                            attractive_field=ATTRACTIVE_FIELD_VAL)) for i in range(VAL_DATA_SIZE)]
    val_loader = DataLoader(val_data_list, batch_size=200)    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN_Network_withEdgeFeatures(msg_passing_iters=MSG_PASSING_ITERS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

loss_func = torch.nn.MSELoss(reduction='sum') #sum of squared errors

def train():
#     print("hi")
    model.train()
    total_loss = 0
    for data in train_loader:
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


def test(plot=False):
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


for epoch in range(1, 5001):
# for epoch in range(1, 501):
    loss = train()
#     print('Epoch {:03d}, Loss: {:.4f}'.format(epoch, loss,))    
    if epoch % 100 == 0:
        test_loss = test(plot=True)
    else:
        test_loss = test(plot=False)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, np.sqrt(loss), np.sqrt(test_loss)))
    scheduler.step()