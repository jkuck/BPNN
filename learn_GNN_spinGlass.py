import os.path as osp
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GINConv

from torch_geometric.data import Data, DataLoader
from ising_model.pytorch_geometric_data import spinGlass_to_torchGeometric
from ising_model.spin_glass_model import SpinGlassModel

from nn_models import GIN_Network_withEdgeFeatures

##########################################################################################################
N_MIN = 10
N_MAX = 10
F_MAX = .1
C_MAX = 5.0
# F_MAX = 1
# C_MAX = 10.0
TRAIN_DATA_SIZE = 50
VAL_DATA_SIZE = 50

train_data_list = [spinGlass_to_torchGeometric(SpinGlassModel(N=random.randint(N_MIN, N_MAX),\
                                                        f=np.random.uniform(low=0, high=F_MAX),\
                                                        c=np.random.uniform(low=0, high=C_MAX))) for i in range(TRAIN_DATA_SIZE)]
train_loader = DataLoader(train_data_list, batch_size=32)

val_data_list = [spinGlass_to_torchGeometric(SpinGlassModel(N=random.randint(N_MIN, N_MAX),\
                                                        f=np.random.uniform(low=0, high=F_MAX),\
                                                        c=np.random.uniform(low=0, high=C_MAX))) for i in range(VAL_DATA_SIZE)]
val_loader = DataLoader(val_data_list, batch_size=32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN_Network_withEdgeFeatures().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

loss_func = torch.nn.MSELoss(reduction='sum') #sum of squared errors

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        pred_ln_Z = model(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, batch=data.batch)
        loss = loss_func(pred_ln_Z, data.ln_Z)
        loss.backward()
        total_loss += loss.item() # * data.num_graphs
#         print("check loss shape, loss.shape:", loss.shape)
#         print("data.batch.size:", data.batch.size())
        optimizer.step()
    return total_loss / TRAIN_DATA_SIZE


# def test():
#     model.eval()
#     logits, accs = model(), []
#     for _, mask in data('train_mask', 'val_mask', 'test_mask'):
#         pred = logits[mask].max(1)[1]
#         acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
#         accs.append(acc)
#     return accs


for epoch in range(1, 201):
    loss = train()
    print('Epoch {:03d}, Loss: {:.4f}'.format(epoch, loss,))    
#     test_acc = test(test_loader)
#     print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(
#         epoch, loss, test_acc))
    scheduler.step()