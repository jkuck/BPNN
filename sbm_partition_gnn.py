import numpy as np
import random
from itertools import combinations
from functools import reduce
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet, S3DIS
import torch_geometric.transforms as T
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.nn import GINConv, EdgeConv, DynamicEdgeConv, DynamicEdgeConv2, global_mean_pool, global_max_pool, DataParallel
from nn_models_sbm import GIN_Network_withEdgeFeatures
import time
import collections
import itertools
import wandb
from community_detection.sbm_data_shuvam import StochasticBlockModel
from community_detection.sbm_libdai import runLBPLibdai, runJT, build_libdaiFactorGraph_from_SBM
import os

USE_WANDB = True
N_TRAIN=20
A_TRAIN=19
B_TRAIN=1
C=2
N_VAL = 20
A_VAL = 19
B_VAL = 1
numLayers = 30
featsize = 8
NUM_TRAIN = 10
NUM_VAL = 4

exp_name = "ginconv_moreweights_sbm_partition"
if USE_WANDB:
    wandb.init(project="GNN_sbm_partition", name=exp_name)
    wandb.config.N_TRAIN = N_TRAIN
    wandb.config.A_TRAIN = A_TRAIN
    wandb.config.B_TRAIN = B_TRAIN
    wandb.config.C = C
    wandb.config.N_VAL = N_VAL
    wandb.config.A_VAL = A_VAL
    wandb.config.B_VAL = B_VAL
    wandb.config.numLayers = numLayers
    wandb.config.MPfeatsize = featsize
    wandb.config.NUM_TRAIN = NUM_TRAIN*N_TRAIN*C
    wandb.config.NUM_VAL = NUM_VAL*N_VAL*C
   


def createSBM(num, p, q, classes):
    prior = [.75, .25]
    labs = np.random.choice(np.arange(classes), num, replace = True, p = prior)
    diff_inds = np.array(list(combinations(np.arange(num), 2)))
    lab_dct = collections.defaultdict(list)
    ind_dct = {}
    for i in range(num):
        lab_dct[labs[i]].append(i)
        ind_dct[i] = labs[i]
    diff_inds = np.array([x for x in diff_inds if ind_dct[x[0]] != ind_dct[x[1]]])
    same_lst = [np.array(list(combinations(lab_dct[curr], 2))) for curr in lab_dct]
    same_inds = reduce(lambda x,y: np.concatenate((x,y), axis = 0), same_lst)
    same_edges = same_inds[np.random.binomial(1, p, same_inds.shape[0]) == 1]
    diff_edges = diff_inds[np.random.binomial(1, q, diff_inds.shape[0]) == 1]
    edges = np.concatenate((same_edges, diff_edges), axis = 0)
    edges = edges[np.argsort(edges[:,0])]
    edges = np.concatenate((edges, np.flip(edges, axis = 1)), axis = 0)
    edge_index = np.flip(edges.T, axis = 0)
    count = collections.Counter(edge_index[1])
    deg_lst = []
    for i in range(num):
        deg_lst.append([count[i]])
    return labs, edge_index, np.array(deg_lst)

def getData(num, p, q, classes, num_examples):
    dataset = []
    for i in range(num_examples):
        labs, edges, data_feat = createSBM(num, p, q, classes)
        edge_mask = torch.ones(edges.shape[1]).unsqueeze(1)
        d = Data(x = torch.Tensor(data_feat), y = torch.Tensor(labs).long(), edge_index = torch.Tensor(edges.copy()).long(), edge_mask = edge_mask)
        dataset.append(d)
    return dataset

def getDataLibdai(num, p, q, classes, num_examples):
    dataset = []
    num_corr = 0
    for i in range(num_examples):
        sbm_model = StochasticBlockModel(num, p, q, classes)
        edge_mask = torch.ones(sbm_model.edge_index_full.shape[1]).unsqueeze(1)
        d = Data(x = torch.Tensor(sbm_model.deg_lst), y = torch.Tensor(sbm_model.gt_variable_labels).long(), edge_index = torch.Tensor(sbm_model.edge_index_full.copy()).long(), edge_mask = edge_mask)
        dataset.append(d)
        var_beliefs = runLBPLibdai(sbm_model)
        corr, _ = getPermInvariantAcc(var_beliefs.max(dim = 1)[1], d.y, classes, device = torch.device('cpu'))
        num_corr += corr
    acc = num_corr / (num * num_examples)
    overlap = (acc - 1 / classes)/(1 - 1 / classes)
    return dataset, overlap
  
def getDataPartition(num, p, q, classes, num_examples, prior_prob):
    normal_edge = np.array([p, q, q, p])
    normal_noedge = np.array([1-p, 1-q, 1-q, 1-p])
    normal_mask = np.ones(4)
    var_1_mask = normal_mask
    var_2_mask = normal_mask
    dataset = []
    init_mse = 0
    for i in range(num_examples):
        sbm_model = StochasticBlockModel(num, p, q, classes, community_probs = prior_prob)
        edge_index = sbm_model.edge_index_full.copy()
        noedge_index = np.concatenate((sbm_model.noedge_index, np.flip(sbm_model.noedge_index, axis = 0)), axis = 1)
        edge_attr = np.repeat(normal_edge[np.newaxis, :], edge_index.shape[1], axis = 0)
        noedge_attr = np.repeat(normal_noedge[np.newaxis, :], noedge_index.shape[1], axis = 0)
        for j in range(num):
            jt_Z_lst = []
            bp_Z_lst = []
            for k in range(classes):
                if k == 0:
                    var_1_mask[:2] = 0
                    var_2_mask[0::2] = 0
                else:
                    var_1_mask[-2:] = 0
                    var_2_mask[1::3] = 0
                edge_mask = np.array([var_1_mask if edge_index[1][m] == j else var_2_mask if edge_index[0][m] == j else normal_mask for m in range(edge_index.shape[1])])
                noedge_mask = np.array([var_1_mask if noedge_index[1][m] == j else var_2_mask if noedge_index[0][m] == j else normal_mask for m in range(noedge_index.shape[1])])
                edge_attr = edge_attr * edge_mask
                noedge_attr = noedge_attr * noedge_mask
                tot_edge_index = np.concatenate((edge_index, noedge_index), axis = 1)
                tot_edge_attr = np.concatenate((edge_attr, noedge_attr), axis = 0)
                libdai_fg = build_libdaiFactorGraph_from_SBM(sbm_model, fixed_var = j, fixed_val = k)
                jt_Z, _ = runJT(libdai_fg, sbm_model)
                _, bp_Z = runLBPLibdai(libdai_fg, sbm_model)
                print(jt_Z, bp_Z)
                jt_Z_lst.append(jt_Z)
                bp_Z_lst.append(bp_Z)
                d = Data(x = torch.Tensor(sbm_model.deg_lst), y = torch.tensor([jt_Z]).float(), edge_index = torch.Tensor(tot_edge_index).long(), edge_attr = torch.Tensor(tot_edge_attr))
                dataset.append(d)
            jt_Z_lst = np.array(jt_Z_lst)
            bp_Z_lst = np.array(bp_Z_lst)
            mse = min(np.sum((jt_Z_lst-bp_Z_lst)**2), np.sum((jt_Z_lst - bp_Z_lst[::-1])**2))
            print(mse)
            init_mse += mse
    return dataset, init_mse / len(dataset)
 
       
'''
def getTrainTest(num, classes, k, d, num_examples, batch_size, test_ind):
    q = np.random.uniform(d-d**.5)
    p = k*d - q
    dataset = getData(num, p/num, q/num, classes, num_examples)
    train_loader = DataListLoader(dataset[test_ind:], batch_size = batch_size, shuffle = True)
    test_loader = DataListLoader(dataset[:test_ind], batch_size = batch_size, shuffle = False)
'''
def getPermInvariantLoss(out, y, n, crit, device):
    perms = list(itertools.permutations(np.arange(n)))
    loss = crit(out, y)
    for p in perms[1:]:
        dct = {i: j for i,j in enumerate(p)}
        perm_lab = list(map(dct.get, y.cpu().numpy()))
        curr_loss = crit(out, torch.Tensor(perm_lab).long().to(device))
        loss = torch.min(loss, curr_loss)
    return loss, crit(out, y)


def getPermInvariantAcc(out, y, n, device):
    perms = list(itertools.permutations(np.arange(n)))
    acc = out.eq(y).sum()
    for p in perms[1:]:
        dct = {i: j for i,j in enumerate(p)}
        perm_lab = list(map(dct.get, y.cpu().numpy()))
        curr_acc = out.eq(torch.Tensor(perm_lab).long().to(device)).sum()
        acc = torch.max(acc, curr_acc)
    return acc.item(), out.eq(y).sum().item()

class DGCNNGeom(nn.Module):
    def __init__(self, num_layers, output_channels, num_features, in_features, aggr = 'max'):
        super(DGCNNGeom, self).__init__()
        '''
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dp1 = Dropout(p=.5)
        self.dp2 = Dropout(p=.5)

        self.conv1 = Seq(Lin(2, 64), self.bn1, nn.LeakyReLU(.2))
        self.conv2 = Seq(Lin(64*2, 64), self.bn2, nn.LeakyReLU(.2))
        self.conv3 = Seq(Lin(64*2, 128), self.bn3, nn.LeakyReLU(.2))
        self.conv4 = Seq(Lin(128*2, 256), self.bn4, nn.LeakyReLU(.2))

        self.linear1 = Seq(Lin(512, 1024), self.bn5, nn.LeakyReLU(.2))
        self.linear2 = Seq(Lin(1024, 512), self.bn6, nn.LeakyReLU(.2), self.dp1)
        self.linear3 = Seq(Lin(512, 256), self.bn7, nn.LeakyReLU(.2), self.dp2)
        self.linear4 = Lin(256, output_channels)

        self.deconv1 = DynamicEdgeConv2(self.conv1, aggr = aggr)
        self.deconv2 = DynamicEdgeConv2(self.conv2, aggr = aggr)
        self.deconv3 = DynamicEdgeConv2(self.conv3, aggr  = aggr)
        self.deconv4 = DynamicEdgeConv2(self.conv4, aggr = aggr)
        '''
        self.num_layers = num_layers
        self.num_classes = output_channels
        self.init_conv = DynamicEdgeConv2(Seq(Lin(in_features*2, num_features), nn.BatchNorm1d(num_features), nn.LeakyReLU(.2)), aggr = aggr)
        for i in range(num_layers-1):
            conv = Seq(Lin(num_features*2, num_features), nn.BatchNorm1d(num_features), nn.LeakyReLU(.2))
            #module = GINConv(conv)      
            module = DynamicEdgeConv2(conv, aggr = aggr)  
            self.add_module('layer{}'.format(i+1), module)
        self.linear1 = Seq(Lin(num_features*num_layers, 128), nn.BatchNorm1d(128), nn.LeakyReLU(.2))
        self.linear2 = Seq(Lin(128, 64), nn.BatchNorm1d(64), nn.LeakyReLU(.2))
        #self.linear3 = Seq(Lin(64, 32), nn.BatchNorm1d(32), nn.LeakyReLU(.2))
        self.linear3 = Lin(64, output_channels)
        self.outputLayer = Seq(Lin(num_features, 16), nn.BatchNorm1d(16), nn.LeakyReLU(.2), Lin(16, output_channels))
    ''' 
    def forward2(self, data):
        #batch_size = data.num_graphs
        #pos = data.pos
        x0 = data.x
        #x0 = torch.cat((pos, x), dim = 1)
        edge_index = data.edge_index
        #edge_mask = data.edge_mask
        #print(edge_mask, edge_index, edge_index.shape, edge_mask.shape)
        #time.sleep(10000)
        x1, prev = self.deconv1(x0, edge_index)
        x2, prev = self.deconv2(x1, edge_index)
        x3, prev = self.deconv3(x2, edge_index)
        x4, prev = self.deconv4(x3, edge_index)
        x = torch.cat((x1, x2, x3, x4), dim = 1)
        x = self.linear1(x)
        x = self.linear4(self.linear3(self.linear2(x)))
        out = F.log_softmax(x, dim=1)
        return out
    '''
    def forward(self, data):
        x0 = data.x
        edge_index = data.edge_index
        edge_mask = data.edge_mask
        x, _ = self.init_conv(x0, edge_index)
        tot = x
        for i in range(self.num_layers - 1):
            x, _ = self._modules['layer{}'.format(i+1)](x, edge_index)
            tot = torch.cat((tot, x), dim = 1)
        #out = self.linear3(self.linear2(self.linear1(tot)))
        #print(out)
        #out = F.log_softmax(out, dim = 1)
        out = self.outputLayer(x)
        return out

def getListLoadery(data):
   y = torch.flatten(torch.Tensor([elem.y.numpy() for elem in data]).long())
   return y.to(device)

def debug(data, out, loss):
    if loss.item() < .1:
        pred = out.max(dim = 1)[1]
        acc, _ = getPermInvariantAcc(pred, getListLoadery(data), C)
        print(loss.item(), acc)
        time.sleep(2)
        
def get_train_acc(data):
    #model.eval()
    tot_acc = 0
    for d in data:
        with torch.no_grad():
            out = model(d)
        acc, _ = getPermInvariantAcc(out.max(dim=1)[1], getListLoadery(d), C)
        tot_acc += acc
        #debug(d, out, loss)
    print("train_acc: " + str(tot_acc / (N_TRAIN*len(data.dataset))))
    return tot_acc / (N_TRAIN*len(data.dataset))

def train(train_loader, model, optimizer, device, loss_func):
    model.train()

    total_loss = 0
    for data in train_loader:
        data.to(device)
        edge_index = data.edge_index
        x = data.x
        true_partition = data.y
        edge_attr = data.edge_attr
        batch = data.batch
        optimizer.zero_grad()
        out = model(x, edge_index, edge_attr, batch)
        loss = loss_func(out, true_partition)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    return total_loss/len(train_loader)


def test(loader, model, device, loss_func):
    #model.eval()
    tot_loss = 0
    for data in loader:
        data.to(device)
        edge_index = data.edge_index
        x = data.x
        edge_attr = data.edge_attr
        true_partition = data.y
        batch = data.batch
        with torch.no_grad():
            out = model(x, edge_index, edge_attr, batch)
            loss = loss_func(out, true_partition)
        tot_loss += loss.item()
    return tot_loss / len(loader)

if __name__ == '__main__':

    random.seed(8)
    np.random.seed(12)
    model = GIN_Network_withEdgeFeatures(input_state_size=1, edge_attr_size=2**C, hidden_size=featsize, msg_passing_iters=numLayers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    #model = DGCNNGeom(numLayers, C, featsize, 1, aggr = 'add').to(device)
    #model = DataParallel(model)
    #optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr = 2e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-3)
    loss_func = torch.nn.MSELoss()   
    train_dataset, libdai_train_mse = getDataPartition(N_TRAIN, A_TRAIN/N_TRAIN, B_TRAIN/N_TRAIN, C, NUM_TRAIN, prior_prob = [.75, .25])
    test_dataset, libdai_test_mse = getDataPartition(N_VAL, A_VAL/N_VAL, B_VAL/N_VAL, C, NUM_VAL, prior_prob = [.75, .25])
        
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)
    print(train_loader)
    init_train_loss = test(train_loader, model, device, loss_func)
    init_test_loss = test(test_loader, model, device, loss_func)
    print("Initial Libdai Train MSE: " + str(libdai_train_mse) + "Initial Libdai Test MSE: " + str(libdai_test_mse))
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(0, init_train_loss, init_test_loss)) 
    if USE_WANDB:
        wandb.log({"Initial Libdai Train MSE": libdai_train_mse, "Initial Libdai Test MSE": libdai_test_mse, "Train Loss": init_train_loss, "Test Loss": init_test_loss})

    save_path = os.path.join('/atlas/u/shuvamc/model_weights', exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for epoch in range(1, 301):
        train_loss = train(train_loader, model, optimizer, device, loss_func)
        test_loss = test(test_loader, model, device, loss_func)
        print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(epoch, train_loss, test_loss))
        if USE_WANDB:
            wandb.log({"Train Loss": train_loss, "Test Loss": test_loss}) 
        torch.save(model.state_dict(), os.path.join(save_path, 'epoch_' + str(epoch) + '.pt'))


