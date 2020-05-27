import numpy as np
from itertools import combinations
from functools import reduce
from torch_geometric.data import Data
from torch_geometric.data import DataListLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential as Seq, Dropout, Linear as Lin
from torch_geometric.datasets import ModelNet, S3DIS
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import GINConv, EdgeConv, DynamicEdgeConv, DynamicEdgeConv2, global_mean_pool, global_max_pool, DataParallel
from nn_models_sbm import 
import time
import collections
import itertools
import wandb
from community_detection.sbm_data_shuvam import StochasticBlockModel
from community_detection.sbm_libdai import runLBPLibdai, runJT, build_libdaiFactorGraph_from_SBM


USE_WANDB = True
N_TRAIN=20
A_TRAIN=19
B_TRAIN=1
C=2
N_VAL = 100
A_VAL = 19
B_VAL = 1
numLayers = 20
featsize = 8
NUM_TRAIN = 10
NUM_VAL = 5

if USE_WANDB:
    wandb.init(project="GNN_sbm", name="DGCNN_SBM_2_class_small_fc")
    wandb.config.N_TRAIN = N_TRAIN
    wandb.config.A_TRAIN = A_TRAIN
    wandb.config.B_TRAIN = B_TRAIN
    wandb.config.C = C
    wandb.config.N_VAL = N_VAL
    wandb.config.A_VAL = A_VAL
    wandb.config.B_VAL = B_VAL
    wandb.config.numLayers = numLayers
    wandb.config.MPfeatsize = featsize
    wandb.config.NUM_TRAIN = NUM_TRAIN
    wandb.config.NUM_VAL = NUM_VAL
   

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
  
def getDataPartition(num, p, q, classes, num_examples, prior_prob)
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
        edge_attr = np.repeat(normal_edge[np.newaxis, :], edge_index.shape[1])
        noedge_attr = np.repeat(normal_noedge[np.newaxis, :], noedge_index.shape[1])
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
                edge_mask = np.array([var_1_mask if edge_index[m][1] == j else var_2_mask if edge_index[m][0] == j else normal_mask for m in range(edge_index.shape[1])])
                noedge_mask = np.array([var_1_mask if noedge_index[m][1] == j else var_2_mask if noedge_index[m][0] == j else normal_mask for m in range(noedge_index.shape[1])])
                edge_attr = edge_attr * edge_mask
                noedge_attr = noedge_attr * noedge_mask
                tot_edge_mask = np.concatenate((edge_mask, noedge_mask), axis = 1)
                tot_edge_attr = np.concatenate((edge_attr, noedge_attr), axis = 1)
                libdai_fg = build_libdaiFactorGraph_from_SBM(sbm_model, fixed_var = j, fixed_val = k)
                jt_Z, _ = runJT(libdai_fg, sbm_model)
                bp_Z, _ = runLBPLibday(libdai_fg, sbm_model)
                print(jt_Z, bp_Z)
                jt_Z_lst.append(jt_Z)
                bp_Z_lst.append(bp_Z)
                d = Data(x = torch.Tensor(sbm_model.deg_lst), y = torch.Tensor(jt_Z), edge_index = torch.Tensor(tot_edge_index).long(), edge_attr = torch.Tensor(tot_edge_attr))
                dataset.append(d)
            jt_Z = np.array(jt_Z)
            bp_Z = np.array(bp_Z)
            mse = min(np.sum((jt_Z-bp_Z)**2), np.sum((jt_Z - bp_Z[::-1])**2))
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
def getPermInvariantLoss(out, y, n, crit):
    perms = list(itertools.permutations(np.arange(n)))
    loss = crit(out, y)
    for p in perms[1:]:
        dct = {i: j for i,j in enumerate(p)}
        perm_lab = list(map(dct.get, y.cpu().numpy()))
        curr_loss = crit(out, torch.Tensor(perm_lab).long().to(device))
        loss = torch.min(loss, curr_loss)
    return loss, crit(out, y)


def getPermInvariantAcc(out, y, n, device = device):
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

def train():
    model.train()

    total_loss = 0
    norm_loss = 0
    for data in train_loader:
        #data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss,norm = getPermInvariantLoss(out, getListLoadery(data), C, nn.CrossEntropyLoss())
        norm_loss += norm.item()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
        #debug(data, out, loss)
    train_acc = get_train_acc(train_loader)
    return total_loss/len(train_loader.dataset), (train_acc-(1/C))/(1-1/C)


def test(loader):
    #model.eval()
    correct = 0
    tot_loss = 0
    for data in loader:
        #data.to(device)
        with torch.no_grad():
            out = model(data)
        acc, norm_curr = getPermInvariantAcc(out.max(dim = 1)[1], getListLoadery(data), C)
        correct += acc
        loss, _ = getPermInvariantLoss(out, getListLoadery(data), C, nn.CrossEntropyLoss())
        tot_loss += loss.item()
    acc = correct / (N_VAL*len(loader.dataset))
    return (acc - (1/C))/(1-1/C), tot_loss / len(loader.dataset)


model = DGCNNGeom(numLayers, C, featsize, 1, aggr = 'add').to(device)
model = DataParallel(model)
#optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
optimizer = torch.optim.Adam(model.parameters(), lr = 2e-3)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=1e-3)

train_dataset = getData(N_TRAIN, A_TRAIN/N_TRAIN, B_TRAIN/N_TRAIN, C, NUM_TRAIN)
test_dataset, init_overlap = getDataLibdai(N_VAL, A_VAL/N_VAL, B_VAL/N_VAL, C, NUM_VAL)
print("Initial BP Overlap: " + str(init_overlap))
if USE_WANDB:
    wandb.log({"Initial BP Overlap": init_overlap})
train_loader = DataListLoader(train_dataset, batch_size = 1, shuffle = True)
test_loader = DataListLoader(test_dataset, batch_size = 1, shuffle = False)

for epoch in range(1, 301):
    #test_acc, norm = test(test_loader)
    #print(test_acc, norm)
    loss, acc = train()
    test_acc, test_loss = test(test_loader)
    #print(norm)
    print('Epoch {:03d}, Loss: {:.4f}, Test: {:.4f}'.format(epoch, loss, test_acc))
    if USE_WANDB:
        wandb.log({"train_loss": loss, "train_overlap": acc, "test_loss": test_loss, "test_overlap": test_acc}) 
    #torch.save(model.state_dict(), '/sailhome/shuvamc/model_weights/baseline_classif_2/epoch_'+str(epoch)+'.pt')
    #scheduler.step()
