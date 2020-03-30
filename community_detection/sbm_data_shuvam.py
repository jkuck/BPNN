import torch
import numpy as np
from itertools import combinations
import collections
from functools import reduce
import time
import sys
sys.path.append('/sailhome/shuvamc/learn_BP')
from factor_graph import FactorGraphData

class StochasticBlockModel:
    def __init__(self, N, P, Q, C, community_probs=None):
        '''
        Sample a stochastic block model
        
        Inputs:
        - N (int): number of nodes
        - P (float): the probability of an edge between vertices in the same community
        - Q (float): the probability of an edge between vertices in different communities
        - C (int): the number of communities
        - community_probs (torch tensor): shape # communities, give the probability that
			each node belongs to each community.  Set to uniform if None is given 
        '''
        self.N = N
        self.p = P
        self.q = Q
        self.C = C
        if community_probs is None:
            self.community_probs = torch.tensor([1.0/C for i in range(C)])
        labs = np.random.choice(np.arange(C), N, replace = True, p = self.community_probs.numpy())
        diff_inds = np.array(list(combinations(np.arange(N), 2)))
        lab_dct = collections.defaultdict(list)
        ind_dct = {}
        for i in range(N):
            lab_dct[labs[i]].append(i)
            ind_dct[i] = labs[i]
        diff_inds = np.array([x for x in diff_inds if ind_dct[x[0]] != ind_dct[x[1]]])
        same_lst = [np.array(list(combinations(lab_dct[curr], 2))) for curr in lab_dct]
        same_inds = reduce(lambda x,y: np.concatenate((x,y), axis = 0), same_lst)
        same_edges = same_inds[np.random.binomial(1, P, same_inds.shape[0]) == 1]
        diff_edges = diff_inds[np.random.binomial(1, Q, diff_inds.shape[0]) == 1]
        edges = np.sort(np.concatenate((same_edges, diff_edges), axis = 0), axis = 0)
        edges_full = np.concatenate((edges, np.flip(edges, axis = 1)), axis = 0)
        edge_index = np.flip(edges.T, axis = 0)
        edge_index_full = np.flip(edges_full.T, axis = 0)
        count = collections.Counter(edge_index_full[1])
        deg_lst = []
        for i in range(N):
            deg_lst.append([count[i]])

        self.edge_index = edge_index
        self.edge_index_full = edge_index_full
        self.deg_lst = np.array(deg_lst)
        self.gt_variable_labels = labs

def build_factorgraph_from_sbm(sbm_model):
    '''
    Convert a sbm model to a factor graph pytorch representation

    Inputs:
    - sbm_model (StochasticBlockModel): defines a stochastic block model

    Outputs:
    - factorgraph (FactorGraphData): 
    '''
    #assert(sbm_model.C == 2)
     
    #Unary Factors
    un_factors = np.repeat(np.repeat(sbm_model.community_probs.numpy()[np.newaxis, :], sbm_model.C, axis = 0)[np.newaxis, :, :], sbm_model.N, axis = 0)
    factor_mask_un = np.ones_like(un_factors)
    factor_mask_un[:,0,:] = 0
    #factor_mask_un = np.repeat(mask_un[np.newaxis, :, :], sbm_model.N, axis = 0)
    factorVarEdgeIndexUn = np.repeat(np.arange(sbm_model.N)[np.newaxis, :], 2, axis = 0) 
    
    #Binary Factors
    same_potential = np.log(sbm_model.p / (2*sbm_model.p + 2*sbm_model.q))
    diff_potential = np.log(sbm_model.q / (2*sbm_model.p + 2*sbm_model.q))
    non_diag = np.full((sbm_model.C, sbm_model.C), diff_potential) * (np.eye(sbm_model.C) * -1 + np.ones(sbm_model.C))
    diag = np.eye(sbm_model.C) * same_potential
    single_factor = diag + non_diag
    #single_factor = np.array([[same_potential, diff_potential], [diff_potential, same_potential]])
    bi_factors = np.repeat(single_factor[np.newaxis, :, :], sbm_model.edge_index.shape[1], axis = 0)
    factor_mask_bi = np.zeros_like(bi_factors)
    edgeToFacBi = np.repeat(np.arange(sbm_model.edge_index.shape[1]), 2) + sbm_model.N
    edgeToVarBi = np.ones(sbm_model.edge_index.shape[1] * 2)
    edgeToVarBi[0::2] = sbm_model.edge_index[1]
    edgeToVarBi[1::2] = sbm_model.edge_index[0]
    factorVarEdgeIndexBi = np.stack((edgeToFacBi, edgeToVarBi))

    #gather everything
    factor_potentials = np.concatenate((un_factors, bi_factors), axis = 0)
    factor_potential_masks = np.concatenate((factor_mask_un, factor_mask_bi), axis = 0)
    factorVarEdgeIndex = np.concatenate((factorVarEdgeIndexUn, factorVarEdgeIndexBi), axis = 1)
    unFacVarLst = np.arange(sbm_model.N)[:, np.newaxis].tolist()
    binFacVarLst = np.flip(sbm_model.edge_index, axis = 0).T.tolist()
    factorToVar = unFacVarLst + binFacVarLst
    edge_var_index_un = np.zeros_like(factorVarEdgeIndexUn[0])
    edge_var_index_bin = np.zeros_like(factorVarEdgeIndexBi[0])
    edge_var_index_bin[1::2] = 1
    edge_var_index = np.concatenate((edge_var_index_un, edge_var_index_bin))
    edge_var_indices = np.stack((edge_var_index, np.ones_like(edge_var_index) * -99))
    factor_graph = FactorGraphData(factor_potentials = torch.Tensor(factor_potentials),
                 factorToVar_edge_index = torch.Tensor(factorVarEdgeIndex), numVars = sbm_model.N, numFactors = factor_potentials.shape[0], 
                 edge_var_indices = torch.Tensor(edge_var_indices), state_dimensions = 2, factor_potential_masks = torch.Tensor(factor_potential_masks),
                 ln_Z=None, factorToVar_double_list = factorToVar, gt_variable_labels = torch.Tensor(sbm_model.gt_variable_labels).long())
    
    return factor_graph
sbm = StochasticBlockModel(100, .1, .01, 5)
fg = build_factorgraph_from_sbm(sbm)
