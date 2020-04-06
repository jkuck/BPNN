import re
import torch
import numpy as np
from collections import defaultdict
from utils import dotdict, neg_inf_to_zero
# from torch_geometric.data import Data, Batch
from data_custom import DataFactorGraph_partial
# from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch_geometric
from parameters import LN_ZERO


def create_scatter_indices_helper(expansion_index, variable_cardinality, state_dimensions, offset):
    '''
    Inputs:
    - expansion_index (int): 
    - 
    - 
    - offset (int): add this value to every index
    '''
    l = torch.tensor([i + offset for i in range(variable_cardinality**state_dimensions)])
    l_shape = [variable_cardinality for i in range(state_dimensions)]
    l = l.reshape(l_shape)
    l = l.transpose(expansion_index, 0)
    return l.flatten()    
#     t = torch.tensor(l)
#     assert((t == l).all())
#     t = t.transpose(expansion_index, 0)
#     return t.flatten()

def create_scatter_indices_varToFactorMsgs(original_indices, variable_cardinality=2, state_dimensions=2):
    #When sending variable to factor messages, variable beliefs must be expanded to have extra redundant dimensions.
    #The approach is to expand all variable beliefs in all outgoing messages, then transpose each belief appropriately 
    #so that the variable it represents lines up in the correct dimension of the factor each message is sent to.
    #This function creates indices, to be used with torch_scatter, to perform this functionality.
    assert((original_indices < state_dimensions).all())
    scatter_indices_list = []
    for position, index in enumerate(original_indices):
        cur_offset = position*(variable_cardinality**state_dimensions)
        cur_indices = create_scatter_indices_helper(expansion_index=index, variable_cardinality=variable_cardinality, 
                          state_dimensions=state_dimensions, offset=cur_offset)
        scatter_indices_list.append(cur_indices)
    scatter_indices = torch.cat(scatter_indices_list)
    return scatter_indices


#this class requires input arguments in the constructor.  This doesn't work nicely
#with the batching implementation, which is why DataFactorGraph_partial was implemented
#A cleaner approach probably exists
class FactorGraphData(DataFactorGraph_partial):
    '''
    Representation of a factor graph in pytorch geometric Data format
    '''
    def __init__(self, factor_potentials, factorToVar_edge_index, numVars, numFactors, 
                 edge_var_indices, state_dimensions, factor_potential_masks, ln_Z=None,
                 factorToVar_double_list=None, gt_variable_labels=None):
        '''
        Inputs:
        - factor_potentials (torch tensor): represents all factor potentials (log base e space) and has 
            shape (num_factors, var_states, ..., var_states), where var_states is the number 
            of states a variable can take (e.g. 2 for an ising model or 3 for community detection
            with 3 communities).  Has state_dimensions + 1 total dimensions (where state_dimensions
            is the number of dimenions in the largest factor), e.g. 3 total 
            dimensions when the largest factor contains 2 variables.  Factors with fewer than 
            state_dimensions dimensions have unused states set to -infinitiy (0 probability)
        - factorToVar_edge_index (torch tensor): shape (2, num_edges), represents all edges in the factor graph.
            factorToVar_edge_index[0, i] is the index of the factor connected by the ith edge
            factorToVar_edge_index[1, i] is the index of the variable connected by the ith edge
        - numVars (int): number of variables in the factor graph
        - numFactors (int): number of factors in the factor graph
        - edge_var_indices (torch tensor): shape (2, num_edges)
            [0, i] indicates the index (0 to factor_degree - 1) of edge i, among all edges originating at the factor which edge i begins at
            [1, i] IS CURRENTLY UNUSED AND BROKEN, BUT IN GENERAL FOR PYTORCH GEOMETRIC SHOULD indicate the index (0 to var_degree - 1) of edge i, among all edges ending at the variable which edge i ends at
        - state_dimensions (int): the number of variables in the largest factor
        - factor_potential_masks (torch tensor): same shape as factor_potentials.  All entries are 0 or 1.
            0: represents that the corresponding location in factor_potentials is valid
            1: represents that the corresponding location in factor_potentials is invalid and should be masked,
               e.g. the factor has fewer than state_dimensions dimensions
        - ln_Z : natural logarithm of the partition function
        - factorToVar_double_list (list of lists): 
            factorToVar_double_list[i] is a list of all variables that factor with index i shares an edge with
            factorToVar_double_list[i][j] is the index of the jth variable that the factor with index i shares an edge with
        - gt_variable_labels (torch tensor): shape (# variables).  ground truth labels for each variable in the stochastic block model
        '''
        super(FactorGraphData, self).__init__()
        if gt_variable_labels is not None:
            self.gt_variable_labels = gt_variable_labels
        if ln_Z is not None:
#             print("check ln_Z:", ln_Z)
            self.ln_Z = torch.tensor([ln_Z], dtype=float)
        
        # (int) the largest node degree
        self.state_dimensions = torch.tensor([state_dimensions])
        
        #potentials defining the factor graph
        self.factor_potentials = factor_potentials 

        # - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge
        #     matrix with shape :obj:`[numFactors, numVars]`
        #     stored as a [2, E] tensor of [factor_idx, var_idx] for each edge factor to variable edge
        if factorToVar_double_list is not None:
            #factorStates_to_varIndices (torch LongTensor): essentially representing edges and pseudo edges (between
            #    junk states and a junk bin)
            #    has shape [num_factors*(2^state_dimensions] with values 
            #    in {0, 1, ..., (number of factor to variable edges)-1, (number of factor to variable edges)}. 
            #    Note (number of factor to variable edges) indicates a 'junk state' and should be output into a 
            #    'junk' bin after scatter operation.            
            self.facStates_to_varIdx, self.facToVar_edge_idx = self.create_factorStates_to_varIndices(factorToVar_double_list)
#             self.facStates_to_varIdx_FIXED, self.facToVar_edge_idx_FIXED = self.create_factorStates_to_varIndices_FIXED(factorToVar_double_list)
        else:
            self.facToVar_edge_idx = factorToVar_edge_index

#         print("facStates_to_varIdx.shape:", self.facStates_to_varIdx.shape)
#         print("facToVar_edge_idx.shape:", self.facToVar_edge_idx.shape)
#         sleep(temp)
        self.edge_index = self.facToVar_edge_idx #hack for batching, see learn_BP_spinGlass.py
        # print("factorToVar_edge_index.shape:", factorToVar_edge_index.shape)
        # print("factorToVar_edge_index.shape:", factorToVar_edge_index.shape)



        # self.factor_degrees[i] stores the number of variables that appear in factor i
        unique_factor_indices, self.factor_degrees = torch.unique(factorToVar_edge_index[0,:], sorted=True, return_counts=True)
        assert((self.factor_degrees >= 1).all())
        assert(unique_factor_indices.shape[0] == numFactors)

        # self.var_degrees[i] stores the number of factors that variables i appears in
        unique_var_indices, self.var_degrees = torch.unique(factorToVar_edge_index[1,:], sorted=True, return_counts=True)
        assert((self.var_degrees >= 1).all())
        assert(unique_var_indices.shape[0] == numVars)

        #when batching, numVars and numFactors record the number of variables and factors for each graph in the batch
        self.numVars = torch.tensor([numVars])
        self.numFactors = torch.tensor([numFactors])
        #when batching, see num_vars and num_factors to access the cumulative number of variables and factors in all
        #graphs in the batch, like num_nodes
        
        
        # edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] IS CURRENTLY UNUSED AND BROKEN, BUT IN GENERAL FOR PYTORCH GEOMETRIC SHOULD indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at
        self.edge_var_indices = edge_var_indices
        assert(self.facToVar_edge_idx.shape == self.edge_var_indices.shape)
        
        self.varToFactorMsg_scatter_indices = create_scatter_indices_varToFactorMsgs(original_indices=self.edge_var_indices[0, :], variable_cardinality=2, state_dimensions=state_dimensions)

        #1 signifies an invalid location (e.g. a dummy dimension in a factor), 0 signifies a valid location 
        self.factor_potential_masks = factor_potential_masks

        
        prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs = self.get_initial_beliefs_and_messages()
        self.prv_varToFactor_messages = prv_varToFactor_messages
        self.prv_factorToVar_messages = prv_factorToVar_messages
        self.prv_factor_beliefs = prv_factor_beliefs
        self.prv_var_beliefs = prv_var_beliefs
        
        assert(self.prv_factor_beliefs.size(0) == self.numFactors)
        assert(self.prv_var_beliefs.size(0) == self.numVars)        
#         print("added prv_varToFactor_messages!!1234")
#         print("prv_varToFactor_messages:", prv_varToFactor_messages)
#         sleep(temp23)
#         for attr, value in self.__dict__.items():
#             if value is not None:
#                 print(attr, type(value))
#             else:
#                 print(attr, value)
#         sleep(check_types)


    def __inc__(self, key, value):
#         if bool(re.search('(index|face)', key)):
#             print('re evaluated True, key:', key)#, 'value:', value)
#         else:
#             print('re evaluated False, key:', key)#, 'value:', value)            
#         return self.num_nodes if bool(re.search('(index|face)', key)) else 0
        if key == 'facStates_to_varIdx':
            return torch.tensor([2*self.edge_var_indices.size(1)]) #2*(number of edges)
        elif key == 'varToFactorMsg_scatter_indices':
            return torch.tensor([self.varToFactorMsg_scatter_indices.size(0)])
        elif key == 'edge_index' or key == 'facToVar_edge_idx':
#             return torch.tensor([self.numFactors, self.numVars])
#             print('hi1, key:', key)
            return torch.tensor([self.prv_factor_beliefs.size(0), self.prv_var_beliefs.size(0)]).unsqueeze(dim=1)
#         elif key == 'state_dimensions':
#             return torch.tensor([0]).unsqueeze(dim=1)
        elif key == 'edge_var_indices':
#             print("hi")
#             sleep(whew)
            return torch.tensor([0, 0]).unsqueeze(dim=1)

        else:
#             print('hi2, key:', key)
            return super(FactorGraphData, self).__inc__(key, value)

    def __cat_dim__(self, key, value):
        if key == 'facToVar_edge_idx' or key == 'edge_var_indices':
            return -1
        else:
            return super(FactorGraphData, self).__cat_dim__(key, value)


#     @property
#     def num_nodes(self):
#         r"""Returns or sets the number of nodes in the graph.
#         .. note::
#             The number of nodes in your data object is typically automatically
#             inferred, *e.g.*, when node features :obj:`x` are present.
#             In some cases however, a graph may only be given by its edge
#             indices :obj:`edge_index`.
#             PyTorch Geometric then *guesses* the number of nodes
#             according to :obj:`edge_index.max().item() + 1`, but in case there
#             exists isolated nodes, this number has not to be correct and can
#             therefore result in unexpected batch-wise behavior.
#             Thus, we recommend to set the number of nodes in your data object
#             explicitly via :obj:`data.num_nodes = ...`.
#             You will be given a warning that requests you to do so.
#         """
#         print("property called!")
#         if hasattr(self, '__num_nodes__'):
#             print("hi1 num_nodes property")
#             return self.__num_nodes__
#         for key, item in self('x', 'pos', 'norm', 'batch'):
#             print("hi2 num_nodes property")            
#             return item.size(self.__cat_dim__(key, item))
# #         if self.face is not None:
# #             warnings.warn(__num_nodes_warn_msg__.format('face'))
# #             return maybe_num_nodes(self.face)
# #         if self.edge_index is not None:
# #             warnings.warn(__num_nodes_warn_msg__.format('edge'))
# #             return maybe_num_nodes(self.edge_index)
#         return None

#     @num_nodes.setter
#     def num_nodes(self, num_nodes):
#         print("hi num_nodes setter called!")                    
#         self.__num_nodes__ = num_nodes        
        


    def create_factorStates_to_varIndices_OLD(self, factorToVar_double_list):
        '''
        Inputs:
        - factorToVar_double_list (list of lists): 
            factorToVar_double_list[i] is a list of all variables that factor with index i shares an edge with
            factorToVar_double_list[i][j] is the index of the jth variable that the factor with index i shares an edge with
            
        Output:
        - factorStates_to_varIndices (torch LongTensor): shape [num_factors*(2^state_dimensions] with values 
            in {0, 1, ..., 2*(number of factor to variable edges)-1, 2*(number of factor to variable edges)}. 
            Note 2*(number of factor to variable edges) indicates a 'junk state' and should be output into a 
            'junk' bin after scatter operation.
            Used with scatter_logsumexp (https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html)
            to marginalize the appropriate factor states for each message
        - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge matrix
            stored as a [2, E] tensor of [factor_idx, var_idx] for each edge factor to variable edge            
        '''
        # the number of (undirected) edges in the factor graph, or messages in one graph wide update
        numMsgs = 0
        for variables_list in factorToVar_double_list:
            numMsgs += len(variables_list)
        
        factorStates_to_varIndices_list = []
        factorToVar_edge_index_list = []
        
#         junk_bin = 2*numMsgs
        junk_bin = -1 #new junk bin for batching
        
        arange_tensor = torch.arange(2**self.state_dimensions.item())
        msg_idx = 0
        for factor_idx, variables_list in enumerate(factorToVar_double_list):
            unused_var_count = self.state_dimensions - len(variables_list)
            
            
            for varIdx_inFac, var_idx in enumerate(variables_list):
                factorToVar_edge_index_list.append(torch.tensor([factor_idx, var_idx]))
                msgIdx_varIs0 = msg_idx
                msgIdx_varIs1 = msg_idx + numMsgs
                msg_idx += 1
                curFact_to_varIndices = -99*torch.ones(2**self.state_dimensions, dtype=torch.long)
                multipler1 = 2**(self.state_dimensions - varIdx_inFac - 1)
                curFact_to_varIndices[((arange_tensor//multipler1) % 2) == 0] = msgIdx_varIs0
                curFact_to_varIndices[((arange_tensor//multipler1) % 2) == 1] = msgIdx_varIs1
                assert(not (curFact_to_varIndices == -99).any())
                #send unused factor states to the junk bin
                if unused_var_count > 0:
                    multiplier2 = 1
                    for unused_var_idx in range(unused_var_count):
                        curFact_to_varIndices[((arange_tensor//multiplier2) % 2) == 1] = junk_bin
                        multiplier2 *= 2
                factorStates_to_varIndices_list.append(curFact_to_varIndices)
        assert(msg_idx == numMsgs)
        assert(len(factorStates_to_varIndices_list) == numMsgs)
        factorStates_to_varIndices = torch.cat(factorStates_to_varIndices_list)
        factorToVar_edge_index = torch.stack(factorToVar_edge_index_list).permute(1,0)
        return factorStates_to_varIndices, factorToVar_edge_index

   


    def create_factorStates_to_varIndices(self, factorToVar_double_list):
        '''
        Inputs:
        - factorToVar_double_list (list of lists): 
            factorToVar_double_list[i] is a list of all variables that factor with index i shares an edge with
            factorToVar_double_list[i][j] is the index of the jth variable that the factor with index i shares an edge with
            
        Output:
        - factorStates_to_varIndices (torch LongTensor): shape [num_factors*(2^state_dimensions] with values 
            in {0, 1, ..., 2*(number of factor to variable edges)-1, 2*(number of factor to variable edges)}. 
            Note 2*(number of factor to variable edges) indicates a 'junk state' and should be output into a 
            'junk' bin after scatter operation.
            Used with scatter_logsumexp (https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html)
            to marginalize the appropriate factor states for each message
        - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge matrix
            stored as a [2, E] tensor of [factor_idx, var_idx] for each edge factor to variable edge            
        '''
        # the number of (undirected) edges in the factor graph, or messages in one graph wide update
        numMsgs = 0
        for variables_list in factorToVar_double_list:
            numMsgs += len(variables_list)
        
        factorStates_to_varIndices_list = []
        factorToVar_edge_index_list = []
        
#         junk_bin = 2*numMsgs
        junk_bin = -1 #new junk bin for batching
        
        arange_tensor = torch.arange(2**self.state_dimensions.item())        
        msg_idx = 0
        for factor_idx, variables_list in enumerate(factorToVar_double_list):
            unused_var_count = self.state_dimensions - len(variables_list)
            
            
            for varIdx_inFac, var_idx in enumerate(variables_list):
                factorToVar_edge_index_list.append(torch.tensor([factor_idx, var_idx]))
                msgIdx_varIs0 = msg_idx
                msgIdx_varIs1 = msg_idx + 1
                msg_idx += 2
                curFact_to_varIndices = -99*torch.ones(2**self.state_dimensions, dtype=torch.long)
                multipler1 = 2**(self.state_dimensions - varIdx_inFac - 1)
                curFact_to_varIndices[((arange_tensor//multipler1) % 2) == 0] = msgIdx_varIs0
                curFact_to_varIndices[((arange_tensor//multipler1) % 2) == 1] = msgIdx_varIs1
                assert(not (curFact_to_varIndices == -99).any())
                #send unused factor states to the junk bin
                if unused_var_count > 0:
                    multiplier2 = 1
                    for unused_var_idx in range(unused_var_count):
                        curFact_to_varIndices[((arange_tensor//multiplier2) % 2) == 1] = junk_bin
                        multiplier2 *= 2
                factorStates_to_varIndices_list.append(curFact_to_varIndices)
        assert(msg_idx == 2*numMsgs), (msg_idx, numMsgs)
        assert(len(factorStates_to_varIndices_list) == numMsgs)
        factorStates_to_varIndices = torch.cat(factorStates_to_varIndices_list)
        factorToVar_edge_index = torch.stack(factorToVar_edge_index_list).permute(1,0)
        return factorStates_to_varIndices, factorToVar_edge_index

        
        
    def get_initial_beliefs_and_messages(self, initialize_randomly=False, device=None):
        edge_count = self.edge_var_indices.shape[1]

        prv_varToFactor_messages = torch.log(torch.stack([torch.ones([2], dtype=torch.float) for j in range(edge_count)], dim=0))
        prv_factorToVar_messages = torch.log(torch.stack([torch.ones([2], dtype=torch.float) for j in range(edge_count)], dim=0))
        prv_factor_beliefs = torch.log(torch.stack([torch.ones([2 for i in range(self.state_dimensions)], dtype=torch.float) for j in range(self.numFactors)], dim=0))
        # prv_factor_beliefs = torch.log(factor_potentials.clone())
        # prv_factor_beliefs = prv_factor_beliefs/torch.logsumexp(prv_factor_beliefs, [i for i in range(1, len(prv_factor_beliefs.size()))])
        prv_var_beliefs = torch.log(torch.stack([torch.ones([2], dtype=torch.float) for j in range(self.numVars)], dim=0))
        if initialize_randomly:
            prv_varToFactor_messages = torch.rand_like(prv_varToFactor_messages)
            prv_factorToVar_messages = torch.rand_like(prv_factorToVar_messages)
            # prv_factor_beliefs = torch.rand_like(prv_factor_beliefs)
            prv_var_beliefs = torch.rand_like(prv_var_beliefs)
        if device is not None:
            prv_varToFactor_messages = prv_varToFactor_messages.to(device)
            prv_factorToVar_messages = prv_factorToVar_messages.to(device)
            prv_factor_beliefs = prv_factor_beliefs.to(device)
            prv_var_beliefs = prv_var_beliefs.to(device)
            
        #These locations are unused, set to LN_ZERO as a safety check for assert statements elsewhere
        assert(prv_factor_beliefs.shape == self.factor_potential_masks.shape)
        prv_factor_beliefs[torch.where(self.factor_potential_masks==1)] = LN_ZERO
            
        return prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs


    def compute_bethe_average_energy(self, factor_beliefs, debug=False):
        '''
        Equation (37) in:
        https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf        
        '''
        assert(self.factor_potentials.shape == factor_beliefs.shape)
        if debug:
            print()
            print('!!!!!!!')
            print("debugging compute_bethe_average_energy")
            print("torch.exp(factor_beliefs):", torch.exp(factor_beliefs))
            print("neg_inf_to_zero(self.factor_potentials):", neg_inf_to_zero(self.factor_potentials))
        bethe_average_energy = -torch.sum(torch.exp(factor_beliefs)*neg_inf_to_zero(self.factor_potentials)) #elementwise multiplication, then sum
        # print("bethe_average_energy:", bethe_average_energy)
        return bethe_average_energy

    def compute_bethe_entropy(self, factor_beliefs, var_beliefs):
        '''
        Equation (38) in:
        https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf        
        '''
        bethe_entropy = -torch.sum(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_beliefs)) #elementwise multiplication, then sum

        assert(var_beliefs.shape == torch.Size([self.numVars, 2])), (var_beliefs.shape, [self.numVars, 2])
        # sum_{x_i} b_i(x_i)*ln(b_i(x_i))
        inner_sum = torch.einsum('ij,ij->i', [torch.exp(var_beliefs), neg_inf_to_zero(var_beliefs)])
        # sum_{i=1}^N (d_i - 1)*inner_sum
        outer_sum = torch.sum((self.var_degrees.float() - 1) * inner_sum)
        # outer_sum = torch.einsum('i,i->', [self.var_degrees - 1, inner_sum])

        bethe_entropy += outer_sum
        # print("bethe_entropy:", bethe_entropy)
        return bethe_entropy

    def compute_bethe_free_energy(self, factor_beliefs, var_beliefs):
        '''
        Compute the Bethe approximation of the free energy.
        - free energy = -ln(Z)
          where Z is the partition function
        - (Bethe approximation of the free energy) = (Bethe average energy) - (Bethe entropy)

        For more details, see page 11 of:
        https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf
        '''
        # print("self.compute_bethe_average_energy():", self.compute_bethe_average_energy())
        # print("self.compute_bethe_entropy():", self.compute_bethe_entropy())
        if torch.isnan(factor_beliefs).any():
            print("values, some should be nan:")
            for val in factor_beliefs.flatten():
                print(val)
        assert(not torch.isnan(factor_beliefs).any()), (factor_beliefs, torch.where(factor_beliefs == torch.tensor(float('nan'))), torch.where(var_beliefs == torch.tensor(float('nan'))))
        assert(not torch.isnan(var_beliefs).any()), var_beliefs
        return self.compute_bethe_average_energy(factor_beliefs) - self.compute_bethe_entropy(factor_beliefs, var_beliefs)

    
def test_create_factorStates_to_varIndices(factorToVar_double_list=[[2,1], [2,3], [1], [3,2,1]], numVars=3):
    '''
    test function
    '''
    numMsgs = 0
    for variables_list in factorToVar_double_list:
        numMsgs += len(variables_list)
        print("numMsgs:", numMsgs)
        
    factorStates_to_varIndices_list = []
    factorToVar_edge_index_list = []
    
    junk_bin = 2*numMsgs
    state_dimensions = 3

    msg_idx = 0
    for factor_idx, variables_list in enumerate(factorToVar_double_list):
        unused_var_count = state_dimensions - len(variables_list)
        for varIdx_inFac, var_idx in enumerate(variables_list):
            factorToVar_edge_index_list.append(torch.tensor([factor_idx, var_idx]))
            print('-'*80)
            print("message number:", msg_idx)
            msgIdx_varIs0 = msg_idx
            msgIdx_varIs1 = msg_idx + numMsgs
            msg_idx += 1
            curFact_to_varIndices = -99*torch.ones(2**state_dimensions, dtype=torch.long)
            multipler1 = 2**(state_dimensions - varIdx_inFac - 1)
            curFact_to_varIndices[((torch.arange(2**state_dimensions)//multipler1) % 2) == 0] = msgIdx_varIs0
            curFact_to_varIndices[((torch.arange(2**state_dimensions)//multipler1) % 2) == 1] = msgIdx_varIs1
            assert(not (curFact_to_varIndices == -99).any())
            #send unused factor states to the junk bin
            if unused_var_count > 0:
                multiplier2 = 1
                for unused_var_idx in range(unused_var_count):
                    curFact_to_varIndices[((torch.arange(2**state_dimensions)//multiplier2) % 2) == 1] = junk_bin
                    multiplier2 *= 2
            factorStates_to_varIndices_list.append(curFact_to_varIndices)
            print("curFact_to_varIndices:", curFact_to_varIndices)
    assert(msg_idx == numMsgs)
    assert(len(factorStates_to_varIndices_list) == numMsgs)
    factorStates_to_varIndices = torch.cat(factorStates_to_varIndices_list)
    factorToVar_edge_index = torch.stack(factorToVar_edge_index_list).permute(1,0)
    
    print("test, factorStates_to_varIndices:", factorStates_to_varIndices)
    print("test, factorToVar_edge_index:", factorToVar_edge_index)
    print("test, factorToVar_edge_index.shape:", factorToVar_edge_index.shape)
    


class Batch_custom(DataFactorGraph_partial):
    #custom incrementing of values for bipartite factor graph with batching
    def __init__(self, **kwargs):       
        super(Batch_custom, self).__init__(**kwargs)

    @staticmethod
    def from_data_list(data_list, follow_batch=[]):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`."""

        keys = [set(data.keys) for data in data_list]
        keys = list(set.union(*keys))
        assert 'batch' not in keys

        batch = Batch_custom()
        batch.__data_class__ = data_list[0].__class__
        batch.__slices__ = {key: [0] for key in keys}
        
        for key in keys:
            batch[key] = []

        for key in follow_batch:
            batch['{}_batch'.format(key)] = []

        cumsum = {key: 0 for key in keys}
        batch.batch = []
        # we have a bipartite graph, so keep track of batches for each set of nodes
        batch.batch_factors = [] #for factor beliefs
        batch.batch_vars = [] #for variable beliefs
        junk_bin_val = 0
        for i, data in enumerate(data_list):
            for key in data.keys:
                item = data[key]
                if torch.is_tensor(item) and item.dtype != torch.bool:
#                     print("key:", key)
#                     print("item:", item)
#                     print("cumsum[key]:", cumsum[key])
#                     print()

                    if key == "facStates_to_varIdx":
#                         print("a item:", item)
#                         print("a item.shape:", item.shape)
#                         print("cumsum[key]:", cumsum[key])                
#                         print("cumsum[key].shape:", cumsum[key].shape)
#                         item = item + cumsum[key]
                        item = item.clone() #without this we edit the data for the next epoch, causing errors
                        item[torch.where(item != -1)] = item[torch.where(item != -1)] + cumsum[key]
#                         print("b item:", item)    
#                         print("b item.shape:", item.shape)    
                    else:
                        item = item + cumsum[key]
                if torch.is_tensor(item):
                    size = item.size(data.__cat_dim__(key, data[key]))
                else:
                    size = 1
                batch.__slices__[key].append(size + batch.__slices__[key][-1])
                if key == "facStates_to_varIdx":
                    num_edges_times2 = data.__inc__(key, item)
                    junk_bin_val += num_edges_times2
                    cumsum[key] += num_edges_times2                   
                else:
                    cumsum[key] += data.__inc__(key, item)
                batch[key].append(item)
                
#                 if key == 'edge_index' or key == 'facToVar_edge_idx':
#                     print("key:", key)
#                     print("cumsum[key]:", cumsum[key])
#                     print()
                    
                if key in follow_batch:
                    item = torch.full((size, ), i, dtype=torch.long)
                    batch['{}_batch'.format(key)].append(item)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long)
                batch.batch.append(item)
                print()
                print("num_nodes item:", item)
                print()
                
            batch.batch_factors.append(torch.full((data.numFactors, ), i, dtype=torch.long))
            batch.batch_vars.append(torch.full((data.numVars, ), i, dtype=torch.long))            

        if num_nodes is None:
            batch.batch = None
            
#         print("1 batch.num_nodes:", batch.num_nodes)
#         print("000 type(batch.num_nodes):", type(batch.num_nodes))

        for key in batch.keys:
#             print()
#             print("key:", key)
            item = batch[key][0]
            if torch.is_tensor(item):                   
                batch[key] = torch.cat(batch[key],
                                       dim=data_list[0].__cat_dim__(key, item))               
                if key == "facStates_to_varIdx":
                    batch[key][torch.where(batch[key] == -1)] = junk_bin_val   
            elif isinstance(item, int) or isinstance(item, float):               
                batch[key] = torch.tensor(batch[key])               
            else:
                raise ValueError('Unsupported attribute type')
                                  
######jdk debugging
#             if key == 'num_nodes':
#                 print("key: num_nodes, batch[key]:", batch[key])
            
#             try:
#                 print("key:", key, "batch.num_nodes:", batch.num_nodes)
#             except:
#                 print("batch.num_nodes doesn't work yet")
            

######end jdk debugging
                
        # Copy custom data functions to batch (does not work yet):
        # if data_list.__class__ != Data:
        #     org_funcs = set(Data.__dict__.keys())
        #     funcs = set(data_list[0].__class__.__dict__.keys())
        #     batch.__custom_funcs__ = funcs.difference(org_funcs)
        #     for func in funcs.difference(org_funcs):
        #         setattr(batch, func, getattr(data_list[0], func))

        if torch_geometric.is_debug_enabled():
            batch.debug()
        
        return batch.contiguous()


    def to_data_list(self):
        r"""Reconstructs the list of :class:`torch_geometric.data.Data` objects
        from the batch object.
        The batch object must have been created via :meth:`from_data_list` in
        order to be able reconstruct the initial objects."""

        if self.__slices__ is None:
            raise RuntimeError(
                ('Cannot reconstruct data list from batch because the batch '
                 'object was not created using Batch.from_data_list()'))

        keys = [key for key in self.keys if key[-5:] != 'batch']
        cumsum = {key: 0 for key in keys}
        data_list = []
        for i in range(len(self.__slices__[keys[0]]) - 1):
            data = self.__data_class__()
            for key in keys:
                if torch.is_tensor(self[key]):
                    data[key] = self[key].narrow(
                        data.__cat_dim__(key,
                                         self[key]), self.__slices__[key][i],
                        self.__slices__[key][i + 1] - self.__slices__[key][i])
                    if self[key].dtype != torch.bool:
                        data[key] = data[key] - cumsum[key]
                else:
                    data[key] = self[key][self.__slices__[key][i]:self.
                                          __slices__[key][i + 1]]
                cumsum[key] = cumsum[key] + data.__inc__(key, data[key])
            data_list.append(data)

        return data_list

    @property
    def num_graphs(self):
        """Returns the number of graphs in the batch."""
        return self.batch[-1].item() + 1    
    
    
    
    
    
class DataLoader_custom(torch.utils.data.DataLoader):
    #copied from pytorch-geometric, only changed to call Batch_custom
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self,
                 dataset,
                 batch_size=1,
                 shuffle=False,
                 follow_batch=[],
                 **kwargs):
        super(DataLoader_custom, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: Batch_custom.from_data_list(
                data_list, follow_batch),
            **kwargs)

    
    
if __name__ == "__main__":
    test_create_factorStates_to_varIndices()
