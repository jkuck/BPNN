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

def create_scatter_indices_varToFactorMsgs(original_indices, variable_cardinality=2, state_dimensions=2, belief_repeats=None):
    #When sending variable to factor messages, variable beliefs must be expanded to have extra redundant dimensions.
    #The approach is to expand all variable beliefs in all outgoing messages, then transpose each belief appropriately 
    #so that the variable it represents lines up in the correct dimension of the factor each message is sent to.
    #This function creates indices, to be used with torch_scatter, to perform this functionality.
    assert((original_indices < state_dimensions).all())
    scatter_indices_list = []
    for position, index in enumerate(original_indices):
        cur_offset = position*belief_repeats*(variable_cardinality**state_dimensions)
        cur_indices = create_scatter_indices_helper(expansion_index=index, variable_cardinality=variable_cardinality, 
                          state_dimensions=state_dimensions, offset=cur_offset)
        for belief_repeat_idx in range(belief_repeats):
            cur_repeat_indices = cur_indices.clone() + belief_repeat_idx*(variable_cardinality**state_dimensions)
            scatter_indices_list.append(cur_repeat_indices)
#             print("cur_repeat_indices:", cur_repeat_indices)
    scatter_indices = torch.cat(scatter_indices_list)
#     print("scatter_indices:", scatter_indices)
#     print()
    return scatter_indices




#this class requires input arguments in the constructor.  This doesn't work nicely
#with the batching implementation, which is why DataFactorGraph_partial was implemented
#A cleaner approach probably exists

##UPDATE: no longer requires arguments, probably can get rid of DataFactorGraph_partial
class FactorGraphData(DataFactorGraph_partial):
    '''
    Representation of a factor graph in pytorch geometric Data format
    '''
    def __init__(self, factor_potentials=None, factorToVar_edge_index=None, numVars=None, numFactors=None, 
                 edge_var_indices=None, state_dimensions=None, factor_potential_masks=None, ln_Z=None,
                 factorToVar_double_list=None, gt_variable_labels=None,
                 var_cardinality=None, belief_repeats=None):
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
        
        - var_cardinality (int): variable cardinality, the number of states each variable can take.
        - belief_repeats (int): to increase node feature size, repeat variable and factor beliefs belief_repeats times
                
        '''
#         print("factor_potentials[0,::]:", factor_potentials[0,::])
#         sleep(1233456543)
        super(FactorGraphData, self).__init__()
        if gt_variable_labels is not None:
            self.gt_variable_labels = gt_variable_labels
        if ln_Z is not None:
#             print("check ln_Z:", ln_Z)
            self.ln_Z = torch.tensor([ln_Z], dtype=float)
        
        # (int) the largest node degree
        if state_dimensions is not None:
            self.state_dimensions = torch.tensor([state_dimensions])
        else:
            self.state_dimensions = None

#         self.var_cardinality = torch.tensor([var_cardinality])
        self.var_cardinality = var_cardinality
        self.belief_repeats = belief_repeats
                
        #final shape is [# of factors, belief_repeats, followed by individual factor shape]
        #transpose required to get belief_repeats in second dimension
        #potentials defining the factor graph
        if factor_potentials is not None:
            expansion_list = [belief_repeats] + list(factor_potentials.shape)        
            self.factor_potentials = factor_potentials.expand(expansion_list).transpose(1, 0)
        else:
            self.factor_potentials = None
        #1 signifies an invalid location (e.g. a dummy dimension in a factor), 0 signifies a valid location 
        if factor_potentials is not None:        
            assert(factor_potentials is not None)
            self.factor_potential_masks = factor_potential_masks.expand(expansion_list).transpose(1, 0)
        else:
            self.factor_potential_masks = None
#         print("self.factor_potentials.shape:", self.factor_potentials.shape)
#         print("self.factor_potential_masks.shape:", self.factor_potential_masks.shape) 
        
#         print("self.factor_potentials[0,0,::].shape:", self.factor_potentials[0,0,::].shape)
#         print("self.factor_potentials[0,0,::]:", self.factor_potentials[0,0,::])
#         print("self.factor_potentials[0,1,::]:", self.factor_potentials[0,1,::])        
        
# #         for potential_idx in range(factor_potentials.shape[0]):
#         for potential_idx in range(3):
#             if (self.factor_potentials[potential_idx,0,::] != self.factor_potentials[potential_idx,1,::]).any():
#                 print("factor potentials differ at idx:", potential_idx)
#                 print("self.factor_potentials[potential_idx,0,::].shape:", self.factor_potentials[potential_idx,0,::].shape)                
#                 print("factor_potentials[potential_idx,0,::]:", self.factor_potentials[potential_idx,0,::])
#                 print("factor_potentials[potential_idx,1,::]:", self.factor_potentials[potential_idx,1,::])
#         print("sum =", torch.sum(self.factor_potentials[:,0,::] - self.factor_potentials[:,1,::]))
#         print(torch.where(factor_potentials[:,0,::] != factor_potentials[:,1,::]))
        if (belief_repeats is not None) and (belief_repeats > 1):
            assert((self.factor_potentials[:,0,::] == self.factor_potentials[:,1,::]).all())
            assert((self.factor_potential_masks[:,0,::] == self.factor_potential_masks[:,1,::]).all())

        # - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge
        #     matrix with shape :obj:`[numFactors, numVars]`
        #     stored as a [2, E] tensor of [factor_idx, var_idx] for each edge factor to variable edge
        if factorToVar_double_list is not None:
            #facStates_to_varIdx (torch LongTensor): essentially representing edges and pseudo edges (between
            #    junk states and a junk bin)
            #    has shape [(number of factor to variable edges)*(2^state_dimensions]  with values 
            #    in {0, 1, ..., (number of factor to variable edges)-1, (number of factor to variable edges)}. 
            #    Note (number of factor to variable edges) indicates a 'junk state' and should be output into a 
            #    'junk' bin after scatter operation.            
            self.facStates_to_varIdx, self.facToVar_edge_idx = self.create_factorStates_to_varIndices(factorToVar_double_list)
#             self.facStates_to_varIdx_FIXED, self.facToVar_edge_idx_FIXED = self.create_factorStates_to_varIndices_FIXED(factorToVar_double_list)
        elif factorToVar_edge_index is not None:
            if factorToVar_double_list is not None:
                assert((self.facToVar_edge_idx == factorToVar_edge_index).all())
            else:
                self.facToVar_edge_idx = factorToVar_edge_index
                self.facStates_to_varIdx = None
        else:
            self.facToVar_edge_idx = None
            self.facStates_to_varIdx = None


#         print("facStates_to_varIdx.shape:", self.facStates_to_varIdx.shape)
#         print("facToVar_edge_idx.shape:", self.facToVar_edge_idx.shape)
#         sleep(temp)
        self.edge_index = self.facToVar_edge_idx #hack for batching, see learn_BP_spinGlass.py
        # print("factorToVar_edge_index.shape:", factorToVar_edge_index.shape)
        # print("factorToVar_edge_index.shape:", factorToVar_edge_index.shape)



        if factorToVar_edge_index is not None:
            # self.factor_degrees[i] stores the number of variables that appear in factor i
            unique_factor_indices, self.factor_degrees = torch.unique(factorToVar_edge_index[0,:], sorted=True, return_counts=True)
            assert((self.factor_degrees >= 1).all())
            assert(unique_factor_indices.shape[0] == numFactors), (unique_factor_indices.shape[0], numFactors)

            # self.var_degrees[i] stores the number of factors that variables i appears in
            unique_var_indices, self.var_degrees = torch.unique(factorToVar_edge_index[1,:], sorted=True, return_counts=True)
            assert((self.var_degrees >= 1).all())
            assert(unique_var_indices.shape[0] == numVars)              
        else:
            self.factor_degrees = None
            self.var_degrees = None



        #when batching, numVars and numFactors record the number of variables and factors for each graph in the batch
        if numVars is not None:
            self.numVars = torch.tensor([numVars])
        else:
            self.numVars = None

        if numFactors is not None:
            self.numFactors = torch.tensor([numFactors])
        else:
            self.numFactors = None
        #when batching, see num_vars and num_factors to access the cumulative number of variables and factors in all
        #graphs in the batch, like num_nodes
        
        
        # edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] IS CURRENTLY UNUSED AND BROKEN, BUT IN GENERAL FOR PYTORCH GEOMETRIC SHOULD indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at
        self.edge_var_indices = edge_var_indices
        if edge_var_indices is not None:
            assert(self.facToVar_edge_idx.shape == self.edge_var_indices.shape)
        
        if edge_var_indices is not None:
            self.varToFactorMsg_scatter_indices = create_scatter_indices_varToFactorMsgs(original_indices=self.edge_var_indices[0, :],\
                                                                                         variable_cardinality=self.var_cardinality,\
                                                                                         state_dimensions=state_dimensions,\
                                                                                         belief_repeats=self.belief_repeats)
        else:
            self.varToFactorMsg_scatter_indices = None
        
        if self.var_cardinality is not None:
            prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs = self.get_initial_beliefs_and_messages()
            self.prv_varToFactor_messages = prv_varToFactor_messages
            self.prv_factorToVar_messages = prv_factorToVar_messages
            self.prv_factor_beliefs = prv_factor_beliefs
            self.prv_var_beliefs = prv_var_beliefs
            assert(self.prv_factor_beliefs.size(0) == self.numFactors)
            assert(self.prv_var_beliefs.size(0) == self.numVars)              
        else:
            self.prv_varToFactor_messages = None
            self.prv_factorToVar_messages = None
            self.prv_factor_beliefs = None
            self.prv_var_beliefs = None           
      
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
            # var_cardinality * belief_repeats * (number of edges)
            return torch.tensor([self.var_cardinality*self.belief_repeats*self.edge_var_indices.size(1)])
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
        
   


    def create_factorStates_to_varIndices(self, factorToVar_double_list):
        '''
        Inputs:
        - factorToVar_double_list (list of lists): 
            factorToVar_double_list[i] is a list of all variables that factor with index i shares an edge with
            factorToVar_double_list[i][j] is the index of the jth variable that the factor with index i shares an edge with
            
        Output:
        - factorStates_to_varIndices (torch LongTensor): 
        
            OLD: shape [(number of factor to variable edges)*(2^state_dimensions] 
            with values in {0, 1, ..., 2*(number of factor to variable edges)-1, 2*(number of factor to variable edges)}. 
            Note 2*(number of factor to variable edges) indicates a 'junk state' and should be output into a 
            'junk' bin after scatter operation.
            Used with scatter_logsumexp (https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html)
            to marginalize the appropriate factor states for each message
            
            NEW: refactor, want it to be: shape [(number of factor to variable edges)*(2^state_dimensions]
            with values in {0, 1, ..., var_cardinality*belief_repeats*(number of factor to variable edges)-1, 2*var_cardinality*belief_repeats*(number of factor to variable edges)}.     
            
        - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge matrix
            stored as a [2, E] tensor of [factor_idx, var_idx] for each edge factor to variable edge            
        '''
#         print("factorToVar_double_list:")
#         print(factorToVar_double_list)
        # the number of (undirected) edges in the factor graph, or messages in one graph wide update
        numMsgs = 0
        for variables_list in factorToVar_double_list:
            numMsgs += len(variables_list)
        
        factorStates_to_varIndices_list = []
        factorToVar_edge_index_list = []
        
        junk_bin = -1 #new junk bin for batching
        
#         self.belief_repeats = 2 #DEBUGGING, DELETE ME!!
#         self.var_cardinality = 2 #DEBUGGING, DELETE ME!!
#         self.state_dimensions = torch.tensor([4]) #DEBUGGING, DELETE ME!!
#         factorToVar_double_list = [[0,1,2], [3,4], [5], [6,7,8,9]] #DEBUGGING, DELETE ME!!
#         factorToVar_double_list = [[0,1,2], [3,4], [5]] #DEBUGGING, DELETE ME!!
        
        arange_tensor = torch.arange(self.var_cardinality**self.state_dimensions.item())        
        msg_idx = 0
        for factor_idx, variables_list in enumerate(factorToVar_double_list):
#             print("variables_list:", variables_list)
            unused_var_count = self.state_dimensions - len(variables_list)
            
#             print("factor_idx:", factor_idx)
            for varIdx_inFac, var_idx in enumerate(variables_list):
                factorToVar_edge_index_list.append(torch.tensor([factor_idx, var_idx]))
                curFact_to_varIndices = -99*torch.ones(self.var_cardinality**self.state_dimensions, dtype=torch.long)
                multiplier1 = self.var_cardinality**(self.state_dimensions - varIdx_inFac - 1)
                for var_state_idx in range(self.var_cardinality):
                    curFact_to_varIndices[((arange_tensor//multiplier1) % self.var_cardinality) == var_state_idx] = msg_idx + var_state_idx
                assert(not (curFact_to_varIndices == -99).any())                    

                #send unused factor states to the junk bin
                if unused_var_count > 0:
                    multiplier2 = 1 #note multiplier2 doubles at each iteration, looping over the variables backwards compared to multiplier1
                    for unused_var_idx in range(unused_var_count):
                        #send all factor states correpsonding to the unused variable being in any state except 0 to the junk bin
                        for var_state_idx in range(1, self.var_cardinality): 
                            curFact_to_varIndices[((arange_tensor//multiplier2) % self.var_cardinality) == var_state_idx] = junk_bin
                        multiplier2 *= self.var_cardinality
                        
                for repeat_idx in range(self.belief_repeats):
                    curRepeat_curFact_to_varIndices = curFact_to_varIndices.clone()
                    factorStates_to_varIndices_list.append(curRepeat_curFact_to_varIndices)
                    msg_idx += self.var_cardinality
                    curFact_to_varIndices[torch.where(curFact_to_varIndices != -1)] += self.var_cardinality
#             print("factorStates_to_varIndices_list:", factorStates_to_varIndices_list)
#             sleep(saldkfjwoei)
#                     print("var_idx:", var_idx, "repeat_idx:", repeat_idx, "curRepeat_curFact_to_varIndices:", curRepeat_curFact_to_varIndices)
                
#             print()
#         print("factorStates_to_varIndices_list:", factorStates_to_varIndices_list)
        
        
        assert(msg_idx == self.var_cardinality*self.belief_repeats*numMsgs), (msg_idx, numMsgs, self.var_cardinality*self.belief_repeats*numMsgs)
        assert(len(factorStates_to_varIndices_list) == numMsgs*self.belief_repeats) 
        factorStates_to_varIndices = torch.cat(factorStates_to_varIndices_list)
        factorToVar_edge_index = torch.stack(factorToVar_edge_index_list).permute(1,0)
#         print("factorStates_to_varIndices:", factorStates_to_varIndices)
#         sleep(check_curFact_to_varIndices) #remove self.var_cardinality = 3 above when removing this line!!!!!!
        
#         print("factorStates_to_varIndices:", factorStates_to_varIndices[:32])
#         sleep(alsdkjflksdj)    
        return factorStates_to_varIndices, factorToVar_edge_index

        
        
    def get_initial_beliefs_and_messages(self, initialize_randomly=False, device=None):
        edge_count = self.edge_var_indices.shape[1]

        prv_varToFactor_messages = torch.log(torch.stack([torch.ones([self.belief_repeats, self.var_cardinality], dtype=torch.float) for j in range(edge_count)], dim=0))
        prv_factorToVar_messages = torch.log(torch.stack([torch.ones([self.belief_repeats, self.var_cardinality], dtype=torch.float) for j in range(edge_count)], dim=0))
        prv_factor_beliefs = torch.log(torch.stack([torch.ones([self.belief_repeats] + [self.var_cardinality for i in range(self.state_dimensions)], dtype=torch.float) for j in range(self.numFactors)], dim=0))
        # prv_factor_beliefs = torch.log(factor_potentials.clone())
        # prv_factor_beliefs = prv_factor_beliefs/torch.logsumexp(prv_factor_beliefs, [i for i in range(1, len(prv_factor_beliefs.size()))])
        prv_var_beliefs = torch.log(torch.stack([torch.ones([self.belief_repeats, self.var_cardinality], dtype=torch.float) for j in range(self.numVars)], dim=0))
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
        assert(prv_factor_beliefs.shape == self.factor_potential_masks.shape), (prv_factor_beliefs.shape, self.factor_potential_masks.shape)
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

    
# def test_create_factorStates_to_varIndices(factorToVar_double_list=[[2,1], [2,3], [1], [3,2,1]], numVars=3):
def test_create_factorStates_to_varIndices(factorToVar_double_list=[[0,1], [1,0]], numVars=3, var_cardinality=2,
                                           state_dimensions = 2, belief_repeats = 1):
    '''
    test function
    '''

    
    numMsgs = 0
    for variables_list in factorToVar_double_list:
        numMsgs += len(variables_list)
        print("numMsgs:", numMsgs)

    # the number of (undirected) edges in the factor graph, or messages in one graph wide update
    numMsgs = 0
    for variables_list in factorToVar_double_list:
        numMsgs += len(variables_list)

    factorStates_to_varIndices_list = []
    factorToVar_edge_index_list = []

    junk_bin = -1 #new junk bin for batching

    arange_tensor = torch.arange(var_cardinality**state_dimensions)        
    msg_idx = 0
    for factor_idx, variables_list in enumerate(factorToVar_double_list):            
        unused_var_count = state_dimensions - len(variables_list)

        for varIdx_inFac, var_idx in enumerate(variables_list):
            factorToVar_edge_index_list.append(torch.tensor([factor_idx, var_idx]))
            curFact_to_varIndices = -99*torch.ones(var_cardinality**state_dimensions, dtype=torch.long)
            multiplier1 = var_cardinality**(state_dimensions - varIdx_inFac - 1)
            for var_state_idx in range(var_cardinality):
                curFact_to_varIndices[((arange_tensor//multiplier1) % var_cardinality) == var_state_idx] = msg_idx + var_state_idx
            assert(not (curFact_to_varIndices == -99).any())                    

            #send unused factor states to the junk bin
            if unused_var_count > 0:
                multiplier2 = 1 #note multiplier2 doubles at each iteration, looping over the variables backwards compared to multiplier1
                for unused_var_idx in range(unused_var_count):
                    #send all factor states correpsonding to the unused variable being in any state except 0 to the junk bin
                    for var_state_idx in range(1, var_cardinality): 
                        curFact_to_varIndices[((arange_tensor//multiplier2) % var_cardinality) == var_state_idx] = junk_bin
                    multiplier2 *= var_cardinality

            for repeat_idx in range(belief_repeats):
                curRepeat_curFact_to_varIndices = curFact_to_varIndices.clone()
                factorStates_to_varIndices_list.append(curRepeat_curFact_to_varIndices)
                msg_idx += var_cardinality
                curFact_to_varIndices[torch.where(curFact_to_varIndices != -1)] += var_cardinality

    assert(msg_idx == var_cardinality*belief_repeats*numMsgs), (msg_idx, numMsgs, var_cardinality*belief_repeats*numMsgs)
    assert(len(factorStates_to_varIndices_list) == numMsgs*belief_repeats) 
    factorStates_to_varIndices = torch.cat(factorStates_to_varIndices_list)
    factorToVar_edge_index = torch.stack(factorToVar_edge_index_list).permute(1,0)
       
    print("test, factorStates_to_varIndices:", factorStates_to_varIndices)
    print("test, factorToVar_edge_index:", factorToVar_edge_index)
    print("test, factorToVar_edge_index.shape:", factorToVar_edge_index.shape)
    
    return factorStates_to_varIndices, factorToVar_edge_index

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
                    facStates_to_varIdx_inc = data.__inc__(key, item)
                    junk_bin_val += facStates_to_varIdx_inc
                    cumsum[key] += facStates_to_varIdx_inc                   
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
                 shuffle=True,
                 follow_batch=[],
                 **kwargs):
        super(DataLoader_custom, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=lambda data_list: Batch_custom.from_data_list(
                data_list, follow_batch),
            **kwargs)

    
#from bpnn_model    
def test_map_beliefs(beliefs, map_type='factor',\
                     facToVar_edge_idx = torch.tensor([[0, 0, 1, 1],
                                                       [0, 1, 1, 0]])):
    '''
    Utility function for propagate().  Maps factor or variable beliefs to edges 
    See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    Inputs:
    - beliefs (tensor): factor_beliefs or var_beliefs, matching map_type
    - map_type: (string) 'factor' or 'var' denotes mapping factor or variable beliefs respectively
    '''
    numFactors = 2
    numVars = 2
    size = [numFactors, numVars]
    

    mapping_indices = {"factor": 0, "var": 1}

    idx = mapping_indices[map_type]
    if isinstance(beliefs, tuple) or isinstance(beliefs, list):
        assert len(beliefs) == 2
        if size[1 - idx] != beliefs[1 - idx].size(0):
            raise ValueError(__size_error_msg__)
        mapped_beliefs = beliefs[idx]
    else:
        mapped_beliefs = beliefs.clone()

    mapped_beliefs = torch.index_select(mapped_beliefs, 0, facToVar_edge_idx[idx])
    return mapped_beliefs    
    
    
from torch_scatter import scatter_logsumexp
from torch_geometric.utils import scatter_
    
if __name__ == "__main__":
    belief_repeats = 1
    var_cardinality = 2
    
    
    factorStates_to_varIndices, factorToVar_edge_index = test_create_factorStates_to_varIndices(factorToVar_double_list=[[0,1, 2], [0,1]], numVars=3,\
                                                         var_cardinality=2, state_dimensions = 3, belief_repeats = 1)
    factor_beliefs = torch.tensor(range(16)).float()
    factor_beliefs = factor_beliefs.view(2,2,2,2)
    factor_beliefs[1,:, :, 1] = -99

#     factorStates_to_varIndices, factorToVar_edge_index = test_create_factorStates_to_varIndices(factorToVar_double_list=[[0,1], [0,1]], numVars=3)
#     factor_beliefs = torch.tensor(range(8)).float()
#     factor_beliefs = factor_beliefs.view(2,2,2)

    print()

    print("factor_beliefs:", factor_beliefs)
    mapped_factor_beliefs = test_map_beliefs(beliefs=factor_beliefs, map_type='factor', facToVar_edge_idx=factorToVar_edge_index)
    print("mapped_factor_beliefs:", mapped_factor_beliefs)
    
    num_edges = factorToVar_edge_index.shape[1]
    print("mapped_factor_beliefs.view(mapped_factor_beliefs.numel())).shape:", mapped_factor_beliefs.view(mapped_factor_beliefs.numel()).shape)
    print("factorStates_to_varIndices.shape:", factorStates_to_varIndices.shape)
    factorStates_to_varIndices[torch.where(factorStates_to_varIndices == -1)] = num_edges*2
    marginalized_states_fast = torch.exp(scatter_logsumexp(src=torch.log(mapped_factor_beliefs.view(mapped_factor_beliefs.numel())), index=factorStates_to_varIndices, dim_size=num_edges*2 + 1))    
    
    print("marginalized_states_fast:", marginalized_states_fast)
    
    old_marginalized_states = marginalized_states_fast[:-1].view((2,num_edges)).permute(1,0)
    new_marginalized_states = marginalized_states_fast[:-1].view(num_edges,belief_repeats,var_cardinality)    
    print("old_marginalized_states:", old_marginalized_states)
    print("new_marginalized_states:", new_marginalized_states)
    
    old_var_beliefs = scatter_('add', old_marginalized_states, factorToVar_edge_index[1])
    new_var_beliefs = scatter_('add', new_marginalized_states, factorToVar_edge_index[1])
    
    print("old_var_beliefs:", old_var_beliefs)
    print("new_var_beliefs:", new_var_beliefs)
    
    old_mapped_var_beliefs = test_map_beliefs(beliefs=old_var_beliefs, map_type='var', facToVar_edge_idx=factorToVar_edge_index)
    new_mapped_var_beliefs = test_map_beliefs(beliefs=new_var_beliefs, map_type='var', facToVar_edge_idx=factorToVar_edge_index)
    print("old_mapped_var_beliefs:", old_mapped_var_beliefs)
    print("new_mapped_var_beliefs:", new_mapped_var_beliefs)
        
    