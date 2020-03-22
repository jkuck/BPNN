import torch
import numpy as np
from collections import defaultdict
from utils import dotdict, neg_inf_to_zero
from torch_geometric.data import Data


class FactorGraphData(Data):
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
        super().__init__()
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
        else:
            self.facToVar_edge_idx = factorToVar_edge_index

        self.edge_index = self.facToVar_edge_idx
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

        self.numVars = torch.tensor([numVars])
        self.numFactors = torch.tensor([numFactors])


        # edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] IS CURRENTLY UNUSED AND BROKEN, BUT IN GENERAL FOR PYTORCH GEOMETRIC SHOULD indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at
        self.edge_var_indices = edge_var_indices
        assert(self.facToVar_edge_idx.shape == self.edge_var_indices.shape)
        


        #1 signifies an invalid location (e.g. a dummy dimension in a factor), 0 signifies a valid location 
        self.factor_potential_masks = factor_potential_masks

        
        prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs = self.get_initial_beliefs_and_messages()
        self.prv_varToFactor_messages = prv_varToFactor_messages
        self.prv_factorToVar_messages = prv_factorToVar_messages
        self.prv_factor_beliefs = prv_factor_beliefs
        self.prv_var_beliefs = prv_var_beliefs
#         print("added prv_varToFactor_messages!!1234")
#         print("prv_varToFactor_messages:", prv_varToFactor_messages)
#         sleep(temp23)
#         for attr, value in self.__dict__.items():
#             if value is not None:
#                 print(attr, type(value))
#             else:
#                 print(attr, value)
#         sleep(check_types)

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
            Used with https://pytorch-scatter.readthedocs.io/en/latest/functions/segment_coo.html
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
        
        junk_bin = 2*numMsgs
        zeros_vec = torch.zeros(2**self.state_dimensions)
        
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
                curFact_to_varIndices[((torch.where(zeros_vec==0)[0]//multipler1) % 2) == 0] = msgIdx_varIs0
                curFact_to_varIndices[((torch.where(zeros_vec==0)[0]//multipler1) % 2) == 1] = msgIdx_varIs1
                assert(not (curFact_to_varIndices == -99).any())
                #send unused factor states to the junk bin
                if unused_var_count > 0:
                    multiplier2 = 1
                    for unused_var_idx in range(unused_var_count):
                        curFact_to_varIndices[((torch.where(zeros_vec==0)[0]//multiplier2) % 2) == 1] = junk_bin
                        multiplier2 *= 2
                factorStates_to_varIndices_list.append(curFact_to_varIndices)
        assert(msg_idx == numMsgs)
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
    zeros_vec = torch.zeros(2**state_dimensions)

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
            curFact_to_varIndices[((torch.where(zeros_vec==0)[0]//multipler1) % 2) == 0] = msgIdx_varIs0
            curFact_to_varIndices[((torch.where(zeros_vec==0)[0]//multipler1) % 2) == 1] = msgIdx_varIs1
            assert(not (curFact_to_varIndices == -99).any())
            #send unused factor states to the junk bin
            if unused_var_count > 0:
                multiplier2 = 1
                for unused_var_idx in range(unused_var_count):
                    curFact_to_varIndices[((torch.where(zeros_vec==0)[0]//multiplier2) % 2) == 1] = junk_bin
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
    

if __name__ == "__main__":
    test_create_factorStates_to_varIndices()
