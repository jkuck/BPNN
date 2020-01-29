import torch
import numpy as np
from collections import defaultdict
from utils import dotdict, neg_inf_to_zero




class FactorGraph(dotdict):
    '''
    Representation of a factor graph
    Inherit from dictionary class for pytorch dataloader
    '''
    def __init__(self, factor_potentials, factorToVar_edge_index, numVars, numFactors, 
                 edge_var_indices, state_dimensions, factor_potential_masks):
        #potentials defining the factor graph
        self.factor_potentials = factor_potentials 

        # - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge
        #     matrix with shape :obj:`[numFactors, numVars]`
        self.factorToVar_edge_index = factorToVar_edge_index

        # print("factorToVar_edge_index.shape:", factorToVar_edge_index.shape)



        # self.factor_degrees[i] stores the number of variables that appear in factor i
        unique_factor_indices, self.factor_degrees = torch.unique(factorToVar_edge_index[0,:], sorted=True, return_counts=True)
        assert((self.factor_degrees >= 1).all())
        assert(unique_factor_indices.shape[0] == numFactors)

        # self.var_degrees[i] stores the number of factors that variables i appears in
        unique_var_indices, self.var_degrees = torch.unique(factorToVar_edge_index[1,:], sorted=True, return_counts=True)
        assert((self.var_degrees >= 1).all())
        assert(unique_var_indices.shape[0] == numVars)

        self.numVars = numVars
        self.numFactors = numFactors


        # edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] IS CURRENTLY UNUSED AND BROKEN, BUT IN GENERAL FOR PYTORCH GEOMETRIC SHOULD indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at
        self.edge_var_indices = edge_var_indices
        # (int) the largest node degree
        self.state_dimensions = state_dimensions

        #1 signifies an invalid location (e.g. a dummy dimension in a factor), 0 signifies a valid location 
        self.factor_potential_masks = factor_potential_masks

    @classmethod
    def init_from_dictionary(cls, arg_dictionary, squeeze_tensors=False):
        '''
        constructor that takes a dictionary as input
        https://stackoverflow.com/questions/2164258/multiple-constructors-in-python
        Inputs:
        - squeeze_tensors (bool): if True, squeeze all input tensors (helpful for batch size 1 testing)
        '''
        for arg in ["factor_potentials", "factorToVar_edge_index", "numVars", "numFactors", 
                      "edge_var_indices", "state_dimensions", "factor_potential_masks"]:
            assert(arg in arg_dictionary), ("Dictionary missing argument:", arg)

        if squeeze_tensors:
            for arg_name, arg_val in arg_dictionary.items():
                if type(arg_val) == torch.Tensor:
                    arg_dictionary[arg_name] = arg_val.squeeze()
                else:
                    print("not squeezing", arg_name, "with type", type(arg_val))
 

        return cls(arg_dictionary["factor_potentials"], arg_dictionary["factorToVar_edge_index"], arg_dictionary["numVars"], arg_dictionary["numFactors"], 
                   arg_dictionary["edge_var_indices"], arg_dictionary["state_dimensions"], arg_dictionary["factor_potential_masks"])

    def get_initial_beliefs_and_messages(self, initialize_randomly=False):
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





