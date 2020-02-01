import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from loopyBP_factorGraph import FactorGraphMsgPassingLayer_NoDoubleCounting
from loopyBP_factorGraph_2layerWorks import FactorGraphMsgPassingLayer_NoDoubleCounting



class lbp_message_passing_network(nn.Module):
    def __init__(self, max_factor_state_dimensions, msg_passing_iters, device=None):
        '''
        Inputs:
        - max_factor_state_dimensions (int): the number of dimensions (variables) the largest factor have.
            -> will have states space of size 2*max_factor_state_dimensions
        - msg_passing_iters (int): the number of iterations of message passing to run (we have this many
            message passing layers with their own learnable parameters)
        '''
        super().__init__()        
        self.message_passing_layers = nn.ModuleList([FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**max_factor_state_dimensions)\
                                       for i in range(msg_passing_iters)])
        self.device = device

    def forward(self, factor_graph):
        prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs = factor_graph.get_initial_beliefs_and_messages(device=self.device)
        for message_passing_layer in self.message_passing_layers:
            prv_varToFactor_messages, prv_factorToVar_messages, prv_var_beliefs, prv_factor_beliefs =\
                message_passing_layer(factor_graph, prv_varToFactor_messages=prv_varToFactor_messages,
                                      prv_factorToVar_messages=prv_factorToVar_messages, prv_factor_beliefs=prv_factor_beliefs)
        bethe_free_energy = factor_graph.compute_bethe_free_energy(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs)
        estimated_ln_partition_function = -bethe_free_energy
        return estimated_ln_partition_function