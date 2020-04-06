import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import remove_self_loops
import numpy as np
from torch.nn import Sequential as Seq, Linear, ReLU
from utils import neg_inf_to_zero, shift_func

import math
from bpnn_model import FactorGraphMsgPassingLayer_NoDoubleCounting
from parameters import SHARE_WEIGHTS, BETHE_MLP

class lbp_message_passing_network(nn.Module):
    def __init__(self, max_factor_state_dimensions, msg_passing_iters, device=None, share_weights=SHARE_WEIGHTS,
                bethe_MLP=BETHE_MLP, map_flag=False):
        '''
        Inputs:
        - max_factor_state_dimensions (int): the number of dimensions (variables) the largest factor have.
            -> will have states space of size 2*max_factor_state_dimensions
        - msg_passing_iters (int): the number of iterations of message passing to run (we have this many
            message passing layers with their own learnable parameters)
        - share_weights (bool): if true, share the same weights across each message passing iteration
        - bethe_MLP (bool): if True, use an MLP to learn a modified Bethe approximation (initialized
                            to the exact Bethe approximation)
        '''
        super().__init__()
        self.share_weights = share_weights
        self.msg_passing_iters = msg_passing_iters
        self.bethe_MLP = bethe_MLP
        self.map_flag = map_flag
        if share_weights:
            self.message_passing_layer = FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**max_factor_state_dimensions, map_flag=map_flag)
        else:
            self.message_passing_layers = nn.ModuleList([FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**max_factor_state_dimensions, map_flag=map_flag)\
                                           for i in range(msg_passing_iters)])
        self.device = device

        if bethe_MLP:
            var_states = 2 #2 for binary variables
            mlp_size =  2*msg_passing_iters*(2**max_factor_state_dimensions) + msg_passing_iters*var_states
#             self.final_mlp = Seq(Linear(mlp_size, mlp_size), ReLU(), Linear(mlp_size, 1))

            self.linear1 = Linear(mlp_size, mlp_size)
            self.linear2 = Linear(mlp_size, 1)
            self.linear1.weight = torch.nn.Parameter(torch.eye(mlp_size))
            self.linear1.bias = torch.nn.Parameter(torch.zeros(self.linear1.bias.shape))
            weight_initialization = torch.zeros((1,mlp_size))
            num_ones = (2*(2**max_factor_state_dimensions)+var_states)
            weight_initialization[0,-num_ones:] = 1
#             print("self.linear2.weight:", self.linear2.weight)
#             print("self.linear2.weight.shape:", self.linear2.weight.shape)


#             print("weight_initialization:", weight_initialization)
            self.linear2.weight = torch.nn.Parameter(weight_initialization)

            self.linear2.bias = torch.nn.Parameter(torch.zeros(self.linear2.bias.shape))
            self.shifted_relu = shift_func(ReLU(), shift=-500)
            self.final_mlp = Seq(self.linear1, self.shifted_relu, self.linear2, self.shifted_relu)
#             self.final_mlp = Seq(self.linear1, self.linear2)





    def forward(self, factor_graph):
#         prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs = factor_graph.get_initial_beliefs_and_messages(device=self.device)
        prv_varToFactor_messages = factor_graph.prv_varToFactor_messages
        prv_factorToVar_messages = factor_graph.prv_factorToVar_messages
        prv_factor_beliefs = factor_graph.prv_factor_beliefs
#         print("prv_factor_beliefs.shape:", prv_factor_beliefs.shape)
#         print("prv_factorToVar_messages.shape:", prv_factorToVar_messages.shape)
#         print("factor_graph.facToVar_edge_idx.shape:", factor_graph.facToVar_edge_idx.shape)


        pooled_states = []

        if self.share_weights:
            for iter in range(self.msg_passing_iters):
                prv_varToFactor_messages, prv_factorToVar_messages, prv_var_beliefs, prv_factor_beliefs =\
                    self.message_passing_layer(factor_graph, prv_varToFactor_messages=prv_varToFactor_messages,
                                          prv_factorToVar_messages=prv_factorToVar_messages, prv_factor_beliefs=prv_factor_beliefs)
                if self.bethe_MLP:
                    cur_pooled_states = self.compute_bethe_free_energy_pooledStates_MLP(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
                    pooled_states.append(cur_pooled_states)
        else:
            for message_passing_layer in self.message_passing_layers:
#                 print("prv_varToFactor_messages:", prv_varToFactor_messages)
#                 print("prv_factorToVar_messages:", prv_factorToVar_messages)
#                 print("prv_factor_beliefs:", prv_factor_beliefs)
#                 print("prv_varToFactor_messages.shape:", prv_varToFactor_messages.shape)
#                 print("prv_factorToVar_messages.shape:", prv_factorToVar_messages.shape)
#                 print("prv_factor_beliefs.shape:", prv_factor_beliefs.shape)
#                 prv_factor_beliefs[torch.where(prv_factor_beliefs==-np.inf)] = 0

                prv_varToFactor_messages, prv_factorToVar_messages, prv_var_beliefs, prv_factor_beliefs =\
                    message_passing_layer(factor_graph, prv_varToFactor_messages=prv_varToFactor_messages,
                                          prv_factorToVar_messages=prv_factorToVar_messages, prv_factor_beliefs=prv_factor_beliefs)
    #                 message_passing_layer(factor_graph, prv_varToFactor_messages=factor_graph.prv_varToFactor_messages,
    #                                       prv_factorToVar_messages=factor_graph.prv_factorToVar_messages, prv_factor_beliefs=factor_graph.prv_factor_beliefs)
                if self.bethe_MLP:
                    cur_pooled_states = self.compute_bethe_free_energy_pooledStates_MLP(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
#                     print("cur_pooled_states:", cur_pooled_states)
#                     print(check_pool)
#                     print("cur_pooled_states.shape:", cur_pooled_states.shape)
                    pooled_states.append(cur_pooled_states)


        if self.bethe_MLP:
#             print("torch.cat(pooled_states).shape:", torch.cat(pooled_states, dim=1).shape)
#             print("torch.cat(pooled_states):", torch.cat(pooled_states, dim=1))
#             sleep(check_pool2)
#             print("torch.cat(pooled_states, dim=1):")
#             print(torch.cat(pooled_states, dim=1))
#             print()
            estimated_ln_partition_function = self.final_mlp(torch.cat(pooled_states, dim=1))

#             bethe_free_energy = compute_bethe_free_energy(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
#             estimated_ln_partition_function_orig = -bethe_free_energy
#             print("estimated_ln_partition_function_orig:", estimated_ln_partition_function_orig)
#             print("estimated_ln_partition_function:", estimated_ln_partition_function)
#             assert(np.isclose(estimated_ln_partition_function_orig.detach().numpy(), estimated_ln_partition_function.detach().numpy(), rtol=1e-03, atol=1e-03)), (estimated_ln_partition_function_orig, estimated_ln_partition_function)
            return estimated_ln_partition_function

        else:
            if False:
                #broken for batch_size > 1
                bethe_free_energy = compute_bethe_free_energy(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
                estimated_ln_partition_function = -bethe_free_energy

                debug=True
                if debug:
                    cur_pooled_states = self.compute_bethe_free_energy_pooledStates_MLP(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
                    check_estimated_ln_partition_function = torch.sum(cur_pooled_states)
    #                 print("check_estimated_ln_partition_function:", check_estimated_ln_partition_function)
    #                 print("estimated_ln_partition_function:", estimated_ln_partition_function)
    #                 sleep(debug_bethe)
                    assert(torch.allclose(check_estimated_ln_partition_function, estimated_ln_partition_function)), (check_estimated_ln_partition_function, estimated_ln_partition_function)
                return estimated_ln_partition_function

            #corrected for batch_size > 1
            cur_pooled_states = self.compute_bethe_free_energy_pooledStates_MLP(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
            estimated_ln_partition_function = torch.sum(cur_pooled_states, dim=1)
            return estimated_ln_partition_function



    def compute_bethe_average_energy_MLP(self, factor_beliefs, factor_potentials, batch_factors, debug=False):
        '''
        Equation (37) in:
        https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf
        '''
        assert(factor_potentials.shape == factor_beliefs.shape)
        if debug:
            print()
            print('!!!!!!!')
            print("debugging compute_bethe_average_energy")
            print("torch.exp(factor_beliefs):", torch.exp(factor_beliefs))
            print("neg_inf_to_zero(factor_potentials):", neg_inf_to_zero(factor_potentials))

        pooled_fac_beleifPotentials = global_add_pool(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials), batch_factors)
        pooled_fac_beleifPotentials = pooled_fac_beleifPotentials.view(pooled_fac_beleifPotentials.shape[0], -1)
        if debug:
            factor_beliefs_shape = factor_beliefs.shape
            pooled_fac_beleifPotentials_orig = torch.sum((torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials)).view(factor_beliefs_shape[0], -1), dim=0)
            print("original pooled_fac_beleifPotentials_orig:", pooled_fac_beleifPotentials_orig)
            print("pooled_fac_beleifPotentials:", pooled_fac_beleifPotentials)
            print("factor_beliefs.shape:", factor_beliefs.shape)
            print("pooled_fac_beleifPotentials_orig.shape:", pooled_fac_beleifPotentials_orig.shape)
            print("pooled_fac_beleifPotentials.shape:", pooled_fac_beleifPotentials.shape)
            print("(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials)).view(factor_beliefs_shape[0], -1).shape:", (torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials)).view(factor_beliefs_shape[0], -1).shape)
        return pooled_fac_beleifPotentials #negate and sum to get average bethe energy


    def compute_bethe_entropy_MLP(self, factor_beliefs, var_beliefs, numVars, var_degrees, batch_factors, batch_vars, debug=False):
        '''
        Equation (38) in:
        https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf
        '''

        pooled_fac_beliefs = -global_add_pool(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_beliefs), batch_factors)
        pooled_fac_beliefs = pooled_fac_beliefs.view(pooled_fac_beliefs.shape[0], -1)
        if debug:
            factor_beliefs_shape = factor_beliefs.shape
            pooled_fac_beliefs_orig = -torch.sum((torch.exp(factor_beliefs)*neg_inf_to_zero(factor_beliefs)).view(factor_beliefs_shape[0], -1), dim=0)
            print("pooled_fac_beliefs_orig:", pooled_fac_beliefs_orig)
            print("pooled_fac_beliefs:", pooled_fac_beliefs)



        var_beliefs_shape = var_beliefs.shape
        pooled_var_beliefs = global_add_pool(torch.exp(var_beliefs)*neg_inf_to_zero(var_beliefs)*(var_degrees.float() - 1).view(var_beliefs_shape[0], -1), batch_vars)

        if debug:
            pooled_var_beliefs_orig = torch.sum(torch.exp(var_beliefs)*neg_inf_to_zero(var_beliefs)*(var_degrees.float() - 1).view(var_beliefs_shape[0], -1), dim=0)
            print("pooled_var_beliefs_orig:", pooled_var_beliefs_orig)
            print("pooled_var_beliefs:", pooled_var_beliefs)
    #         sleep(SHAPECHECK)

        return pooled_fac_beliefs, pooled_var_beliefs

    def compute_bethe_free_energy_pooledStates_MLP(self, factor_beliefs, var_beliefs, factor_graph):
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
        pooled_fac_beleifPotentials = self.compute_bethe_average_energy_MLP(factor_beliefs=factor_beliefs,\
                                      factor_potentials=factor_graph.factor_potentials, batch_factors=factor_graph.batch_factors)
        pooled_fac_beliefs, pooled_var_beliefs = self.compute_bethe_entropy_MLP(factor_beliefs=factor_beliefs, var_beliefs=var_beliefs, numVars=torch.sum(factor_graph.numVars), var_degrees=factor_graph.var_degrees, batch_factors=factor_graph.batch_factors, batch_vars=factor_graph.batch_vars)

        if len(pooled_fac_beleifPotentials.shape) > 1:
            cat_dim = 1
        else:
            cat_dim = 0

        return torch.cat([pooled_fac_beleifPotentials, pooled_fac_beliefs, pooled_var_beliefs], dim=cat_dim)



def compute_bethe_average_energy(factor_beliefs, factor_potentials, debug=False):
    '''
    Equation (37) in:
    https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf
    '''
    assert(factor_potentials.shape == factor_beliefs.shape)
    if debug:
        print()
        print('!!!!!!!')
        print("debugging compute_bethe_average_energy")
        print("torch.exp(factor_beliefs):", torch.exp(factor_beliefs))
        print("neg_inf_to_zero(factor_potentials):", neg_inf_to_zero(factor_potentials))
    bethe_average_energy = -torch.sum(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials)) #elementwise multiplication, then sum
#     print("bethe_average_energy:", bethe_average_energy)
    return bethe_average_energy

def compute_bethe_entropy(factor_beliefs, var_beliefs, numVars, var_degrees):
    '''
    Equation (38) in:
    https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf
    '''
    bethe_entropy = -torch.sum(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_beliefs)) #elementwise multiplication, then sum

#     print("numVars:", numVars)
    assert(var_beliefs.shape == torch.Size([numVars, 2])), (var_beliefs.shape, [numVars, 2])
    # sum_{x_i} b_i(x_i)*ln(b_i(x_i))
    inner_sum = torch.einsum('ij,ij->i', [torch.exp(var_beliefs), neg_inf_to_zero(var_beliefs)])
    # sum_{i=1}^N (d_i - 1)*inner_sum
    outer_sum = torch.sum((var_degrees.float() - 1) * inner_sum)
    # outer_sum = torch.einsum('i,i->', [var_degrees - 1, inner_sum])

    bethe_entropy += outer_sum
#     print("bethe_entropy:", bethe_entropy)
    return bethe_entropy

def compute_bethe_free_energy(factor_beliefs, var_beliefs, factor_graph):
    '''
    BROKEN FOR BATCH SIZE > 1
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
    return (compute_bethe_average_energy(factor_beliefs=factor_beliefs, factor_potentials=factor_graph.factor_potentials)\
            - compute_bethe_entropy(factor_beliefs=factor_beliefs, var_beliefs=var_beliefs, numVars=torch.sum(factor_graph.numVars), var_degrees=factor_graph.var_degrees))

class GIN_Network_withEdgeFeatures(nn.Module):
    def __init__(self, input_state_size=1, edge_attr_size=1, hidden_size=4, msg_passing_iters=5, feat_all_layers=True, edgedevice=None):
        '''
        Inputs:
        - msg_passing_iters (int): the number of iterations of message passing to run (we have this many
            message passing layers with their own learnable parameters)
        - feat_all_layers (bool): if True, use concatenation of sum of all node features after every layer as input to final MLP
        '''
        super().__init__()
        layers = [GINConv_withEdgeFeatures(nn1=Seq(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)),
                                          nn2=Seq(Linear(input_state_size + edge_attr_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)))] + \
                 [GINConv_withEdgeFeatures(nn1=Seq(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)),
                                          nn2=Seq(Linear(hidden_size + edge_attr_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)))\
                                          for i in range(msg_passing_iters - 1)]
        self.message_passing_layers = nn.ModuleList(layers)
        self.feat_all_layers = feat_all_layers
        if self.feat_all_layers:
            self.final_mlp = Seq(Linear(msg_passing_iters*hidden_size, msg_passing_iters*hidden_size), ReLU(), Linear(msg_passing_iters*hidden_size, 1))
        else:
            self.final_mlp = Seq(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        if self.feat_all_layers:
            summed_node_features_all_layers = []
            for message_passing_layer in self.message_passing_layers:
                x = message_passing_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
                summed_node_features_all_layers.append(global_add_pool(x, batch))

#             print("torch.cat(summed_node_features_all_layers, dim=0).shape:", torch.cat(summed_node_features_all_layers, dim=1).shape)
#             print("global_add_pool(x, batch).shape:", global_add_pool(x, batch).shape)
            return self.final_mlp(torch.cat(summed_node_features_all_layers, dim=1))
        else:
            for message_passing_layer in self.message_passing_layers:
                x = message_passing_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

            return self.final_mlp(global_add_pool(x, batch))

class GINConv_withEdgeFeatures(MessagePassing):
    r"""Modification of the graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, which uses
    edge features.

    .. math::
        \mathbf{x}^{\prime}_i = \text{MLP}_1 \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \text{MLP}_2 \left( \mathbf{x}_j, \mathbf{e}_{i,j} \right) \right),

    Args:
        nn1 (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        nn2 (torch.nn.Module): A neural network mapping shape [-1, in_channels + edge_features]
            to [-1, in_channels]
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn1, nn2, eps=0, train_eps=False, **kwargs):
        super(GINConv_withEdgeFeatures, self).__init__(aggr='add', **kwargs)
        self.nn1 = nn1
        self.nn2 = nn2
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn1)
        self.eps.data.fill_(self.initial_eps)


    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.nn1((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out


    def message(self, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # edge_attr has shape [E, edge_features]
        tmp = torch.cat([x_j, edge_attr], dim=1)  # tmp has shape [E, in_channels + edge_features]
        return self.nn2(tmp)
