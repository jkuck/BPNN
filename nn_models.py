import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import remove_self_loops

from torch.nn import Sequential as Seq, Linear, ReLU

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
    
    
class GIN_Network_withEdgeFeatures(nn.Module):
    def __init__(self, input_state_size=1, edge_attr_size=1, hidden_size=128, msg_passing_iters=5, edgedevice=None):
        '''
        Inputs:
        - msg_passing_iters (int): the number of iterations of message passing to run (we have this many
            message passing layers with their own learnable parameters)
        '''
        super().__init__()
        layers = [GINConv_withEdgeFeatures(nn1=Seq(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)),
                                          nn2=Seq(Linear(input_state_size + edge_attr_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)))] + \
                 [GINConv_withEdgeFeatures(nn1=Seq(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)),
                                          nn2=Seq(Linear(hidden_size + edge_attr_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)))\
                                          for i in range(msg_passing_iters - 1)]
        self.message_passing_layers = nn.ModuleList(layers)
        self.final_mlp = Seq(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, 1))
        
    def forward(self, x, edge_index, edge_attr, batch):
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
