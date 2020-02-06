import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import remove_self_loops

from torch.nn import Sequential as Seq, Linear, ReLU
from utils import neg_inf_to_zero

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
#                 message_passing_layer(factor_graph, prv_varToFactor_messages=factor_graph.prv_varToFactor_messages,
#                                       prv_factorToVar_messages=factor_graph.prv_factorToVar_messages, prv_factor_beliefs=factor_graph.prv_factor_beliefs)
          
        bethe_free_energy = compute_bethe_free_energy(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
        estimated_ln_partition_function = -bethe_free_energy
        return estimated_ln_partition_function
    
    
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
    # print("bethe_average_energy:", bethe_average_energy)
    return bethe_average_energy

def compute_bethe_entropy(factor_beliefs, var_beliefs, numVars, var_degrees):
    '''
    Equation (38) in:
    https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf        
    '''
    bethe_entropy = -torch.sum(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_beliefs)) #elementwise multiplication, then sum

    assert(var_beliefs.shape == torch.Size([numVars, 2])), (var_beliefs.shape, [numVars, 2])
    # sum_{x_i} b_i(x_i)*ln(b_i(x_i))
    inner_sum = torch.einsum('ij,ij->i', [torch.exp(var_beliefs), neg_inf_to_zero(var_beliefs)])
    # sum_{i=1}^N (d_i - 1)*inner_sum
    outer_sum = torch.sum((var_degrees.float() - 1) * inner_sum)
    # outer_sum = torch.einsum('i,i->', [var_degrees - 1, inner_sum])

    bethe_entropy += outer_sum
    # print("bethe_entropy:", bethe_entropy)
    return bethe_entropy

def compute_bethe_free_energy(factor_beliefs, var_beliefs, factor_graph):
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
    return (compute_bethe_average_energy(factor_beliefs=factor_beliefs, factor_potentials=factor_graph.factor_potentials)\
            - compute_bethe_entropy(factor_beliefs=factor_beliefs, var_beliefs=var_beliefs, numVars=factor_graph.numVars, var_degrees=factor_graph.var_degrees))

class GIN_Network_withEdgeFeatures(nn.Module):
    def __init__(self, input_state_size=1, edge_attr_size=1, hidden_size=4, msg_passing_iters=5, edgedevice=None):
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
