import inspect

import torch
from torch_geometric.utils import scatter_

special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')


class MessagePassing_NoDoubleCounting(torch.nn.Module):
    r"""Base class for creating message passing layers

    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),

    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.

    Args:
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"` or :obj:`"max"`).
            (default: :obj:`"add"`)
        flow (string, optional): The flow direction of message passing
            (:obj:`"source_to_target"` or :obj:`"target_to_source"`).
            (default: :obj:`"source_to_target"`)
    """

    def __init__(self, aggr='add', flow='source_to_target'):
        super(MessagePassing_NoDoubleCounting, self).__init__()

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.__message_args_factorToVar__ = inspect.getfullargspec(self.message_factorToVar)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args_factorToVar__)
                                 if arg in special_args]
        self.__message_args_factorToVar__ = [
            arg for arg in self.__message_args_factorToVar__ if arg not in special_args
        ]
        self.__message_args_varToFactor__ = inspect.getfullargspec(self.message_varToFactor)[0][1:]

        self.__message_args_varToFactor__ = [

            arg for arg in self.__message_args_varToFactor__ if arg not in special_args

        ]        
        self.__update_args__ = inspect.getfullargspec(self.update)[0][2:]

    def propagate(self, edge_index, size=None, factor_potentials, **kwargs):
        r"""The initial call to start propagating messages.

        Args:
            edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            size (list or tuple, optional): The size :obj:`[N, M]` of the
                assignment matrix. If set to :obj:`None`, the size is tried to
                get automatically inferrred. (default: :obj:`None`)
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """

        size = [None, None] if size is None else list(size)
        assert len(size) == 2

        i, j = (0, 1) if self.flow == 'target_to_source' else (1, 0)
        ij = {"_i": i, "_j": j}

        message_args_factorToVar = []
        for arg in self.__message_args_factorToVar__:
            if arg[-2:] in ij.keys():
                tmp = kwargs[arg[:-2]]
                if tmp is None:  # pragma: no cover
                    message_args_factorToVar.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if size[1 - idx] is None:
                            size[1 - idx] = tmp[1 - idx].size(0)
                        if size[1 - idx] != tmp[1 - idx].size(0):
                            raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args_factorToVar.append(tmp)
            else:
                message_args_factorToVar.append(kwargs[arg])

        message_args_varToFactor = []
        for arg in self.__message_args_varToFactor__:
            if arg[-2:] in ij.keys():
                tmp = kwargs[arg[:-2]]
                if tmp is None:  # pragma: no cover
                    message_args_varToFactor.append(tmp)
                else:
                    idx = ij[arg[-2:]]
                    if isinstance(tmp, tuple) or isinstance(tmp, list):
                        assert len(tmp) == 2
                        if size[1 - idx] is None:
                            size[1 - idx] = tmp[1 - idx].size(0)
                        if size[1 - idx] != tmp[1 - idx].size(0):
                            raise ValueError(__size_error_msg__)
                        tmp = tmp[idx]

                    if size[idx] is None:
                        size[idx] = tmp.size(0)
                    if size[idx] != tmp.size(0):
                        raise ValueError(__size_error_msg__)

                    tmp = torch.index_select(tmp, 0, edge_index[idx])
                    message_args_varToFactor.append(tmp)
            else:
                message_args_varToFactor.append(kwargs[arg])

        size[0] = size[1] if size[0] is None else size[0]
        size[1] = size[0] if size[1] is None else size[1]

        kwargs['edge_index'] = edge_index
        kwargs['size'] = size

        for (idx, arg) in self.__special_args__:
            if arg[-2:] in ij.keys():
                message_args_factorToVar.insert(idx, kwargs[arg[:-2]][ij[arg[-2:]]])
            else:
                message_args_factorToVar.insert(idx, kwargs[arg])

        update_args = [kwargs[arg] for arg in self.__update_args__]

        #update variable beliefs
        factorToVar_messages = self.message_factorToVar(*message_args_factorToVar)
        new_factor_beliefs = scatter_('add', factorToVar_messages, edge_index[i], dim_size=size[i])


        #update factor beliefs
        varToFactor_expandedMessages = self.message_varToFactor(*message_args_factorToVar)
        new_factor_beliefs = scatter_('add', varToFactor_expandedMessages, edge_index[i], dim_size=size[i])
        new_factor_beliefs += factor_potentials #factor_potentials previously x_base

#        print("cur_messages:", cur_messages)
#        print("aggregated_messages:", aggregated_messages)

        return new_state, cur_messages

    def message_factorToVar(self, x_j, prv_messages, edge_var_indices, state_dimensions):
        # note, we assume each edge has a matching edge in the opposite direction (e.g. edge (i,j) has a corresponding edge (j,i)) and each pair is stored at locations i and i + 1 where i is even
        # x_j has shape [E, X.shape] (double check)
        # prv_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at

        # b = torch.tensor([[[2, -np.inf], [-np.inf, -np.inf]], [[4, -np.inf], [-np.inf, -np.inf]]])
        # var_idx = 0
        # torch.logsumexp(b, dim=tuple([i for i in range(len(b.shape)) if i != var_idx]))

        # marginalized_states = torch.logsumexp(x_j[node_idx], dim=tuple([i for i in range(len(b.shape)) if i != var_idx]))

        # print("torch.exp(x_j):", torch.exp(x_j))
        # print("prv_messages:", prv_messages)
        # print("edge_var_indices:", edge_var_indices)
        # print("state_dimensions:", state_dimensions)

        # tensor with dimensions [edges, 2] for binary variables
        marginalized_states = torch.stack([
            torch.logsumexp(node_state, dim=tuple([i for i in range(len(node_state.shape)) if i != edge_var_indices[0, edge_idx]])) for edge_idx, node_state in enumerate(torch.unbind(x_j, dim=0))
                                          ], dim=0)

        #swap neighboring messages so that each state lines up with the message it sent, rather than received        
        assert(prv_messages.shape[0] % 2 == 0)
        swap_indices = [i+1 if (i % 2 == 0) else i-1 for i in range(prv_messages.shape[0])]
        destination_to_source_messages = torch.zeros_like(prv_messages)
        destination_to_source_messages[swap_indices] = prv_messages
        #swap axes so that the single variable of interest is on the correct dimension for the sending node rather than the receiving node
        destination_to_source_messages_sourceVarOrder = torch.stack([
            message_state.transpose(edge_var_indices[1, message_idx], edge_var_indices[0, message_idx]) for message_idx, message_state in enumerate(torch.unbind(destination_to_source_messages, dim=0))
                                          ], dim=0)
        #to avoid dividing by zero messages use: (could assert that corresponding states are -infinity too)
        destination_to_source_messages_sourceVarOrder[destination_to_source_messages_sourceVarOrder == -np.inf] = 0
        # print("torch.exp(destination_to_source_messages_sourceVarOrder):", torch.exp(destination_to_source_messages_sourceVarOrder))

        #avoid double counting
        messages = marginalized_states - destination_to_source_messages_sourceVarOrder

        # print("torch.exp(messages):", torch.exp(messages))
#        print("destination_to_source_messages_sourceVarOrder:", torch.exp(destination_to_source_messages_sourceVarOrder))
#        print("messages:", torch.exp(messages))


        return messages

    def message_varToFactor(self, x_j, prv_messages, edge_var_indices, state_dimensions):
        # note, we assume each edge has a matching edge in the opposite direction (e.g. edge (i,j) has a corresponding edge (j,i)) and each pair is stored at locations i and i + 1 where i is even
        # x_j has shape [E, X.shape] (double check)
        # prv_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at

        # b = torch.tensor([[[2, -np.inf], [-np.inf, -np.inf]], [[4, -np.inf], [-np.inf, -np.inf]]])
        # var_idx = 0
        # torch.logsumexp(b, dim=tuple([i for i in range(len(b.shape)) if i != var_idx]))

        # marginalized_states = torch.logsumexp(x_j[node_idx], dim=tuple([i for i in range(len(b.shape)) if i != var_idx]))

        # print("torch.exp(x_j):", torch.exp(x_j))
        # print("prv_messages:", prv_messages)
        # print("edge_var_indices:", edge_var_indices)
        # print("state_dimensions:", state_dimensions)

        # print("torch.exp(x_j):", torch.exp(x_j))


        # print("torch.exp(x_j):", torch.exp(x_j))
        # print("prv_messages:", prv_messages)
        # print("torch.exp(prv_messages):", torch.exp(prv_messages))

        # divide previous messages to avoid double counting
        # might have dimensions mismatch !!
#        print("x_j.shape:", x_j.shape)
#        print("prv_messages.shape:", prv_messages.shape)

        #swap neighboring messages so that each state lines up with the message it sent, rather than received        
        assert(prv_messages.shape[0] % 2 == 0)
        swap_indices = [i+1 if (i % 2 == 0) else i-1 for i in range(prv_messages.shape[0])]
        destination_to_source_messages = torch.zeros_like(prv_messages)
        destination_to_source_messages[swap_indices] = prv_messages


        #swap axes so that the single variable of interest is on the correct dimension for the sending node rather than the receiving node
        destination_to_source_messages_sourceVarOrder = torch.stack([
            message_state.transpose(edge_var_indices[1, message_idx], edge_var_indices[0, message_idx]) for message_idx, message_state in enumerate(torch.unbind(destination_to_source_messages, dim=0))
                                          ], dim=0)
        #to avoid dividing by zero messages use: (could assert that corresponding states are -infinity too)
        destination_to_source_messages_sourceVarOrder[destination_to_source_messages_sourceVarOrder == -np.inf] = 0
        # print("torch.exp(destination_to_source_messages_sourceVarOrder):", torch.exp(destination_to_source_messages_sourceVarOrder))

        messages = x_j - destination_to_source_messages_sourceVarOrder
        # print("torch.exp(messages):", torch.exp(messages))

#        print("expanded_states:", torch.exp(expanded_states))
#        print("destination_to_source_messages_sourceVarOrder:", torch.exp(destination_to_source_messages_sourceVarOrder))
#        print("messages:", torch.exp(messages))


        expansion_list = [2 for i in range(state_dimensions - 1)] + [-1,]

        # #debug

        # for message_idx, message_state in enumerate(torch.unbind(x_j, dim=0)):
        #     temp_message_state = message_state.expand(expansion_list)
        #     print("expansion_list:", expansion_list)
        #     print("temp_message_state.shape:", temp_message_state.shape)
        #     print("edge_var_indices[1, message_idx]", edge_var_indices[1, message_idx])
        #     print("state_dimensions-1:", state_dimensions-1)
        #     message_state.expand(expansion_list).transpose(edge_var_indices[1, message_idx], state_dimensions-1)
        # #end debug


        expanded_messages = torch.stack([
            message_state.expand(expansion_list).transpose(edge_var_indices[1, message_idx], state_dimensions-1) for message_idx, message_state in enumerate(torch.unbind(messages, dim=0))
                                          ], dim=0)

        return expanded_messages


    def update(self, aggr_out, x_base):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out + x_base
