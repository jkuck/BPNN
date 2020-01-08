import inspect

import torch
from torch_geometric.utils import scatter_
from models import build_factorgraph_from_SATproblem

special_args = [
    'edge_index', 'edge_index_i', 'edge_index_j', 'size', 'size_i', 'size_j'
]
__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')


def map_beliefs(tensor, map_type, factorToVar_edge_index, size):
    '''
    Utility function for propogate().  Maps factor or variable beliefs to edges 
    See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    Inputs:
    - tensor: (torch tensor) with N entries
    - map_type: (string) 'factor' or 'var' denotes mapping factor or variable beliefs respectively
    - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge
        matrix with shape :obj:`[numFactors, numVars]`
    - size (list): [numFactors, numVars]
    '''
    mapping_indices = {"factor": 0, "var": 1}

    idx = mapping_indices[map_type]
    if isinstance(tensor, tuple) or isinstance(tensor, list):
        assert len(tensor) == 2
        if size[1 - idx] != tensor[1 - idx].size(0):
            raise ValueError(__size_error_msg__)
        return_tensor = tensor[idx]
    else
        return_tensor = tensor.clone()

    if size is not None and size[idx] != return_tensor.size(0):
        raise ValueError(__size_error_msg__)

    return_tensor = torch.index_select(return_tensor, 0, factorToVar_edge_index[idx])
    return return_tensor



class MessagePassing_factorGraph(torch.nn.Module):
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


    """

    def __init__(self):
        super(MessagePassing_factorGraph, self).__init__()



    #FIX ME, copied from old code/loopy_BP.py
    def forward(self, x, x_base, edge_index, prv_messages, edge_var_indices, state_dimensions):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # prv_messages has shape [E], e.g. one message for each edge on the last iteration
        # state_dimensions (int) the largest node degree

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, prv_messages=prv_messages,\
                              edge_var_indices=edge_var_indices, state_dimensions=state_dimensions, x_base=x_base)


    def propagate(self, factor_beliefs, prv_varToFactor_messages, factorToVar_edge_index, numVars,\
                  numFactors, factor_potentials, edge_var_indices, state_dimensions):
        r"""The initial call to start propagating messages.

        Args:
            factorToVar_edge_index (Tensor): The indices of a general (sparse) assignment
                matrix with shape :obj:`[N, M]` (can be directed or
                undirected).
            numVars (int): the number of variables in the factor graph
            numFactors (int): the number of factors in the factor graph
            **kwargs: Any additional data which is needed to construct messages
                and to update node embeddings.
        """
        kwargs['factorToVar_edge_index'] = factorToVar_edge_index

        #update variable beliefs
        mapped_factor_beliefs = map_beliefs(factor_beliefs, 'factor', factorToVar_edge_index, size=[numFactors, numVars])
        factorToVar_messages = self.message_factorToVar(mapped_factor_beliefs, prv_varToFactor_messages, edge_var_indices, state_dimensions)
        new_var_beliefs = scatter_('add', factorToVar_messages, factorToVar_edge_index[1], dim_size=numVars)


        #update factor beliefs
        mapped_var_beliefs = map_beliefs(new_var_beliefs, 'var', factorToVar_edge_index, size=[numFactors, numVars])
        varToFactor_messages = self.message_varToFactor(mapped_var_beliefs, factorToVar_messages, edge_var_indices, state_dimensions)
        expansion_list = [2 for i in range(kwargs['state_dimensions'] - 1)] + [-1,] #messages have states for one variable, add dummy dimensions for the other variables in factors
        varToFactor_expandedMessages = torch.stack([
            message_state.expand(expansion_list).transpose(kwargs['edge_var_indices'][1, message_idx], kwargs['state_dimensions']-1) for message_idx, message_state in enumerate(torch.unbind(varToFactor_messages, dim=0))
                                          ], dim=0)

        new_factor_beliefs = scatter_('add', varToFactor_expandedMessages, factorToVar_edge_index[0], dim_size=numFactors)
        new_factor_beliefs += factor_potentials #factor_potentials previously x_base

        return new_var_beliefs, new_factor_beliefs, varToFactor_messages

    def message_factorToVar(self, factor_beliefs, prv_varToFactor_messages, edge_var_indices, state_dimensions):
        # factor_beliefs has shape [E, X.shape] (double check)
        # prv_varToFactor_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at


        # tensor with dimensions [edges, 2] for binary variables
        marginalized_states = torch.stack([
            torch.logsumexp(node_state, dim=tuple([i for i in range(len(node_state.shape)) if i != edge_var_indices[0, edge_idx]])) for edge_idx, node_state in enumerate(torch.unbind(factor_beliefs, dim=0))
                                          ], dim=0)

        #SHOULDN'T HAVE TO DO THIS WITH 1d messages??
        #swap axes so that the single variable of interest is on the correct dimension for the sending node rather than the receiving node
        destination_to_source_messages_sourceVarOrder = torch.stack([
            message_state.transpose(edge_var_indices[1, message_idx], edge_var_indices[0, message_idx]) for message_idx, message_state in enumerate(torch.unbind(prv_varToFactor_messages, dim=0))
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

    def message_varToFactor(self, var_beliefs, prv_factorToVar_messages, edge_var_indices, state_dimensions):
        # var_beliefs has shape [E, X.shape] (double check)
        # prv_factorToVar_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at

        #SHOULDN'T HAVE TO DO THIS WITH 1d messages??
        #swap axes so that the single variable of interest is on the correct dimension for the sending node rather than the receiving node
        destination_to_source_messages_sourceVarOrder = torch.stack([
            message_state.transpose(edge_var_indices[1, message_idx], edge_var_indices[0, message_idx]) for message_idx, message_state in enumerate(torch.unbind(prv_factorToVar_messages, dim=0))
                                          ], dim=0)
        
        #to avoid dividing by zero messages use: (could assert that corresponding states are -infinity too)
        destination_to_source_messages_sourceVarOrder[destination_to_source_messages_sourceVarOrder == -np.inf] = 0
        # print("torch.exp(destination_to_source_messages_sourceVarOrder):", torch.exp(destination_to_source_messages_sourceVarOrder))

        messages = var_beliefs - destination_to_source_messages_sourceVarOrder
        # print("torch.exp(messages):", torch.exp(messages))
#        print("expanded_states:", torch.exp(expanded_states))
#        print("destination_to_source_messages_sourceVarOrder:", torch.exp(destination_to_source_messages_sourceVarOrder))
#        print("messages:", torch.exp(messages))
        return messages

def test_LoopyBP_ForSAT_automatedGraphConstruction():
    # clauses = [[1, -2], [-2, 3], [1, 3]] # count=4
    # clauses = [[1, -2], [-1, 2]] # count=2
    # clauses = [[1, -2, 3], [-2, 4], [3]] # count=6
    # clauses = [[1, -2], [-2, 4], [3]] # count=5
    clauses = [[1, 2], [-2, 3]] # count=4
    # clauses = [[1, 2]] # count=3

    data, x_base, edge_var_indices, state_dimensions = build_factorgraph_from_SATproblem(clauses)

    conv = MessagePassing_factorGraph()

    # print('data.x:', torch.exp(data.x))
    # print('data.edge_attr:', torch.exp(data.edge_attr))
    # print('-'*80)
    # print()
    # sleep(temp)


    for itr in range(3):
        data.x, data.edge_attr = conv(x=data.x, x_base=torch.log(x_base), edge_index=data.edge_index, prv_messages=data.edge_attr,\
                                         edge_var_indices=edge_var_indices, state_dimensions=state_dimensions)        
        print('data.x:', torch.exp(data.x))
        print('data.edge_attr:', torch.exp(data.edge_attr))
        # print("partition function estimates:")
        print("#"*10)
    for state in data.x:
        log_Z_est = torch.logsumexp(state, dim=tuple(range(len(state.shape))))
        print('log(Z):', log_Z_est, 'Z:', torch.exp(log_Z_est))
    print('-'*80)
    print()

if __name__ == "__main__":
    # calculate_partition_function_exact()
    # sleep(BruteForce)

    # test_LoopyBP_ForSAT2()
    test_LoopyBP_ForSAT_automatedGraphConstruction()
    sleep(temp)
    # sleep(34)
    # SAT_filename = "/atlas/u/jkuck/approxmc/counting2/s641_3_2.cnf.gz.no_w.cnf"
    # SAT_filename = '/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/tire-1.cnf'
    SAT_filename = '/atlas/u/jkuck/pytorch_geometric/jdk_examples/SAT_problems/test3.cnf'
    run_LoopyBP_ForSAT_onCNF(filename=SAT_filename)
