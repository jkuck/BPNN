import torch
import numpy as np
from torch_geometric.utils import scatter_
from models import build_factorgraph_from_SATproblem

import mrftools


__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')


def map_beliefs(factor_graph, map_type):
    '''
    Utility function for propogate().  Maps factor or variable beliefs to edges 
    See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    Inputs:
    - factor_graph: (FactorGraph, defined in models.py) the factor graph whose beliefs we are mapping
    - map_type: (string) 'factor' or 'var' denotes mapping factor or variable beliefs respectively
    '''
    if map_type == 'factor':
        beliefs = factor_graph.factor_beliefs
    elif map_type == 'var':
        beliefs = factor_graph.var_beliefs
    else:
        assert(False), "Error, incorrect map_type"
    size = [factor_graph.numFactors, factor_graph.numVars]


    mapping_indices = {"factor": 0, "var": 1}

    idx = mapping_indices[map_type]
    if isinstance(beliefs, tuple) or isinstance(beliefs, list):
        assert len(beliefs) == 2
        if size[1 - idx] != beliefs[1 - idx].size(0):
            raise ValueError(__size_error_msg__)
        mapped_beliefs = beliefs[idx]
    else:
        mapped_beliefs = beliefs.clone()

    if size is not None and size[idx] != mapped_beliefs.size(0):
        raise ValueError(__size_error_msg__)

    mapped_beliefs = torch.index_select(mapped_beliefs, 0, factor_graph.factorToVar_edge_index[idx])
    return mapped_beliefs



class FactorGraphMsgPassingLayer_NoDoubleCounting(torch.nn.Module):
    r"""Perform message passing in factor graphs without 'double counting'
    i.e. divide by previously sent messages as in loopy belief propagation
    """

    def __init__(self):
        super(FactorGraphMsgPassingLayer_NoDoubleCounting, self).__init__()



    def forward(self, factor_graph):
        '''
        Inputs:
        - factor_graph: (FactorGraph, defined in models.py) the factor graph we will perform one
            iteration of message passing on.
        '''
        # Step 3-5: Start propagating messages.
        self.propagate(factor_graph)


    def propagate(self, factor_graph, debug=False):
        r"""Perform one iteration of message passing.  Pass messages from factors to variables, then
        from variables to factors.

        Inputs:
        - factor_graph: (FactorGraph, defined in models.py) the factor graph we will perform one
            iteration of message passing on.

        Outputs:
        - none, but the state of the factor graph will be updated, specifically the following will be changed:
            factor_graph.prv_varToFactor_messages
            factor_graph.prv_factorToVar_messages
            factor_graph.factor_beliefs
            factor_graph.var_beliefs
        """

        #update variable beliefs
        factorToVar_messages = self.message_factorToVar(factor_graph)
        factor_graph.var_beliefs = scatter_('add', factorToVar_messages, factor_graph.factorToVar_edge_index[1], dim_size=factor_graph.numVars)
        factor_graph.prv_factorToVar_messages = factorToVar_messages

        #update factor beliefs
        varToFactor_messages = self.message_varToFactor(factor_graph)
        expansion_list = [2 for i in range(factor_graph.state_dimensions - 1)] + [-1,] #messages have states for one variable, add dummy dimensions for the other variables in factors
        varToFactor_expandedMessages = torch.stack([
            # message_state.expand(expansion_list).transpose(factor_graph.edge_var_indices[0, message_idx], factor_graph.edge_var_indices[0, message_idx]) for message_idx, message_state in enumerate(torch.unbind(varToFactor_messages, dim=0))
            message_state.expand(expansion_list).transpose(factor_graph.edge_var_indices[0, message_idx], factor_graph.state_dimensions-1) for message_idx, message_state in enumerate(torch.unbind(varToFactor_messages, dim=0))
                                          ], dim=0)


        #debug
        if debug:
            print("factor_graph.edge_var_indices[0]", factor_graph.edge_var_indices[0])
            for message_idx, message_state in enumerate(torch.unbind(varToFactor_messages, dim=0)):
                print("###")
                print("message_idx:", message_idx)
                print("factor_graph.edge_var_indices[0, message_idx]:", factor_graph.edge_var_indices[0, message_idx])
                print("factor_graph.state_dimensions-1:", factor_graph.state_dimensions-1)
                print("factor_graph.edge_var_indices[0, message_idx]:", factor_graph.edge_var_indices[0, message_idx])
                print("factor_graph.edge_var_indices[1, message_idx]:", factor_graph.edge_var_indices[1, message_idx])
                print("message_state.expand(expansion_list):", torch.exp(message_state.expand(expansion_list)))
                print(torch.exp(message_state.expand(expansion_list).transpose(factor_graph.edge_var_indices[1, message_idx], factor_graph.edge_var_indices[0, message_idx])))
                print
            print("varToFactor_expandedMessages:", torch.exp(varToFactor_expandedMessages))
        #end debug

        factor_graph.factor_beliefs = scatter_('add', varToFactor_expandedMessages, factor_graph.factorToVar_edge_index[0], dim_size=factor_graph.numFactors)
        if debug:
            print("1 factor_graph.factor_beliefs:", torch.exp(factor_graph.factor_beliefs))
        
        factor_graph.factor_beliefs += factor_graph.factor_potentials #factor_potentials previously x_base
        if debug:
            print("3 factor_graph.factor_beliefs:", torch.exp(factor_graph.factor_beliefs))
            print("factor_graph.factorToVar_edge_index[0]:", factor_graph.factorToVar_edge_index[0])
            print("factor_graph.factorToVar_edge_index:", factor_graph.factorToVar_edge_index)
            sleep(debug34)
        
        factor_graph.prv_varToFactor_messages = varToFactor_messages

    def message_factorToVar(self, factor_graph):
        # factor_graph.factor_beliefs has shape [E, X.shape] (double check)
        # factor_graph.prv_varToFactor_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # factor_graph.edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at

        mapped_factor_beliefs = map_beliefs(factor_graph, 'factor')

        # tensor with dimensions [edges, 2] for binary variables
        marginalized_states = torch.stack([
            torch.logsumexp(node_state, dim=tuple([i for i in range(len(node_state.shape)) if i != factor_graph.edge_var_indices[0, edge_idx]])) for edge_idx, node_state in enumerate(torch.unbind(mapped_factor_beliefs, dim=0))
                                          ], dim=0)

        # #debug!!
        # print("factor_graph.factor_beliefs:", factor_graph.factor_beliefs)
        # print("mapped_factor_beliefs:", mapped_factor_beliefs)
        # for edge_idx, node_state in enumerate(torch.unbind(mapped_factor_beliefs, dim=0)):
        #     print("-----")
        #     print("node state:", node_state)
        #     print("dim:", tuple([i for i in range(len(node_state.shape)) if i != factor_graph.edge_var_indices[0, edge_idx]]))
        #     print("logsumexp:", torch.logsumexp(node_state, dim=tuple([i for i in range(len(node_state.shape)) if i != factor_graph.edge_var_indices[0, edge_idx]])))
        #     print()
        # print()
        # print("factorToVar marginalized_states:", torch.exp(marginalized_states))

        # #end debug!!

        if False:
            #SHOULDN'T HAVE TO DO THIS WITH 1d messages??
            #swap axes so that the single variable of interest is on the correct dimension for the sending node rather than the receiving node
            destination_to_source_messages_sourceVarOrder = torch.stack([
                message_state.transpose(factor_graph.edge_var_indices[1, message_idx], factor_graph.edge_var_indices[0, message_idx]) for message_idx, message_state in enumerate(torch.unbind(factor_graph.prv_varToFactor_messages, dim=0))
                                              ], dim=0)

            #to avoid dividing by zero messages use: (could assert that corresponding states are -infinity too)
            destination_to_source_messages_sourceVarOrder[destination_to_source_messages_sourceVarOrder == -np.inf] = 0
            # print("torch.exp(destination_to_source_messages_sourceVarOrder):", torch.exp(destination_to_source_messages_sourceVarOrder))
    
            #avoid double counting
            messages = marginalized_states - destination_to_source_messages_sourceVarOrder
        else:
            prv_varToFactor_messages = factor_graph.prv_varToFactor_messages.clone()
            prv_varToFactor_messages[prv_varToFactor_messages == -np.inf] = 0
            #avoid double counting
            messages = marginalized_states - prv_varToFactor_messages

        # print("factorToVar messages:", messages)
        # sleep(debug12)
        # print("torch.exp(messages):", torch.exp(messages))
#        print("destination_to_source_messages_sourceVarOrder:", torch.exp(destination_to_source_messages_sourceVarOrder))
#        print("messages:", torch.exp(messages))
        return messages

    def message_varToFactor(self, factor_graph):
        # factor_graph.var_beliefs has shape [E, X.shape] (double check)
        # factor_graph.prv_factorToVar_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # factor_graph.edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at

        mapped_var_beliefs = map_beliefs(factor_graph, 'var')

        if False:
            #SHOULDN'T HAVE TO DO THIS WITH 1d messages??
            #swap axes so that the single variable of interest is on the correct dimension for the sending node rather than the receiving node
            destination_to_source_messages_sourceVarOrder = torch.stack([
                message_state.transpose(factor_graph.edge_var_indices[1, message_idx], factor_graph.edge_var_indices[0, message_idx]) for message_idx, message_state in enumerate(torch.unbind(factor_graph.prv_factorToVar_messages, dim=0))
                                              ], dim=0)
        
            #to avoid dividing by zero messages use: (could assert that corresponding states are -infinity too)
            destination_to_source_messages_sourceVarOrder[destination_to_source_messages_sourceVarOrder == -np.inf] = 0
            # print("torch.exp(destination_to_source_messages_sourceVarOrder):", torch.exp(destination_to_source_messages_sourceVarOrder))

            messages = mapped_var_beliefs - destination_to_source_messages_sourceVarOrder
        else:
            prv_factorToVar_messages = factor_graph.prv_factorToVar_messages.clone()
            prv_factorToVar_messages[prv_factorToVar_messages == -np.inf] = 0
            #avoid double counting
            messages = mapped_var_beliefs - prv_factorToVar_messages            

        # print("torch.exp(messages):", torch.exp(messages))
#        print("expanded_states:", torch.exp(expanded_states))
#        print("destination_to_source_messages_sourceVarOrder:", torch.exp(destination_to_source_messages_sourceVarOrder))
#        print("messages:", torch.exp(messages))
        return messages

def test_LoopyBP_ForSAT_automatedGraphConstruction():
    # clauses = [[1, -2], [-2, 3], [1, 3]] # count=4
    clauses = [[-1, 2], [1, -2]] # count=2
    # clauses = [[1, -2, 3], [-2, 4], [3]] # count=6
    # clauses = [[1, -2], [-2, 4], [3]] # count=5
    # clauses = [[1, 2], [-2, 3]] # count=4
    # clauses = [[1, 2]] # count=3

    factor_graph = build_factorgraph_from_SATproblem(clauses)

    msg_passing_layer = FactorGraphMsgPassingLayer_NoDoubleCounting()

    # print('data.x:', torch.exp(data.x))
    # print('data.edge_attr:', torch.exp(data.edge_attr))
    # print('-'*80)
    # print()
    # sleep(temp)


    for itr in range(30):
        print('variable beliefs:', torch.exp(factor_graph.var_beliefs))
        print('factor beliefs:', torch.exp(factor_graph.factor_beliefs))
        print('prv_factorToVar_messages:', torch.exp(factor_graph.prv_factorToVar_messages))
        print('prv_varToFactor_messages:', torch.exp(factor_graph.prv_varToFactor_messages))

        msg_passing_layer(factor_graph)        

        # print("partition function estimates:")
        print("#"*10)
    for state in factor_graph.var_beliefs:
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
