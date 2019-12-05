import torch
import numpy as np
# from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data

# import sys
# sys.path.insert(0, "/atlas/u/jkuck/pytorch_geometric/torch_geometric/nn/conv")
from message_passing_no_double_counting import MessagePassing_NoDoubleCounting

from collections import defaultdict

import mrftools

# $ cd /atlas/u/jkuck/virtual_environments/pytorch_geometric
# $ source bin/activate
# # cd /atlas/u/jkuck/pytorch_geometric/jdk_examples


class LoopyBP(MessagePassing_NoDoubleCounting):
    def __init__(self, in_channels, out_channels):
        super(LoopyBP, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, edge_index, prv_messages):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # prv_messages has shape [E], e.g. one message for each edge on the last iteration

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, prv_messages=prv_messages)

    def message(self, x_j, prv_messages):
        # x_j has shape [E, out_channels]
        # prv_messages has shape [E], e.g. one message for each edge on the last iteration

        # divide previous messages to avoid double counting, take log to use addition
        return x_j - prv_messages

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out


class LoopyBP_ForSAT(MessagePassing_NoDoubleCounting):
    '''

    edge_attr: must store:
        - the variable index in that particular clause, 0 to clause_degree - 1

    x: (node states) must store:
        - 2 states for variable nodes
        - 2^clause_degree states for clause nodes



    '''
    def __init__(self, in_channels, out_channels):
        super(LoopyBP_ForSAT, self).__init__(aggr='add')  # "Add" aggregation.

    def forward(self, x, x_base, edge_index, prv_messages, edge_var_indices, state_dimensions):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # prv_messages has shape [E], e.g. one message for each edge on the last iteration
        # state_dimensions (int) the largest node degree

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, prv_messages=prv_messages,\
                              edge_var_indices=edge_var_indices, state_dimensions=state_dimensions, x_base=x_base)

    def message(self, x_j, prv_messages, edge_var_indices, state_dimensions):
        # note, we assume each edge has a matching edge in the opposite direction (e.g. edge (i,j) has a corresponding edge (j,i)) and each pair is stored at locations i and i + 1 where i is even
        # x_j has shape [E, X.shape] (double check)
        # prv_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of the edge among all edges originating at the node edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of the edge among all edges ending at the node edge i ends at

        # b = torch.tensor([[[2, -np.inf], [-np.inf, -np.inf]], [[4, -np.inf], [-np.inf, -np.inf]]])
        # var_idx = 0
        # torch.logsumexp(b, dim=tuple([i for i in range(len(b.shape)) if i != var_idx]))

        # marginalized_states = torch.logsumexp(x_j[node_idx], dim=tuple([i for i in range(len(b.shape)) if i != var_idx]))

        # tensor with dimensions [edges, 2] for binary variables
        marginalized_states = torch.stack([
            torch.logsumexp(node_state, dim=tuple([i for i in range(len(node_state.shape)) if i != edge_var_indices[0, edge_idx]])) for edge_idx, node_state in enumerate(torch.unbind(x_j, dim=0))
                                          ], dim=0)


        expansion_list = [2 for i in range(state_dimensions - 1)] + [-1,]

        # #debug

        # for message_idx, message_state in enumerate(torch.unbind(marginalized_states, dim=0)):
        #     temp_message_state = message_state.expand(expansion_list)
        #     print("expansion_list:", expansion_list)
        #     print("temp_message_state.shape:", temp_message_state.shape)
        #     print("edge_var_indices[1, message_idx]", edge_var_indices[1, message_idx])
        #     print("state_dimensions-1:", state_dimensions-1)
        #     message_state.expand(expansion_list).transpose(edge_var_indices[1, message_idx], state_dimensions-1)
        # #end debug


        expanded_states = torch.stack([
            message_state.expand(expansion_list).transpose(edge_var_indices[1, message_idx], state_dimensions-1) for message_idx, message_state in enumerate(torch.unbind(marginalized_states, dim=0))
                                          ], dim=0)


        # divide previous messages to avoid double counting
        # might have dimensions mismatch !!
#        print("expanded_states.shape:", expanded_states.shape)
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

        messages = expanded_states - destination_to_source_messages_sourceVarOrder

#        print("expanded_states:", torch.exp(expanded_states))
#        print("destination_to_source_messages_sourceVarOrder:", torch.exp(destination_to_source_messages_sourceVarOrder))
#        print("messages:", torch.exp(messages))


        return messages


    def update(self, aggr_out, x_base):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out + x_base


def test_LoopyBP():
    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1]], dtype=torch.long)
    x = torch.tensor([[2], [1], [1]], dtype=torch.float)
    edge_attr =  torch.tensor([[1], [1], [1], [1]], dtype=torch.float)

    data = Data(x=torch.log(x), edge_index=edge_index.t().contiguous(), edge_attr=torch.log(edge_attr))

    conv = LoopyBP(1, 1)

    for itr in range(20):
        cur_state, data.edge_attr = conv(data.x, data.edge_index, prv_messages=data.edge_attr)        
        print('data.x:', torch.exp(data.x))
        print('data.xcur_state:', torch.exp(cur_state))
        print('data.edge_attr:', torch.exp(data.edge_attr))
        print('-'*80)
        print()


def test_LoopyBP_ForSAT():
    edge_index = torch.tensor([[0, 1],
                               [1, 0],
                               [1, 2],
                               [2, 1]], dtype=torch.long)
    x = torch.tensor([[[1, 0],
                       [1, 0],],
                      [[1, 1],
                       [0, 1],],
                      [[1, 0],
                       [1, 0],]], dtype=torch.float)

    x_base = torch.tensor([[[1, 0],
                       [1, 0],],
                      [[1, 1],
                       [0, 1],],
                      [[1, 0],
                       [1, 0],]], dtype=torch.float)    

    edge_var_indices = torch.tensor([[0, 0, 1, 0],
                                     [0, 0, 0, 1]])

    edge_attr = torch.tensor([[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]], dtype=torch.float)

    data = Data(x=torch.log(x), edge_index=edge_index.t().contiguous(), edge_attr=torch.log(edge_attr))

    conv = LoopyBP_ForSAT(1, 1)

    print('data.x:', torch.exp(data.x))
    print('data.edge_attr:', torch.exp(data.edge_attr))
    print('-'*80)
    print()


    for itr in range(5):
        data.x, data.edge_attr = conv(x=data.x, x_base=torch.log(x_base), edge_index=data.edge_index, prv_messages=data.edge_attr,\
                                         edge_var_indices=edge_var_indices, state_dimensions=2)        
        print('data.x:', torch.exp(data.x))
        print('data.edge_attr:', torch.exp(data.edge_attr))
        print('-'*80)
        print()

def test_LoopyBP_ForSAT2():
    edge_index = torch.tensor([[3, 0],
                               [0, 3],
                               [4, 0],
                               [0, 4],
                               [5, 0],
                               [0, 5],
                               [4, 1],
                               [1, 4],
                               [6, 1],
                               [1, 6],
                               [5, 2],
                               [2, 5]], dtype=torch.long)
    x = torch.tensor([[[[1, 1],
                        [0, 1],],
                       [[1, 1],
                        [1, 1],]],
                      [[[1, 0],
                        [1, 0],],
                       [[0, 0],
                        [1, 0],]],
                      [[[0, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]],
                      [[[1, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]],                        
                      [[[1, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]],  
                      [[[1, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]],  
                      [[[1, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]]], dtype=torch.float)

    x_base = torch.tensor([[[[1, 1],
                        [0, 1],],
                       [[1, 1],
                        [1, 1],]],
                      [[[1, 0],
                        [1, 0],],
                       [[0, 0],
                        [1, 0],]],
                      [[[0, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]],
                      [[[1, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]],                        
                      [[[1, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]],  
                      [[[1, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]],  
                      [[[1, 0],
                        [0, 0],],
                       [[1, 0],
                        [0, 0],]]], dtype=torch.float)

    edge_var_indices = torch.tensor([[0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0],
                                     [0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0]])

    edge_attr = torch.tensor([[[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]], [[[1, 1], [1, 1]], [[1, 1], [1, 1]]]], dtype=torch.float)

    data = Data(x=torch.log(x), edge_index=edge_index.t().contiguous(), edge_attr=torch.log(edge_attr))

    conv = LoopyBP_ForSAT(1, 1)

    print('data.x:', torch.exp(data.x))
    print('data.edge_attr:', torch.exp(data.edge_attr))
    print('-'*80)
    print()


    for itr in range(5):
        data.x, data.edge_attr = conv(x=data.x, x_base=torch.log(x_base), edge_index=data.edge_index, prv_messages=data.edge_attr,\
                                         edge_var_indices=edge_var_indices, state_dimensions=3)        
        print('data.x:', torch.exp(data.x))
        print('data.edge_attr:', torch.exp(data.edge_attr))
        print('-'*80)
        print()


def build_clause_node_state(clause, state_dimensions):
    '''
    The ith variable in a clause corresponds to the ith dimension in the tensor representation of the state.
    '''
    #Create a tensor for the 2^state_dimensions states
    state = torch.zeros([2 for i in range(state_dimensions)])
    #Iterate over the 2^state_dimensions variable assignments and set those to 1 that satisfy the clause
    for indices in np.ndindex(state.shape):
        junk_location = False
        for dimension in range(len(clause), state_dimensions):
            if indices[dimension] == 1:
                junk_location = True #this dimension is unused by this clause, set to 0
        if junk_location:
            continue
        set_to_1 = False
        for dimension, index_val in enumerate(indices):
            if dimension >= len(clause):
                break
            if clause[dimension] > 0 and index_val == 1:
                set_to_1 = True
            elif clause[dimension] < 0 and index_val == 0:
                set_to_1 = True
        if set_to_1:
            state[indices] = 1
    return state

def test_build_clause_node_state():
    for new_dimensions in range(3,7):
        state = build_clause_node_state([1, -2, 3], new_dimensions)
        print(state)
        for indices in np.ndindex(state.shape):
            print(tuple(reversed(indices)), state[tuple(reversed(indices))])

def build_variable_node_state(state_dimensions):
    #This is the same for any variable node
    #Create a tensor for the 2^state_dimensions states
    state = torch.zeros([2 for i in range(state_dimensions)])
    #set 2 values in first dimension to 1
    state[tuple([0] + [0 for i in range(state_dimensions - 1)])] = 1
    state[tuple([1] + [0 for i in range(state_dimensions - 1)])] = 1
    assert(torch.sum(state) == 2)
    return state

def build_edge_var_indices(clauses, max_clause_degree=None):
    print("max_clause_degree:", max_clause_degree)
    indices_at_source_node = []
    indices_at_destination_node = []
    for clause in clauses:
        for var_idx in range(len(clause)):
            indices_at_source_node.append(0)
            indices_at_source_node.append(var_idx)
            indices_at_destination_node.append(var_idx)
            indices_at_destination_node.append(0)
            if max_clause_degree is not None:
                assert(var_idx < max_clause_degree), (var_idx, max_clause_degree)
    edge_var_indices = torch.tensor([indices_at_source_node, indices_at_destination_node])
    return edge_var_indices

def test_build_edge_var_indices():
    clauses = [[1, -2, 3], [-2, 4], [3]]
    edge_var_indices = build_edge_var_indices(clauses)
    expected_edge_var_indices = torch.tensor([[0, 0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0],
                                              [0, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0]])
    assert(torch.all(torch.eq(edge_var_indices, expected_edge_var_indices)))
    print(edge_var_indices)
    print(expected_edge_var_indices)

def build_graph_from_SAT_problem(clauses):
    '''
    Take a SAT problem in CNF form (specified by clauses) and return a graph representation
    whose partition function is the number of satisfying solutions

    Inputs:
    - clauses: (list of list of ints) variables should be numbered 1 to N with no gaps
    '''
    num_clauses = len(clauses)
    edge_index_list = []
    dictionary_of_vars = defaultdict(int)
    for clause_idx, clause in enumerate(clauses):
        for literal in clause:
            var_node_idx = num_clauses - 1 + np.abs(literal)
            edge_index_list.append([var_node_idx, clause_idx])
            edge_index_list.append([clause_idx, var_node_idx])
            dictionary_of_vars[np.abs(literal)] += 1

    print("a")

    # check variables are numbered 1 to N with no gaps
    N = -1 #number of variables
    max_var_degree = -1
    for var_name, var_degree in dictionary_of_vars.items():
        if var_name > N:
            N = var_name
        if var_degree > max_var_degree:
            max_var_degree = var_degree
    assert(N == len(dictionary_of_vars))

    print("b")

    # get largest clause degree
    max_clause_degree = -1
    for clause in clauses:
        if len(clause) > max_clause_degree:
            max_clause_degree = len(clause)
    # state_dimensions = max(max_clause_degree, max_var_degree)
    state_dimensions = max_clause_degree

    edge_index = torch.tensor(edge_index_list, dtype=torch.long)

    print("c")


####################    # create local indices of variable nodes for each clause node
####################    # the first variable appearing in a clause has index 0, the second variable
####################    # appearing in a clause has index 1, etc.  It is important to keep track
####################    # of this because these variable indices are different between clauses
####################
####################    clause_node_variable_indices = []


    x = torch.stack([build_clause_node_state(clause=clause, state_dimensions=state_dimensions) for clause in clauses] +\
                    [build_variable_node_state(state_dimensions=state_dimensions) for i in range(N)], dim=0)

   
    print("d")


    x_base = torch.zeros_like(x)
    x_base.copy_(x)

    edge_var_indices = build_edge_var_indices(clauses, max_clause_degree=max_clause_degree)
    print("state_dimensions:", state_dimensions)

    edge_count = edge_var_indices.shape[1]
    edge_attr = torch.stack([torch.ones([2 for i in range(state_dimensions)]) for j in range(edge_count)], dim=0)
    edge_attr = torch.rand_like(edge_attr)

    data = Data(x=torch.log(x), edge_index=edge_index.t().contiguous(), edge_attr=torch.log(edge_attr))

    return data, x_base, edge_var_indices, state_dimensions

def test_LoopyBP_ForSAT_automatedGraphConstruction():
    # clauses = [[1, -2, 3], [-2, 4], [3]] # count=6
    clauses = [[1, -2], [-2, 4], [3]] # count=5
    # clauses = [[1, 2], [-2, 3]] # count=4
    # clauses = [[1, 2]] # count=3

    data, x_base, edge_var_indices, state_dimensions = build_graph_from_SAT_problem(clauses)

    conv = LoopyBP_ForSAT(1, 1)

    print('data.x:', torch.exp(data.x))
    print('data.edge_attr:', torch.exp(data.edge_attr))
    print('-'*80)
    print()


    for itr in range(10):
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

def parse_dimacs(filename):
    clauses = []
    dictionary_of_vars = defaultdict(int)    
    with open(filename, 'r') as f:    
        for line in f:
            line_as_list = line.strip().split(" ")
            if line_as_list[0] == "p":
                n_vars = int(line_as_list[2])
                n_clauses = int(line_as_list[3])
            elif line_as_list[0] == "c":
                continue
            else:
                cur_clause = [int(s) for s in line_as_list[:-1]]
                for var in cur_clause:
                    dictionary_of_vars[int(abs(var))] += 1
                clauses.append(cur_clause)
    # assert(n_clauses == len(clauses)), (n_clauses, len(clauses), filename)
    if(n_clauses != len(clauses)):
        print("actual clause count doesn't match expected clause count!!")
    #check variables are numbered 1 to n_vars with no gaps        
    num_vars_check = -1
    for var_name, var_degree in dictionary_of_vars.items():
        if var_name > num_vars_check:
            num_vars_check = var_name
    assert(num_vars_check == n_vars) #make sure largest variable is named n_vars
    assert(len(dictionary_of_vars) == n_vars) #make sure we actually have this many variables
    return n_vars, clauses

def run_LoopyBP_ForSAT_onCNF(filename):
    n_vars, clauses = parse_dimacs(filename)
    print('building graph')
    data, x_base, edge_var_indices, state_dimensions = build_graph_from_SAT_problem(clauses)
    print('graph built')

    conv = LoopyBP_ForSAT(1, 1)

    # print('data.x:', torch.exp(data.x))
    # print('data.edge_attr:', torch.exp(data.edge_attr))
    # print('-'*80)
    # print()


    for itr in range(50000):
        data.x, data.edge_attr = conv(x=data.x, x_base=torch.log(x_base), edge_index=data.edge_index, prv_messages=data.edge_attr,\
                                         edge_var_indices=edge_var_indices, state_dimensions=state_dimensions)        
        # print('data.x:', torch.exp(data.x))
        # print('data.edge_attr:', torch.exp(data.edge_attr))
        # print("partition function estimates:")
        for state in data.x:
            log_Z_est = torch.logsumexp(state, dim=tuple(range(len(state.shape))))
            print('log(Z):', log_Z_est, 'Z:', torch.exp(log_Z_est),)
        print()
        print('-'*80)
        print()


def calculate_partition_function_exact():
    # clauses = [[1, -2, 3], [-2, 4], [3]] # count=6
    # clauses = [[1, -2], [-2, 4], [3]] # count=5
    # clauses = [[1, 2], [-2, 3]] # count=4
    clauses = [[1, 2]] # count=3

    sat_FactorGraph = mrftools.MarkovNet()

    variables = {}
    for clause in clauses:
        for literal in clause:
            variables[np.abs(literal)] = 1

    for variable, _ in variables.items():
        factor = np.log(np.array([1, 1]))
        sat_FactorGraph.set_unary_factor(variable, factor)

    for clause in clauses:
        if len(clause) == 1:
            literal = clause[0]
            variable = np.abs(literal)
            if literal > 0:
                factor = np.log(np.array([0, 1]))
            else:
                factor = np.log(np.array([1, 0]))
            sat_FactorGraph.set_unary_factor(variable, factor)
        else:
            vars_in_clause = [np.abs(literal) for literal in clause]
            clause_state = np.log(build_clause_node_state(clause=clause, state_dimensions=len(clause)).numpy())
            print("vars_in_clause:", vars_in_clause)
            print("clause_state.shape:", clause_state.shape)
            sat_FactorGraph.set_edge_factor(tuple(vars_in_clause), clause_state)

    sat_FactorGraph.create_matrices()

    bf = mrftools.BruteForce(sat_FactorGraph)
    exact_z = bf.compute_z()
    print('Exact partition sum:', exact_z)


    bp = mrftools.BeliefPropagator(sat_FactorGraph)
    bp.infer(display='full')

    print("Bethe energy functional: %f" % bp.compute_energy_functional())

    print("Brute force log partition function: %f" % np.log(bf.compute_z()))



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
