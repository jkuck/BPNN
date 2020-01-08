def build_factorPotential_fromClause(clause, state_dimensions):
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

def test_build_factorPotential_fromClause():
    for new_dimensions in range(3,7):
        state = build_factorPotential_fromClause([1, -2, 3], new_dimensions)
        print(state)
        for indices in np.ndindex(state.shape):
            print(tuple(reversed(indices)), state[tuple(reversed(indices))])


def build_factorgraph_from_SATproblem(clauses):
    '''
    Take a SAT problem in CNF form (specified by clauses) and return a factor graph representation
    whose partition function is the number of satisfying solutions

    Inputs:
    - clauses: (list of list of ints) variables should be numbered 1 to N with no gaps
    '''
    num_clauses = len(clauses)
    factorToVar_edge_index_list = []
    varToFactor_edge_index_list = []
    dictionary_of_vars = defaultdict(int)
    for clause_idx, clause in enumerate(clauses):
        for literal in clause:
            var_node_idx = np.abs(literal) - 1
            varToFactor_edge_index_list.append([var_node_idx, clause_idx])
            factorToVar_edge_index_list.append([clause_idx, var_node_idx])
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

    factorToVar_edge_index = torch.tensor(factorToVar_edge_index_list, dtype=torch.long)
    varToFactor_edge_index = torch.tensor(varToFactor_edge_index_list, dtype=torch.long)

    print("c")


####################    # create local indices of variable nodes for each clause node
####################    # the first variable appearing in a clause has index 0, the second variable
####################    # appearing in a clause has index 1, etc.  It is important to keep track
####################    # of this because these variable indices are different between clauses
####################
####################    clause_node_variable_indices = []


    factor_potentials = torch.stack([build_factorPotential_fromClause(clause=clause, state_dimensions=state_dimensions) for clause in clauses], dim=0)

   
    print("d")


    x_base = torch.zeros_like(factor_potentials)
    x_base.copy_(factor_potentials)

    edge_var_indices = build_edge_var_indices(clauses, max_clause_degree=max_clause_degree)
    print("state_dimensions:", state_dimensions)

    edge_count = edge_var_indices.shape[1]

    edge_attr = torch.stack([torch.ones([2 for i in range(state_dimensions)]) for j in range(edge_count)], dim=0)
    if INITIALIZE_MESSAGES_RANDOMLY:
        edge_attr = torch.rand_like(edge_attr)

    data = Data(factor_potentials=torch.log(factor_potentials), factorToVar_edge_index=factorToVar_edge_index.t().contiguous(),\
                varToFactor_edge_index=varToFactor_edge_index.t().contiguous(), edge_attr=torch.log(edge_attr))

    return data, x_base, edge_var_indices, state_dimensions