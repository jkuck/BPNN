

def build_factorgraph_from_SpinGlassModel(spin_glass_model, initialize_randomly=False, epsilon=0, max_factor_dimensions=5,
                                      local_state_dim=False):
    '''
    

    Inputs:

    - initialize_randomly: (bool) if true randomly initialize beliefs and previous messages
        if false initialize to 1
    - epsilon (float): set states with potential 0 to epsilon for numerical stability
    - max_factor_dimensions (int): do not construct a factor graph if the largest factor (clause) contains
        more than this many variables
    - local_state_dim (bool): if True, then the number of dimensions in each factor is set to the number of 
        variables in the largest clause in /this/ problem.  If False, then the number of dimensions in each factor
        is set to max_factor_dimensions for compatibility with other SAT problems.

    Outputs:
    - factorgraph (FactorGraph): or None if there is a clause containing more than max_factor_dimensions variables
    '''
    num_factors = len(clauses)
    factorToVar_edge_index_list = []
    dictionary_of_vars = defaultdict(int)
    for clause_idx, clause in enumerate(clauses):
        for literal in clause:
            var_node_idx = np.abs(literal) - 1
            factorToVar_edge_index_list.append([clause_idx, var_node_idx])
            dictionary_of_vars[np.abs(literal)] += 1

    # print("a")

    # check largest variable name equals the number of variables
    N = -1 #number of variables
    max_var_degree = -1
    for var_name, var_degree in dictionary_of_vars.items():
        if var_name > N:
            N = var_name
        if var_degree > max_var_degree:
            max_var_degree = var_degree
    if(N != len(dictionary_of_vars)):
        for var_idx in range(1, N+1):
            if var_idx not in dictionary_of_vars:
                print(var_idx, "missing from dictionary_of_vars")
    assert(N == len(dictionary_of_vars)), (N, len(dictionary_of_vars))

    # print("b")

    # get largest clause degree
    max_clause_degree = -1
    for clause in clauses:
        if len(clause) > max_clause_degree:
            max_clause_degree = len(clause)
    if max_clause_degree > max_factor_dimensions:
        return None
    # state_dimensions = max(max_clause_degree, max_var_degree)
    if local_state_dim:
        state_dimensions = max_clause_degree
    else:
        state_dimensions = max_factor_dimensions

    factorToVar_edge_index = torch.tensor(factorToVar_edge_index_list, dtype=torch.long)

    # print("c")


####################    # create local indices of variable nodes for each clause node
####################    # the first variable appearing in a clause has index 0, the second variable
####################    # appearing in a clause has index 1, etc.  It is important to keep track
####################    # of this because these variable indices are different between clauses
####################
####################    clause_node_variable_indices = []

    # factor_potentials = torch.stack([build_factorPotential_fromClause(clause=clause, state_dimensions=state_dimensions, epsilon=epsilon) for clause in clauses], dim=0)
    states = []
    masks = []
    for clause in clauses:
        state, mask = build_factorPotential_fromClause(clause=clause, state_dimensions=state_dimensions, epsilon=epsilon)
        states.append(state)
        masks.append(mask)
    factor_potentials = torch.stack(states, dim=0)
    factor_potential_masks = torch.stack(masks, dim=0)
 


   
    # print("d")


    x_base = torch.zeros_like(factor_potentials)
    x_base.copy_(factor_potentials)

    edge_var_indices = build_edge_var_indices(clauses, max_clause_degree=max_clause_degree)
    # print("state_dimensions:", state_dimensions)

    edge_count = edge_var_indices.shape[1]



    factor_graph = FactorGraph(factor_potentials=torch.log(factor_potentials),
                 factorToVar_edge_index=factorToVar_edge_index.t().contiguous(), numVars=N, numFactors=num_factors, 
                 edge_var_indices=edge_var_indices, state_dimensions=state_dimensions, factor_potential_masks=factor_potential_masks)

    return factor_graph