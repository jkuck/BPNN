class FactorGraph():
    '''
    Representation of a factor graph
    '''
    def __init__(self, factor_potentials, factorToVar_edge_index, numVars, numFactors, 
                 edge_var_indices, state_dimensions, factor_beliefs, var_beliefs,
                 prv_varToFactor_messages, prv_factorToVar_messages,):
        #potentials defining the factor graph
        self.factor_potentials = factor_potentials 

        #factorToVar_edge_index (Tensor): The indices of a general (sparse) assignment
        #    matrix with shape :obj:`[N, M]` (can be directed or
        #    undirected).
        self.factorToVar_edge_index = factorToVar_edge_index
        self.numVars = numVars
        self.numFactors = numFactors

        # - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge
        #     matrix with shape :obj:`[numFactors, numVars]`
        self.edge_var_indices = edge_var_indices
        # (int) the largest node degree
        self.state_dimensions = state_dimensions

        self.factor_beliefs = factor_beliefs # initially junk
        self.var_beliefs = var_beliefs # initially junk
        self.prv_varToFactor_messages = prv_varToFactor_messages # initially junk
        self.prv_factorToVar_messages = prv_factorToVar_messages # initially junk

        self.aggr = aggr
        assert self.aggr in ['add', 'mean', 'max']

        self.flow = flow
        assert self.flow in ['source_to_target', 'target_to_source']

        self.__message_args__ = inspect.getfullargspec(self.message)[0][1:]
        self.__special_args__ = [(i, arg)
                                 for i, arg in enumerate(self.__message_args__)
                                 if arg in special_args]
        self.__message_args__ = [
            arg for arg in self.__message_args__ if arg not in special_args
        ]
        self.__update_args__ = inspect.getfullargspec(self.update)[0][2:]


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


def build_edge_var_indices(clauses, max_clause_degree=None):
    print("max_clause_degree:", max_clause_degree)
    indices_at_source_node = []
    indices_at_destination_node = []
    for clause in clauses:
        for var_idx in range(len(clause)):
            indices_at_source_node.append(var_idx) #source node is the factor
            indices_at_destination_node.append(0) #destination node is the variable
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

def build_factorgraph_from_SATproblem(clauses):
    '''
    Take a SAT problem in CNF form (specified by clauses) and return a factor graph representation
    whose partition function is the number of satisfying solutions

    Inputs:
    - clauses: (list of list of ints) variables should be numbered 1 to N with no gaps
    '''
    num_clauses = len(clauses)
    factorToVar_edge_index_list = []
    dictionary_of_vars = defaultdict(int)
    for clause_idx, clause in enumerate(clauses):
        for literal in clause:
            var_node_idx = np.abs(literal) - 1
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

    factor_graph = {'factor_log_potentials': torch.log(factor_potentials),
                    'factorToVar_edge_index': factorToVar_edge_index.t().contiguous(),
                    }

    data = Data(factor_potentials=torch.log(factor_potentials), factorToVar_edge_index=factorToVar_edge_index.t().contiguous(),\
                edge_attr=torch.log(edge_attr))

    return data, x_base, edge_var_indices, state_dimensions