import random


#Pytorch dataset
class SpinGlassDataset(Dataset):
    def __init__(self, dataset_size, N_min, N_max, f_max, c_max):
        '''
        Inputs:
        - N_min, N_max: ints, each model in the dataset will be a grid with shape (NxN), where 
            N is sampled uniformly from [N_min, N_min+1, ..., N_max]
        - f_max: float, local field parameters (theta_i) for each model in the dataset
            will be drawn uniformly at random from [-f, f] for each node in the grid.  
            f (for each model in the dataset) is drawn uniformly at random from [0, f_max]
        - c: float, coupling parameters (theta_ij) for each model in the dataset
             will be drawn uniformly at random from [0, c) for each edge in 
             the grid. c (for each model in the dataset) is drawn uniformly at random
             from [0, c_max]

        '''

        self.spin_glass_problems = []
        self.ln_partition_functions = []
        for idx in range(dataset_size):
            cur_N = random.randint(N_min, N_max)
            cur_f = np.random.uniform(low=0, high=f_max)
            cur_c = np.random.uniform(low=0, high=c_max)
            cur_sg_model = SpinGlassModel(N=cur_N, f=cur_f, c=cur_c)
            cur_ln_Z = cur_sg_model.junction_tree_libdai()
            sg_as_factor_graph = build_factorgraph_from_SpinGlassModel(cur_sg_model)

            self.spin_glass_problems.append(sg_as_factor_graph)\
            self.ln_partition_functions.append(cur_ln_Z)
        assert(dataset_size == len(self.spin_glass_problems))            
        assert(dataset_size == len(self.ln_partition_functions))

    def __len__(self):
        return len(self.ln_partition_functions)

    def __getitem__(self, index):
        '''
        Outputs:
        - sg_problem (FactorGraph, defined in factor_graph.py): factor graph representation of spin glass problem
        - ln_solution_count (float): natural logarithm(partition function of sg_problem)
        '''
        sg_problem = self.spin_glass_problems[index]
        ln_solution_count = self.ln_partition_functions[index]
        return sg_problem, ln_solution_count

def build_unary_factor(var_idx, f, state_dimensions):
    '''
    create single variable factor for pytorch

    Inputs:
    - var_idx: (int) variable index, 0 to N-1, for this factor's node
    - f: (float) local field at this node

    Outputs:
    - factor (torch.tensor): the factor (already in logspace)
    - mask (torch.tensor): set to 0 for used locations ([:, 0, 0, ..., 0])
         set to 1 for all other unused locations
    '''
    #only use 1 dimension for a single variable factor 
    used_dimension_count = 1
    #initialize to all 0's, -infinity in log space
    state = -np.inf*torch.ones([2 for i in range(state_dimensions)])
    mask = torch.zeros([2 for i in range(state_dimensions)])


    # set state[:, 0, 0, ..., 0] = [-f, f]
    # set all values of mask to 1, except for mask[:, 0, 0, ..., 0]
    for indices in np.ndindex(state.shape):
        junk_location = False
        for dimension in range(used_dimension_count, state_dimensions):
            if indices[dimension] == 1:
                junk_location = True #this dimension is unused by this clause, set to 0
        if junk_location:
            mask[indices] = 1
            continue
        for idx in range(used_dimension_count, len(indices)):
            assert(indices[idx] == 0)
        if indices[0] == 0:
            state[indices] = -f
        elif indices[0] == 1:
            state[indices] = f
        else:
            assert(False)
    return state, mask

def build_pairwise_factor(var_idx, c, state_dimensions):
    '''
    create factor over two variables for pytorch

    Inputs:
    - var_idx: (int) variable index, 0 to N-1, for this factor's node
    - f: (float) local field at this node

    Outputs:
    - factor (torch.tensor): the factor (already in logspace)
    - mask (torch.tensor): set to 0 for used locations ([:, :, 0, ..., 0])
         set to 1 for all other unused locations

    '''
    #only use 1 dimension for a single variable factor 
    used_dimension_count = 2
    #initialize to all 0's, -infinity in log space
    state = -np.inf*torch.ones([2 for i in range(state_dimensions)])
    mask = torch.zeros([2 for i in range(state_dimensions)])


    # set state[:, :, 0, ..., 0] = [[ c, -c],
    #                               [-c,  c]]
    # set all values of mask to 1, except for mask[:, :, 0, 0, ..., 0]
    for indices in np.ndindex(state.shape):
        junk_location = False
        for dimension in range(used_dimension_count, state_dimensions):
            if indices[dimension] == 1:
                junk_location = True #this dimension is unused by this clause, set to 0
        if junk_location:
            mask[indices] = 1
            continue
        for idx in range(used_dimension_count, len(indices)):
            assert(indices[idx] == 0)
        if indices[0] == indices[1]:
            state[indices] = c
        else:
            state[indices] = -c

    return state, mask    


def build_edge_var_indices(sg_model):
    # edge_var_indices has shape [2, E]. 
    #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
    #   [1, i] IS CURRENTLY UNUSED AND BROKEN, BUT IN GENERAL FOR PYTORCH GEOMETRIC SHOULD indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at
    N = sg_model.N

    indices_at_source_node = []
    indices_at_destination_node = []

    # Create edge indices for single variable factors
    for var_idx in range(N**2):
        indices_at_source_node.append(0) #source node is the factor
        indices_at_destination_node.append(-99) #destination node is the variable

    # Create edge indices for horizontal pairwise factors
    for row_idx in range(N):
        for col_idx in range(N-1):
            indices_at_source_node.append(0) #source node is the factor
            indices_at_source_node.append(1) #source node is the factor
            indices_at_destination_node.append(-99) #destination node is the variable
            indices_at_destination_node.append(-99) #destination node is the variable

    # reate edge indices for vertical pairwise factors
    for row_idx in range(N-1):
        for col_idx in range(N):
            indices_at_source_node.append(0) #source node is the factor
            indices_at_source_node.append(1) #source node is the factor
            indices_at_destination_node.append(-99) #destination node is the variable
            indices_at_destination_node.append(-99) #destination node is the variable

    edge_var_indices = torch.tensor([indices_at_source_node, indices_at_destination_node])
    return edge_var_indices

def build_factorgraph_from_SpinGlassModel(sg_model):
    '''
    Convert a spin glass model to a factor graph pytorch representation

    Inputs:
    - sg_model (SpinGlassModel): defines a sping glass model
    Outputs:
    - factorgraph (FactorGraph): or None if there is a clause containing more than max_factor_dimensions variables
    '''
    assert(len(sg_model.fixed_variables) == 0)
    state_dimensions = 2
    N = sg_model.N
    num_vars = N**2
    #(N-1)*N horizontal coupling factors, (N-1)*N vertical coupling factors, and N**2 single variable factors
    num_factors = 2*(N - 1)*N + N**2

    #Variable indexing: variable with indices [row_idx, col_idx] (e.g. with lcl_fld_param given by lcl_fld_params[row_idx,col_idx]) has index row_idx*N+col_idx
    #Factor indexing: 
    #   - single variable factors: have the same index as their variable index 
    #   - horizontal coupling factors: factor between [row_idx, col_idx] and [row_idx, col_idx+1] has index (N**2 + row_idx*(N-1) + col_idx)
    #   - vertical coupling factors: factor between [row_idx, col_idx] and [row_idx+1, col_idx] has index (N**2 + N*(N-1) + row_idx*N + col_idx)
    factorToVar_edge_index_list = []
    # add unary factor to variable edges
    for unary_factor_idx in range(N**2):
        var_idx = unary_factor_idx
        factorToVar_edge_index_list.append([unary_factor_idx, var_idx])
    # add horizontal factor to variable edges
    for row_idx in range(N):
        for col_idx in range(N-1):
            horizontal_factor_idx = (N**2 + row_idx*(N-1) + col_idx)
            var_idx1 = row_idx*N + col_idx
            factorToVar_edge_index_list.append([horizontal_factor_idx, var_idx1])
            var_idx2 = row_idx*N + col_idx + 1
            factorToVar_edge_index_list.append([horizontal_factor_idx, var_idx2])
    # add vertical factor to variable edges
    for row_idx in range(N-1):
        for col_idx in range(N):
            vertical_factor_idx = (N**2 + N*(N-1) + row_idx*N + col_idx)
            var_idx1 = row_idx*N + col_idx
            factorToVar_edge_index_list.append([vertical_factor_idx, var_idx1])
            var_idx2 = (row_idx+1)*N + col_idx
            factorToVar_edge_index_list.append([vertical_factor_idx, var_idx2])
    assert(len(factorToVar_edge_index_list) == 4*(N - 1)*N + N**2)
    # - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge
    #     matrix with shape :obj:`[numFactors, numVars]`
    factorToVar_edge_index = torch.tensor(factorToVar_edge_index_list, dtype=torch.long)

    # Create pytorch tensor factors for each single variable factor
    for var_idx in range(N**2):
        r = var_idx//N
        c = var_idx%N
        state, mask = build_single_node_factor(mn, fixed_variables, var_idx, f=sg_model.lcl_fld_params[r,c])
        states.append(state)
        masks.append(mask)

    # Create pytorch tensor factors for each horizontal pairwise factor
    for row_idx in range(N):
        for col_idx in range(N-1):
            var_idx1 = row_idx*N + col_idx
            var_idx2 = row_idx*N + col_idx + 1
            state, mask = build_pairwise_factor(mn, fixed_variables, var_idx1=var_idx1, var_idx2=var_idx2, c=sg_model.cpl_params_h[row_idx,col_idx])
            states.append(state)
            masks.append(mask)

    # Create pytorch tensor factors for each vertical pairwise factor
    for row_idx in range(N-1):
        for col_idx in range(N):
            var_idx1 = row_idx*N + col_idx
            var_idx2 = (row_idx+1)*N + col_idx
            state, mask = build_pairwise_factor(mn, fixed_variables, var_idx1=var_idx1, var_idx2=var_idx2, c=sg_model.cpl_params_v[row_idx,col_idx])
            states.append(state)
            masks.append(mask)

    factor_potentials = torch.stack(states, dim=0)
    factor_potential_masks = torch.stack(masks, dim=0)

    edge_var_indices = build_edge_var_indices(clauses, max_clause_degree=max_clause_degree)
    # print("state_dimensions:", state_dimensions)

    edge_count = edge_var_indices.shape[1]



    factor_graph = FactorGraph(factor_potentials=factor_potentials,
                 factorToVar_edge_index=factorToVar_edge_index.t().contiguous(), numVars=num_vars, numFactors=num_factors, 
                 edge_var_indices=edge_var_indices, state_dimensions=state_dimensions, factor_potential_masks=factor_potential_masks)

    return factor_graph