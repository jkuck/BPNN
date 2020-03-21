######### Stochastic Block Model Data #########
# This file implements generating community detection data sampled 
# from the stochastic block mode.  It also converts the data to
# a factor graph data format that is compatible with pytorch
# geometric (class FactorGraphData)



######### Sample SBM #########
#sample a single stochastic block model represented by this class
#this is equivalent to SpinGlassModel in learn_BP/ising_model/spin_glass_model.py
class StochasticBlockModel:
    def __init__(self, N, P, Q, C, community_probs=None):
    	'''
        Sample a stochastic block model
        
        Inputs:
        - N (int): number of nodes
        - P (float): the probability of an edge between vertices in the same community
        - Q (float): the probability of an edge between vertices in different communities
        - C (int): the number of communities
        - community_probs (torch tensor): shape # communities, give the probability that
        	each node belongs to each community.  Set to uniform if None is given 
        '''    	
        self.N = N
		self.P = P
		self.Q = Q
		self.C = C
		if community_probs is None:
			self.community_probs = torch.tensor([1.0/C for i in range(C)])
		# add code here for sampling node labels
		# self.gt_variable_labels = 

######### Convert SBM to FactorGraphData #########
# convert StochasticBlockModel to FactorGraphData representation
# should do the same thing as build_factorgraph_from_SpinGlassModel
# in learn_BP/ising_model/pytorch_dataset.py
def build_factorgraph_from_sbm(sbm_model):
    '''
    Convert a spin glass model to a factor graph pytorch representation

    Inputs:
    - sbm_model (StochasticBlockModel): defines a stochastic block model

    Outputs:
    - factorgraph (FactorGraphData): 
    '''
    state_dimensions = 2 #factors are pairwise at most in SBM
    num_vars = sbm_model.N
    #(num_vars*(num_vars-1)/2 pairwise factors, num_vars unary factors
    num_factors = num_vars*(num_vars-1)/2 + num_vars

    #Variable indexing: O to num_vars
    #Factor indexing: 
    #   - single variable factors: have indices [0, num_vars - 1] 
    #   - pairwise factors: have indices [num_vars, num_vars + num_vars*(num_vars-1)/2 - 1]
    
    #list of [factor_idx, var_idx] for each edge
    factorToVar_edge_index_list = []
    # factorToVar_double_list[i] is a list of all variables that factor with index i shares an edge with
    # factorToVar_double_list[i][j] is the index of the jth variable that the factor with index i shares an edge with
    factorToVar_double_list = []
    # add unary factor to variable edges
    for unary_factor_idx in range(num_vars):
        var_idx = unary_factor_idx
        factorToVar_edge_index_list.append([unary_factor_idx, var_idx])
        factorToVar_double_list.append([var_idx])
    # add pairwise factors to variable edges
    pairwise_factor_idx = 0
    for var_idx1 in range(num_vars):
        for var_idx2 in range(var_idx1+1, num_vars):
            factorToVar_edge_index_list.append([horizontal_factor_idx, var_idx1])
            factorToVar_edge_index_list.append([horizontal_factor_idx, var_idx2])
            assert(len(factorToVar_double_list) == horizontal_factor_idx)
            factorToVar_double_list.append([var_idx1, var_idx2])
            pairwise_factor_idx += 1

    assert(len(factorToVar_edge_index_list) == num_vars + 2*num_vars*(num_vars-1)/2)
    assert(len(factorToVar_edge_index_list) == num_vars + num_vars*(num_vars-1)/2)

    # - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge
    #     matrix with shape :obj:`[numFactors, numVars]`
    factorToVar_edge_index = torch.tensor(factorToVar_edge_index_list, dtype=torch.long)

    factor_potentials_list = []
    masks_list = []
    # Create pytorch tensor factors for each single variable factor
    for var_idx in range(num_vars):
        factor_potential, mask = build_unary_factor(sbm_model.community_probs, state_dimensions=state_dimensions, sbm_model.C)
        factor_potentials_list.append(factor_potential)
        masks_list.append(mask)

    # Create pytorch tensor factors for pairwise factors
    for row_idx in range(N):
        for col_idx in range(N-1):
            factor_potential, mask = build_pairwise_factor(p=sbm_model.p, q=sbm_model.q, state_dimensions=state_dimensions, C=sbm_model.C)
            factor_potentials_list.append(factor_potential)
            masks_list.append(mask)


    factor_potentials = torch.stack(factor_potentials_list, dim=0)
    factor_potential_masks = torch.stack(masks_list, dim=0)

    edge_var_indices = build_edge_var_indices(sbm_model=sbm_model)

    assert(factorToVar_edge_index.shape == edge_var_indices.shape), "shapes should match!"

    factor_graph = FactorGraphData(factor_potentials=factor_potentials,
                 factorToVar_edge_index=factorToVar_edge_index.t().contiguous(), numVars=num_vars, numFactors=num_factors, 
                 edge_var_indices=edge_var_indices, state_dimensions=state_dimensions, factor_potential_masks=factor_potential_masks,
#                      ln_Z=ln_Z)
                 ln_Z=None, factorToVar_double_list=factorToVar_double_list,
                 gt_variable_labels=sbm_model.gt_variable_labels)

        
    return factor_graph

def build_unary_factor(community_probs, state_dimensions, C):
    '''
    helper function for build_factorgraph_from_sbm
    create single variable factor for pytorch

    Inputs:
    - community_probs (torch.tensor): probabilities for each community, should have shape (# communities)
	- C (int): number of communities
	- state_dimensions: number of variables in largest factor

    Outputs:
    - factor (torch.tensor): the factor (already in logspace)
    - mask (torch.tensor): set to 0 for used locations ([:, 0, 0, ..., 0])
         set to 1 for all other unused locations
    '''
    #only use 1 dimension for a single variable factor 
    used_dimension_count = 1
    #initialize to all 0's, -infinity in log space
    factor_potential = -np.inf*torch.ones([C for i in range(state_dimensions)])
    mask = torch.zeros([C for i in range(state_dimensions)])


    # set factor_potential[:, 0, 0, ..., 0] = [-f, f]
    # set all values of mask to 1, except for mask[:, 0, 0, ..., 0]
    for indices in np.ndindex(factor_potential.shape):
        junk_location = False
        for dimension in range(used_dimension_count, state_dimensions):
            if indices[dimension] == 1:
                junk_location = True #this dimension is unused by this clause, set to 0
        if junk_location:
            mask[indices] = 1
            continue
        for idx in range(used_dimension_count, len(indices)):
            assert(indices[idx] == 0)
        factor_potential[indices] = community_probs[indices[0]]
    return factor_potential, mask

def build_pairwise_factor(p, q, state_dimensions, C):
    '''
	helper function for build_factorgraph_from_sbm.
    create factor over two variables for pytorch

    Inputs:
    - p (float): probabity of an edge for nodes in the same community
    - q (float): probabity of an edge for nodes in different communities
	- C (int): number of communities
	- state_dimensions: number of variables in largest factor

    Outputs:
    - factor (torch.tensor): the factor (already in logspace)
    - mask (torch.tensor): set to 0 for used locations ([:, :, 0, ..., 0])
         set to 1 for all other unused locations

    '''
    assert(C==2), "build_pairwise_factor not implemented for > 2 communities!"
    #only use 1 dimension for a single variable factor 
    used_dimension_count = 2
    #initialize to all 0's, -infinity in log space
    factor_potential = -np.inf*torch.ones([C for i in range(state_dimensions)])
    mask = torch.zeros([C for i in range(state_dimensions)])


    # set factor_potential[:, :, 0, ..., 0] = [[ln(p/(2p+2q)), ln(2/(2p+2q))],
    #                               		   [ln(q/(2p+2q)), ln(p/(2p+2q))]]
    # set all values of mask to 1, except for mask[:, :, 0, 0, ..., 0]
    for indices in np.ndindex(factor_potential.shape):
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
            factor_potential[indices] = np.log(p/(2p+2q)) #don't have to normalize the factor
        else:
            factor_potential[indices] = np.log(q/(2p+2q)) #don't have to normalize the factor

    return factor_potential, mask    


def build_edge_var_indices(sbm_model):
	# helper function for build_factorgraph_from_sbm
    # edge_var_indices has shape [2, E]. 
    #   [0, i] indicates the index (0 to factor_degree - 1) of edge i, among all edges originating at the factor which edge i begins at
    #   [1, i] IS CURRENTLY UNUSED AND BROKEN, BUT IN GENERAL FOR PYTORCH GEOMETRIC SHOULD indicate the index (0 to var_degree - 1) of edge i, among all edges ending at the variable which edge i ends at
 
    num_vars = sbm_model.N

    indices_at_source_node = []
    indices_at_destination_node = []

    # Create edge indices for single variable factors
    for var_idx in range(num_vars):
        indices_at_source_node.append(0) #source node is the factor
        indices_at_destination_node.append(-99) #destination node is the variable, JUNK

    # Create edge indices for pairwise factors
    for var_idx1 in range(num_vars):
        for var_idx2 in range(var_idx1+1, num_vars):
            indices_at_source_node.append(0) #source node is the factor
            indices_at_source_node.append(1) #source node is the factor
            indices_at_destination_node.append(-99) #destination node is the variable, JUNK
            indices_at_destination_node.append(-99) #destination node is the variable, JUNK

    edge_var_indices = torch.tensor([indices_at_source_node, indices_at_destination_node])
    return edge_var_indices


