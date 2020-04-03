import random
from torch.utils.data import Dataset
import numpy as np
import torch

from factor_graph import FactorGraphData
from .spin_glass_model import SpinGlassModel



def build_unary_factor(f, state_dimensions):
    '''
    create single variable factor for pytorch

    Inputs:
    - f: (float) local field at this node

    Outputs:
    - factor (torch.tensor): the factor (already in logspace)
    - mask (torch.tensor): set to 0 for used locations ([:, 0, 0, ..., 0])
         set to 1 for all other unused locations
    '''
    #only use 1 dimension for a single variable factor 
    used_dimension_count = 1
    #initialize to all 0's, -infinity in log space
    factor_potential = -np.inf*torch.ones([2 for i in range(state_dimensions)])
#     factor_potential = -np.nan*torch.ones([2 for i in range(state_dimensions)]) #for debugging
    
    mask = torch.zeros([2 for i in range(state_dimensions)])


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
        if indices[0] == 0:
            factor_potential[indices] = -f
        elif indices[0] == 1:
            factor_potential[indices] = f
        else:
            assert(False)
    return factor_potential, mask

def build_pairwise_factor(c, state_dimensions):
    '''
    create factor over two variables for pytorch

    Inputs:
    - f: (float) local field at this node

    Outputs:
    - factor (torch.tensor): the factor (already in logspace)
    - mask (torch.tensor): set to 0 for used locations ([:, :, 0, ..., 0])
         set to 1 for all other unused locations

    '''
    #only use 1 dimension for a single variable factor 
    used_dimension_count = 2
    #initialize to all 0's, -infinity in log space
    factor_potential = -np.inf*torch.ones([2 for i in range(state_dimensions)])
    mask = torch.zeros([2 for i in range(state_dimensions)])


    # set factor_potential[:, :, 0, ..., 0] = [[ c, -c],
    #                               [-c,  c]]
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
            factor_potential[indices] = c
        else:
            factor_potential[indices] = -c

    return factor_potential, mask    


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

    if sg_model.contains_higher_order_potentials:
        for potential_idx in range(sg_model.ho_potential_count):
            for var_idx in range(sg_model.ho_potential_degree):
                indices_at_source_node.append(var_idx) #source node is the factor
                indices_at_destination_node.append(-99) #destination node is the variable

    edge_var_indices = torch.tensor([indices_at_source_node, indices_at_destination_node])
    return edge_var_indices

def build_factorgraph_from_SpinGlassModel(sg_model):
    '''
    Convert a spin glass model to a factor graph pytorch representation

    Inputs:
    - sg_model (SpinGlassModel): defines a sping glass model

    Outputs:
    - factorgraph (FactorGraphData): 
    '''
    state_dimensions = 2
    N = sg_model.N
    num_vars = N**2
    #(N-1)*N horizontal coupling factors, (N-1)*N vertical coupling factors, and N**2 single variable factors
    if sg_model.contains_higher_order_potentials:
        num_factors = 2*(N - 1)*N + N**2 + sg_model.ho_potential_count
    else:
        num_factors = 2*(N - 1)*N + N**2

    #Variable indexing: variable with indices [row_idx, col_idx] (e.g. with lcl_fld_param given by lcl_fld_params[row_idx,col_idx]) has index row_idx*N+col_idx
    #Factor indexing: 
    #   - single variable factors: have the same index as their variable index 
    #   - horizontal coupling factors: factor between [row_idx, col_idx] and [row_idx, col_idx+1] has index (N**2 + row_idx*(N-1) + col_idx)
    #   - vertical coupling factors: factor between [row_idx, col_idx] and [row_idx+1, col_idx] has index (N**2 + N*(N-1) + row_idx*N + col_idx)
    
    #list of [factor_idx, var_idx] for each edge
    factorToVar_edge_index_list = []
    # factorToVar_double_list[i] is a list of all variables that factor with index i shares an edge with
    # factorToVar_double_list[i][j] is the index of the jth variable that the factor with index i shares an edge with
    factorToVar_double_list = []
    # add unary factor to variable edges
    for unary_factor_idx in range(N**2):
        var_idx = unary_factor_idx
        factorToVar_edge_index_list.append([unary_factor_idx, var_idx])
        factorToVar_double_list.append([var_idx])
    # add horizontal factor to variable edges
    for row_idx in range(N):
        for col_idx in range(N-1):
            horizontal_factor_idx = (N**2 + row_idx*(N-1) + col_idx)
            var_idx1 = row_idx*N + col_idx
            factorToVar_edge_index_list.append([horizontal_factor_idx, var_idx1])
            var_idx2 = row_idx*N + col_idx + 1
            factorToVar_edge_index_list.append([horizontal_factor_idx, var_idx2])
            assert(len(factorToVar_double_list) == horizontal_factor_idx)
            factorToVar_double_list.append([var_idx1, var_idx2])
            
    # add vertical factor to variable edges
    for row_idx in range(N-1):
        for col_idx in range(N):
            vertical_factor_idx = (N**2 + N*(N-1) + row_idx*N + col_idx)
            var_idx1 = row_idx*N + col_idx
            factorToVar_edge_index_list.append([vertical_factor_idx, var_idx1])
            var_idx2 = (row_idx+1)*N + col_idx
            factorToVar_edge_index_list.append([vertical_factor_idx, var_idx2])
            
            assert(len(factorToVar_double_list) == vertical_factor_idx)
            factorToVar_double_list.append([var_idx1, var_idx2])  
            
    assert(len(factorToVar_edge_index_list) == 4*(N - 1)*N + N**2)
    if sg_model.contains_higher_order_potentials:
        for higher_order_idx in range(sg_model.ho_potential_count):
            factor_idx =  2*(N - 1)*N + N**2 + higher_order_idx
            for variable_idx in sg_model.higher_order_potentials_variables[higher_order_idx]:
                factorToVar_edge_index_list.append([factor_idx, variable_idx])
            # pass
            
            assert(len(factorToVar_double_list) == factor_idx)
            factorToVar_double_list.append(list(sg_model.higher_order_potentials_variables[higher_order_idx]))

    # - factorToVar_edge_index (Tensor): The indices of a general (sparse) edge
    #     matrix with shape :obj:`[numFactors, numVars]`
    factorToVar_edge_index = torch.tensor(factorToVar_edge_index_list, dtype=torch.long)

    factor_potentials_list = []
    masks_list = []
    # Create pytorch tensor factors for each single variable factor
    for var_idx in range(N**2):
        r = var_idx//N
        c = var_idx%N
        factor_potential, mask = build_unary_factor(f=sg_model.lcl_fld_params[r,c], state_dimensions=state_dimensions)
        factor_potentials_list.append(factor_potential)
        masks_list.append(mask)

    # Create pytorch tensor factors for each horizontal pairwise factor
    for row_idx in range(N):
        for col_idx in range(N-1):
            var_idx1 = row_idx*N + col_idx
            var_idx2 = row_idx*N + col_idx + 1
            factor_potential, mask = build_pairwise_factor(c=sg_model.cpl_params_h[row_idx,col_idx], state_dimensions=state_dimensions)
            factor_potentials_list.append(factor_potential)
            masks_list.append(mask)

    # Create pytorch tensor factors for each vertical pairwise factor
    for row_idx in range(N-1):
        for col_idx in range(N):
            var_idx1 = row_idx*N + col_idx
            var_idx2 = (row_idx+1)*N + col_idx
            factor_potential, mask = build_pairwise_factor(c=sg_model.cpl_params_v[row_idx,col_idx], state_dimensions=state_dimensions)
            factor_potentials_list.append(factor_potential)
            masks_list.append(mask)

    if sg_model.contains_higher_order_potentials:
        # Add higher order factors
        for potential_idx in range(sg_model.ho_potential_count):
            factor_potentials_list.append(torch.tensor(sg_model.higher_order_potentials[potential_idx]))
            masks_list.append(torch.zeros_like(sg_model.higher_order_potentials[potential_idx]))

    factor_potentials = torch.stack(factor_potentials_list, dim=0)
    factor_potential_masks = torch.stack(masks_list, dim=0)

    edge_var_indices = build_edge_var_indices(sg_model=sg_model)
    # print("state_dimensions:", state_dimensions)

    edge_count = edge_var_indices.shape[1]

    ln_Z = sg_model.junction_tree_libdai()

    factor_graph = FactorGraphData(factor_potentials=factor_potentials,
                 factorToVar_edge_index=factorToVar_edge_index.t().contiguous(), numVars=num_vars, numFactors=num_factors, 
                 edge_var_indices=edge_var_indices, state_dimensions=state_dimensions, factor_potential_masks=factor_potential_masks,
#                      ln_Z=ln_Z)
                 ln_Z=ln_Z, factorToVar_double_list=factorToVar_double_list)

        
    return factor_graph