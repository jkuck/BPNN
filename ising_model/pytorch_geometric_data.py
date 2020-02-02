from torch_geometric.data import Data
import torch

def spinGlass_to_torchGeometric(sg_model):
    '''
    Convert a spin glass model represented as a SpinGlassModel object
    to pytorch geometric Data
    Inputs:
    - sg_model (SpinGlassModel): representation of a spin glass model
    
    Outputs:
    - sg_model_torchGeom (torch_geometric.data.Data): representation of a spin glass model
        as pytorch geometric Data
    '''
    
    edge_index, edge_attr = construct_edges(sg_model)
    unary_potentials = torch.tensor(sg_model.lcl_fld_params, dtype=torch.float).flatten()
    assert(unary_potentials.shape == (sg_model.N**2,)), (unary_potentials.shape, sg_model.N**2)
    x = unary_potentials.clone()
    sg_model_torchGeom = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    sg_model_torchGeom.unary_potentials = unary_potentials
    sg_model_torchGeom.ln_Z = torch.tensor([sg_model.junction_tree_libdai()])
    return sg_model_torchGeom
    
    
def construct_edges(sg_model):
    '''
    Inputs:
    - sg_model (SpinGlassModel): representation of a spin glass model
    
    Outputs:
    - edge_index (LongTensor): Graph connectivity in COO format with shape [2, num_edges].
    - edge_attr (Tensor): Edge feature matrix with shape [num_edges, num_edge_features].
        These are horizontal coupling potentials followed by vertical coupling potentials.
        Note, each is repeated twice because we represent undirected edges as 2 directed edges.
    '''
    
    

    N = sg_model.N    
    edge_attr_list = []
    edge_index_list = []
    #Variable indexing: variable with indices [row_idx, col_idx] (e.g. with lcl_fld_param given by lcl_fld_params[row_idx,col_idx]) has index row_idx*N+col_idx
    # add horizontal edges
    for row_idx in range(N):
        for col_idx in range(N-1):
            var_idx1 = row_idx*N + col_idx
            var_idx2 = row_idx*N + col_idx + 1
            edge_index_list.append([var_idx1, var_idx2])
            edge_index_list.append([var_idx2, var_idx1])
            edge_attr_list.append([sg_model.cpl_params_h[row_idx, col_idx]])
            edge_attr_list.append([sg_model.cpl_params_h[row_idx, col_idx]])
            
    # add vertical edges
    for row_idx in range(N-1):
        for col_idx in range(N):
            var_idx1 = row_idx*N + col_idx
            var_idx2 = (row_idx+1)*N + col_idx
            edge_index_list.append([var_idx1, var_idx2])
            edge_index_list.append([var_idx2, var_idx1])
            edge_attr_list.append([sg_model.cpl_params_v[row_idx, col_idx]])
            edge_attr_list.append([sg_model.cpl_params_v[row_idx, col_idx]])
            
    assert(len(edge_index_list) == 4*(N - 1)*N)
    assert(len(edge_attr_list) == 4*(N - 1)*N)
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).permute(1,0)
    edge_attr = torch.tensor(edge_attr_list)
    assert(edge_index.shape == (2, 4*(N - 1)*N))
    assert(edge_attr.shape == (4*(N - 1)*N,1)), (edge_attr.shape, (4*(N - 1)*N,1), edge_index.shape, (2, 4*(N - 1)*N))
    
    return edge_index, edge_attr