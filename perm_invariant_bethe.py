import random
import torch

def var_idx_perm_equivariant_2dfactor_all_helper1(mlp, input_tensor):
    # no permutation
    input_tensor_shape = input_tensor.shape
    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 10)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 4]
    facPotentials = input_tensor[:, :, :4]
    facBeliefs = input_tensor[:, :, 4:8]
    varBeliefs = input_tensor[:, :, 8:]
    fp_input = facPotentials
    fb_input = facBeliefs
    permuted_input_tensor = torch.cat([fp_input, fb_input, varBeliefs], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def var_idx_perm_equivariant_2dfactor_all_helper2(mlp, input_tensor):
    #single permutation
    input_tensor_shape = input_tensor.shape
    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 10)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 4]
    facPotentials = input_tensor[:, :, :4]
    facBeliefs = input_tensor[:, :, 4:8]
    varBeliefs = input_tensor[:, :, 8:]
    facPotentials = facPotentials.reshape(factor_shape)
    fp_input = facPotentials.permute(0,1,3,2)
    fp_input = fp_input.reshape(flat_factor_shape)
    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input = facBeliefs.permute(0,1,3,2)
    fb_input = fb_input.reshape(flat_factor_shape)
    permuted_input_tensor = torch.cat([fp_input, fb_input, varBeliefs], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output
def var_idx_perm_equivariant_2dfactor_all(mlp, input_tensor):
    output = var_idx_perm_equivariant_2dfactor_all_helper1(mlp, input_tensor) +\
             var_idx_perm_equivariant_2dfactor_all_helper2(mlp, input_tensor)
    output = output/2
    return output
