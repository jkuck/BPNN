import random
import torch

#this version operates on final beliefs before aggregated across the factor graph
#original version that operated on aggregated beliefs wasn't actually invariant
#this version doesn't have variable beliefs as input, since the number of
#variables doesn't match up with the number of factors

def var_idx_perm_equivariant_2dfactor_all_helper1(mlp, input_tensor):
    # no permutation

    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 8)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 4]
    facPotentials = input_tensor[:, :, :4]
    facBeliefs = input_tensor[:, :, 4]

    fp_input = facPotentials

    fb_input = facBeliefs

    permuted_input_tensor = torch.cat([fp_input, fb_input], dim=2)
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

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input = facPotentials.permute(0,1,3,2)
    fp_input = fp_input.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input = facBeliefs.permute(0,1,3,2)
    fb_input = fb_input.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input, fb_input], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output



def var_idx_perm_equivariant_2dfactor_all(mlp, input_tensor):
    #for factors with 2 variables
    output = var_idx_perm_equivariant_2dfactor_all_helper1(mlp, input_tensor) +\
             var_idx_perm_equivariant_2dfactor_all_helper2(mlp, input_tensor)
    output = output/2
    return output



def permute_dim1112_initImplementation(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    #should have shape (batch_size, message_passing_layers, 66)
    #66=2^5 + 2^5 + 2 for pooled states
    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    fp_input_tensor1 = facPotentials.reshape(factor_shape)
    fp_input_tensor11 = fp_input_tensor1
    fp_input_tensor111 = fp_input_tensor11
    fp_input_tensor1112 = fp_input_tensor111.permute(0,1,2,3,4,6,5)
    fp_input_tensor1112 = fp_input_tensor1112.reshape(flat_factor_shape)

    fb_input_tensor1 = facBeliefs.reshape(factor_shape)
    fb_input_tensor11 = fb_input_tensor1
    fb_input_tensor111 = fb_input_tensor11
    fb_input_tensor1112 = fb_input_tensor111.permute(0,1,2,3,4,6,5)
    fb_input_tensor1112 = fb_input_tensor1112.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1112, fb_input_tensor1112], dim=2)
    output = mlp(permuted_input_tensor.reshape(input_tensor_shape[0], -1))

    return output

def permute_dim1112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor11 = fp_input_tensor1
    fp_input_tensor111 = fp_input_tensor11
    fp_input_tensor1112 = fp_input_tensor111.permute(0,1,2,3,4,6,5)
    fp_input_tensor1112 = fp_input_tensor1112.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor11 = fb_input_tensor1
    fb_input_tensor111 = fb_input_tensor11
    fb_input_tensor1112 = fb_input_tensor111.permute(0,1,2,3,4,6,5)
    fb_input_tensor1112 = fb_input_tensor1112.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1112, fb_input_tensor1112], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output


def permute_dim1111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape
    assert(len(input_tensor_shape) == 3), input_tensor_shape
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    # print("factor_shape:", factor_shape)
    # print("input_tensor_shape:", input_tensor_shape)
    # sleep(asldkf)


    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor11 = fp_input_tensor1
    fp_input_tensor111 = fp_input_tensor11
    fp_input_tensor1111 = fp_input_tensor111
    fp_input_tensor1111 = fp_input_tensor1111.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor11 = fb_input_tensor1
    fb_input_tensor111 = fb_input_tensor11
    fb_input_tensor1111 = fb_input_tensor111
    fb_input_tensor1111 = fb_input_tensor1111.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1111, fb_input_tensor1111], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output



def permute_dim1121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor11 = fp_input_tensor1
    fp_input_tensor112 = fp_input_tensor11.permute(0,1,2,3,5,4,6) 
    fp_input_tensor1121 = fp_input_tensor112
    fp_input_tensor1121 = fp_input_tensor1121.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor11 = fb_input_tensor1
    fb_input_tensor112 = fb_input_tensor11.permute(0,1,2,3,5,4,6) 
    fb_input_tensor1121 = fb_input_tensor112
    fb_input_tensor1121 = fb_input_tensor1121.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1121, fb_input_tensor1121], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor11 = fp_input_tensor1
    fp_input_tensor112 = fp_input_tensor11.permute(0,1,2,3,5,4,6) 
    fp_input_tensor1122 = fp_input_tensor112.permute(0,1,2,3,4,6,5)
    fp_input_tensor1122 = fp_input_tensor1122.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor11 = fb_input_tensor1
    fb_input_tensor112 = fb_input_tensor11.permute(0,1,2,3,5,4,6) 
    fb_input_tensor1122 = fb_input_tensor112.permute(0,1,2,3,4,6,5)
    fb_input_tensor1122 = fb_input_tensor1122.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1122, fb_input_tensor1122], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor11 = fp_input_tensor1
    fp_input_tensor113 = fp_input_tensor11.permute(0,1,2,3,6,5,4)
    fp_input_tensor1131 = fp_input_tensor113
    fp_input_tensor1131 = fp_input_tensor1131.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor11 = fb_input_tensor1
    fb_input_tensor113 = fb_input_tensor11.permute(0,1,2,3,6,5,4)
    fb_input_tensor1131 = fb_input_tensor113
    fb_input_tensor1131 = fb_input_tensor1131.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1131, fb_input_tensor1131], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor11 = fp_input_tensor1
    fp_input_tensor113 = fp_input_tensor11.permute(0,1,2,3,6,5,4)
    fp_input_tensor1132 = fp_input_tensor113.permute(0,1,2,3,4,6,5)
    fp_input_tensor1132 = fp_input_tensor1132.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor11 = fb_input_tensor1
    fb_input_tensor113 = fb_input_tensor11.permute(0,1,2,3,6,5,4)
    fb_input_tensor1132 = fb_input_tensor113.permute(0,1,2,3,4,6,5)
    fb_input_tensor1132 = fb_input_tensor1132.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1132, fb_input_tensor1132], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor12 = fp_input_tensor1.permute(0,1,2,4,3,5,6)
    fp_input_tensor121 = fp_input_tensor12
    fp_input_tensor1211 = fp_input_tensor121
    fp_input_tensor1211 = fp_input_tensor1211.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor12 = fb_input_tensor1.permute(0,1,2,4,3,5,6)
    fb_input_tensor121 = fb_input_tensor12
    fb_input_tensor1211 = fb_input_tensor121
    fb_input_tensor1211 = fb_input_tensor1211.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1211, fb_input_tensor1211], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor12 = fp_input_tensor1.permute(0,1,2,4,3,5,6)
    fp_input_tensor121 = fp_input_tensor12
    fp_input_tensor1212 = fp_input_tensor121.permute(0,1,2,3,4,6,5)
    fp_input_tensor1212 = fp_input_tensor1212.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor12 = fb_input_tensor1.permute(0,1,2,4,3,5,6)
    fb_input_tensor121 = fb_input_tensor12
    fb_input_tensor1212 = fb_input_tensor121.permute(0,1,2,3,4,6,5)
    fb_input_tensor1212 = fb_input_tensor1212.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1212, fb_input_tensor1212], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor12 = fp_input_tensor1.permute(0,1,2,4,3,5,6)
    fp_input_tensor122 = fp_input_tensor12.permute(0,1,2,3,5,4,6) 
    fp_input_tensor1221 = fp_input_tensor122
    fp_input_tensor1221 = fp_input_tensor1221.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor12 = fb_input_tensor1.permute(0,1,2,4,3,5,6)
    fb_input_tensor122 = fb_input_tensor12.permute(0,1,2,3,5,4,6) 
    fb_input_tensor1221 = fb_input_tensor122
    fb_input_tensor1221 = fb_input_tensor1221.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1221, fb_input_tensor1221], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor12 = fp_input_tensor1.permute(0,1,2,4,3,5,6)
    fp_input_tensor122 = fp_input_tensor12.permute(0,1,2,3,5,4,6) 
    fp_input_tensor1222 = fp_input_tensor122.permute(0,1,2,3,4,6,5)
    fp_input_tensor1222 = fp_input_tensor1222.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor12 = fb_input_tensor1.permute(0,1,2,4,3,5,6)
    fb_input_tensor122 = fb_input_tensor12.permute(0,1,2,3,5,4,6) 
    fb_input_tensor1222 = fb_input_tensor122.permute(0,1,2,3,4,6,5)
    fb_input_tensor1222 = fb_input_tensor1222.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1222, fb_input_tensor1222], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor12 = fp_input_tensor1.permute(0,1,2,4,3,5,6)
    fp_input_tensor123 = fp_input_tensor12.permute(0,1,2,3,6,5,4)
    fp_input_tensor1231 = fp_input_tensor123
    fp_input_tensor1231 = fp_input_tensor1231.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor12 = fb_input_tensor1.permute(0,1,2,4,3,5,6)
    fb_input_tensor123 = fb_input_tensor12.permute(0,1,2,3,6,5,4)
    fb_input_tensor1231 = fb_input_tensor123
    fb_input_tensor1231 = fb_input_tensor1231.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1231, fb_input_tensor1231], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor12 = fp_input_tensor1.permute(0,1,2,4,3,5,6)
    fp_input_tensor123 = fp_input_tensor12.permute(0,1,2,3,6,5,4)
    fp_input_tensor1232 = fp_input_tensor123.permute(0,1,2,3,4,6,5)
    fp_input_tensor1232 = fp_input_tensor1232.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor12 = fb_input_tensor1.permute(0,1,2,4,3,5,6)
    fb_input_tensor123 = fb_input_tensor12.permute(0,1,2,3,6,5,4)
    fb_input_tensor1232 = fb_input_tensor123.permute(0,1,2,3,4,6,5)
    fb_input_tensor1232 = fb_input_tensor1232.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1232, fb_input_tensor1232], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor13 = fp_input_tensor1.permute(0,1,2,5,4,3,6)
    fp_input_tensor131 = fp_input_tensor13
    fp_input_tensor1311 = fp_input_tensor131
    fp_input_tensor1311 = fp_input_tensor1311.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor13 = fb_input_tensor1.permute(0,1,2,5,4,3,6)
    fb_input_tensor131 = fb_input_tensor13
    fb_input_tensor1311 = fb_input_tensor131
    fb_input_tensor1311 = fb_input_tensor1311.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1311, fb_input_tensor1311], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor13 = fp_input_tensor1.permute(0,1,2,5,4,3,6)
    fp_input_tensor131 = fp_input_tensor13
    fp_input_tensor1312 = fp_input_tensor131.permute(0,1,2,3,4,6,5)
    fp_input_tensor1312 = fp_input_tensor1312.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor13 = fb_input_tensor1.permute(0,1,2,5,4,3,6)
    fb_input_tensor131 = fb_input_tensor13
    fb_input_tensor1312 = fb_input_tensor131.permute(0,1,2,3,4,6,5)
    fb_input_tensor1312 = fb_input_tensor1312.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1312, fb_input_tensor1312], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor13 = fp_input_tensor1.permute(0,1,2,5,4,3,6)
    fp_input_tensor132 = fp_input_tensor13.permute(0,1,2,3,5,4,6) 
    fp_input_tensor1321 = fp_input_tensor132
    fp_input_tensor1321 = fp_input_tensor1321.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor13 = fb_input_tensor1.permute(0,1,2,5,4,3,6)
    fb_input_tensor132 = fb_input_tensor13.permute(0,1,2,3,5,4,6) 
    fb_input_tensor1321 = fb_input_tensor132
    fb_input_tensor1321 = fb_input_tensor1321.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1321, fb_input_tensor1321], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor13 = fp_input_tensor1.permute(0,1,2,5,4,3,6)
    fp_input_tensor132 = fp_input_tensor13.permute(0,1,2,3,5,4,6) 
    fp_input_tensor1322 = fp_input_tensor132.permute(0,1,2,3,4,6,5)
    fp_input_tensor1322 = fp_input_tensor1322.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor13 = fb_input_tensor1.permute(0,1,2,5,4,3,6)
    fb_input_tensor132 = fb_input_tensor13.permute(0,1,2,3,5,4,6) 
    fb_input_tensor1322 = fb_input_tensor132.permute(0,1,2,3,4,6,5)
    fb_input_tensor1322 = fb_input_tensor1322.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1322, fb_input_tensor1322], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor13 = fp_input_tensor1.permute(0,1,2,5,4,3,6)
    fp_input_tensor133 = fp_input_tensor13.permute(0,1,2,3,6,5,4)
    fp_input_tensor1331 = fp_input_tensor133
    fp_input_tensor1331 = fp_input_tensor1331.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor13 = fb_input_tensor1.permute(0,1,2,5,4,3,6)
    fb_input_tensor133 = fb_input_tensor13.permute(0,1,2,3,6,5,4)
    fb_input_tensor1331 = fb_input_tensor133
    fb_input_tensor1331 = fb_input_tensor1331.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1331, fb_input_tensor1331], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor13 = fp_input_tensor1.permute(0,1,2,5,4,3,6)
    fp_input_tensor133 = fp_input_tensor13.permute(0,1,2,3,6,5,4)
    fp_input_tensor1332 = fp_input_tensor133.permute(0,1,2,3,4,6,5)
    fp_input_tensor1332 = fp_input_tensor1332.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor13 = fb_input_tensor1.permute(0,1,2,5,4,3,6)
    fb_input_tensor133 = fb_input_tensor13.permute(0,1,2,3,6,5,4)
    fb_input_tensor1332 = fb_input_tensor133.permute(0,1,2,3,4,6,5)
    fb_input_tensor1332 = fb_input_tensor1332.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1332, fb_input_tensor1332], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor14 = fp_input_tensor1.permute(0,1,2,6,4,5,3)
    fp_input_tensor141 = fp_input_tensor14
    fp_input_tensor1411 = fp_input_tensor141
    fp_input_tensor1411 = fp_input_tensor1411.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor14 = fb_input_tensor1.permute(0,1,2,6,4,5,3)
    fb_input_tensor141 = fb_input_tensor14
    fb_input_tensor1411 = fb_input_tensor141
    fb_input_tensor1411 = fb_input_tensor1411.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1411, fb_input_tensor1411], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor14 = fp_input_tensor1.permute(0,1,2,6,4,5,3)
    fp_input_tensor141 = fp_input_tensor14
    fp_input_tensor1412 = fp_input_tensor141.permute(0,1,2,3,4,6,5)
    fp_input_tensor1412 = fp_input_tensor1412.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor14 = fb_input_tensor1.permute(0,1,2,6,4,5,3)
    fb_input_tensor141 = fb_input_tensor14
    fb_input_tensor1412 = fb_input_tensor141.permute(0,1,2,3,4,6,5)
    fb_input_tensor1412 = fb_input_tensor1412.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1412, fb_input_tensor1412], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor14 = fp_input_tensor1.permute(0,1,2,6,4,5,3)
    fp_input_tensor142 = fp_input_tensor14.permute(0,1,2,3,5,4,6) 
    fp_input_tensor1421 = fp_input_tensor142
    fp_input_tensor1421 = fp_input_tensor1421.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor14 = fb_input_tensor1.permute(0,1,2,6,4,5,3)
    fb_input_tensor142 = fb_input_tensor14.permute(0,1,2,3,5,4,6) 
    fb_input_tensor1421 = fb_input_tensor142
    fb_input_tensor1421 = fb_input_tensor1421.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1421, fb_input_tensor1421], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor14 = fp_input_tensor1.permute(0,1,2,6,4,5,3)
    fp_input_tensor142 = fp_input_tensor14.permute(0,1,2,3,5,4,6) 
    fp_input_tensor1422 = fp_input_tensor142.permute(0,1,2,3,4,6,5)
    fp_input_tensor1422 = fp_input_tensor1422.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor14 = fb_input_tensor1.permute(0,1,2,6,4,5,3)
    fb_input_tensor142 = fb_input_tensor14.permute(0,1,2,3,5,4,6) 
    fb_input_tensor1422 = fb_input_tensor142.permute(0,1,2,3,4,6,5)
    fb_input_tensor1422 = fb_input_tensor1422.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1422, fb_input_tensor1422], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor14 = fp_input_tensor1.permute(0,1,2,6,4,5,3)
    fp_input_tensor143 = fp_input_tensor14.permute(0,1,2,3,6,5,4)
    fp_input_tensor1431 = fp_input_tensor143
    fp_input_tensor1431 = fp_input_tensor1431.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor14 = fb_input_tensor1.permute(0,1,2,6,4,5,3)
    fb_input_tensor143 = fb_input_tensor14.permute(0,1,2,3,6,5,4)
    fb_input_tensor1431 = fb_input_tensor143
    fb_input_tensor1431 = fb_input_tensor1431.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1431, fb_input_tensor1431], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim1432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor1 = facPotentials
    fp_input_tensor14 = fp_input_tensor1.permute(0,1,2,6,4,5,3)
    fp_input_tensor143 = fp_input_tensor14.permute(0,1,2,3,6,5,4)
    fp_input_tensor1432 = fp_input_tensor143.permute(0,1,2,3,4,6,5)
    fp_input_tensor1432 = fp_input_tensor1432.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor1 = facBeliefs
    fb_input_tensor14 = fb_input_tensor1.permute(0,1,2,6,4,5,3)
    fb_input_tensor143 = fb_input_tensor14.permute(0,1,2,3,6,5,4)
    fb_input_tensor1432 = fb_input_tensor143.permute(0,1,2,3,4,6,5)
    fb_input_tensor1432 = fb_input_tensor1432.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor1432, fb_input_tensor1432], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor21 = fp_input_tensor2
    fp_input_tensor211 = fp_input_tensor21
    fp_input_tensor2111 = fp_input_tensor211
    fp_input_tensor2111 = fp_input_tensor2111.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor21 = fb_input_tensor2
    fb_input_tensor211 = fb_input_tensor21
    fb_input_tensor2111 = fb_input_tensor211
    fb_input_tensor2111 = fb_input_tensor2111.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2111, fb_input_tensor2111], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor21 = fp_input_tensor2
    fp_input_tensor211 = fp_input_tensor21
    fp_input_tensor2112 = fp_input_tensor211.permute(0,1,2,3,4,6,5)
    fp_input_tensor2112 = fp_input_tensor2112.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor21 = fb_input_tensor2
    fb_input_tensor211 = fb_input_tensor21
    fb_input_tensor2112 = fb_input_tensor211.permute(0,1,2,3,4,6,5)
    fb_input_tensor2112 = fb_input_tensor2112.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2112, fb_input_tensor2112], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor21 = fp_input_tensor2
    fp_input_tensor212 = fp_input_tensor21.permute(0,1,2,3,5,4,6) 
    fp_input_tensor2121 = fp_input_tensor212
    fp_input_tensor2121 = fp_input_tensor2121.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor21 = fb_input_tensor2
    fb_input_tensor212 = fb_input_tensor21.permute(0,1,2,3,5,4,6) 
    fb_input_tensor2121 = fb_input_tensor212
    fb_input_tensor2121 = fb_input_tensor2121.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2121, fb_input_tensor2121], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor21 = fp_input_tensor2
    fp_input_tensor212 = fp_input_tensor21.permute(0,1,2,3,5,4,6) 
    fp_input_tensor2122 = fp_input_tensor212.permute(0,1,2,3,4,6,5)
    fp_input_tensor2122 = fp_input_tensor2122.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor21 = fb_input_tensor2
    fb_input_tensor212 = fb_input_tensor21.permute(0,1,2,3,5,4,6) 
    fb_input_tensor2122 = fb_input_tensor212.permute(0,1,2,3,4,6,5)
    fb_input_tensor2122 = fb_input_tensor2122.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2122, fb_input_tensor2122], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor21 = fp_input_tensor2
    fp_input_tensor213 = fp_input_tensor21.permute(0,1,2,3,6,5,4)
    fp_input_tensor2131 = fp_input_tensor213
    fp_input_tensor2131 = fp_input_tensor2131.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor21 = fb_input_tensor2
    fb_input_tensor213 = fb_input_tensor21.permute(0,1,2,3,6,5,4)
    fb_input_tensor2131 = fb_input_tensor213
    fb_input_tensor2131 = fb_input_tensor2131.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2131, fb_input_tensor2131], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor21 = fp_input_tensor2
    fp_input_tensor213 = fp_input_tensor21.permute(0,1,2,3,6,5,4)
    fp_input_tensor2132 = fp_input_tensor213.permute(0,1,2,3,4,6,5)
    fp_input_tensor2132 = fp_input_tensor2132.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor21 = fb_input_tensor2
    fb_input_tensor213 = fb_input_tensor21.permute(0,1,2,3,6,5,4)
    fb_input_tensor2132 = fb_input_tensor213.permute(0,1,2,3,4,6,5)
    fb_input_tensor2132 = fb_input_tensor2132.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2132, fb_input_tensor2132], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor22 = fp_input_tensor2.permute(0,1,2,4,3,5,6)
    fp_input_tensor221 = fp_input_tensor22
    fp_input_tensor2211 = fp_input_tensor221
    fp_input_tensor2211 = fp_input_tensor2211.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor22 = fb_input_tensor2.permute(0,1,2,4,3,5,6)
    fb_input_tensor221 = fb_input_tensor22
    fb_input_tensor2211 = fb_input_tensor221
    fb_input_tensor2211 = fb_input_tensor2211.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2211, fb_input_tensor2211], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor22 = fp_input_tensor2.permute(0,1,2,4,3,5,6)
    fp_input_tensor221 = fp_input_tensor22
    fp_input_tensor2212 = fp_input_tensor221.permute(0,1,2,3,4,6,5)
    fp_input_tensor2212 = fp_input_tensor2212.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor22 = fb_input_tensor2.permute(0,1,2,4,3,5,6)
    fb_input_tensor221 = fb_input_tensor22
    fb_input_tensor2212 = fb_input_tensor221.permute(0,1,2,3,4,6,5)
    fb_input_tensor2212 = fb_input_tensor2212.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2212, fb_input_tensor2212], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor22 = fp_input_tensor2.permute(0,1,2,4,3,5,6)
    fp_input_tensor222 = fp_input_tensor22.permute(0,1,2,3,5,4,6) 
    fp_input_tensor2221 = fp_input_tensor222
    fp_input_tensor2221 = fp_input_tensor2221.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor22 = fb_input_tensor2.permute(0,1,2,4,3,5,6)
    fb_input_tensor222 = fb_input_tensor22.permute(0,1,2,3,5,4,6) 
    fb_input_tensor2221 = fb_input_tensor222
    fb_input_tensor2221 = fb_input_tensor2221.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2221, fb_input_tensor2221], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor22 = fp_input_tensor2.permute(0,1,2,4,3,5,6)
    fp_input_tensor222 = fp_input_tensor22.permute(0,1,2,3,5,4,6) 
    fp_input_tensor2222 = fp_input_tensor222.permute(0,1,2,3,4,6,5)
    fp_input_tensor2222 = fp_input_tensor2222.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor22 = fb_input_tensor2.permute(0,1,2,4,3,5,6)
    fb_input_tensor222 = fb_input_tensor22.permute(0,1,2,3,5,4,6) 
    fb_input_tensor2222 = fb_input_tensor222.permute(0,1,2,3,4,6,5)
    fb_input_tensor2222 = fb_input_tensor2222.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2222, fb_input_tensor2222], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor22 = fp_input_tensor2.permute(0,1,2,4,3,5,6)
    fp_input_tensor223 = fp_input_tensor22.permute(0,1,2,3,6,5,4)
    fp_input_tensor2231 = fp_input_tensor223
    fp_input_tensor2231 = fp_input_tensor2231.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor22 = fb_input_tensor2.permute(0,1,2,4,3,5,6)
    fb_input_tensor223 = fb_input_tensor22.permute(0,1,2,3,6,5,4)
    fb_input_tensor2231 = fb_input_tensor223
    fb_input_tensor2231 = fb_input_tensor2231.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2231, fb_input_tensor2231], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor22 = fp_input_tensor2.permute(0,1,2,4,3,5,6)
    fp_input_tensor223 = fp_input_tensor22.permute(0,1,2,3,6,5,4)
    fp_input_tensor2232 = fp_input_tensor223.permute(0,1,2,3,4,6,5)
    fp_input_tensor2232 = fp_input_tensor2232.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor22 = fb_input_tensor2.permute(0,1,2,4,3,5,6)
    fb_input_tensor223 = fb_input_tensor22.permute(0,1,2,3,6,5,4)
    fb_input_tensor2232 = fb_input_tensor223.permute(0,1,2,3,4,6,5)
    fb_input_tensor2232 = fb_input_tensor2232.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2232, fb_input_tensor2232], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor23 = fp_input_tensor2.permute(0,1,2,5,4,3,6)
    fp_input_tensor231 = fp_input_tensor23
    fp_input_tensor2311 = fp_input_tensor231
    fp_input_tensor2311 = fp_input_tensor2311.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor23 = fb_input_tensor2.permute(0,1,2,5,4,3,6)
    fb_input_tensor231 = fb_input_tensor23
    fb_input_tensor2311 = fb_input_tensor231
    fb_input_tensor2311 = fb_input_tensor2311.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2311, fb_input_tensor2311], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor23 = fp_input_tensor2.permute(0,1,2,5,4,3,6)
    fp_input_tensor231 = fp_input_tensor23
    fp_input_tensor2312 = fp_input_tensor231.permute(0,1,2,3,4,6,5)
    fp_input_tensor2312 = fp_input_tensor2312.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor23 = fb_input_tensor2.permute(0,1,2,5,4,3,6)
    fb_input_tensor231 = fb_input_tensor23
    fb_input_tensor2312 = fb_input_tensor231.permute(0,1,2,3,4,6,5)
    fb_input_tensor2312 = fb_input_tensor2312.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2312, fb_input_tensor2312], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor23 = fp_input_tensor2.permute(0,1,2,5,4,3,6)
    fp_input_tensor232 = fp_input_tensor23.permute(0,1,2,3,5,4,6) 
    fp_input_tensor2321 = fp_input_tensor232
    fp_input_tensor2321 = fp_input_tensor2321.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor23 = fb_input_tensor2.permute(0,1,2,5,4,3,6)
    fb_input_tensor232 = fb_input_tensor23.permute(0,1,2,3,5,4,6) 
    fb_input_tensor2321 = fb_input_tensor232
    fb_input_tensor2321 = fb_input_tensor2321.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2321, fb_input_tensor2321], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor23 = fp_input_tensor2.permute(0,1,2,5,4,3,6)
    fp_input_tensor232 = fp_input_tensor23.permute(0,1,2,3,5,4,6) 
    fp_input_tensor2322 = fp_input_tensor232.permute(0,1,2,3,4,6,5)
    fp_input_tensor2322 = fp_input_tensor2322.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor23 = fb_input_tensor2.permute(0,1,2,5,4,3,6)
    fb_input_tensor232 = fb_input_tensor23.permute(0,1,2,3,5,4,6) 
    fb_input_tensor2322 = fb_input_tensor232.permute(0,1,2,3,4,6,5)
    fb_input_tensor2322 = fb_input_tensor2322.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2322, fb_input_tensor2322], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor23 = fp_input_tensor2.permute(0,1,2,5,4,3,6)
    fp_input_tensor233 = fp_input_tensor23.permute(0,1,2,3,6,5,4)
    fp_input_tensor2331 = fp_input_tensor233
    fp_input_tensor2331 = fp_input_tensor2331.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor23 = fb_input_tensor2.permute(0,1,2,5,4,3,6)
    fb_input_tensor233 = fb_input_tensor23.permute(0,1,2,3,6,5,4)
    fb_input_tensor2331 = fb_input_tensor233
    fb_input_tensor2331 = fb_input_tensor2331.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2331, fb_input_tensor2331], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor23 = fp_input_tensor2.permute(0,1,2,5,4,3,6)
    fp_input_tensor233 = fp_input_tensor23.permute(0,1,2,3,6,5,4)
    fp_input_tensor2332 = fp_input_tensor233.permute(0,1,2,3,4,6,5)
    fp_input_tensor2332 = fp_input_tensor2332.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor23 = fb_input_tensor2.permute(0,1,2,5,4,3,6)
    fb_input_tensor233 = fb_input_tensor23.permute(0,1,2,3,6,5,4)
    fb_input_tensor2332 = fb_input_tensor233.permute(0,1,2,3,4,6,5)
    fb_input_tensor2332 = fb_input_tensor2332.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2332, fb_input_tensor2332], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor24 = fp_input_tensor2.permute(0,1,2,6,4,5,3)
    fp_input_tensor241 = fp_input_tensor24
    fp_input_tensor2411 = fp_input_tensor241
    fp_input_tensor2411 = fp_input_tensor2411.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor24 = fb_input_tensor2.permute(0,1,2,6,4,5,3)
    fb_input_tensor241 = fb_input_tensor24
    fb_input_tensor2411 = fb_input_tensor241
    fb_input_tensor2411 = fb_input_tensor2411.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2411, fb_input_tensor2411], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor24 = fp_input_tensor2.permute(0,1,2,6,4,5,3)
    fp_input_tensor241 = fp_input_tensor24
    fp_input_tensor2412 = fp_input_tensor241.permute(0,1,2,3,4,6,5)
    fp_input_tensor2412 = fp_input_tensor2412.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor24 = fb_input_tensor2.permute(0,1,2,6,4,5,3)
    fb_input_tensor241 = fb_input_tensor24
    fb_input_tensor2412 = fb_input_tensor241.permute(0,1,2,3,4,6,5)
    fb_input_tensor2412 = fb_input_tensor2412.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2412, fb_input_tensor2412], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor24 = fp_input_tensor2.permute(0,1,2,6,4,5,3)
    fp_input_tensor242 = fp_input_tensor24.permute(0,1,2,3,5,4,6) 
    fp_input_tensor2421 = fp_input_tensor242
    fp_input_tensor2421 = fp_input_tensor2421.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor24 = fb_input_tensor2.permute(0,1,2,6,4,5,3)
    fb_input_tensor242 = fb_input_tensor24.permute(0,1,2,3,5,4,6) 
    fb_input_tensor2421 = fb_input_tensor242
    fb_input_tensor2421 = fb_input_tensor2421.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2421, fb_input_tensor2421], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor24 = fp_input_tensor2.permute(0,1,2,6,4,5,3)
    fp_input_tensor242 = fp_input_tensor24.permute(0,1,2,3,5,4,6) 
    fp_input_tensor2422 = fp_input_tensor242.permute(0,1,2,3,4,6,5)
    fp_input_tensor2422 = fp_input_tensor2422.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor24 = fb_input_tensor2.permute(0,1,2,6,4,5,3)
    fb_input_tensor242 = fb_input_tensor24.permute(0,1,2,3,5,4,6) 
    fb_input_tensor2422 = fb_input_tensor242.permute(0,1,2,3,4,6,5)
    fb_input_tensor2422 = fb_input_tensor2422.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2422, fb_input_tensor2422], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor24 = fp_input_tensor2.permute(0,1,2,6,4,5,3)
    fp_input_tensor243 = fp_input_tensor24.permute(0,1,2,3,6,5,4)
    fp_input_tensor2431 = fp_input_tensor243
    fp_input_tensor2431 = fp_input_tensor2431.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor24 = fb_input_tensor2.permute(0,1,2,6,4,5,3)
    fb_input_tensor243 = fb_input_tensor24.permute(0,1,2,3,6,5,4)
    fb_input_tensor2431 = fb_input_tensor243
    fb_input_tensor2431 = fb_input_tensor2431.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2431, fb_input_tensor2431], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim2432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor2 = facPotentials.permute(0,1,3,2,4,5,6)
    fp_input_tensor24 = fp_input_tensor2.permute(0,1,2,6,4,5,3)
    fp_input_tensor243 = fp_input_tensor24.permute(0,1,2,3,6,5,4)
    fp_input_tensor2432 = fp_input_tensor243.permute(0,1,2,3,4,6,5)
    fp_input_tensor2432 = fp_input_tensor2432.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor2 = facBeliefs.permute(0,1,3,2,4,5,6)
    fb_input_tensor24 = fb_input_tensor2.permute(0,1,2,6,4,5,3)
    fb_input_tensor243 = fb_input_tensor24.permute(0,1,2,3,6,5,4)
    fb_input_tensor2432 = fb_input_tensor243.permute(0,1,2,3,4,6,5)
    fb_input_tensor2432 = fb_input_tensor2432.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor2432, fb_input_tensor2432], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor31 = fp_input_tensor3
    fp_input_tensor311 = fp_input_tensor31
    fp_input_tensor3111 = fp_input_tensor311
    fp_input_tensor3111 = fp_input_tensor3111.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor31 = fb_input_tensor3
    fb_input_tensor311 = fb_input_tensor31
    fb_input_tensor3111 = fb_input_tensor311
    fb_input_tensor3111 = fb_input_tensor3111.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3111, fb_input_tensor3111], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor31 = fp_input_tensor3
    fp_input_tensor311 = fp_input_tensor31
    fp_input_tensor3112 = fp_input_tensor311.permute(0,1,2,3,4,6,5)
    fp_input_tensor3112 = fp_input_tensor3112.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor31 = fb_input_tensor3
    fb_input_tensor311 = fb_input_tensor31
    fb_input_tensor3112 = fb_input_tensor311.permute(0,1,2,3,4,6,5)
    fb_input_tensor3112 = fb_input_tensor3112.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3112, fb_input_tensor3112], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor31 = fp_input_tensor3
    fp_input_tensor312 = fp_input_tensor31.permute(0,1,2,3,5,4,6) 
    fp_input_tensor3121 = fp_input_tensor312
    fp_input_tensor3121 = fp_input_tensor3121.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor31 = fb_input_tensor3
    fb_input_tensor312 = fb_input_tensor31.permute(0,1,2,3,5,4,6) 
    fb_input_tensor3121 = fb_input_tensor312
    fb_input_tensor3121 = fb_input_tensor3121.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3121, fb_input_tensor3121], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor31 = fp_input_tensor3
    fp_input_tensor312 = fp_input_tensor31.permute(0,1,2,3,5,4,6) 
    fp_input_tensor3122 = fp_input_tensor312.permute(0,1,2,3,4,6,5)
    fp_input_tensor3122 = fp_input_tensor3122.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor31 = fb_input_tensor3
    fb_input_tensor312 = fb_input_tensor31.permute(0,1,2,3,5,4,6) 
    fb_input_tensor3122 = fb_input_tensor312.permute(0,1,2,3,4,6,5)
    fb_input_tensor3122 = fb_input_tensor3122.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3122, fb_input_tensor3122], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor31 = fp_input_tensor3
    fp_input_tensor313 = fp_input_tensor31.permute(0,1,2,3,6,5,4)
    fp_input_tensor3131 = fp_input_tensor313
    fp_input_tensor3131 = fp_input_tensor3131.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor31 = fb_input_tensor3
    fb_input_tensor313 = fb_input_tensor31.permute(0,1,2,3,6,5,4)
    fb_input_tensor3131 = fb_input_tensor313
    fb_input_tensor3131 = fb_input_tensor3131.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3131, fb_input_tensor3131], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor31 = fp_input_tensor3
    fp_input_tensor313 = fp_input_tensor31.permute(0,1,2,3,6,5,4)
    fp_input_tensor3132 = fp_input_tensor313.permute(0,1,2,3,4,6,5)
    fp_input_tensor3132 = fp_input_tensor3132.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor31 = fb_input_tensor3
    fb_input_tensor313 = fb_input_tensor31.permute(0,1,2,3,6,5,4)
    fb_input_tensor3132 = fb_input_tensor313.permute(0,1,2,3,4,6,5)
    fb_input_tensor3132 = fb_input_tensor3132.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3132, fb_input_tensor3132], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor32 = fp_input_tensor3.permute(0,1,2,4,3,5,6)
    fp_input_tensor321 = fp_input_tensor32
    fp_input_tensor3211 = fp_input_tensor321
    fp_input_tensor3211 = fp_input_tensor3211.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor32 = fb_input_tensor3.permute(0,1,2,4,3,5,6)
    fb_input_tensor321 = fb_input_tensor32
    fb_input_tensor3211 = fb_input_tensor321
    fb_input_tensor3211 = fb_input_tensor3211.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3211, fb_input_tensor3211], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor32 = fp_input_tensor3.permute(0,1,2,4,3,5,6)
    fp_input_tensor321 = fp_input_tensor32
    fp_input_tensor3212 = fp_input_tensor321.permute(0,1,2,3,4,6,5)
    fp_input_tensor3212 = fp_input_tensor3212.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor32 = fb_input_tensor3.permute(0,1,2,4,3,5,6)
    fb_input_tensor321 = fb_input_tensor32
    fb_input_tensor3212 = fb_input_tensor321.permute(0,1,2,3,4,6,5)
    fb_input_tensor3212 = fb_input_tensor3212.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3212, fb_input_tensor3212], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor32 = fp_input_tensor3.permute(0,1,2,4,3,5,6)
    fp_input_tensor322 = fp_input_tensor32.permute(0,1,2,3,5,4,6) 
    fp_input_tensor3221 = fp_input_tensor322
    fp_input_tensor3221 = fp_input_tensor3221.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor32 = fb_input_tensor3.permute(0,1,2,4,3,5,6)
    fb_input_tensor322 = fb_input_tensor32.permute(0,1,2,3,5,4,6) 
    fb_input_tensor3221 = fb_input_tensor322
    fb_input_tensor3221 = fb_input_tensor3221.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3221, fb_input_tensor3221], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor32 = fp_input_tensor3.permute(0,1,2,4,3,5,6)
    fp_input_tensor322 = fp_input_tensor32.permute(0,1,2,3,5,4,6) 
    fp_input_tensor3222 = fp_input_tensor322.permute(0,1,2,3,4,6,5)
    fp_input_tensor3222 = fp_input_tensor3222.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor32 = fb_input_tensor3.permute(0,1,2,4,3,5,6)
    fb_input_tensor322 = fb_input_tensor32.permute(0,1,2,3,5,4,6) 
    fb_input_tensor3222 = fb_input_tensor322.permute(0,1,2,3,4,6,5)
    fb_input_tensor3222 = fb_input_tensor3222.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3222, fb_input_tensor3222], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor32 = fp_input_tensor3.permute(0,1,2,4,3,5,6)
    fp_input_tensor323 = fp_input_tensor32.permute(0,1,2,3,6,5,4)
    fp_input_tensor3231 = fp_input_tensor323
    fp_input_tensor3231 = fp_input_tensor3231.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor32 = fb_input_tensor3.permute(0,1,2,4,3,5,6)
    fb_input_tensor323 = fb_input_tensor32.permute(0,1,2,3,6,5,4)
    fb_input_tensor3231 = fb_input_tensor323
    fb_input_tensor3231 = fb_input_tensor3231.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3231, fb_input_tensor3231], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor32 = fp_input_tensor3.permute(0,1,2,4,3,5,6)
    fp_input_tensor323 = fp_input_tensor32.permute(0,1,2,3,6,5,4)
    fp_input_tensor3232 = fp_input_tensor323.permute(0,1,2,3,4,6,5)
    fp_input_tensor3232 = fp_input_tensor3232.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor32 = fb_input_tensor3.permute(0,1,2,4,3,5,6)
    fb_input_tensor323 = fb_input_tensor32.permute(0,1,2,3,6,5,4)
    fb_input_tensor3232 = fb_input_tensor323.permute(0,1,2,3,4,6,5)
    fb_input_tensor3232 = fb_input_tensor3232.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3232, fb_input_tensor3232], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor33 = fp_input_tensor3.permute(0,1,2,5,4,3,6)
    fp_input_tensor331 = fp_input_tensor33
    fp_input_tensor3311 = fp_input_tensor331
    fp_input_tensor3311 = fp_input_tensor3311.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor33 = fb_input_tensor3.permute(0,1,2,5,4,3,6)
    fb_input_tensor331 = fb_input_tensor33
    fb_input_tensor3311 = fb_input_tensor331
    fb_input_tensor3311 = fb_input_tensor3311.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3311, fb_input_tensor3311], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor33 = fp_input_tensor3.permute(0,1,2,5,4,3,6)
    fp_input_tensor331 = fp_input_tensor33
    fp_input_tensor3312 = fp_input_tensor331.permute(0,1,2,3,4,6,5)
    fp_input_tensor3312 = fp_input_tensor3312.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor33 = fb_input_tensor3.permute(0,1,2,5,4,3,6)
    fb_input_tensor331 = fb_input_tensor33
    fb_input_tensor3312 = fb_input_tensor331.permute(0,1,2,3,4,6,5)
    fb_input_tensor3312 = fb_input_tensor3312.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3312, fb_input_tensor3312], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor33 = fp_input_tensor3.permute(0,1,2,5,4,3,6)
    fp_input_tensor332 = fp_input_tensor33.permute(0,1,2,3,5,4,6) 
    fp_input_tensor3321 = fp_input_tensor332
    fp_input_tensor3321 = fp_input_tensor3321.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor33 = fb_input_tensor3.permute(0,1,2,5,4,3,6)
    fb_input_tensor332 = fb_input_tensor33.permute(0,1,2,3,5,4,6) 
    fb_input_tensor3321 = fb_input_tensor332
    fb_input_tensor3321 = fb_input_tensor3321.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3321, fb_input_tensor3321], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor33 = fp_input_tensor3.permute(0,1,2,5,4,3,6)
    fp_input_tensor332 = fp_input_tensor33.permute(0,1,2,3,5,4,6) 
    fp_input_tensor3322 = fp_input_tensor332.permute(0,1,2,3,4,6,5)
    fp_input_tensor3322 = fp_input_tensor3322.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor33 = fb_input_tensor3.permute(0,1,2,5,4,3,6)
    fb_input_tensor332 = fb_input_tensor33.permute(0,1,2,3,5,4,6) 
    fb_input_tensor3322 = fb_input_tensor332.permute(0,1,2,3,4,6,5)
    fb_input_tensor3322 = fb_input_tensor3322.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3322, fb_input_tensor3322], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor33 = fp_input_tensor3.permute(0,1,2,5,4,3,6)
    fp_input_tensor333 = fp_input_tensor33.permute(0,1,2,3,6,5,4)
    fp_input_tensor3331 = fp_input_tensor333
    fp_input_tensor3331 = fp_input_tensor3331.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor33 = fb_input_tensor3.permute(0,1,2,5,4,3,6)
    fb_input_tensor333 = fb_input_tensor33.permute(0,1,2,3,6,5,4)
    fb_input_tensor3331 = fb_input_tensor333
    fb_input_tensor3331 = fb_input_tensor3331.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3331, fb_input_tensor3331], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor33 = fp_input_tensor3.permute(0,1,2,5,4,3,6)
    fp_input_tensor333 = fp_input_tensor33.permute(0,1,2,3,6,5,4)
    fp_input_tensor3332 = fp_input_tensor333.permute(0,1,2,3,4,6,5)
    fp_input_tensor3332 = fp_input_tensor3332.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor33 = fb_input_tensor3.permute(0,1,2,5,4,3,6)
    fb_input_tensor333 = fb_input_tensor33.permute(0,1,2,3,6,5,4)
    fb_input_tensor3332 = fb_input_tensor333.permute(0,1,2,3,4,6,5)
    fb_input_tensor3332 = fb_input_tensor3332.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3332, fb_input_tensor3332], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor34 = fp_input_tensor3.permute(0,1,2,6,4,5,3)
    fp_input_tensor341 = fp_input_tensor34
    fp_input_tensor3411 = fp_input_tensor341
    fp_input_tensor3411 = fp_input_tensor3411.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor34 = fb_input_tensor3.permute(0,1,2,6,4,5,3)
    fb_input_tensor341 = fb_input_tensor34
    fb_input_tensor3411 = fb_input_tensor341
    fb_input_tensor3411 = fb_input_tensor3411.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3411, fb_input_tensor3411], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor34 = fp_input_tensor3.permute(0,1,2,6,4,5,3)
    fp_input_tensor341 = fp_input_tensor34
    fp_input_tensor3412 = fp_input_tensor341.permute(0,1,2,3,4,6,5)
    fp_input_tensor3412 = fp_input_tensor3412.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor34 = fb_input_tensor3.permute(0,1,2,6,4,5,3)
    fb_input_tensor341 = fb_input_tensor34
    fb_input_tensor3412 = fb_input_tensor341.permute(0,1,2,3,4,6,5)
    fb_input_tensor3412 = fb_input_tensor3412.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3412, fb_input_tensor3412], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor34 = fp_input_tensor3.permute(0,1,2,6,4,5,3)
    fp_input_tensor342 = fp_input_tensor34.permute(0,1,2,3,5,4,6) 
    fp_input_tensor3421 = fp_input_tensor342
    fp_input_tensor3421 = fp_input_tensor3421.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor34 = fb_input_tensor3.permute(0,1,2,6,4,5,3)
    fb_input_tensor342 = fb_input_tensor34.permute(0,1,2,3,5,4,6) 
    fb_input_tensor3421 = fb_input_tensor342
    fb_input_tensor3421 = fb_input_tensor3421.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3421, fb_input_tensor3421], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor34 = fp_input_tensor3.permute(0,1,2,6,4,5,3)
    fp_input_tensor342 = fp_input_tensor34.permute(0,1,2,3,5,4,6) 
    fp_input_tensor3422 = fp_input_tensor342.permute(0,1,2,3,4,6,5)
    fp_input_tensor3422 = fp_input_tensor3422.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor34 = fb_input_tensor3.permute(0,1,2,6,4,5,3)
    fb_input_tensor342 = fb_input_tensor34.permute(0,1,2,3,5,4,6) 
    fb_input_tensor3422 = fb_input_tensor342.permute(0,1,2,3,4,6,5)
    fb_input_tensor3422 = fb_input_tensor3422.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3422, fb_input_tensor3422], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor34 = fp_input_tensor3.permute(0,1,2,6,4,5,3)
    fp_input_tensor343 = fp_input_tensor34.permute(0,1,2,3,6,5,4)
    fp_input_tensor3431 = fp_input_tensor343
    fp_input_tensor3431 = fp_input_tensor3431.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor34 = fb_input_tensor3.permute(0,1,2,6,4,5,3)
    fb_input_tensor343 = fb_input_tensor34.permute(0,1,2,3,6,5,4)
    fb_input_tensor3431 = fb_input_tensor343
    fb_input_tensor3431 = fb_input_tensor3431.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3431, fb_input_tensor3431], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim3432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor3 = facPotentials.permute(0,1,4,3,2,5,6)
    fp_input_tensor34 = fp_input_tensor3.permute(0,1,2,6,4,5,3)
    fp_input_tensor343 = fp_input_tensor34.permute(0,1,2,3,6,5,4)
    fp_input_tensor3432 = fp_input_tensor343.permute(0,1,2,3,4,6,5)
    fp_input_tensor3432 = fp_input_tensor3432.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor3 = facBeliefs.permute(0,1,4,3,2,5,6)
    fb_input_tensor34 = fb_input_tensor3.permute(0,1,2,6,4,5,3)
    fb_input_tensor343 = fb_input_tensor34.permute(0,1,2,3,6,5,4)
    fb_input_tensor3432 = fb_input_tensor343.permute(0,1,2,3,4,6,5)
    fb_input_tensor3432 = fb_input_tensor3432.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor3432, fb_input_tensor3432], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor41 = fp_input_tensor4
    fp_input_tensor411 = fp_input_tensor41
    fp_input_tensor4111 = fp_input_tensor411
    fp_input_tensor4111 = fp_input_tensor4111.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor41 = fb_input_tensor4
    fb_input_tensor411 = fb_input_tensor41
    fb_input_tensor4111 = fb_input_tensor411
    fb_input_tensor4111 = fb_input_tensor4111.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4111, fb_input_tensor4111], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor41 = fp_input_tensor4
    fp_input_tensor411 = fp_input_tensor41
    fp_input_tensor4112 = fp_input_tensor411.permute(0,1,2,3,4,6,5)
    fp_input_tensor4112 = fp_input_tensor4112.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor41 = fb_input_tensor4
    fb_input_tensor411 = fb_input_tensor41
    fb_input_tensor4112 = fb_input_tensor411.permute(0,1,2,3,4,6,5)
    fb_input_tensor4112 = fb_input_tensor4112.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4112, fb_input_tensor4112], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor41 = fp_input_tensor4
    fp_input_tensor412 = fp_input_tensor41.permute(0,1,2,3,5,4,6) 
    fp_input_tensor4121 = fp_input_tensor412
    fp_input_tensor4121 = fp_input_tensor4121.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor41 = fb_input_tensor4
    fb_input_tensor412 = fb_input_tensor41.permute(0,1,2,3,5,4,6) 
    fb_input_tensor4121 = fb_input_tensor412
    fb_input_tensor4121 = fb_input_tensor4121.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4121, fb_input_tensor4121], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor41 = fp_input_tensor4
    fp_input_tensor412 = fp_input_tensor41.permute(0,1,2,3,5,4,6) 
    fp_input_tensor4122 = fp_input_tensor412.permute(0,1,2,3,4,6,5)
    fp_input_tensor4122 = fp_input_tensor4122.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor41 = fb_input_tensor4
    fb_input_tensor412 = fb_input_tensor41.permute(0,1,2,3,5,4,6) 
    fb_input_tensor4122 = fb_input_tensor412.permute(0,1,2,3,4,6,5)
    fb_input_tensor4122 = fb_input_tensor4122.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4122, fb_input_tensor4122], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor41 = fp_input_tensor4
    fp_input_tensor413 = fp_input_tensor41.permute(0,1,2,3,6,5,4)
    fp_input_tensor4131 = fp_input_tensor413
    fp_input_tensor4131 = fp_input_tensor4131.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor41 = fb_input_tensor4
    fb_input_tensor413 = fb_input_tensor41.permute(0,1,2,3,6,5,4)
    fb_input_tensor4131 = fb_input_tensor413
    fb_input_tensor4131 = fb_input_tensor4131.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4131, fb_input_tensor4131], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor41 = fp_input_tensor4
    fp_input_tensor413 = fp_input_tensor41.permute(0,1,2,3,6,5,4)
    fp_input_tensor4132 = fp_input_tensor413.permute(0,1,2,3,4,6,5)
    fp_input_tensor4132 = fp_input_tensor4132.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor41 = fb_input_tensor4
    fb_input_tensor413 = fb_input_tensor41.permute(0,1,2,3,6,5,4)
    fb_input_tensor4132 = fb_input_tensor413.permute(0,1,2,3,4,6,5)
    fb_input_tensor4132 = fb_input_tensor4132.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4132, fb_input_tensor4132], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor42 = fp_input_tensor4.permute(0,1,2,4,3,5,6)
    fp_input_tensor421 = fp_input_tensor42
    fp_input_tensor4211 = fp_input_tensor421
    fp_input_tensor4211 = fp_input_tensor4211.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor42 = fb_input_tensor4.permute(0,1,2,4,3,5,6)
    fb_input_tensor421 = fb_input_tensor42
    fb_input_tensor4211 = fb_input_tensor421
    fb_input_tensor4211 = fb_input_tensor4211.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4211, fb_input_tensor4211], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor42 = fp_input_tensor4.permute(0,1,2,4,3,5,6)
    fp_input_tensor421 = fp_input_tensor42
    fp_input_tensor4212 = fp_input_tensor421.permute(0,1,2,3,4,6,5)
    fp_input_tensor4212 = fp_input_tensor4212.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor42 = fb_input_tensor4.permute(0,1,2,4,3,5,6)
    fb_input_tensor421 = fb_input_tensor42
    fb_input_tensor4212 = fb_input_tensor421.permute(0,1,2,3,4,6,5)
    fb_input_tensor4212 = fb_input_tensor4212.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4212, fb_input_tensor4212], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor42 = fp_input_tensor4.permute(0,1,2,4,3,5,6)
    fp_input_tensor422 = fp_input_tensor42.permute(0,1,2,3,5,4,6) 
    fp_input_tensor4221 = fp_input_tensor422
    fp_input_tensor4221 = fp_input_tensor4221.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor42 = fb_input_tensor4.permute(0,1,2,4,3,5,6)
    fb_input_tensor422 = fb_input_tensor42.permute(0,1,2,3,5,4,6) 
    fb_input_tensor4221 = fb_input_tensor422
    fb_input_tensor4221 = fb_input_tensor4221.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4221, fb_input_tensor4221], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor42 = fp_input_tensor4.permute(0,1,2,4,3,5,6)
    fp_input_tensor422 = fp_input_tensor42.permute(0,1,2,3,5,4,6) 
    fp_input_tensor4222 = fp_input_tensor422.permute(0,1,2,3,4,6,5)
    fp_input_tensor4222 = fp_input_tensor4222.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor42 = fb_input_tensor4.permute(0,1,2,4,3,5,6)
    fb_input_tensor422 = fb_input_tensor42.permute(0,1,2,3,5,4,6) 
    fb_input_tensor4222 = fb_input_tensor422.permute(0,1,2,3,4,6,5)
    fb_input_tensor4222 = fb_input_tensor4222.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4222, fb_input_tensor4222], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor42 = fp_input_tensor4.permute(0,1,2,4,3,5,6)
    fp_input_tensor423 = fp_input_tensor42.permute(0,1,2,3,6,5,4)
    fp_input_tensor4231 = fp_input_tensor423
    fp_input_tensor4231 = fp_input_tensor4231.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor42 = fb_input_tensor4.permute(0,1,2,4,3,5,6)
    fb_input_tensor423 = fb_input_tensor42.permute(0,1,2,3,6,5,4)
    fb_input_tensor4231 = fb_input_tensor423
    fb_input_tensor4231 = fb_input_tensor4231.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4231, fb_input_tensor4231], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor42 = fp_input_tensor4.permute(0,1,2,4,3,5,6)
    fp_input_tensor423 = fp_input_tensor42.permute(0,1,2,3,6,5,4)
    fp_input_tensor4232 = fp_input_tensor423.permute(0,1,2,3,4,6,5)
    fp_input_tensor4232 = fp_input_tensor4232.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor42 = fb_input_tensor4.permute(0,1,2,4,3,5,6)
    fb_input_tensor423 = fb_input_tensor42.permute(0,1,2,3,6,5,4)
    fb_input_tensor4232 = fb_input_tensor423.permute(0,1,2,3,4,6,5)
    fb_input_tensor4232 = fb_input_tensor4232.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4232, fb_input_tensor4232], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor43 = fp_input_tensor4.permute(0,1,2,5,4,3,6)
    fp_input_tensor431 = fp_input_tensor43
    fp_input_tensor4311 = fp_input_tensor431
    fp_input_tensor4311 = fp_input_tensor4311.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor43 = fb_input_tensor4.permute(0,1,2,5,4,3,6)
    fb_input_tensor431 = fb_input_tensor43
    fb_input_tensor4311 = fb_input_tensor431
    fb_input_tensor4311 = fb_input_tensor4311.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4311, fb_input_tensor4311], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor43 = fp_input_tensor4.permute(0,1,2,5,4,3,6)
    fp_input_tensor431 = fp_input_tensor43
    fp_input_tensor4312 = fp_input_tensor431.permute(0,1,2,3,4,6,5)
    fp_input_tensor4312 = fp_input_tensor4312.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor43 = fb_input_tensor4.permute(0,1,2,5,4,3,6)
    fb_input_tensor431 = fb_input_tensor43
    fb_input_tensor4312 = fb_input_tensor431.permute(0,1,2,3,4,6,5)
    fb_input_tensor4312 = fb_input_tensor4312.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4312, fb_input_tensor4312], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor43 = fp_input_tensor4.permute(0,1,2,5,4,3,6)
    fp_input_tensor432 = fp_input_tensor43.permute(0,1,2,3,5,4,6) 
    fp_input_tensor4321 = fp_input_tensor432
    fp_input_tensor4321 = fp_input_tensor4321.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor43 = fb_input_tensor4.permute(0,1,2,5,4,3,6)
    fb_input_tensor432 = fb_input_tensor43.permute(0,1,2,3,5,4,6) 
    fb_input_tensor4321 = fb_input_tensor432
    fb_input_tensor4321 = fb_input_tensor4321.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4321, fb_input_tensor4321], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor43 = fp_input_tensor4.permute(0,1,2,5,4,3,6)
    fp_input_tensor432 = fp_input_tensor43.permute(0,1,2,3,5,4,6) 
    fp_input_tensor4322 = fp_input_tensor432.permute(0,1,2,3,4,6,5)
    fp_input_tensor4322 = fp_input_tensor4322.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor43 = fb_input_tensor4.permute(0,1,2,5,4,3,6)
    fb_input_tensor432 = fb_input_tensor43.permute(0,1,2,3,5,4,6) 
    fb_input_tensor4322 = fb_input_tensor432.permute(0,1,2,3,4,6,5)
    fb_input_tensor4322 = fb_input_tensor4322.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4322, fb_input_tensor4322], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor43 = fp_input_tensor4.permute(0,1,2,5,4,3,6)
    fp_input_tensor433 = fp_input_tensor43.permute(0,1,2,3,6,5,4)
    fp_input_tensor4331 = fp_input_tensor433
    fp_input_tensor4331 = fp_input_tensor4331.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor43 = fb_input_tensor4.permute(0,1,2,5,4,3,6)
    fb_input_tensor433 = fb_input_tensor43.permute(0,1,2,3,6,5,4)
    fb_input_tensor4331 = fb_input_tensor433
    fb_input_tensor4331 = fb_input_tensor4331.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4331, fb_input_tensor4331], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor43 = fp_input_tensor4.permute(0,1,2,5,4,3,6)
    fp_input_tensor433 = fp_input_tensor43.permute(0,1,2,3,6,5,4)
    fp_input_tensor4332 = fp_input_tensor433.permute(0,1,2,3,4,6,5)
    fp_input_tensor4332 = fp_input_tensor4332.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor43 = fb_input_tensor4.permute(0,1,2,5,4,3,6)
    fb_input_tensor433 = fb_input_tensor43.permute(0,1,2,3,6,5,4)
    fb_input_tensor4332 = fb_input_tensor433.permute(0,1,2,3,4,6,5)
    fb_input_tensor4332 = fb_input_tensor4332.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4332, fb_input_tensor4332], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor44 = fp_input_tensor4.permute(0,1,2,6,4,5,3)
    fp_input_tensor441 = fp_input_tensor44
    fp_input_tensor4411 = fp_input_tensor441
    fp_input_tensor4411 = fp_input_tensor4411.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor44 = fb_input_tensor4.permute(0,1,2,6,4,5,3)
    fb_input_tensor441 = fb_input_tensor44
    fb_input_tensor4411 = fb_input_tensor441
    fb_input_tensor4411 = fb_input_tensor4411.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4411, fb_input_tensor4411], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor44 = fp_input_tensor4.permute(0,1,2,6,4,5,3)
    fp_input_tensor441 = fp_input_tensor44
    fp_input_tensor4412 = fp_input_tensor441.permute(0,1,2,3,4,6,5)
    fp_input_tensor4412 = fp_input_tensor4412.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor44 = fb_input_tensor4.permute(0,1,2,6,4,5,3)
    fb_input_tensor441 = fb_input_tensor44
    fb_input_tensor4412 = fb_input_tensor441.permute(0,1,2,3,4,6,5)
    fb_input_tensor4412 = fb_input_tensor4412.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4412, fb_input_tensor4412], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor44 = fp_input_tensor4.permute(0,1,2,6,4,5,3)
    fp_input_tensor442 = fp_input_tensor44.permute(0,1,2,3,5,4,6) 
    fp_input_tensor4421 = fp_input_tensor442
    fp_input_tensor4421 = fp_input_tensor4421.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor44 = fb_input_tensor4.permute(0,1,2,6,4,5,3)
    fb_input_tensor442 = fb_input_tensor44.permute(0,1,2,3,5,4,6) 
    fb_input_tensor4421 = fb_input_tensor442
    fb_input_tensor4421 = fb_input_tensor4421.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4421, fb_input_tensor4421], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor44 = fp_input_tensor4.permute(0,1,2,6,4,5,3)
    fp_input_tensor442 = fp_input_tensor44.permute(0,1,2,3,5,4,6) 
    fp_input_tensor4422 = fp_input_tensor442.permute(0,1,2,3,4,6,5)
    fp_input_tensor4422 = fp_input_tensor4422.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor44 = fb_input_tensor4.permute(0,1,2,6,4,5,3)
    fb_input_tensor442 = fb_input_tensor44.permute(0,1,2,3,5,4,6) 
    fb_input_tensor4422 = fb_input_tensor442.permute(0,1,2,3,4,6,5)
    fb_input_tensor4422 = fb_input_tensor4422.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4422, fb_input_tensor4422], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor44 = fp_input_tensor4.permute(0,1,2,6,4,5,3)
    fp_input_tensor443 = fp_input_tensor44.permute(0,1,2,3,6,5,4)
    fp_input_tensor4431 = fp_input_tensor443
    fp_input_tensor4431 = fp_input_tensor4431.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor44 = fb_input_tensor4.permute(0,1,2,6,4,5,3)
    fb_input_tensor443 = fb_input_tensor44.permute(0,1,2,3,6,5,4)
    fb_input_tensor4431 = fb_input_tensor443
    fb_input_tensor4431 = fb_input_tensor4431.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4431, fb_input_tensor4431], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim4432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor4 = facPotentials.permute(0,1,5,3,4,2,6)
    fp_input_tensor44 = fp_input_tensor4.permute(0,1,2,6,4,5,3)
    fp_input_tensor443 = fp_input_tensor44.permute(0,1,2,3,6,5,4)
    fp_input_tensor4432 = fp_input_tensor443.permute(0,1,2,3,4,6,5)
    fp_input_tensor4432 = fp_input_tensor4432.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor4 = facBeliefs.permute(0,1,5,3,4,2,6)
    fb_input_tensor44 = fb_input_tensor4.permute(0,1,2,6,4,5,3)
    fb_input_tensor443 = fb_input_tensor44.permute(0,1,2,3,6,5,4)
    fb_input_tensor4432 = fb_input_tensor443.permute(0,1,2,3,4,6,5)
    fb_input_tensor4432 = fb_input_tensor4432.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor4432, fb_input_tensor4432], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5111(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor51 = fp_input_tensor5
    fp_input_tensor511 = fp_input_tensor51
    fp_input_tensor5111 = fp_input_tensor511
    fp_input_tensor5111 = fp_input_tensor5111.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor51 = fb_input_tensor5
    fb_input_tensor511 = fb_input_tensor51
    fb_input_tensor5111 = fb_input_tensor511
    fb_input_tensor5111 = fb_input_tensor5111.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5111, fb_input_tensor5111], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5112(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor51 = fp_input_tensor5
    fp_input_tensor511 = fp_input_tensor51
    fp_input_tensor5112 = fp_input_tensor511.permute(0,1,2,3,4,6,5)
    fp_input_tensor5112 = fp_input_tensor5112.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor51 = fb_input_tensor5
    fb_input_tensor511 = fb_input_tensor51
    fb_input_tensor5112 = fb_input_tensor511.permute(0,1,2,3,4,6,5)
    fb_input_tensor5112 = fb_input_tensor5112.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5112, fb_input_tensor5112], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5121(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor51 = fp_input_tensor5
    fp_input_tensor512 = fp_input_tensor51.permute(0,1,2,3,5,4,6) 
    fp_input_tensor5121 = fp_input_tensor512
    fp_input_tensor5121 = fp_input_tensor5121.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor51 = fb_input_tensor5
    fb_input_tensor512 = fb_input_tensor51.permute(0,1,2,3,5,4,6) 
    fb_input_tensor5121 = fb_input_tensor512
    fb_input_tensor5121 = fb_input_tensor5121.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5121, fb_input_tensor5121], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5122(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor51 = fp_input_tensor5
    fp_input_tensor512 = fp_input_tensor51.permute(0,1,2,3,5,4,6) 
    fp_input_tensor5122 = fp_input_tensor512.permute(0,1,2,3,4,6,5)
    fp_input_tensor5122 = fp_input_tensor5122.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor51 = fb_input_tensor5
    fb_input_tensor512 = fb_input_tensor51.permute(0,1,2,3,5,4,6) 
    fb_input_tensor5122 = fb_input_tensor512.permute(0,1,2,3,4,6,5)
    fb_input_tensor5122 = fb_input_tensor5122.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5122, fb_input_tensor5122], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5131(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor51 = fp_input_tensor5
    fp_input_tensor513 = fp_input_tensor51.permute(0,1,2,3,6,5,4)
    fp_input_tensor5131 = fp_input_tensor513
    fp_input_tensor5131 = fp_input_tensor5131.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor51 = fb_input_tensor5
    fb_input_tensor513 = fb_input_tensor51.permute(0,1,2,3,6,5,4)
    fb_input_tensor5131 = fb_input_tensor513
    fb_input_tensor5131 = fb_input_tensor5131.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5131, fb_input_tensor5131], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5132(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor51 = fp_input_tensor5
    fp_input_tensor513 = fp_input_tensor51.permute(0,1,2,3,6,5,4)
    fp_input_tensor5132 = fp_input_tensor513.permute(0,1,2,3,4,6,5)
    fp_input_tensor5132 = fp_input_tensor5132.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor51 = fb_input_tensor5
    fb_input_tensor513 = fb_input_tensor51.permute(0,1,2,3,6,5,4)
    fb_input_tensor5132 = fb_input_tensor513.permute(0,1,2,3,4,6,5)
    fb_input_tensor5132 = fb_input_tensor5132.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5132, fb_input_tensor5132], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5211(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor52 = fp_input_tensor5.permute(0,1,2,4,3,5,6)
    fp_input_tensor521 = fp_input_tensor52
    fp_input_tensor5211 = fp_input_tensor521
    fp_input_tensor5211 = fp_input_tensor5211.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor52 = fb_input_tensor5.permute(0,1,2,4,3,5,6)
    fb_input_tensor521 = fb_input_tensor52
    fb_input_tensor5211 = fb_input_tensor521
    fb_input_tensor5211 = fb_input_tensor5211.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5211, fb_input_tensor5211], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5212(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor52 = fp_input_tensor5.permute(0,1,2,4,3,5,6)
    fp_input_tensor521 = fp_input_tensor52
    fp_input_tensor5212 = fp_input_tensor521.permute(0,1,2,3,4,6,5)
    fp_input_tensor5212 = fp_input_tensor5212.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor52 = fb_input_tensor5.permute(0,1,2,4,3,5,6)
    fb_input_tensor521 = fb_input_tensor52
    fb_input_tensor5212 = fb_input_tensor521.permute(0,1,2,3,4,6,5)
    fb_input_tensor5212 = fb_input_tensor5212.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5212, fb_input_tensor5212], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5221(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor52 = fp_input_tensor5.permute(0,1,2,4,3,5,6)
    fp_input_tensor522 = fp_input_tensor52.permute(0,1,2,3,5,4,6) 
    fp_input_tensor5221 = fp_input_tensor522
    fp_input_tensor5221 = fp_input_tensor5221.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor52 = fb_input_tensor5.permute(0,1,2,4,3,5,6)
    fb_input_tensor522 = fb_input_tensor52.permute(0,1,2,3,5,4,6) 
    fb_input_tensor5221 = fb_input_tensor522
    fb_input_tensor5221 = fb_input_tensor5221.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5221, fb_input_tensor5221], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5222(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor52 = fp_input_tensor5.permute(0,1,2,4,3,5,6)
    fp_input_tensor522 = fp_input_tensor52.permute(0,1,2,3,5,4,6) 
    fp_input_tensor5222 = fp_input_tensor522.permute(0,1,2,3,4,6,5)
    fp_input_tensor5222 = fp_input_tensor5222.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor52 = fb_input_tensor5.permute(0,1,2,4,3,5,6)
    fb_input_tensor522 = fb_input_tensor52.permute(0,1,2,3,5,4,6) 
    fb_input_tensor5222 = fb_input_tensor522.permute(0,1,2,3,4,6,5)
    fb_input_tensor5222 = fb_input_tensor5222.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5222, fb_input_tensor5222], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5231(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor52 = fp_input_tensor5.permute(0,1,2,4,3,5,6)
    fp_input_tensor523 = fp_input_tensor52.permute(0,1,2,3,6,5,4)
    fp_input_tensor5231 = fp_input_tensor523
    fp_input_tensor5231 = fp_input_tensor5231.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor52 = fb_input_tensor5.permute(0,1,2,4,3,5,6)
    fb_input_tensor523 = fb_input_tensor52.permute(0,1,2,3,6,5,4)
    fb_input_tensor5231 = fb_input_tensor523
    fb_input_tensor5231 = fb_input_tensor5231.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5231, fb_input_tensor5231], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5232(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor52 = fp_input_tensor5.permute(0,1,2,4,3,5,6)
    fp_input_tensor523 = fp_input_tensor52.permute(0,1,2,3,6,5,4)
    fp_input_tensor5232 = fp_input_tensor523.permute(0,1,2,3,4,6,5)
    fp_input_tensor5232 = fp_input_tensor5232.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor52 = fb_input_tensor5.permute(0,1,2,4,3,5,6)
    fb_input_tensor523 = fb_input_tensor52.permute(0,1,2,3,6,5,4)
    fb_input_tensor5232 = fb_input_tensor523.permute(0,1,2,3,4,6,5)
    fb_input_tensor5232 = fb_input_tensor5232.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5232, fb_input_tensor5232], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5311(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor53 = fp_input_tensor5.permute(0,1,2,5,4,3,6)
    fp_input_tensor531 = fp_input_tensor53
    fp_input_tensor5311 = fp_input_tensor531
    fp_input_tensor5311 = fp_input_tensor5311.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor53 = fb_input_tensor5.permute(0,1,2,5,4,3,6)
    fb_input_tensor531 = fb_input_tensor53
    fb_input_tensor5311 = fb_input_tensor531
    fb_input_tensor5311 = fb_input_tensor5311.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5311, fb_input_tensor5311], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5312(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor53 = fp_input_tensor5.permute(0,1,2,5,4,3,6)
    fp_input_tensor531 = fp_input_tensor53
    fp_input_tensor5312 = fp_input_tensor531.permute(0,1,2,3,4,6,5)
    fp_input_tensor5312 = fp_input_tensor5312.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor53 = fb_input_tensor5.permute(0,1,2,5,4,3,6)
    fb_input_tensor531 = fb_input_tensor53
    fb_input_tensor5312 = fb_input_tensor531.permute(0,1,2,3,4,6,5)
    fb_input_tensor5312 = fb_input_tensor5312.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5312, fb_input_tensor5312], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5321(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor53 = fp_input_tensor5.permute(0,1,2,5,4,3,6)
    fp_input_tensor532 = fp_input_tensor53.permute(0,1,2,3,5,4,6) 
    fp_input_tensor5321 = fp_input_tensor532
    fp_input_tensor5321 = fp_input_tensor5321.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor53 = fb_input_tensor5.permute(0,1,2,5,4,3,6)
    fb_input_tensor532 = fb_input_tensor53.permute(0,1,2,3,5,4,6) 
    fb_input_tensor5321 = fb_input_tensor532
    fb_input_tensor5321 = fb_input_tensor5321.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5321, fb_input_tensor5321], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5322(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor53 = fp_input_tensor5.permute(0,1,2,5,4,3,6)
    fp_input_tensor532 = fp_input_tensor53.permute(0,1,2,3,5,4,6) 
    fp_input_tensor5322 = fp_input_tensor532.permute(0,1,2,3,4,6,5)
    fp_input_tensor5322 = fp_input_tensor5322.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor53 = fb_input_tensor5.permute(0,1,2,5,4,3,6)
    fb_input_tensor532 = fb_input_tensor53.permute(0,1,2,3,5,4,6) 
    fb_input_tensor5322 = fb_input_tensor532.permute(0,1,2,3,4,6,5)
    fb_input_tensor5322 = fb_input_tensor5322.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5322, fb_input_tensor5322], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5331(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor53 = fp_input_tensor5.permute(0,1,2,5,4,3,6)
    fp_input_tensor533 = fp_input_tensor53.permute(0,1,2,3,6,5,4)
    fp_input_tensor5331 = fp_input_tensor533
    fp_input_tensor5331 = fp_input_tensor5331.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor53 = fb_input_tensor5.permute(0,1,2,5,4,3,6)
    fb_input_tensor533 = fb_input_tensor53.permute(0,1,2,3,6,5,4)
    fb_input_tensor5331 = fb_input_tensor533
    fb_input_tensor5331 = fb_input_tensor5331.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5331, fb_input_tensor5331], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5332(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor53 = fp_input_tensor5.permute(0,1,2,5,4,3,6)
    fp_input_tensor533 = fp_input_tensor53.permute(0,1,2,3,6,5,4)
    fp_input_tensor5332 = fp_input_tensor533.permute(0,1,2,3,4,6,5)
    fp_input_tensor5332 = fp_input_tensor5332.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor53 = fb_input_tensor5.permute(0,1,2,5,4,3,6)
    fb_input_tensor533 = fb_input_tensor53.permute(0,1,2,3,6,5,4)
    fb_input_tensor5332 = fb_input_tensor533.permute(0,1,2,3,4,6,5)
    fb_input_tensor5332 = fb_input_tensor5332.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5332, fb_input_tensor5332], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5411(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor54 = fp_input_tensor5.permute(0,1,2,6,4,5,3)
    fp_input_tensor541 = fp_input_tensor54
    fp_input_tensor5411 = fp_input_tensor541
    fp_input_tensor5411 = fp_input_tensor5411.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor54 = fb_input_tensor5.permute(0,1,2,6,4,5,3)
    fb_input_tensor541 = fb_input_tensor54
    fb_input_tensor5411 = fb_input_tensor541
    fb_input_tensor5411 = fb_input_tensor5411.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5411, fb_input_tensor5411], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5412(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor54 = fp_input_tensor5.permute(0,1,2,6,4,5,3)
    fp_input_tensor541 = fp_input_tensor54
    fp_input_tensor5412 = fp_input_tensor541.permute(0,1,2,3,4,6,5)
    fp_input_tensor5412 = fp_input_tensor5412.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor54 = fb_input_tensor5.permute(0,1,2,6,4,5,3)
    fb_input_tensor541 = fb_input_tensor54
    fb_input_tensor5412 = fb_input_tensor541.permute(0,1,2,3,4,6,5)
    fb_input_tensor5412 = fb_input_tensor5412.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5412, fb_input_tensor5412], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5421(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor54 = fp_input_tensor5.permute(0,1,2,6,4,5,3)
    fp_input_tensor542 = fp_input_tensor54.permute(0,1,2,3,5,4,6) 
    fp_input_tensor5421 = fp_input_tensor542
    fp_input_tensor5421 = fp_input_tensor5421.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor54 = fb_input_tensor5.permute(0,1,2,6,4,5,3)
    fb_input_tensor542 = fb_input_tensor54.permute(0,1,2,3,5,4,6) 
    fb_input_tensor5421 = fb_input_tensor542
    fb_input_tensor5421 = fb_input_tensor5421.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5421, fb_input_tensor5421], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5422(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor54 = fp_input_tensor5.permute(0,1,2,6,4,5,3)
    fp_input_tensor542 = fp_input_tensor54.permute(0,1,2,3,5,4,6) 
    fp_input_tensor5422 = fp_input_tensor542.permute(0,1,2,3,4,6,5)
    fp_input_tensor5422 = fp_input_tensor5422.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor54 = fb_input_tensor5.permute(0,1,2,6,4,5,3)
    fb_input_tensor542 = fb_input_tensor54.permute(0,1,2,3,5,4,6) 
    fb_input_tensor5422 = fb_input_tensor542.permute(0,1,2,3,4,6,5)
    fb_input_tensor5422 = fb_input_tensor5422.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5422, fb_input_tensor5422], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5431(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor54 = fp_input_tensor5.permute(0,1,2,6,4,5,3)
    fp_input_tensor543 = fp_input_tensor54.permute(0,1,2,3,6,5,4)
    fp_input_tensor5431 = fp_input_tensor543
    fp_input_tensor5431 = fp_input_tensor5431.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor54 = fb_input_tensor5.permute(0,1,2,6,4,5,3)
    fb_input_tensor543 = fb_input_tensor54.permute(0,1,2,3,6,5,4)
    fb_input_tensor5431 = fb_input_tensor543
    fb_input_tensor5431 = fb_input_tensor5431.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5431, fb_input_tensor5431], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def permute_dim5432(mlp, input_tensor):
    input_tensor_shape = input_tensor.shape

    assert(len(input_tensor_shape) == 3)
    assert(input_tensor_shape[2] == 64)
    batch_size = input_tensor_shape[0]
    msg_passing_layers = input_tensor_shape[1]
    factor_shape = [batch_size, msg_passing_layers, 2, 2, 2, 2, 2]
    flat_factor_shape = [batch_size, msg_passing_layers, 32]
    facPotentials = input_tensor[:, :, :32]
    facBeliefs = input_tensor[:, :, 32:]

    facPotentials = facPotentials.reshape(factor_shape)
    fp_input_tensor5 = facPotentials.permute(0,1,6,3,4,5,2)
    fp_input_tensor54 = fp_input_tensor5.permute(0,1,2,6,4,5,3)
    fp_input_tensor543 = fp_input_tensor54.permute(0,1,2,3,6,5,4)
    fp_input_tensor5432 = fp_input_tensor543.permute(0,1,2,3,4,6,5)
    fp_input_tensor5432 = fp_input_tensor5432.reshape(flat_factor_shape)

    facBeliefs = facBeliefs.reshape(factor_shape)
    fb_input_tensor5 = facBeliefs.permute(0,1,6,3,4,5,2)
    fb_input_tensor54 = fb_input_tensor5.permute(0,1,2,6,4,5,3)
    fb_input_tensor543 = fb_input_tensor54.permute(0,1,2,3,6,5,4)
    fb_input_tensor5432 = fb_input_tensor543.permute(0,1,2,3,4,6,5)
    fb_input_tensor5432 = fb_input_tensor5432.reshape(flat_factor_shape)

    permuted_input_tensor = torch.cat([fp_input_tensor5432, fb_input_tensor5432], dim=2)
    output = mlp(permuted_input_tensor.reshape(batch_size, -1))
    return output

def var_idx_perm_equivariant_5dfactor_sample(mlp, input_tensor, functions_to_sample=8):
    '''
    A factor containing 5 variables has 5! orderings of the variables.
    Sample functions_to_sample permutations of input_tensor dimensions, apply input mlp to each flattened tensor and unpermute, and return mean to make mlp invariant to variable ordering
    '''
    permutation_functions = [permute_dim1111, permute_dim1112, permute_dim1121, permute_dim1122, permute_dim1131, permute_dim1132, permute_dim1211, permute_dim1212, permute_dim1221, permute_dim1222, permute_dim1231, permute_dim1232, permute_dim1311, permute_dim1312, permute_dim1321, permute_dim1322, permute_dim1331, permute_dim1332, permute_dim1411, permute_dim1412, permute_dim1421, permute_dim1422, permute_dim1431, permute_dim1432, permute_dim2111, permute_dim2112, permute_dim2121, permute_dim2122, permute_dim2131, permute_dim2132, permute_dim2211, permute_dim2212, permute_dim2221, permute_dim2222, permute_dim2231, permute_dim2232, permute_dim2311, permute_dim2312, permute_dim2321, permute_dim2322, permute_dim2331, permute_dim2332, permute_dim2411, permute_dim2412, permute_dim2421, permute_dim2422, permute_dim2431, permute_dim2432, permute_dim3111, permute_dim3112, permute_dim3121, permute_dim3122, permute_dim3131, permute_dim3132, permute_dim3211, permute_dim3212, permute_dim3221, permute_dim3222, permute_dim3231, permute_dim3232, permute_dim3311, permute_dim3312, permute_dim3321, permute_dim3322, permute_dim3331, permute_dim3332, permute_dim3411, permute_dim3412, permute_dim3421, permute_dim3422, permute_dim3431, permute_dim3432, permute_dim4111, permute_dim4112, permute_dim4121, permute_dim4122, permute_dim4131, permute_dim4132, permute_dim4211, permute_dim4212, permute_dim4221, permute_dim4222, permute_dim4231, permute_dim4232, permute_dim4311, permute_dim4312, permute_dim4321, permute_dim4322, permute_dim4331, permute_dim4332, permute_dim4411, permute_dim4412, permute_dim4421, permute_dim4422, permute_dim4431, permute_dim4432, permute_dim5111, permute_dim5112, permute_dim5121, permute_dim5122, permute_dim5131, permute_dim5132, permute_dim5211, permute_dim5212, permute_dim5221, permute_dim5222, permute_dim5231, permute_dim5232, permute_dim5311, permute_dim5312, permute_dim5321, permute_dim5322, permute_dim5331, permute_dim5332, permute_dim5411, permute_dim5412, permute_dim5421, permute_dim5422, permute_dim5431, permute_dim5432]
    sampled_functions = random.sample(permutation_functions, k=functions_to_sample)
    output = sampled_functions[0](mlp, input_tensor)
    for sampled_function in sampled_functions[1:]:
        output += sampled_function(mlp, input_tensor)
    return output/functions_to_sample


def var_idx_perm_equivariant_5dfactor_all(mlp, input_tensor):
    '''
    A factor containing 5 variables has 5! orderings of the variables.
    Sample functions_to_sample permutations of input_tensor dimensions, apply input mlp to each flattened tensor and unpermute, and return mean to make mlp invariant to variable ordering
    '''
    # permutation_functions = [permute_dim1111, permute_dim1112, permute_dim1121, permute_dim1122, permute_dim1131, permute_dim1132, permute_dim1211, permute_dim1212, permute_dim1221, permute_dim1222, permute_dim1231, permute_dim1232, permute_dim1311, permute_dim1312, permute_dim1321, permute_dim1322, permute_dim1331, permute_dim1332, permute_dim1411, permute_dim1412, permute_dim1421, permute_dim1422, permute_dim1431, permute_dim1432, permute_dim2111, permute_dim2112, permute_dim2121, permute_dim2122, permute_dim2131, permute_dim2132, permute_dim2211, permute_dim2212, permute_dim2221, permute_dim2222, permute_dim2231, permute_dim2232, permute_dim2311, permute_dim2312, permute_dim2321, permute_dim2322, permute_dim2331, permute_dim2332, permute_dim2411, permute_dim2412, permute_dim2421, permute_dim2422, permute_dim2431, permute_dim2432, permute_dim3111, permute_dim3112, permute_dim3121, permute_dim3122, permute_dim3131, permute_dim3132, permute_dim3211, permute_dim3212, permute_dim3221, permute_dim3222, permute_dim3231, permute_dim3232, permute_dim3311, permute_dim3312, permute_dim3321, permute_dim3322, permute_dim3331, permute_dim3332, permute_dim3411, permute_dim3412, permute_dim3421, permute_dim3422, permute_dim3431, permute_dim3432, permute_dim4111, permute_dim4112, permute_dim4121, permute_dim4122, permute_dim4131, permute_dim4132, permute_dim4211, permute_dim4212, permute_dim4221, permute_dim4222, permute_dim4231, permute_dim4232, permute_dim4311, permute_dim4312, permute_dim4321, permute_dim4322, permute_dim4331, permute_dim4332, permute_dim4411, permute_dim4412, permute_dim4421, permute_dim4422, permute_dim4431, permute_dim4432, permute_dim5111, permute_dim5112, permute_dim5121, permute_dim5122, permute_dim5131, permute_dim5132, permute_dim5211, permute_dim5212, permute_dim5221, permute_dim5222, permute_dim5231, permute_dim5232, permute_dim5311, permute_dim5312, permute_dim5321, permute_dim5322, permute_dim5331, permute_dim5332, permute_dim5411, permute_dim5412, permute_dim5421, permute_dim5422, permute_dim5431, permute_dim5432]
    # output = permutation_functions[0](mlp, input_tensor)
    # for function in permutation_functions[1:]:
    #     output += function(mlp, input_tensor)
    # return output/len(permutation_functions)

    #no for loop in case this is faster although looks awful as is..
    output = (permute_dim1111(mlp, input_tensor) + permute_dim1112(mlp, input_tensor) + permute_dim1121(mlp, input_tensor) + permute_dim1122(mlp, input_tensor) + permute_dim1131(mlp, input_tensor) + permute_dim1132(mlp, input_tensor) + permute_dim1211(mlp, input_tensor) + permute_dim1212(mlp, input_tensor) + permute_dim1221(mlp, input_tensor) + permute_dim1222(mlp, input_tensor) + permute_dim1231(mlp, input_tensor) + permute_dim1232(mlp, input_tensor) + permute_dim1311(mlp, input_tensor) + permute_dim1312(mlp, input_tensor) + permute_dim1321(mlp, input_tensor) + permute_dim1322(mlp, input_tensor) + permute_dim1331(mlp, input_tensor) + permute_dim1332(mlp, input_tensor) + permute_dim1411(mlp, input_tensor) + permute_dim1412(mlp, input_tensor) + permute_dim1421(mlp, input_tensor) + permute_dim1422(mlp, input_tensor) + permute_dim1431(mlp, input_tensor) + permute_dim1432(mlp, input_tensor) + permute_dim2111(mlp, input_tensor) + permute_dim2112(mlp, input_tensor) + permute_dim2121(mlp, input_tensor) + permute_dim2122(mlp, input_tensor) + permute_dim2131(mlp, input_tensor) + permute_dim2132(mlp, input_tensor) + permute_dim2211(mlp, input_tensor) + permute_dim2212(mlp, input_tensor) + permute_dim2221(mlp, input_tensor) + permute_dim2222(mlp, input_tensor) + permute_dim2231(mlp, input_tensor) + permute_dim2232(mlp, input_tensor) + permute_dim2311(mlp, input_tensor) + permute_dim2312(mlp, input_tensor) + permute_dim2321(mlp, input_tensor) + permute_dim2322(mlp, input_tensor) + permute_dim2331(mlp, input_tensor) + permute_dim2332(mlp, input_tensor) + permute_dim2411(mlp, input_tensor) + permute_dim2412(mlp, input_tensor) + permute_dim2421(mlp, input_tensor) + permute_dim2422(mlp, input_tensor) + permute_dim2431(mlp, input_tensor) + permute_dim2432(mlp, input_tensor) + permute_dim3111(mlp, input_tensor) + permute_dim3112(mlp, input_tensor) + permute_dim3121(mlp, input_tensor) + permute_dim3122(mlp, input_tensor) + permute_dim3131(mlp, input_tensor) + permute_dim3132(mlp, input_tensor) + permute_dim3211(mlp, input_tensor) + permute_dim3212(mlp, input_tensor) + permute_dim3221(mlp, input_tensor) + permute_dim3222(mlp, input_tensor) + permute_dim3231(mlp, input_tensor) + permute_dim3232(mlp, input_tensor) + permute_dim3311(mlp, input_tensor) + permute_dim3312(mlp, input_tensor) + permute_dim3321(mlp, input_tensor) + permute_dim3322(mlp, input_tensor) + permute_dim3331(mlp, input_tensor) + permute_dim3332(mlp, input_tensor) + permute_dim3411(mlp, input_tensor) + permute_dim3412(mlp, input_tensor) + permute_dim3421(mlp, input_tensor) + permute_dim3422(mlp, input_tensor) + permute_dim3431(mlp, input_tensor) + permute_dim3432(mlp, input_tensor) + permute_dim4111(mlp, input_tensor) + permute_dim4112(mlp, input_tensor) + permute_dim4121(mlp, input_tensor) + permute_dim4122(mlp, input_tensor) + permute_dim4131(mlp, input_tensor) + permute_dim4132(mlp, input_tensor) + permute_dim4211(mlp, input_tensor) + permute_dim4212(mlp, input_tensor) + permute_dim4221(mlp, input_tensor) + permute_dim4222(mlp, input_tensor) + permute_dim4231(mlp, input_tensor) + permute_dim4232(mlp, input_tensor) + permute_dim4311(mlp, input_tensor) + permute_dim4312(mlp, input_tensor) + permute_dim4321(mlp, input_tensor) + permute_dim4322(mlp, input_tensor) + permute_dim4331(mlp, input_tensor) + permute_dim4332(mlp, input_tensor) + permute_dim4411(mlp, input_tensor) + permute_dim4412(mlp, input_tensor) + permute_dim4421(mlp, input_tensor) + permute_dim4422(mlp, input_tensor) + permute_dim4431(mlp, input_tensor) + permute_dim4432(mlp, input_tensor) + permute_dim5111(mlp, input_tensor) + permute_dim5112(mlp, input_tensor) + permute_dim5121(mlp, input_tensor) + permute_dim5122(mlp, input_tensor) + permute_dim5131(mlp, input_tensor) + permute_dim5132(mlp, input_tensor) + permute_dim5211(mlp, input_tensor) + permute_dim5212(mlp, input_tensor) + permute_dim5221(mlp, input_tensor) + permute_dim5222(mlp, input_tensor) + permute_dim5231(mlp, input_tensor) + permute_dim5232(mlp, input_tensor) + permute_dim5311(mlp, input_tensor) + permute_dim5312(mlp, input_tensor) + permute_dim5321(mlp, input_tensor) + permute_dim5322(mlp, input_tensor) + permute_dim5331(mlp, input_tensor) + permute_dim5332(mlp, input_tensor) + permute_dim5411(mlp, input_tensor) + permute_dim5412(mlp, input_tensor) + permute_dim5421(mlp, input_tensor) + permute_dim5422(mlp, input_tensor) + permute_dim5431(mlp, input_tensor) + permute_dim5432(mlp, input_tensor))
    output = output/120
    return output
