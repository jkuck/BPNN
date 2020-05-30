import torch
import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    # https://github.com/aparo/pyes/issues/183
    def __getattr__(self, attr):
        if attr.startswith('__'):
            raise AttributeError
        return self.get(attr, None)

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)

def neg_inf_to_zero(tensor):
    '''
    return tensor with negative infinity values replaced with zeros
    '''
    return_tensor = tensor.clone()
    return_tensor[return_tensor == -float('inf')] = 0
    return return_tensor


def logminusexp(a, b):
    '''
    return log(exp(a) - exp(b)) elementwise
    Inputs:
    - a: (tensor) must have the same shape as b
    - b: (tensor) must have the same shape as a
    '''
    assert(a.shape == b.shape)
    element_wise_max = torch.max(a,b)
    a_prime = a - element_wise_max
    b_prime = b - element_wise_max
    return element_wise_max + torch.log(torch.exp(a_prime) - torch.exp(b_prime))


def log_normalize(tensor):
    '''
    normalize the input tensor such that sum(exp(tensor[i,:,:,...,:]))=1 for any i
    we assume that sum(exp(tensor[i,:,:,...,:]))>0 for the input tensor and all i in the assert statement
    '''
    normalization_view = [1 for i in range(len(tensor.shape))]
    normalization_view[0] = -1
    log_sum_exp_factor_beliefs = logsumexp_multipleDim(tensor, dim=0).view(normalization_view)
    # for factor_idx in range(factor_beliefs.shape[0]):
    #     assert((factor_beliefs[factor_idx] != -np.inf).all()), (factor_beliefs[factor_idx], log_sum_exp_factor_beliefs[factor_idx], (factor_beliefs[factor_idx] != -np.inf))    
    assert((log_sum_exp_factor_beliefs != -np.inf).all()) #debugging, assumes every index should contain non-zero entries
    normalized_tensor = tensor - log_sum_exp_factor_beliefs#normalize factor beliefs
   
    # #DEBUG
    # logsumexp_normalized_tensor = logsumexp_multipleDim(normalized_tensor, dim=0)
    # assert((logsumexp_normalized_tensor == 0.0).all()), (logsumexp_normalized_tensor, tensor, normalized_tensor)
    # #END DEBUG

    return normalized_tensor


def max_multipleDim(input, axes, keepdim=False):
    '''
    modified from https://github.com/pytorch/pytorch/issues/2006
    Take the maximum across multiple axes.  For example, say the input has dimensions [a,b,c,d] and
    axes = [0,2].  The output (with keepdim=False) will have dimensions [b,d] and output[i,j] will
    be the maximum value of input[:, i, :, j]
    '''
    # probably some check for uniqueness of axes
    if keepdim:
        for ax in axes:
            input = input.max(ax, keepdim=True)[0]
    else:
        for ax in sorted(axes, reverse=True):
            input = input.max(ax)[0]
    return input

def logsumexp_multipleDim(tensor, dim_to_keep=None):
    """
    Compute log(sum(exp(tensor), aggregate_dimensions)) in a numerically stable way.

    Inputs:
    - tensor (tensor): input tensor
    - dim_to_keep (list of ints): the only dimensions to keep in the output.  i.e. 
        for a 4d input tensor with dim_to_keep=[2] (0-indexed): return_tensor[i] = logsumexp(tensor[:,:,i,:])

    Outputs:
    - return_tensor (tensor): logsumexp of input tensor along specified dimensions (those not appearing in dim_to_keep).
        Has same number of dimensions as the original tensor, but those not appearing in dim_to_keep have size 1

    """
    assert(not torch.isnan(tensor).any())

    tensor_dimensions = len(tensor.shape)
    assert((torch.tensor([dim_to_keep]) < tensor_dimensions).all())
    assert((torch.tensor([dim_to_keep]) >= 0).all())    
    aggregate_dimensions = [i for i in range(tensor_dimensions) if (i not in dim_to_keep)]
    # print("aggregate_dimensions:", aggregate_dimensions)
    # print("tensor:", tensor)
    max_values = max_multipleDim(tensor, axes=aggregate_dimensions, keepdim=True)
#     print("max_values:", max_values)
    max_values[torch.where(max_values == -np.inf)] = 0
    assert(not torch.isnan(max_values).any())
    assert((max_values > -np.inf).all())
    assert(not torch.isnan(tensor - max_values).any())
    assert(not torch.isnan(torch.exp(tensor - max_values)).any())
    assert(not torch.isnan(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions)).any())
    # assert(not (torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions) > 0).all())
    assert(not torch.isnan(torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions))).any())

    # print("max_values:", max_values)
    # print("tensor - max_values", tensor - max_values)
    # print("torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions)):", torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions)))
#     print("tensor.shape", tensor.shape)
#     print("max_values.shape", max_values.shape)
#     sleep(temp)
    return_tensor = torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions, keepdim=True)) + max_values
    # print("return_tensor:", return_tensor)
    assert(not torch.isnan(return_tensor).any())
    return return_tensor



class shift_func(torch.nn.Module):
    '''
    take a function y = f(x) and shift it (in x and y) such that
    y' = f'(x) = shift + f(x-shift)
    '''    
    def __init__(self, func, shift):
        super().__init__()        
        self.func = func
        self.shift = shift

    def forward(self, x):
        return self.shift + self.func(x - self.shift)



class reflect_xy(torch.nn.Module):
    '''
    reflect a function over the x-axis and y-axis
    '''    
    def __init__(self, func):
        super().__init__()        
        self.func = func

    def forward(self, x):
        return  -self.func(-x)


def var_idx_perm_equivariant_5dfactor(mlp, input_tensor):
    '''

    '''
    input_tensor_shape = input_tensor.shape

    #5 options for first variable index
    input_tensor1 = input_tensor
    input_tensor2 = input_tensor.permute(0,1,3,2,4,5,6)
    input_tensor3 = input_tensor.permute(0,1,4,3,2,5,6)
    input_tensor4 = input_tensor.permute(0,1,5,3,4,2,6)
    input_tensor5 = input_tensor.permute(0,1,6,3,4,5,2)

    #4*5 options for second variable index
    input_tensor11 = input_tensor1
    input_tensor12 = input_tensor1.permute(0,1,2,4,3,5,6)
    input_tensor13 = input_tensor1.permute(0,1,2,5,4,3,6)
    input_tensor14 = input_tensor1.permute(0,1,2,6,4,5,3)

    input_tensor21 = input_tensor2
    input_tensor22 = input_tensor2.permute(0,1,2,4,3,5,6)
    input_tensor23 = input_tensor2.permute(0,1,2,5,4,3,6)
    input_tensor24 = input_tensor2.permute(0,1,2,6,4,5,3)

    input_tensor31 = input_tensor3
    input_tensor32 = input_tensor3.permute(0,1,2,4,3,5,6)
    input_tensor33 = input_tensor3.permute(0,1,2,5,4,3,6)
    input_tensor34 = input_tensor3.permute(0,1,2,6,4,5,3)

    input_tensor41 = input_tensor4
    input_tensor42 = input_tensor4.permute(0,1,2,4,3,5,6)
    input_tensor43 = input_tensor4.permute(0,1,2,5,4,3,6)
    input_tensor44 = input_tensor4.permute(0,1,2,6,4,5,3)

    input_tensor51 = input_tensor5
    input_tensor52 = input_tensor5.permute(0,1,2,4,3,5,6)
    input_tensor53 = input_tensor5.permute(0,1,2,5,4,3,6)
    input_tensor54 = input_tensor5.permute(0,1,2,6,4,5,3)

    #3*4*5 options for third variable index
    input_tensor111 = input_tensor11
    input_tensor112 = input_tensor11.permute(0,1,2,3,5,4,6) 
    input_tensor113 = input_tensor11.permute(0,1,2,3,6,5,4)

    input_tensor121 = input_tensor12
    input_tensor122 = input_tensor12.permute(0,1,2,3,5,4,6) 
    input_tensor123 = input_tensor12.permute(0,1,2,3,6,5,4)

    input_tensor131 = input_tensor13
    input_tensor132 = input_tensor13.permute(0,1,2,3,5,4,6) 
    input_tensor133 = input_tensor13.permute(0,1,2,3,6,5,4)

    input_tensor141 = input_tensor14
    input_tensor142 = input_tensor14.permute(0,1,2,3,5,4,6) 
    input_tensor143 = input_tensor14.permute(0,1,2,3,6,5,4)

    input_tensor211 = input_tensor21
    input_tensor212 = input_tensor21.permute(0,1,2,3,5,4,6) 
    input_tensor213 = input_tensor21.permute(0,1,2,3,6,5,4)

    input_tensor221 = input_tensor22
    input_tensor222 = input_tensor22.permute(0,1,2,3,5,4,6) 
    input_tensor223 = input_tensor22.permute(0,1,2,3,6,5,4)

    input_tensor231 = input_tensor23
    input_tensor232 = input_tensor23.permute(0,1,2,3,5,4,6) 
    input_tensor233 = input_tensor23.permute(0,1,2,3,6,5,4)

    input_tensor241 = input_tensor24
    input_tensor242 = input_tensor24.permute(0,1,2,3,5,4,6) 
    input_tensor243 = input_tensor24.permute(0,1,2,3,6,5,4)

    input_tensor311 = input_tensor31
    input_tensor312 = input_tensor31.permute(0,1,2,3,5,4,6) 
    input_tensor313 = input_tensor31.permute(0,1,2,3,6,5,4)

    input_tensor321 = input_tensor32
    input_tensor322 = input_tensor32.permute(0,1,2,3,5,4,6) 
    input_tensor323 = input_tensor32.permute(0,1,2,3,6,5,4)

    input_tensor331 = input_tensor33
    input_tensor332 = input_tensor33.permute(0,1,2,3,5,4,6) 
    input_tensor333 = input_tensor33.permute(0,1,2,3,6,5,4)

    input_tensor341 = input_tensor34
    input_tensor342 = input_tensor34.permute(0,1,2,3,5,4,6) 
    input_tensor343 = input_tensor34.permute(0,1,2,3,6,5,4)

    input_tensor411 = input_tensor41
    input_tensor412 = input_tensor41.permute(0,1,2,3,5,4,6) 
    input_tensor413 = input_tensor41.permute(0,1,2,3,6,5,4)

    input_tensor421 = input_tensor42
    input_tensor422 = input_tensor42.permute(0,1,2,3,5,4,6) 
    input_tensor423 = input_tensor42.permute(0,1,2,3,6,5,4)

    input_tensor431 = input_tensor43
    input_tensor432 = input_tensor43.permute(0,1,2,3,5,4,6) 
    input_tensor433 = input_tensor43.permute(0,1,2,3,6,5,4)

    input_tensor441 = input_tensor44
    input_tensor442 = input_tensor44.permute(0,1,2,3,5,4,6) 
    input_tensor443 = input_tensor44.permute(0,1,2,3,6,5,4)

    input_tensor511 = input_tensor51
    input_tensor512 = input_tensor51.permute(0,1,2,3,5,4,6) 
    input_tensor513 = input_tensor51.permute(0,1,2,3,6,5,4)

    input_tensor521 = input_tensor52
    input_tensor522 = input_tensor52.permute(0,1,2,3,5,4,6) 
    input_tensor523 = input_tensor52.permute(0,1,2,3,6,5,4)

    input_tensor531 = input_tensor53
    input_tensor532 = input_tensor53.permute(0,1,2,3,5,4,6) 
    input_tensor533 = input_tensor53.permute(0,1,2,3,6,5,4)

    input_tensor541 = input_tensor54
    input_tensor542 = input_tensor54.permute(0,1,2,3,5,4,6) 
    input_tensor543 = input_tensor54.permute(0,1,2,3,6,5,4)


    #2*3*4*5 options for third variable index
    input_tensor1111 = input_tensor111
    input_tensor1112 = input_tensor111.permute(0,1,2,3,4,6,5)

    input_tensor1121 = input_tensor112
    input_tensor1122 = input_tensor112.permute(0,1,2,3,4,6,5)

    input_tensor1131 = input_tensor113
    input_tensor1132 = input_tensor113.permute(0,1,2,3,4,6,5)

    input_tensor1211 = input_tensor121
    input_tensor1212 = input_tensor121.permute(0,1,2,3,4,6,5)

    input_tensor1221 = input_tensor122
    input_tensor1222 = input_tensor122.permute(0,1,2,3,4,6,5)

    input_tensor1231 = input_tensor123
    input_tensor1232 = input_tensor123.permute(0,1,2,3,4,6,5)

    input_tensor1311 = input_tensor131
    input_tensor1312 = input_tensor131.permute(0,1,2,3,4,6,5)

    input_tensor1321 = input_tensor132
    input_tensor1322 = input_tensor132.permute(0,1,2,3,4,6,5)

    input_tensor1331 = input_tensor133
    input_tensor1332 = input_tensor133.permute(0,1,2,3,4,6,5)

    input_tensor1411 = input_tensor141
    input_tensor1412 = input_tensor141.permute(0,1,2,3,4,6,5)

    input_tensor1421 = input_tensor142
    input_tensor1422 = input_tensor142.permute(0,1,2,3,4,6,5)

    input_tensor1431 = input_tensor143
    input_tensor1432 = input_tensor143.permute(0,1,2,3,4,6,5)

    input_tensor2111 = input_tensor211
    input_tensor2112 = input_tensor211.permute(0,1,2,3,4,6,5)

    input_tensor2121 = input_tensor212
    input_tensor2122 = input_tensor212.permute(0,1,2,3,4,6,5)

    input_tensor2131 = input_tensor213
    input_tensor2132 = input_tensor213.permute(0,1,2,3,4,6,5)

    input_tensor2211 = input_tensor221
    input_tensor2212 = input_tensor221.permute(0,1,2,3,4,6,5)

    input_tensor2221 = input_tensor222
    input_tensor2222 = input_tensor222.permute(0,1,2,3,4,6,5)

    input_tensor2231 = input_tensor223
    input_tensor2232 = input_tensor223.permute(0,1,2,3,4,6,5)

    input_tensor2311 = input_tensor231
    input_tensor2312 = input_tensor231.permute(0,1,2,3,4,6,5)

    input_tensor2321 = input_tensor232
    input_tensor2322 = input_tensor232.permute(0,1,2,3,4,6,5)

    input_tensor2331 = input_tensor233
    input_tensor2332 = input_tensor233.permute(0,1,2,3,4,6,5)

    input_tensor2411 = input_tensor241
    input_tensor2412 = input_tensor241.permute(0,1,2,3,4,6,5)

    input_tensor2421 = input_tensor242
    input_tensor2422 = input_tensor242.permute(0,1,2,3,4,6,5)

    input_tensor2431 = input_tensor243
    input_tensor2432 = input_tensor243.permute(0,1,2,3,4,6,5)

    input_tensor3111 = input_tensor311
    input_tensor3112 = input_tensor311.permute(0,1,2,3,4,6,5)

    input_tensor3121 = input_tensor312
    input_tensor3122 = input_tensor312.permute(0,1,2,3,4,6,5)

    input_tensor3131 = input_tensor313
    input_tensor3132 = input_tensor313.permute(0,1,2,3,4,6,5)

    input_tensor3211 = input_tensor321
    input_tensor3212 = input_tensor321.permute(0,1,2,3,4,6,5)

    input_tensor3221 = input_tensor322
    input_tensor3222 = input_tensor322.permute(0,1,2,3,4,6,5)

    input_tensor3231 = input_tensor323
    input_tensor3232 = input_tensor323.permute(0,1,2,3,4,6,5)

    input_tensor3311 = input_tensor331
    input_tensor3312 = input_tensor331.permute(0,1,2,3,4,6,5)

    input_tensor3321 = input_tensor332
    input_tensor3322 = input_tensor332.permute(0,1,2,3,4,6,5)

    input_tensor3331 = input_tensor333
    input_tensor3332 = input_tensor333.permute(0,1,2,3,4,6,5)

    input_tensor3411 = input_tensor341
    input_tensor3412 = input_tensor341.permute(0,1,2,3,4,6,5)

    input_tensor3421 = input_tensor342
    input_tensor3422 = input_tensor342.permute(0,1,2,3,4,6,5)

    input_tensor3431 = input_tensor343
    input_tensor3432 = input_tensor343.permute(0,1,2,3,4,6,5)

    input_tensor4111 = input_tensor411
    input_tensor4112 = input_tensor411.permute(0,1,2,3,4,6,5)

    input_tensor4121 = input_tensor412
    input_tensor4122 = input_tensor412.permute(0,1,2,3,4,6,5)

    input_tensor4131 = input_tensor413
    input_tensor4132 = input_tensor413.permute(0,1,2,3,4,6,5)

    input_tensor4211 = input_tensor421
    input_tensor4212 = input_tensor421.permute(0,1,2,3,4,6,5)

    input_tensor4221 = input_tensor422
    input_tensor4222 = input_tensor422.permute(0,1,2,3,4,6,5)

    input_tensor4231 = input_tensor423
    input_tensor4232 = input_tensor423.permute(0,1,2,3,4,6,5)

    input_tensor4311 = input_tensor431
    input_tensor4312 = input_tensor431.permute(0,1,2,3,4,6,5)

    input_tensor4321 = input_tensor432
    input_tensor4322 = input_tensor432.permute(0,1,2,3,4,6,5)

    input_tensor4331 = input_tensor433
    input_tensor4332 = input_tensor433.permute(0,1,2,3,4,6,5)

    input_tensor4411 = input_tensor441
    input_tensor4412 = input_tensor441.permute(0,1,2,3,4,6,5)

    input_tensor4421 = input_tensor442
    input_tensor4422 = input_tensor442.permute(0,1,2,3,4,6,5)

    input_tensor4431 = input_tensor443
    input_tensor4432 = input_tensor443.permute(0,1,2,3,4,6,5)

    input_tensor5111 = input_tensor511
    input_tensor5112 = input_tensor511.permute(0,1,2,3,4,6,5)

    input_tensor5121 = input_tensor512
    input_tensor5122 = input_tensor512.permute(0,1,2,3,4,6,5)

    input_tensor5131 = input_tensor513
    input_tensor5132 = input_tensor513.permute(0,1,2,3,4,6,5)

    input_tensor5211 = input_tensor521
    input_tensor5212 = input_tensor521.permute(0,1,2,3,4,6,5)

    input_tensor5221 = input_tensor522
    input_tensor5222 = input_tensor522.permute(0,1,2,3,4,6,5)

    input_tensor5231 = input_tensor523
    input_tensor5232 = input_tensor523.permute(0,1,2,3,4,6,5)

    input_tensor5311 = input_tensor531
    input_tensor5312 = input_tensor531.permute(0,1,2,3,4,6,5)

    input_tensor5321 = input_tensor532
    input_tensor5322 = input_tensor532.permute(0,1,2,3,4,6,5)

    input_tensor5331 = input_tensor533
    input_tensor5332 = input_tensor533.permute(0,1,2,3,4,6,5)

    input_tensor5411 = input_tensor541
    input_tensor5412 = input_tensor541.permute(0,1,2,3,4,6,5)

    input_tensor5421 = input_tensor542
    input_tensor5422 = input_tensor542.permute(0,1,2,3,4,6,5)

    input_tensor5431 = input_tensor543
    input_tensor5432 = input_tensor543.permute(0,1,2,3,4,6,5)

    #pass all inputs through mlp and undo permutations
    output_tensor1111 = mlp(input_tensor1111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape)
    output_tensor1112 = mlp(input_tensor1112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5)
    output_tensor1121 = mlp(input_tensor1121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6)
    output_tensor1122 = mlp(input_tensor1122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6)
    output_tensor1131 = mlp(input_tensor1131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4)
    output_tensor1132 = mlp(input_tensor1132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4)
    output_tensor1211 = mlp(input_tensor1211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6)
    output_tensor1212 = mlp(input_tensor1212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6)
    output_tensor1221 = mlp(input_tensor1221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6)
    output_tensor1222 = mlp(input_tensor1222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6)
    output_tensor1231 = mlp(input_tensor1231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6)
    output_tensor1232 = mlp(input_tensor1232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6)
    output_tensor1311 = mlp(input_tensor1311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6)
    output_tensor1312 = mlp(input_tensor1312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6)
    output_tensor1321 = mlp(input_tensor1321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6)
    output_tensor1322 = mlp(input_tensor1322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6)
    output_tensor1331 = mlp(input_tensor1331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6)
    output_tensor1332 = mlp(input_tensor1332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6)
    output_tensor1411 = mlp(input_tensor1411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3)
    output_tensor1412 = mlp(input_tensor1412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3)
    output_tensor1421 = mlp(input_tensor1421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3)
    output_tensor1422 = mlp(input_tensor1422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3)
    output_tensor1431 = mlp(input_tensor1431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3)
    output_tensor1432 = mlp(input_tensor1432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3)
    output_tensor2111 = mlp(input_tensor2111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,3,2,4,5,6)
    output_tensor2112 = mlp(input_tensor2112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,3,2,4,5,6)
    output_tensor2121 = mlp(input_tensor2121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,3,2,4,5,6)
    output_tensor2122 = mlp(input_tensor2122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,3,2,4,5,6)
    output_tensor2131 = mlp(input_tensor2131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,3,2,4,5,6)
    output_tensor2132 = mlp(input_tensor2132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,3,2,4,5,6)
    output_tensor2211 = mlp(input_tensor2211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    output_tensor2212 = mlp(input_tensor2212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    output_tensor2221 = mlp(input_tensor2221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    output_tensor2222 = mlp(input_tensor2222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    output_tensor2231 = mlp(input_tensor2231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    output_tensor2232 = mlp(input_tensor2232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,3,2,4,5,6)
    output_tensor2311 = mlp(input_tensor2311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    output_tensor2312 = mlp(input_tensor2312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    output_tensor2321 = mlp(input_tensor2321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    output_tensor2322 = mlp(input_tensor2322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    output_tensor2331 = mlp(input_tensor2331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    output_tensor2332 = mlp(input_tensor2332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,3,2,4,5,6)
    output_tensor2411 = mlp(input_tensor2411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    output_tensor2412 = mlp(input_tensor2412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    output_tensor2421 = mlp(input_tensor2421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    output_tensor2422 = mlp(input_tensor2422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    output_tensor2431 = mlp(input_tensor2431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    output_tensor2432 = mlp(input_tensor2432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,3,2,4,5,6)
    output_tensor3111 = mlp(input_tensor3111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,4,3,2,5,6)
    output_tensor3112 = mlp(input_tensor3112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,4,3,2,5,6)
    output_tensor3121 = mlp(input_tensor3121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,4,3,2,5,6)
    output_tensor3122 = mlp(input_tensor3122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,4,3,2,5,6)
    output_tensor3131 = mlp(input_tensor3131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,4,3,2,5,6)
    output_tensor3132 = mlp(input_tensor3132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,4,3,2,5,6)
    output_tensor3211 = mlp(input_tensor3211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    output_tensor3212 = mlp(input_tensor3212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    output_tensor3221 = mlp(input_tensor3221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    output_tensor3222 = mlp(input_tensor3222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    output_tensor3231 = mlp(input_tensor3231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    output_tensor3232 = mlp(input_tensor3232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,4,3,2,5,6)
    output_tensor3311 = mlp(input_tensor3311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    output_tensor3312 = mlp(input_tensor3312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    output_tensor3321 = mlp(input_tensor3321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    output_tensor3322 = mlp(input_tensor3322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    output_tensor3331 = mlp(input_tensor3331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    output_tensor3332 = mlp(input_tensor3332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,4,3,2,5,6)
    output_tensor3411 = mlp(input_tensor3411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    output_tensor3412 = mlp(input_tensor3412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    output_tensor3421 = mlp(input_tensor3421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    output_tensor3422 = mlp(input_tensor3422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    output_tensor3431 = mlp(input_tensor3431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    output_tensor3432 = mlp(input_tensor3432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,4,3,2,5,6)
    output_tensor4111 = mlp(input_tensor4111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,5,3,4,2,6)
    output_tensor4112 = mlp(input_tensor4112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,5,3,4,2,6)
    output_tensor4121 = mlp(input_tensor4121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,5,3,4,2,6)
    output_tensor4122 = mlp(input_tensor4122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,5,3,4,2,6)
    output_tensor4131 = mlp(input_tensor4131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,5,3,4,2,6)
    output_tensor4132 = mlp(input_tensor4132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,5,3,4,2,6)
    output_tensor4211 = mlp(input_tensor4211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    output_tensor4212 = mlp(input_tensor4212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    output_tensor4221 = mlp(input_tensor4221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    output_tensor4222 = mlp(input_tensor4222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    output_tensor4231 = mlp(input_tensor4231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    output_tensor4232 = mlp(input_tensor4232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,5,3,4,2,6)
    output_tensor4311 = mlp(input_tensor4311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    output_tensor4312 = mlp(input_tensor4312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    output_tensor4321 = mlp(input_tensor4321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    output_tensor4322 = mlp(input_tensor4322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    output_tensor4331 = mlp(input_tensor4331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    output_tensor4332 = mlp(input_tensor4332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,5,3,4,2,6)
    output_tensor4411 = mlp(input_tensor4411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    output_tensor4412 = mlp(input_tensor4412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    output_tensor4421 = mlp(input_tensor4421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    output_tensor4422 = mlp(input_tensor4422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    output_tensor4431 = mlp(input_tensor4431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    output_tensor4432 = mlp(input_tensor4432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,5,3,4,2,6)
    output_tensor5111 = mlp(input_tensor5111.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,6,3,4,5,2)
    output_tensor5112 = mlp(input_tensor5112.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,6,3,4,5,2)
    output_tensor5121 = mlp(input_tensor5121.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,6,3,4,5,2)
    output_tensor5122 = mlp(input_tensor5122.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,6,3,4,5,2)
    output_tensor5131 = mlp(input_tensor5131.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,6,3,4,5,2)
    output_tensor5132 = mlp(input_tensor5132.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,6,3,4,5,2)
    output_tensor5211 = mlp(input_tensor5211.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    output_tensor5212 = mlp(input_tensor5212.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    output_tensor5221 = mlp(input_tensor5221.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    output_tensor5222 = mlp(input_tensor5222.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    output_tensor5231 = mlp(input_tensor5231.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    output_tensor5232 = mlp(input_tensor5232.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,4,3,5,6).permute(0,1,6,3,4,5,2)
    output_tensor5311 = mlp(input_tensor5311.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    output_tensor5312 = mlp(input_tensor5312.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    output_tensor5321 = mlp(input_tensor5321.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    output_tensor5322 = mlp(input_tensor5322.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    output_tensor5331 = mlp(input_tensor5331.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    output_tensor5332 = mlp(input_tensor5332.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,5,4,3,6).permute(0,1,6,3,4,5,2)
    output_tensor5411 = mlp(input_tensor5411.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    output_tensor5412 = mlp(input_tensor5412.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    output_tensor5421 = mlp(input_tensor5421.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    output_tensor5422 = mlp(input_tensor5422.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,5,4,6).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    output_tensor5431 = mlp(input_tensor5431.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)
    output_tensor5432 = mlp(input_tensor5432.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape).permute(0,1,2,3,4,6,5).permute(0,1,2,3,6,5,4).permute(0,1,2,6,4,5,3).permute(0,1,6,3,4,5,2)

    # assert(output_tensor1111 == output_tensor2111).all(), (output_tensor1111 -output_tensor1112)
    # for idx,cur_output in enumerate([output_tensor1111,output_tensor1112,output_tensor1121,output_tensor1122,output_tensor1131,output_tensor1132,output_tensor1211,output_tensor1212,output_tensor1221,output_tensor1222,output_tensor1231,output_tensor1232,output_tensor1311,output_tensor1312,output_tensor1321,output_tensor1322,output_tensor1331,output_tensor1332,output_tensor1411,output_tensor1412,output_tensor1421,output_tensor1422,output_tensor1431,output_tensor1432,\
    # output_tensor2111,output_tensor2112,output_tensor2121,output_tensor2122,output_tensor2131,output_tensor2132,output_tensor2211,output_tensor2212,output_tensor2221,output_tensor2222,output_tensor2231,output_tensor2232,output_tensor2311,output_tensor2312,output_tensor2321,output_tensor2322,output_tensor2331,output_tensor2332,output_tensor2411,output_tensor2412,output_tensor2421,output_tensor2422,output_tensor2431,output_tensor2432,\
    # output_tensor3111,output_tensor3112,output_tensor3121,output_tensor3122,output_tensor3131,output_tensor3132,output_tensor3211,output_tensor3212,output_tensor3221,output_tensor3222,output_tensor3231,output_tensor3232,output_tensor3311,output_tensor3312,output_tensor3321,output_tensor3322,output_tensor3331,output_tensor3332,output_tensor3411,output_tensor3412,output_tensor3421,output_tensor3422,output_tensor3431,output_tensor3432,\
    # output_tensor4111,output_tensor4112,output_tensor4121,output_tensor4122,output_tensor4131,output_tensor4132,output_tensor4211,output_tensor4212,output_tensor4221,output_tensor4222,output_tensor4231,output_tensor4232,output_tensor4311,output_tensor4312,output_tensor4321,output_tensor4322,output_tensor4331,output_tensor4332,output_tensor4411,output_tensor4412,output_tensor4421,output_tensor4422,output_tensor4431,output_tensor4432,\
    # output_tensor5111,output_tensor5112,output_tensor5121,output_tensor5122,output_tensor5131,output_tensor5132,output_tensor5211,output_tensor5212,output_tensor5221,output_tensor5222,output_tensor5231,output_tensor5232,output_tensor5311,output_tensor5312,output_tensor5321,output_tensor5322,output_tensor5331,output_tensor5332,output_tensor5411,output_tensor5412,output_tensor5421,output_tensor5422,output_tensor5431,output_tensor5432]):
    #     assert(output_tensor1111 == cur_output).all(), ((output_tensor1111 -cur_output), idx)

    combined_output = (output_tensor1111+output_tensor1112+output_tensor1121+output_tensor1122+output_tensor1131+output_tensor1132+output_tensor1211+output_tensor1212+output_tensor1221+output_tensor1222+output_tensor1231+output_tensor1232+output_tensor1311+output_tensor1312+output_tensor1321+output_tensor1322+output_tensor1331+output_tensor1332+output_tensor1411+output_tensor1412+output_tensor1421+output_tensor1422+output_tensor1431+output_tensor1432+\
    output_tensor2111+output_tensor2112+output_tensor2121+output_tensor2122+output_tensor2131+output_tensor2132+output_tensor2211+output_tensor2212+output_tensor2221+output_tensor2222+output_tensor2231+output_tensor2232+output_tensor2311+output_tensor2312+output_tensor2321+output_tensor2322+output_tensor2331+output_tensor2332+output_tensor2411+output_tensor2412+output_tensor2421+output_tensor2422+output_tensor2431+output_tensor2432+\
    output_tensor3111+output_tensor3112+output_tensor3121+output_tensor3122+output_tensor3131+output_tensor3132+output_tensor3211+output_tensor3212+output_tensor3221+output_tensor3222+output_tensor3231+output_tensor3232+output_tensor3311+output_tensor3312+output_tensor3321+output_tensor3322+output_tensor3331+output_tensor3332+output_tensor3411+output_tensor3412+output_tensor3421+output_tensor3422+output_tensor3431+output_tensor3432+\
    output_tensor4111+output_tensor4112+output_tensor4121+output_tensor4122+output_tensor4131+output_tensor4132+output_tensor4211+output_tensor4212+output_tensor4221+output_tensor4222+output_tensor4231+output_tensor4232+output_tensor4311+output_tensor4312+output_tensor4321+output_tensor4322+output_tensor4331+output_tensor4332+output_tensor4411+output_tensor4412+output_tensor4421+output_tensor4422+output_tensor4431+output_tensor4432+\
    output_tensor5111+output_tensor5112+output_tensor5121+output_tensor5122+output_tensor5131+output_tensor5132+output_tensor5211+output_tensor5212+output_tensor5221+output_tensor5222+output_tensor5231+output_tensor5232+output_tensor5311+output_tensor5312+output_tensor5321+output_tensor5322+output_tensor5331+output_tensor5332+output_tensor5411+output_tensor5412+output_tensor5421+output_tensor5422+output_tensor5431+output_tensor5432)/120

    return combined_output


def variable_state_equivariant_22(mlp, input_tensor):
    '''
    Make an MLP that operates on factors of 2 variables with cardinality 2 equivariant
    to the variable indexing within the factor and the indexing of states within variables
    '''
    input_tensor_shape = input_tensor.shape
    
    #get all equivariant factor belief orderings
    input_tensor0 = input_tensor
    input_tensor1 = torch.index_select(input_tensor, dim=2, index=torch.tensor([1,0], device='cuda'))
    input_tensor2 = torch.index_select(input_tensor, dim=3, index=torch.tensor([1,0], device='cuda'))
    input_tensor3 = torch.index_select(input_tensor1, dim=3, index=torch.tensor([1,0], device='cuda'))

    input_tensor4 = input_tensor.permute(0,1,3,2)
    input_tensor5 = torch.index_select(input_tensor4, dim=2, index=torch.tensor([1,0], device='cuda'))
    input_tensor6 = torch.index_select(input_tensor4, dim=3, index=torch.tensor([1,0], device='cuda'))
    input_tensor7 = torch.index_select(input_tensor5, dim=3, index=torch.tensor([1,0], device='cuda'))


    outputs0 = mlp(input_tensor0.view(input_tensor_shape[0], -1)).view(input_tensor_shape)

    outputs1 = mlp(input_tensor1.view(input_tensor_shape[0], -1)).view(input_tensor_shape)
    outputs1 = torch.index_select(outputs1, dim=2, index=torch.tensor([1,0], device='cuda'))

    outputs2 = mlp(input_tensor2.view(input_tensor_shape[0], -1)).view(input_tensor_shape)
    outputs2 = torch.index_select(outputs2, dim=3, index=torch.tensor([1,0], device='cuda'))

    outputs3 = mlp(input_tensor3.view(input_tensor_shape[0], -1)).view(input_tensor_shape)
    outputs3 = torch.index_select(outputs3, dim=2, index=torch.tensor([1,0], device='cuda'))
    outputs3 = torch.index_select(outputs3, dim=3, index=torch.tensor([1,0], device='cuda'))
    
    outputs4 = mlp(input_tensor4.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape)
    outputs4 = outputs4.permute(0,1,3,2)

    outputs5 = mlp(input_tensor5.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape)
    outputs5 = torch.index_select(outputs5, dim=2, index=torch.tensor([1,0], device='cuda'))
    outputs5 = outputs5.permute(0,1,3,2)

    outputs6 = mlp(input_tensor6.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape)
    outputs6 = torch.index_select(outputs6, dim=3, index=torch.tensor([1,0], device='cuda'))
    outputs6 = outputs6.permute(0,1,3,2)

    outputs7 = mlp(input_tensor7.reshape(input_tensor_shape[0], -1)).reshape(input_tensor_shape)
    outputs7 = torch.index_select(outputs7, dim=2, index=torch.tensor([1,0], device='cuda'))
    outputs7 = torch.index_select(outputs7, dim=3, index=torch.tensor([1,0], device='cuda'))
    outputs7 = outputs7.permute(0,1,3,2)

    combined_output = (outputs0 + outputs1 + outputs2 + outputs3 +\
                        outputs4 + outputs5 + outputs6 + outputs7)/8

    return combined_output

        
