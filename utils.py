import torch
import numpy as np

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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

def logsumexp_multipleDim(tensor, dim=None):
    """
    Compute log(sum(exp(tensor), dim)) in a numerically stable way.

    Inputs:
    - tensor (tensor): input tensor
    - dim (int): the only dimension to keep in the output.  i.e. for a 4d input tensor with dim=2 (0-indexed):
        return_tensor[i] = logsumexp(tensor[:,:,i,:])

    Outputs:
    - return_tensor (1d tensor): logsumexp of input tensor along specified dimension

    """
    assert(not torch.isnan(tensor).any())

    tensor_dimensions = len(tensor.shape)
    assert(dim < tensor_dimensions and dim >= 0)
    aggregate_dimensions = [i for i in range(tensor_dimensions) if i != dim]
    # print("aggregate_dimensions:", aggregate_dimensions)
    # print("tensor:", tensor)
    max_values = max_multipleDim(tensor, axes=aggregate_dimensions, keepdim=True)
    max_values[torch.where(max_values == -np.inf)] = 0
    assert(not torch.isnan(max_values).any())
    assert((max_values > -np.inf).any())
    assert(not torch.isnan(tensor - max_values).any())
    assert(not torch.isnan(torch.exp(tensor - max_values)).any())
    assert(not torch.isnan(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions)).any())
    # assert(not (torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions) > 0).all())
    assert(not torch.isnan(torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions))).any())

    # print("max_values:", max_values)
    # print("tensor - max_values", tensor - max_values)
    # print("torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions)):", torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions)))
    return_tensor = torch.log(torch.sum(torch.exp(tensor - max_values), dim=aggregate_dimensions)) + max_values.squeeze()
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
