import torch

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