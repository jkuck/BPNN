import torch
from torch.nn import Sequential as Seq, Linear, ReLU
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.utils import scatter_
from factor_graph import FactorGraph
from sat_data import parse_dimacs, SatProblems, build_factorgraph_from_SATproblem
from utils import dotdict, logminusexp
import matplotlib.pyplot as plt
import matplotlib
import mrftools


__size_error_msg__ = ('All tensors which should get mapped to the same source '
                      'or target nodes must be of same size in dimension 0.')

SAT_PROBLEM_DIRECTORY = '/Users/jkuck/research/learn_BP/SAT_problems/'
# SAT_PROBLEM_DIRECTORY = '/atlas/u/jkuck/pytorch_geometric/jdk_examples/SAT_problems/'

def map_beliefs(beliefs, factor_graph, map_type):
    '''
    Utility function for propogate().  Maps factor or variable beliefs to edges 
    See https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    Inputs:
    - beliefs (tensor): factor_beliefs or var_beliefs, matching map_type
    - factor_graph: (FactorGraph, defined in factor_graph.py) the factor graph whose beliefs we are mapping
    - map_type: (string) 'factor' or 'var' denotes mapping factor or variable beliefs respectively
    '''

    size = [factor_graph.numFactors, factor_graph.numVars]


    mapping_indices = {"factor": 0, "var": 1}

    idx = mapping_indices[map_type]
    if isinstance(beliefs, tuple) or isinstance(beliefs, list):
        assert len(beliefs) == 2
        if size[1 - idx] != beliefs[1 - idx].size(0):
            raise ValueError(__size_error_msg__)
        mapped_beliefs = beliefs[idx]
    else:
        mapped_beliefs = beliefs.clone()

    if size is not None and size[idx] != mapped_beliefs.size(0):
        # print("factor_graph:", factor_graph)
        # print("beliefs:", beliefs)
        # print("beliefs.shape:", beliefs.shape)
        # print("size:", size)
        # print("idx:", idx)
        # print("mapped_beliefs.size(0):", mapped_beliefs.size(0))
        raise ValueError(__size_error_msg__)

    mapped_beliefs = torch.index_select(mapped_beliefs, 0, factor_graph.factorToVar_edge_index[idx])
    return mapped_beliefs

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


class FactorGraphMsgPassingLayer_NoDoubleCounting(torch.nn.Module):
    r"""Perform message passing in factor graphs without 'double counting'
    i.e. divide by previously sent messages as in loopy belief propagation

    Inputs:
    - learn_BP (bool): if False run standard looph belief propagation, if True
        insert a neural network into message aggregation for learning
    - logspace_mlp (bool): if True mlp runs in log space (trouble with good results), 
        if False we take exponent then run mlp then take log
    """

    def __init__(self, learn_BP=True, factor_state_space=None, avoid_nans=True, logspace_mlp=False):
        super(FactorGraphMsgPassingLayer_NoDoubleCounting, self).__init__()

        self.learn_BP = learn_BP
        self.avoid_nans = avoid_nans
        self.logspace_mlp = logspace_mlp
        print("learn_BP:", learn_BP)
        if learn_BP:
            assert(factor_state_space is not None)
            # print("factor_state_space:", factor_state_space)
            # sleep(float)
            self.linear1 = Linear(factor_state_space, factor_state_space)
            self.linear2 = Linear(factor_state_space, factor_state_space)
            self.linear1.weight = torch.nn.Parameter(torch.eye(factor_state_space))
            self.linear1.bias = torch.nn.Parameter(torch.zeros(self.linear1.bias.shape))
            self.linear2.weight = torch.nn.Parameter(torch.eye(factor_state_space))
            self.linear2.bias = torch.nn.Parameter(torch.zeros(self.linear2.bias.shape))

            if self.logspace_mlp:
                self.shifted_relu = shift_func(ReLU(), shift=-50)
                # self.shifted_relu = shift_func(ReLU(), shift=np.exp(-99))

                self.mlp = Seq(self.linear1, self.shifted_relu, self.linear2)

            else:
                self.shifted_relu = shift_func(ReLU(), shift=.0000000000000000001) #we'll get NaN's if we take the log of 0 or a negative number when going back to log space                
                self.mlp = Seq(self.linear1, ReLU(), self.linear2, self.shifted_relu)
                # self.mlp = Seq(self.linear1, ReLU(), self.linear2)

            # self.relu = ReLU()                
            # self.mlp = Seq(self.linear1, ReLU(), self.linear2, self.shifted_relu)
            # self.mlp = self.linear1
            # self.mlp = Seq(self.linear1, self.shifted_relu, self.linear2, self.shifted_relu)
            # self.mlp = Seq(self.linear1, self.shifted_relu, self.linear2, self.shifted_relu)
            # self.mlp = Seq(self.linear1, ReLU(), self.linear2, ReLU())

            # self.mlp = Seq(Linear(factor_state_space, factor_state_space),
            #                ReLU(),
            #                Linear(factor_state_space, factor_state_space))

    def forward(self, factor_graph, prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs):
        '''
        Inputs:
        - factor_graph: (FactorGraph, defined in factor_graph.py) the factor graph we will perform one
            iteration of message passing on.
        '''
        # Step 3-5: Start propagating messages.
        return self.propagate(factor_graph=factor_graph, prv_varToFactor_messages=prv_varToFactor_messages,
                              prv_factorToVar_messages=prv_factorToVar_messages, prv_factor_beliefs=prv_factor_beliefs)


    def propagate(self, factor_graph, prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, alpha=.5, debug=False):
        r"""Perform one iteration of message passing.  Pass messages from factors to variables, then
        from variables to factors.

        Inputs:
        - factor_graph: (FactorGraph, defined in factor_graph.py) the factor graph we will perform one
            iteration of message passing on.
        - prv_varToFactor_messages (tensor): varToFactor_messages from the last message passing iteration
        - prv_factor_beliefs (tensor): factor beliefs from the last message passing iteration
        - alpha (float): residual weighting for factorToVar_messages

        Outputs:
        - varToFactor_messages (tensor): varToFactor_messages in this message passing iteration 
        - factorToVar_messages (tensor): factorToVar_messages in this message passing iteration 
        - factor_beliefs (tensor): updated factor_beliefs 
        - var_beliefs (tensor): updated var_beliefs 
        """
        assert(not torch.isnan(prv_factor_beliefs).any()), prv_factor_beliefs
        assert(not torch.isnan(prv_factor_beliefs).any()), prv_factor_beliefs
        #update variable beliefs
        factorToVar_messages = alpha*self.message_factorToVar(prv_factor_beliefs, factor_graph, prv_varToFactor_messages) +\
                               (1 - alpha)*prv_factorToVar_messages
        assert(not torch.isnan(factorToVar_messages).any()), prv_factor_beliefs

        var_beliefs = scatter_('add', factorToVar_messages, factor_graph.factorToVar_edge_index[1], dim_size=factor_graph.numVars)
        if debug:
            print("var_beliefs pre norm:", torch.exp(var_beliefs))
        assert(len(var_beliefs.shape) == 2)
        var_beliefs = var_beliefs - logsumexp_multipleDim(var_beliefs, dim=0).view(-1,1)#normalize variable beliefs
        if debug:
            print("var_beliefs post norm:", torch.exp(var_beliefs))
        assert(not torch.isnan(var_beliefs).any()), var_beliefs

        #update factor beliefs
        varToFactor_messages = self.message_varToFactor(var_beliefs, factor_graph, prv_factorToVar_messages=factorToVar_messages) 
        assert(not torch.isnan(varToFactor_messages).any()), prv_factor_beliefs
        expansion_list = [2 for i in range(factor_graph.state_dimensions - 1)] + [-1,] #messages have states for one variable, add dummy dimensions for the other variables in factors
        varToFactor_expandedMessages = torch.stack([
            # message_state.expand(expansion_list).transpose(factor_graph.edge_var_indices[0, message_idx], factor_graph.edge_var_indices[0, message_idx]) for message_idx, message_state in enumerate(torch.unbind(varToFactor_messages, dim=0))
            message_state.expand(expansion_list).transpose(factor_graph.edge_var_indices[0, message_idx], factor_graph.state_dimensions-1) for message_idx, message_state in enumerate(torch.unbind(varToFactor_messages, dim=0))
                                          ], dim=0)


        #debug
        if debug:
            print("factor_graph.edge_var_indices[0]", factor_graph.edge_var_indices[0])
            for message_idx, message_state in enumerate(torch.unbind(varToFactor_messages, dim=0)):
                print("###")
                print("message_idx:", message_idx)
                print("factor_graph.edge_var_indices[0, message_idx]:", factor_graph.edge_var_indices[0, message_idx])
                print("factor_graph.state_dimensions-1:", factor_graph.state_dimensions-1)
                print("factor_graph.edge_var_indices[0, message_idx]:", factor_graph.edge_var_indices[0, message_idx])
                print("factor_graph.edge_var_indices[1, message_idx]:", factor_graph.edge_var_indices[1, message_idx])
                print("message_state.expand(expansion_list):", torch.exp(message_state.expand(expansion_list)))
                print(torch.exp(message_state.expand(expansion_list).transpose(factor_graph.edge_var_indices[1, message_idx], factor_graph.edge_var_indices[0, message_idx])))
                print
            print("varToFactor_expandedMessages:", torch.exp(varToFactor_expandedMessages))
        #end debug

        factor_beliefs = scatter_('add', varToFactor_expandedMessages, factor_graph.factorToVar_edge_index[0], dim_size=factor_graph.numFactors)
        
        assert(not torch.isnan(factor_beliefs).any()), factor_beliefs
      
        if debug:
            print("1 factor_beliefs:", torch.exp(factor_beliefs))
        
        if self.learn_BP:
            
            normalization_view = [1 for i in range(len(factor_beliefs.shape))]
            normalization_view[0] = -1
            factor_beliefs = factor_beliefs - logsumexp_multipleDim(factor_beliefs, dim=0).view(normalization_view)#normalize factor beliefs
            

            # print("factor_beliefs.shape:", factor_beliefs.shape)
            factor_beliefs_shape = factor_beliefs.shape
            # print("1 factor_beliefs:", factor_beliefs)
            # print("factor_beliefs.shape:", factor_beliefs.shape)
            assert(not torch.isnan(factor_beliefs).any()), factor_beliefs
            if self.avoid_nans:
                if self.logspace_mlp: #another try, stay with logspace
                    # pass
                    # assert((factor_beliefs==-np.inf).any())
                    masked_locations = torch.where(factor_graph.factor_potential_masks==1)
                    # factor_beliefs_neg_inf_locations = torch.where(factor_beliefs==-np.inf)
                    non_inf_belief_potential_locations = torch.where((factor_graph.factor_potential_masks==0) & (factor_beliefs!=-np.inf))
                    neg_inf_belief_locations = torch.where((factor_graph.factor_potential_masks==0) & (factor_beliefs==-np.inf)) #we set potentials to epsilon, so this is where beliefs are zero in 'valid' (i.e. not extra variable) locations
                    factor_beliefs_clone = factor_beliefs.clone()
                    factor_beliefs_clone[masked_locations] = 0
                    factor_beliefs_clone[neg_inf_belief_locations] = -99
                    
                    assert(not torch.isnan(factor_beliefs).any()), factor_beliefs
                    assert(not (factor_beliefs[non_inf_belief_potential_locations] == -np.inf).any()), factor_beliefs
                    check_factor_beliefs(factor_beliefs) #debugging
                    assert(not (factor_beliefs_clone==-np.inf).any()), factor_beliefs_temp
                    factor_beliefs_temp = self.mlp(factor_beliefs_clone.view(factor_beliefs_shape[0], -1)).view(factor_beliefs_shape)
                    # assert((factor_beliefs_temp == factor_beliefs_clone).all()), (factor_beliefs_temp, factor_beliefs_clone)
                    check_factor_beliefs(factor_beliefs) #debugging
                    assert(not torch.isnan(factor_beliefs_temp).any()), factor_beliefs_temp
                    
                    factor_beliefs = -np.inf*torch.ones_like(factor_beliefs)
                    factor_beliefs[non_inf_belief_potential_locations] = factor_beliefs_temp[non_inf_belief_potential_locations]

                else:
                    factor_beliefs = torch.exp(factor_beliefs) #go from log-space to standard probability space to avoid negative numbers, getting NaN's without this
                    factor_beliefs_clone = factor_beliefs.clone()
                    
                    check_factor_beliefs(factor_beliefs) #debugging
                    factor_beliefs_temp = self.mlp(factor_beliefs_clone.view(factor_beliefs_shape[0], -1)).view(factor_beliefs_shape)
                    # factor_beliefs = factor_beliefs_temp.clone()
                    # factor_beliefs = factor_beliefs.view(factor_beliefs_shape)
                    # factor_beliefs[factor_beliefs_neg_inf_locations] = -np.inf
                    # assert((factor_beliefs>0).all())

                    # factor_beliefs = torch.zeros(factor_beliefs_shape, dtype=factor_beliefs_temp.dtype, layout=factor_beliefs_temp.layout, device=factor_beliefs_temp.device)
                    # factor_beliefs = torch.zeros_like(factor_beliefs)
                    # valid_locations = torch.where((factor_beliefs!=-np.inf) & (factor_beliefs_temp>0))
                    # factor_beliefs[valid_locations] = factor_beliefs_temp[valid_locations]
                    # factor_beliefs = torch.log(factor_beliefs) #go back to log-space


                    valid_locations = torch.where((factor_graph.factor_potential_masks==0) & (factor_beliefs!=-np.inf))
                    # valid_locations = torch.where((factor_beliefs!=-np.inf) & (factor_beliefs_temp>0))
                    factor_beliefs = -np.inf*torch.ones_like(factor_beliefs)
                    factor_beliefs[valid_locations] = torch.log(factor_beliefs_temp[valid_locations])
                    
                    # check_factor_beliefs(factor_beliefs) #debugging
                    assert(not torch.isnan(factor_beliefs).any()), factor_beliefs
                    # factor_beliefs = self.mlp(factor_beliefs.view(factor_beliefs_shape[0], -1)) + .01
                    # factor_beliefs = factor_beliefs.view(factor_beliefs_shape)
                    

            else:
                factor_beliefs = torch.exp(factor_beliefs) #go from log-space to standard probability space to avoid negative numbers, getting NaN's without this
                factor_beliefs = self.mlp(factor_beliefs.view(factor_beliefs_shape[0], -1))
                factor_beliefs = factor_beliefs.view(factor_beliefs_shape)
                factor_beliefs = torch.log(factor_beliefs) #go back to log-space
                
            # factor_beliefs = self.mlp(factor_beliefs.view(factor_beliefs_shape[0], -1)) + .01
            # assert((factor_beliefs>=0).all())
            
            # factor_beliefs[torch.where(factor_beliefs == 0)] += .000001
            # print("factor_beliefs:", factor_beliefs)
            # factor_beliefs[torch.nonzero(factor_beliefs == 0, as_tuple=True)] += .000001
            # factor_beliefs = self.linear1(factor_beliefs.view(factor_beliefs_shape[0], -1))
            # print("2 factor_beliefs:", factor_beliefs)
            # print("factor_beliefs.shape:", factor_beliefs.shape)
            # sleep(chekc1)

        factor_beliefs += factor_graph.factor_potentials #factor_potentials previously x_base

        assert(not torch.isnan(factor_beliefs).any()), factor_beliefs
        if debug:
            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("factor_beliefs pre norm:", torch.exp(factor_beliefs))
        normalization_view = [1 for i in range(len(factor_beliefs.shape))]
        normalization_view[0] = -1
        log_sum_exp_factor_beliefs = logsumexp_multipleDim(factor_beliefs, dim=0).view(normalization_view)
        factor_beliefs = factor_beliefs - log_sum_exp_factor_beliefs#normalize factor beliefs
        assert((log_sum_exp_factor_beliefs != -np.inf).all()) #debugging
        # for factor_idx in range(factor_beliefs.shape[0]):
        #     assert((factor_beliefs[factor_idx] != -np.inf).all()), (factor_beliefs[factor_idx], log_sum_exp_factor_beliefs[factor_idx], (factor_beliefs[factor_idx] != -np.inf))
        check_factor_beliefs(factor_beliefs) #debugging
        assert(not torch.isnan(factor_beliefs).any()), (factor_beliefs, factor_graph.numFactors, factor_graph.numVars)
        if debug:
            print("factor_beliefs post norm:", torch.exp(factor_beliefs))
            print("torch.sum(factor_beliefs) post norm:", torch.sum(torch.exp(factor_beliefs)))
            sleep(check_norms2)

        if debug:
            print("3 factor_beliefs:", torch.exp(factor_beliefs))
            print("factor_graph.factorToVar_edge_index[0]:", factor_graph.factorToVar_edge_index[0])
            print("factor_graph.factorToVar_edge_index:", factor_graph.factorToVar_edge_index)
            sleep(debug34)
        
        return varToFactor_messages, factorToVar_messages, var_beliefs, factor_beliefs


# def logsumexp(tensor, dim):
#     tensor_exp = tor

    def message_factorToVar(self, prv_factor_beliefs, factor_graph, prv_varToFactor_messages):
        # prv_factor_beliefs has shape [E, X.shape] (double check)
        # factor_graph.prv_varToFactor_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # factor_graph.edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at

        mapped_factor_beliefs = map_beliefs(prv_factor_beliefs, factor_graph, 'factor')
        mapped_factor_potentials_masks = map_beliefs(factor_graph.factor_potential_masks, factor_graph, 'factor')

        if self.avoid_nans:
            #best idea: set to say 0, then log sum exp with -# of infinities precomputed tensor to correct

            # mapped_factor_beliefs[torch.where(mapped_factor_beliefs==-np.inf)] = -99
            mapped_factor_beliefs[torch.where((mapped_factor_potentials_masks==0) & (mapped_factor_beliefs==-np.inf))] = -99 #leave invalid beliefs at -inf

            # factor_beliefs_neg_inf_locations = torch.where(mapped_factor_beliefs==-np.inf)
            # mapped_factor_beliefs[factor_beliefs_neg_inf_locations] = 0
            # adjustment_tensor = torch.zeros_like(mapped_factor_beliefs)
            # adjustment_tensor[factor_beliefs_neg_inf_locations] = 1

            # tensor with dimensions [edges, 2] for binary variables
            marginalized_states = torch.stack([
                torch.logsumexp(node_state, dim=tuple([i for i in range(len(node_state.shape)) if i != factor_graph.edge_var_indices[0, edge_idx]])) for edge_idx, node_state in enumerate(torch.unbind(mapped_factor_beliefs, dim=0))
                                              ], dim=0)

            # marginalized_adjustment_tensor = torch.stack([
            #     torch.logsumexp(node_adjustment, dim=tuple([i for i in range(len(node_adjustment.shape)) if i != factor_graph.edge_var_indices[0, edge_idx]])) for edge_idx, node_adjustment in enumerate(torch.unbind(adjustment_tensor, dim=0))
            #                                   ], dim=0)

            # marginalized_states = logminusexp(marginalized_states, marginalized_adjustment_tensor)

        else:
            # tensor with dimensions [edges, 2] for binary variables
            marginalized_states = torch.stack([
                torch.logsumexp(node_state, dim=tuple([i for i in range(len(node_state.shape)) if i != factor_graph.edge_var_indices[0, edge_idx]])) for edge_idx, node_state in enumerate(torch.unbind(mapped_factor_beliefs, dim=0))
                # torch.logsumexp(node_state, dim=tuple([i for i in range(len(node_state.shape)) if i != factor_graph.edge_var_indices[0, edge_idx]])) for edge_idx, node_state in enumerate(torch.unbind(mapped_factor_beliefs, dim=0))
                                              ], dim=0)

        prv_varToFactor_messages_zeroed = prv_varToFactor_messages.clone()
        prv_varToFactor_messages_zeroed[prv_varToFactor_messages_zeroed == -np.inf] = 0
        #avoid double counting
        messages = marginalized_states - prv_varToFactor_messages_zeroed

        return messages

    def message_varToFactor(self, var_beliefs, factor_graph, prv_factorToVar_messages):
        # var_beliefs has shape [E, X.shape] (double check)
        # factor_graph.prv_factorToVar_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # factor_graph.edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at

        mapped_var_beliefs = map_beliefs(var_beliefs, factor_graph, 'var')

        prv_factorToVar_messages_zeroed = prv_factorToVar_messages.clone()
        prv_factorToVar_messages_zeroed[prv_factorToVar_messages_zeroed == -np.inf] = 0
        #avoid double counting
        messages = mapped_var_beliefs - prv_factorToVar_messages_zeroed            

        return messages

def check_factor_beliefs(factor_beliefs):
    '''
    Check that none of the factors have all beliefs set to -infinity
    For debugging
    '''
    for factor_idx in range(factor_beliefs.shape[0]):
        assert((factor_beliefs[factor_idx] != -np.inf).any())

def test_LoopyBP_ForSAT_automatedGraphConstruction(filename=None, learn_BP=True):
    if filename is None:
        clauses = [[1, -2], [-2, 3], [1, 3]] # count=4
        # clauses = [[-1, 2], [1, -2]] # count=2
        # clauses = [[-1, 2]] # count=3
        # clauses = [[1, -2, 3], [-2, 4], [3]] # count=6
        # clauses = [[1, -2], [-2, 4], [3]] # count=5
        # clauses = [[1, 2], [-2, 3]] # count=4
        # clauses = [[1, 2]] # count=3
    else:
        n_vars, clauses = parse_dimacs(filename)

    factor_graph = build_factorgraph_from_SATproblem(clauses)
    prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs = factor_graph.get_initial_beliefs_and_messages()
    run_loopy_bp(factor_graph, prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs, iters, learn_BP)

def run_loopy_bp(factor_graph, prv_varToFactor_messages, prv_factor_beliefs, iters, learn_BP, verbose=False):
    # print("factor_graph:", factor_graph)
    # print("factor_graph['state_dimensions']:", factor_graph['state_dimensions'])
    # print("factor_graph.state_dimensions:", factor_graph.state_dimensions)
    msg_passing_layer = FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=learn_BP, factor_state_space=2**factor_graph.state_dimensions.item())

    for itr in range(iters):
        if verbose:
            print('variable beliefs:', torch.exp(var_beliefs))
            print('factor beliefs:', torch.exp(factor_beliefs))
            print('prv_factorToVar_messages:', torch.exp(factor_graph.prv_factorToVar_messages))
            print('prv_varToFactor_messages:', torch.exp(factor_graph.prv_varToFactor_messages))
        
        prv_varToFactor_messages, prv_factorToVar_messages, prv_var_beliefs, prv_factor_beliefs = msg_passing_layer(factor_graph, prv_varToFactor_messages, prv_factor_beliefs)        
    bethe_free_energy = factor_graph.compute_bethe_free_energy(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs)
    z_estimate_bethe = torch.exp(-bethe_free_energy)

    if verbose:
        print()
        print('-'*80)
        print("Bethe free energy =", bethe_free_energy)
        print("Partition function estimate from Bethe free energy =", z_estimate_bethe)    
    return bethe_free_energy, z_estimate_bethe

def plot_lbp_vs_exactCount(dataset_size=10, verbose=True, lbp_iters=4):
    #fix problems to load!!
    problems_to_load = ["01A-1","01B-1","01B-2","01B-3"]
    sat_data = SatProblems(problems_to_load=problems_to_load,
               counts_dir_name="/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/SAT_problems_solved_counts",
               problems_dir_name="/atlas/u/jkuck/GNN_sharpSAT/data/SAT_problems_under_5k/training_generated/SAT_problems_solved",
               dataset_size=dataset_size)
    data_loader = DataLoader(sat_data, batch_size=1)

    exact_solution_counts = []
    lbp_estimated_counts = []
    for sat_problem, log_solution_count in data_loader:
        # sat_problem.compute_bethe_free_energy()
        sat_problem = FactorGraph.init_from_dictionary(sat_problem, squeeze_tensors=True)
        prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs = sat_problem.get_initial_beliefs_and_messages()
        assert(sat_problem.state_dimensions == 5)
        exact_solution_counts.append(log_solution_count)


        bethe_free_energy, z_estimate_bethe = run_loopy_bp(factor_graph=sat_problem, prv_varToFactor_messages=prv_varToFactor_messages, 
                                                           prv_factor_beliefs=prv_factor_beliefs, iters=lbp_iters, learn_BP=True)
        lbp_estimated_counts.append(-bethe_free_energy)

        if verbose:
            print("exact log_solution_count:", log_solution_count)
            print("bethe estimate log_solution_count:", -bethe_free_energy)
            # print("sat_problem:", sat_problem)
            print("sat_problem.factor_potentials.shape:", sat_problem.factor_potentials.shape)
            print("sat_problem.numVars.shape:", sat_problem.numVars.shape)
            print("sat_problem.numFactors.shape:", sat_problem.numFactors.shape)
            print("prv_factor_beliefs.shape:", prv_factor_beliefs.shape)
            print("prv_var_beliefs.shape:", prv_var_beliefs.shape)
            print()  

    plt.plot(exact_solution_counts, lbp_estimated_counts, 'x', c='b', label='Negative Bethe Free Energy, %d iters' % lbp_iters)
    plt.plot([min(exact_solution_counts), max(exact_solution_counts)], [min(exact_solution_counts), max(exact_solution_counts)], '-', c='g', label='Exact Estimate')

    # plt.axhline(y=math.log(2)*log_2_Z[PROBLEM_NAME], color='y', label='Ground Truth ln(Set Size)') 
    plt.xlabel('ln(Exact Model Count)', fontsize=14)
    plt.ylabel('ln(Estimated Model Count)', fontsize=14)
    plt.title('Exact Model Count vs. Bethe Estimate', fontsize=20)
    plt.legend(fontsize=12)    
    #make the font bigger
    matplotlib.rcParams.update({'font.size': 10})        

    plt.grid(True)
    # Shrink current axis's height by 10% on the bottom
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0 + box.height * 0.1,
    #                 box.width, box.height * 0.9])
    #fig.savefig('/Users/jkuck/Downloads/temp.png', bbox_extra_artists=(lgd,), bbox_inches='tight')    

    plt.show()


if __name__ == "__main__":
    plot_lbp_vs_exactCount()
    sleep(check_data_loader)

    test_LoopyBP_ForSAT_automatedGraphConstruction()
    sleep(temp34)
    # SAT_filename = "/atlas/u/jkuck/approxmc/counting2/s641_3_2.cnf.gz.no_w.cnf"
    # SAT_filename = '/atlas/u/jkuck/low_density_parity_checks/SAT_problems_cnf/tire-1.cnf'
    SAT_filename = SAT_PROBLEM_DIRECTORY + 'test4.cnf'
    test_LoopyBP_ForSAT_automatedGraphConstruction(filename=SAT_filename, learn_BP=False)
