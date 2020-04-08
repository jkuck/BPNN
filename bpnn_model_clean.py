import torch
from torch.nn import Sequential as Seq, Linear, ReLU
import numpy as np
from torch.utils.data import DataLoader
from torch_geometric.utils import scatter_
from torch_scatter import scatter_logsumexp
from sat_helpers.sat_data import parse_dimacs, SatProblems, build_factorgraph_from_SATproblem
from utils import dotdict, logminusexp, shift_func #wrote a helper function that is not used in this file: log_normalize

import matplotlib.pyplot as plt
import matplotlib
import mrftools
from parameters import alpha, alpha2, LN_ZERO

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
    - factor_graph: (FactorGraphData, defined in factor_graph.py) the factor graph whose beliefs we are mapping
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

#     if size is not None and size[idx] != mapped_beliefs.size(0):
#         print("factor_graph:", factor_graph)
#         print("beliefs:", beliefs)
#         print("beliefs.shape:", beliefs.shape)
#         print("size:", size)
#         print("idx:", idx)
#         print("mapped_beliefs.size(0):", mapped_beliefs.size(0))
#         raise ValueError(__size_error_msg__)

#     print(type(beliefs), type(mapped_beliefs), type(factor_graph.facToVar_edge_idx))
#     print(beliefs.device, mapped_beliefs.device, factor_graph.facToVar_edge_idx.device)
    assert(factor_graph is not None)
    mapped_beliefs = torch.index_select(mapped_beliefs, 0, factor_graph.facToVar_edge_idx[idx])
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




class FactorGraphMsgPassingLayer_NoDoubleCounting(torch.nn.Module):
    r"""Perform message passing in factor graphs without 'double counting'
    i.e. divide by previously sent messages as in loopy belief propagation

    Inputs:
    - learn_BP (bool): if False run standard looph belief propagation, if True
        insert a neural network into message aggregation for learning
    - lne_mlp (bool): if False mlp runs in log space (had trouble with good results), 
        if True we take exponent then run mlp then take log
    """

    def __init__(self, learn_BP=True, factor_state_space=None, avoid_nans=True, lne_mlp=False, use_MLP1=True, use_MLP2=True):
        super(FactorGraphMsgPassingLayer_NoDoubleCounting, self).__init__()

        self.use_MLP1 = use_MLP1
        self.use_MLP2 = use_MLP2
        self.learn_BP = learn_BP
        self.avoid_nans = avoid_nans
        self.lne_mlp = lne_mlp
        print("learn_BP:", learn_BP)
        if learn_BP:
            assert(factor_state_space is not None)     
            self.linear1 = Linear(factor_state_space, factor_state_space)
            self.linear2 = Linear(factor_state_space, factor_state_space)
            self.linear1.weight = torch.nn.Parameter(torch.eye(factor_state_space))
            self.linear1.bias = torch.nn.Parameter(torch.zeros(self.linear1.bias.shape))
            self.linear2.weight = torch.nn.Parameter(torch.eye(factor_state_space))
            self.linear2.bias = torch.nn.Parameter(torch.zeros(self.linear2.bias.shape))

            self.shifted_relu = shift_func(ReLU(), shift=.0000000000000000001) #we'll get NaN's if we take the log of 0 or a negative number when going back to log space           
            if lne_mlp:            
                self.mlp1 = Seq(self.linear1, ReLU(), self.linear2, self.shifted_relu)  
            else:
                self.mlp1 = Seq(self.linear1, self.linear2)  


            #add factor potential as part of MLP
#                 self.linear3 = Linear(factor_state_space*2, factor_state_space)
#                 self.linear4 = Linear(factor_state_space, factor_state_space)
#                 self.linear3.weight = torch.nn.Parameter(torch.cat([torch.eye(factor_state_space), torch.eye(factor_state_space)], 1))
#                 self.linear3.bias = torch.nn.Parameter(torch.zeros(self.linear1.bias.shape))
#                 self.linear4.weight = torch.nn.Parameter(torch.eye(factor_state_space))
#                 self.linear4.bias = torch.nn.Parameter(torch.zeros(self.linear2.bias.shape))

#                 self.shifted_relu1 = shift_func(ReLU(), shift=-50) #allow beliefs less than 0    

            #add factor potential after MLP
            self.linear3 = Linear(factor_state_space, factor_state_space)
            self.linear4 = Linear(factor_state_space, factor_state_space)
            self.linear3.weight = torch.nn.Parameter(torch.eye(factor_state_space))
            self.linear3.bias = torch.nn.Parameter(torch.zeros(self.linear1.bias.shape))
            self.linear4.weight = torch.nn.Parameter(torch.eye(factor_state_space))
            self.linear4.bias = torch.nn.Parameter(torch.zeros(self.linear2.bias.shape))

            self.shifted_relu1 = shift_func(ReLU(), shift=.0000000000000000001) #we'll get NaN's if we take the log of 0 or a negative number when going back to log space   
            if lne_mlp:
                self.mlp2 = Seq(self.linear3, ReLU(), self.linear4, self.shifted_relu1) 
            else:
                self.mlp2 = Seq(self.linear3, self.linear4)  


    def forward(self, factor_graph, prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs):
        '''
        Inputs:
        - factor_graph: (FactorGraphData, defined in factor_graph.py) the factor graph we will perform one
            iteration of message passing on.
        '''
        # Step 3-5: Start propagating messages.
        return self.propagate(factor_graph=factor_graph, prv_varToFactor_messages=prv_varToFactor_messages,
                              prv_factorToVar_messages=prv_factorToVar_messages, prv_factor_beliefs=prv_factor_beliefs)

    
    
    #testing a simplified version, modelling GIN, that preserves no double counting computation graph
    def propagate(self, factor_graph, prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs,\
                  alpha=alpha, alpha2=alpha2, debug=False, normalize_messages=True, normalize_beliefs=False):
        r"""Perform one iteration of message passing.  Pass messages from factors to variables, then
        from variables to factors.

        Notes:
        - safe to clamp messages to be larger than LN_ZERO constant, but don't clamp or mess with beliefs.
          Beliefs should just be the sum of messages during message passing and are only computed to make
          the computation more efficient. (sum once, then subtract out messages, do this exactly)

        Inputs:
        - factor_graph: (FactorGraphData, defined in factor_graph.py) the factor graph we will perform one
            iteration of message passing on.
        - prv_varToFactor_messages (tensor): varToFactor_messages from the last message passing iteration
        - prv_factor_beliefs (tensor): factor beliefs from the last message passing iteration
        - alpha (float): residual weighting for factorToVar_messages. alpha=1 corresponds to no skip connection, 
            alpha=0 removes the neural network

        Outputs:
        - varToFactor_messages (tensor): varToFactor_messages in this message passing iteration 
        - factorToVar_messages (tensor): factorToVar_messages in this message passing iteration 
        - factor_beliefs (tensor): updated factor_beliefs 
        - var_beliefs (tensor): updated var_beliefs 
        """
        assert(not torch.isnan(prv_factor_beliefs).any()), prv_factor_beliefs
        assert(not torch.isnan(prv_factor_beliefs).any()), prv_factor_beliefs
        assert(factor_graph is not None)
        #update variable beliefs
        factorToVar_messages = self.message_factorToVar(prv_factor_beliefs=prv_factor_beliefs, factor_graph=factor_graph,\
                                                              prv_varToFactor_messages=prv_varToFactor_messages,\
                                                              prv_factorToVar_messages=prv_factorToVar_messages, alpha=alpha,\
                                                              normalize_messages=normalize_messages)

        var_beliefs = scatter_('add', factorToVar_messages, factor_graph.facToVar_edge_idx[1], dim_size=factor_graph.num_vars)
        assert(len(var_beliefs.shape) == 2)
        if normalize_beliefs:
            var_beliefs = var_beliefs - logsumexp_multipleDim(var_beliefs, dim=0).view(-1,1)#normalize variable beliefs
        assert(not torch.isnan(var_beliefs).any()), var_beliefs
        
        #update factor beliefs
        varToFactor_messages = self.message_varToFactor(var_beliefs, factor_graph, prv_factorToVar_messages=factorToVar_messages,\
                                                        normalize_messages=normalize_messages)
        
        expansion_list = [2 for i in range(factor_graph.state_dimensions - 1)] + [-1,] #messages have states for one variable, add dummy dimensions for the other variables in factors
        new_shape = [varToFactor_messages.shape[0]] + expansion_list
        assert(len(varToFactor_messages.shape) == 2)
        fast_expansion_list = [2 for i in range(factor_graph.state_dimensions.item() - 1)] + list(varToFactor_messages.shape)
        varToFactor_messages_expand = varToFactor_messages.expand(fast_expansion_list)
        varToFactor_messages_expand = varToFactor_messages_expand.transpose(-2, 0).transpose(-1, 1) #expanded dimensions are prepended, this moves them to the to back
        varToFactor_messages_expand_flatten = varToFactor_messages_expand.flatten()
        varToFactor_expandedMessages = scatter_('add', src=varToFactor_messages_expand_flatten, index=factor_graph.varToFactorMsg_scatter_indices, dim=0)            
        varToFactor_expandedMessages = varToFactor_expandedMessages.reshape(new_shape)
        #print("123debug2 varToFactor_expandedMessages =", varToFactor_expandedMessages)


########## begin if self.learn_BP: if statement
#         if self.learn_BP:
#             if self.num_mlps == 2:
                
#         normalization_view = [1 for i in range(len(varToFactor_expandedMessages.shape))]
#         normalization_view[0] = -1

#         # FIX ME, DO NOT THINK WE SHUOLD NORMALIZE AFTER EXPANDING!! actually don't think anything wrong with this, just a weird place...
#         varToFactor_expandedMessages = varToFactor_expandedMessages - logsumexp_multipleDim(varToFactor_expandedMessages, dim=0).view(normalization_view)#normalize factor beliefs


        if self.use_MLP1:
            if self.lne_mlp:
                varToFactor_expandedMessages_shape = varToFactor_expandedMessages.shape
                assert(not torch.isnan(varToFactor_expandedMessages).any()), varToFactor_expandedMessages

                varToFactor_expandedMessages = torch.exp(varToFactor_expandedMessages) #go from log-space to standard probability space to avoid negative numbers, getting NaN's without this
                varToFactor_expandedMessages_clone = varToFactor_expandedMessages.clone()
                varToFactor_expandedMessages_temp = (1-alpha2)*self.mlp1(varToFactor_expandedMessages_clone.view(varToFactor_expandedMessages_shape[0], -1)).view(varToFactor_expandedMessages_shape) + alpha2*varToFactor_expandedMessages_clone

                valid_locations = torch.where((varToFactor_expandedMessages>LN_ZERO))
                # valid_locations = torch.where((varToFactor_expandedMessages>LN_ZERO) & (varToFactor_expandedMessages_temp>0))
                varToFactor_expandedMessages = LN_ZERO*torch.ones_like(varToFactor_expandedMessages)
                varToFactor_expandedMessages[valid_locations] = torch.log(varToFactor_expandedMessages_temp[valid_locations])

            else:
                varToFactor_expandedMessages_shape = varToFactor_expandedMessages.shape
                assert(not torch.isnan(varToFactor_expandedMessages).any()), varToFactor_expandedMessages

                varToFactor_expandedMessages = (1-alpha2)*self.mlp1(varToFactor_expandedMessages.view(varToFactor_expandedMessages_shape[0], -1)).view(varToFactor_expandedMessages_shape) + alpha2*varToFactor_expandedMessages
                
            
        assert(not torch.isnan(varToFactor_expandedMessages).any()), varToFactor_expandedMessages

#                 factor_beliefs = scatter_('add', varToFactor_expandedMessages, factor_graph.facToVar_edge_idx[0], dim_size=factor_graph.numFactors)
#         factor_beliefs = scatter_('add', varToFactor_expandedMessages, factor_graph.facToVar_edge_idx[0]) #for batching
        factor_beliefs = scatter_('add', varToFactor_expandedMessages, factor_graph.facToVar_edge_idx[0], dim_size=factor_graph.num_factors) #for batching

    
    
        num_factors = torch.sum(factor_graph.numFactors)            

        assert(num_factors == factor_beliefs.shape[0]), (num_factors, factor_beliefs.shape[0])
        assert(num_factors == factor_graph.factor_potentials.shape[0]), (num_factors, factor_graph.factor_potentials.shape[0])

        if self.use_MLP2:
            if self.lne_mlp:
                factor_beliefs_shape = factor_beliefs.shape
        #                 factor_beliefs = self.mlp2(torch.cat([factor_beliefs.view(num_factors,-1), factor_graph.factor_potentials.view(num_factors,-1)], 1)).view(factor_beliefs_shape)
        #                 factor_beliefs = self.mlp2(factor_beliefs.view(num_factors,-1)).view(factor_beliefs_shape)

                factor_beliefs = torch.exp(factor_beliefs) #go from log-space to standard probability space to avoid negative numbers, getting NaN's without this
                factor_beliefs_clone = factor_beliefs.clone()

                check_factor_beliefs(factor_beliefs) #debugging
                factor_beliefs_temp = (1-alpha2)*self.mlp2(factor_beliefs_clone.view(factor_beliefs_shape[0], -1)).view(factor_beliefs_shape) + alpha2*factor_beliefs_clone
                valid_locations = torch.where((factor_graph.factor_potential_masks==0) & (factor_beliefs>LN_ZERO))
                # valid_locations = torch.where((factor_beliefs>LN_ZERO) & (factor_beliefs_temp>0))
                factor_beliefs = LN_ZERO*torch.ones_like(factor_beliefs)
                factor_beliefs[valid_locations] = torch.log(factor_beliefs_temp[valid_locations])

                # check_factor_beliefs(factor_beliefs) #debugging
                assert(not torch.isnan(factor_beliefs).any()), factor_beliefs

            else:
                factor_beliefs_shape = factor_beliefs.shape
                check_factor_beliefs(factor_beliefs) #debugging
                factor_beliefs = (1-alpha2)*self.mlp2(factor_beliefs.view(factor_beliefs_shape[0], -1)).view(factor_beliefs_shape) + alpha2*factor_beliefs

                # check_factor_beliefs(factor_beliefs) #debugging
                assert(not torch.isnan(factor_beliefs).any()), factor_beliefs
                

        factor_beliefs += factor_graph.factor_potentials #factor_potentials previously x_base
        factor_beliefs[torch.where(factor_graph.factor_potential_masks==1)] = LN_ZERO
        
########## end if self.learn_BP: if statement

                
            

        assert(not torch.isnan(factor_beliefs).any()), factor_beliefs

        normalization_view = [1 for i in range(len(factor_beliefs.shape))]
        normalization_view[0] = -1
        log_sum_exp_factor_beliefs = logsumexp_multipleDim(factor_beliefs, dim=0).view(normalization_view)
        if normalize_beliefs:
            factor_beliefs = factor_beliefs - log_sum_exp_factor_beliefs#normalize factor beliefs
        
        
        assert((log_sum_exp_factor_beliefs != -np.inf).all()) #debugging
        assert((log_sum_exp_factor_beliefs >= LN_ZERO).all()) #debugging
        check_factor_beliefs(factor_beliefs) #debugging
        assert(not torch.isnan(factor_beliefs).any()), (factor_beliefs, factor_graph.numFactors, factor_graph.numVars)
        factor_beliefs[torch.where(factor_graph.factor_potential_masks==1)] = LN_ZERO
#         assert((factor_beliefs[torch.where(factor_graph.factor_potential_masks==1)] == LN_ZERO).all())

        return varToFactor_messages, factorToVar_messages, var_beliefs, factor_beliefs

    
    
# def logsumexp(tensor, dim):
#     tensor_exp = tor

    def message_factorToVar(self, prv_factor_beliefs, factor_graph, prv_varToFactor_messages, prv_factorToVar_messages,\
                            alpha, normalize_messages, debug=False, fast_logsumexp=True):
        #subtract previous messages from current factor beliefs to get factor to variable messages
        # prv_factor_beliefs has shape [E, X.shape] (double check)
        # factor_graph.prv_varToFactor_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # factor_graph.edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at

        
        assert((prv_factor_beliefs[torch.where(factor_graph.factor_potential_masks==1)] == LN_ZERO).all())
        
        mapped_factor_beliefs = map_beliefs(prv_factor_beliefs, factor_graph, 'factor')
        mapped_factor_potentials_masks = map_beliefs(factor_graph.factor_potential_masks, factor_graph, 'factor')

        
        #print("123debug10 mapped_factor_beliefs =", mapped_factor_beliefs)
        #best idea: set to say 0, then log sum exp with -# of infinities precomputed tensor to correct

        # mapped_factor_beliefs[torch.where(mapped_factor_beliefs==-np.inf)] = -99

        #was using 2/4/2020
#         mapped_factor_beliefs[torch.where((mapped_factor_potentials_masks==0) & (mapped_factor_beliefs==-np.inf))] = -99 #leave invalid beliefs at -inf

        #new 4/4/2020
#         print("mapped_factor_beliefs[torch.where(mapped_factor_potentials_masks==1)]:")
#         print(mapped_factor_beliefs[torch.where(mapped_factor_potentials_masks==1)])
        
#         print("mapped_factor_beliefs:")
#         print(mapped_factor_beliefs)        
        
        assert((mapped_factor_beliefs[torch.where(mapped_factor_potentials_masks==1)] == LN_ZERO).all())
        mapped_factor_beliefs[torch.where(mapped_factor_beliefs<LN_ZERO)] = LN_ZERO
        
        num_edges = factor_graph.facToVar_edge_idx.shape[1]
        assert(mapped_factor_beliefs.view(mapped_factor_beliefs.numel()).shape == factor_graph.facStates_to_varIdx.shape)
        assert((factor_graph.facStates_to_varIdx <= num_edges*2).all())
        marginalized_states_fast = scatter_logsumexp(src=mapped_factor_beliefs.view(mapped_factor_beliefs.numel()), index=factor_graph.facStates_to_varIdx, dim_size=num_edges*2 + 1) 
        marginalized_states = marginalized_states_fast[:-1].view(num_edges,2)

        assert((prv_varToFactor_messages >= LN_ZERO).all()), (torch.min(prv_varToFactor_messages))

        #avoid double counting
        factorToVar_messages = marginalized_states - prv_varToFactor_messages
        
        # FIX ME
#         TO DO:
#             - normalize messages
#             - check if cloning in following is necessary 
#         assert(False), "finish these todos"
#         #for numerical stability, don't let log messages get smaller than LN_ZERO (constant set in parameters.py)
        factorToVar_messages = torch.clamp(factorToVar_messages, min=LN_ZERO)
        factorToVar_messages = alpha*factorToVar_messages + (1 - alpha)*prv_factorToVar_messages
        if normalize_messages:
#             print("pre normalization factorToVar_messages:")
#             print(factorToVar_messages)
            
            factorToVar_messages = factorToVar_messages - logsumexp_multipleDim(factorToVar_messages, dim=0).view(-1,1)#normalize variable beliefs
            check_messages = torch.sum(torch.exp(factorToVar_messages), dim=-1)
            assert(torch.max(torch.abs(check_messages-1)) < .00001), (torch.sum(torch.abs(check_messages-1)), torch.max(torch.abs(check_messages-1)), check_messages)

#             print("post normalization factorToVar_messages:")
#             print(factorToVar_messages)
#             check_messages = torch.sum(torch.exp(factorToVar_messages), dim=-1)
#             print("check_messages:", check_messages)
#             print("factorToVar_messages.shape:", factorToVar_messages.shape)
#             print("check_messages.shape:", check_messages.shape)
#             sleep(lskafdjlks)           
        factorToVar_messages = torch.clamp(factorToVar_messages, min=LN_ZERO)
        assert(not torch.isnan(factorToVar_messages).any()), prv_factor_beliefs        
        return factorToVar_messages

    def message_varToFactor(self, var_beliefs, factor_graph, prv_factorToVar_messages, normalize_messages):
        #subtract previous messages from current variable beliefs to get variable to factor messages
        # var_beliefs has shape [E, X.shape] (double check)
        # factor_graph.prv_factorToVar_messages has shape [2, E], e.g. two messages (for each binary variable state) for each edge on the last iteration
        # factor_graph.edge_var_indices has shape [2, E]. 
        #   [0, i] indicates the index (0 to var_degree_origin - 1) of edge i, among all edges originating at the node which edge i begins at
        #   [1, i] indicates the index (0 to var_degree_end - 1) of edge i, among all edges ending at the node which edge i ends at

        mapped_var_beliefs = map_beliefs(var_beliefs, factor_graph, 'var')
        assert((prv_factorToVar_messages >= LN_ZERO).all())

        #avoid double counting
        varToFactor_messages = mapped_var_beliefs - prv_factorToVar_messages            

        # FIX ME
#         TO DO:
#             - normalize messages
#         assert(False), "finish these todos"
            
        #for numerical stability, don't let log messages get smaller than LN_ZERO (constant set in parameters.py)
        varToFactor_messages = torch.clamp(varToFactor_messages, min=LN_ZERO)
        if normalize_messages:
            varToFactor_messages = varToFactor_messages - logsumexp_multipleDim(varToFactor_messages, dim=0).view(-1,1)#normalize variable beliefs
#             print("post normalization varToFactor_messages:")
#             print(varToFactor_messages)
            check_messages = torch.sum(torch.exp(varToFactor_messages), dim=-1)
            assert(torch.max(torch.abs(check_messages-1)) < .00001), (torch.sum(torch.abs(check_messages-1)), torch.max(torch.abs(check_messages-1)), check_messages)

        varToFactor_messages = torch.clamp(varToFactor_messages, min=LN_ZERO)
        assert(not torch.isnan(varToFactor_messages).any()), prv_factor_beliefs            
        return varToFactor_messages    
    
def check_factor_beliefs(factor_beliefs):
    '''
    Check that none of the factors have all beliefs set to -infinity
    For debugging
    '''
    pass
#     for factor_idx in range(factor_beliefs.shape[0]):
#         assert((factor_beliefs[factor_idx] != -np.inf).any())

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

        # switched from FactorGraph to FactorGraphData.  Removed this line, might be a bug somewhere because of this though
        # sat_problem = FactorGraph.init_from_dictionary(sat_problem, squeeze_tensors=True)

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
