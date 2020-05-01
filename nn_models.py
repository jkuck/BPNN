import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import remove_self_loops
import numpy as np
from torch.nn import Sequential as Seq, Linear, ReLU
from utils import neg_inf_to_zero, shift_func
import math
import time

# from bpnn_model import FactorGraphMsgPassingLayer_NoDoubleCounting
# from bpnn_model_partialRefactorNoBeliefRepeats import FactorGraphMsgPassingLayer_NoDoubleCounting
USE_OLD_CODE = False
if USE_OLD_CODE:
    from bpnn_model_partialRefactorNoBeliefRepeats import FactorGraphMsgPassingLayer_NoDoubleCounting
#     from bpnn_model_partialRefactor import FactorGraphMsgPassingLayer_NoDoubleCounting
    from bpnn_model_clean import logsumexp_multipleDim
else:
    from bpnn_model_clean import FactorGraphMsgPassingLayer_NoDoubleCounting, logsumexp_multipleDim

from parameters import alpha2
import time
class lbp_message_passing_network(nn.Module):
    def __init__(self, max_factor_state_dimensions, msg_passing_iters, lne_mlp, use_MLP1, use_MLP2, use_MLP3, use_MLP4,
                 subtract_prv_messages, share_weights, bethe_MLP,
                belief_repeats=None, var_cardinality=None, learn_bethe_residual_weight=False,
                 initialize_to_exact_bethe = True, alpha_damping_FtoV=None, alpha_damping_VtoF=None, use_old_bethe=None,
                 APPLY_BP_POST_BPNN=False, APPLY_BP_EVERY_ITER=False, BPNN_layers_per_shared_weight_layer=1):
        '''
        Inputs:
        - max_factor_state_dimensions (int): the number of dimensions (variables) the largest factor have.
            -> will have states space of size 2*max_factor_state_dimensions
        - msg_passing_iters (int): the number of iterations of message passing to run (we have this many
            message passing layers with their own learnable parameters)
            
        - lne_mlp (bool): if True message passing mlps operate in standard space rather than log space
        - use_MLP1 (bool): one of the original MLPs that operate on factor beliefs (problematic because they're not index invariant)            
        - use_MLP2 (bool): one of the original MLPs that operate on factor beliefs (problematic because they're not index invariant)            
        - use_MLP3 (bool): one of the new MLPs that operate on variable beliefs
        - use_MLP4 (bool): one of the new MLPs that operate on variable beliefs        
        - subtract_prv_messages (bool): if true, subtract previously sent messages (to avoid 'double counting')
            
        - share_weights (bool): if true, share the same weights across each message passing iteration
        - bethe_MLP (string): ['shifted','standard','linear','none']
                            if 'none', then use the standard bethe approximation with no learning.
                            otherwise, use an MLP to learn a modified Bethe approximation where this argument
                            describes (potential) non linearities in the MLP
        
        - learn_bethe_residual_weight (bool): if True, (and bethe_MLP is true) learn use the bethe_MLP
            to predict the residual between then Bethe approximation and the exact partition function
        - initialize_to_exact_bethe (bool): if True initialize the bethe_MLP to perform exact computation
            of the bethe approximation (for beliefs from the last round of message passing).  may make
            training worse.  should be False if learn_bethe_residual_weight=True
            
        - APPLY_BP_POST_BPNN (bool): if True, apply standard BP message passing iterations (no learned MLPs) after BPNN layers
        - APPLY_BP_EVERY_ITER (bool): if True, apply a standard BP message passing interation (no learned MLPS) after every shared weight BPNN layer
        - BPNN_layers_per_shared_weight_layer (int): apply this many BP layers (with different weights) in every shared weight layer
        '''
        super().__init__()        
        self.share_weights = share_weights
        self.msg_passing_iters = msg_passing_iters
        self.bethe_MLP = bethe_MLP
        self.belief_repeats = belief_repeats
        self.learn_bethe_residual_weight = learn_bethe_residual_weight
        self.use_old_bethe = use_old_bethe
        self.APPLY_BP_POST_BPNN = APPLY_BP_POST_BPNN
        self.APPLY_BP_EVERY_ITER = APPLY_BP_EVERY_ITER
        if learn_bethe_residual_weight:
            self.alpha_betheMLP = torch.nn.Parameter(alpha2*torch.ones(1))
            assert(initialize_to_exact_bethe == False), "Set initialize_to_exact_bethe=False when learn_bethe_residual_weight=True"
            
        if USE_OLD_CODE:
            if share_weights:
                self.message_passing_layer = FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**max_factor_state_dimensions)
            else:
                self.message_passing_layers = nn.ModuleList([FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**max_factor_state_dimensions)\
                                               for i in range(msg_passing_iters)])
        else:
            if share_weights:
                # self.message_passing_layer = FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**max_factor_state_dimensions,
                #     var_cardinality=var_cardinality, belief_repeats=belief_repeats, lne_mlp=lne_mlp, use_MLP1=use_MLP1, use_MLP2=use_MLP2, 
                #     use_MLP3=use_MLP3, use_MLP4=use_MLP4, subtract_prv_messages=subtract_prv_messages, alpha_damping_FtoV=alpha_damping_FtoV, alpha_damping_VtoF=alpha_damping_VtoF)
                
                self.message_passing_layers = nn.ModuleList([\
                    FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**max_factor_state_dimensions,
                        var_cardinality=var_cardinality, belief_repeats=belief_repeats, lne_mlp=lne_mlp, use_MLP1=use_MLP1, use_MLP2=use_MLP2, 
                        use_MLP3=use_MLP3, use_MLP4=use_MLP4, subtract_prv_messages=subtract_prv_messages, alpha_damping_FtoV=alpha_damping_FtoV, alpha_damping_VtoF=alpha_damping_VtoF)
                                                             for i in range(BPNN_layers_per_shared_weight_layer)])
                

                self.fixed_BP_layer = FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**max_factor_state_dimensions,
                    var_cardinality=var_cardinality, belief_repeats=belief_repeats, lne_mlp=lne_mlp, use_MLP1=False, use_MLP2=False, 
                    use_MLP3=False, use_MLP4=False, subtract_prv_messages=True, alpha_damping_FtoV=alpha_damping_FtoV, alpha_damping_VtoF=alpha_damping_VtoF) 
            else:
                self.message_passing_layers = nn.ModuleList([\
                    FactorGraphMsgPassingLayer_NoDoubleCounting(learn_BP=True, factor_state_space=2**max_factor_state_dimensions,
                        var_cardinality=var_cardinality, belief_repeats=belief_repeats, lne_mlp=lne_mlp, use_MLP1=use_MLP1, use_MLP2=use_MLP2, 
                        use_MLP3=use_MLP3, use_MLP4=use_MLP4, subtract_prv_messages=subtract_prv_messages, alpha_damping_FtoV=alpha_damping_FtoV, alpha_damping_VtoF=alpha_damping_VtoF)\
                                                             for i in range(msg_passing_iters)])

        if bethe_MLP != 'none':
            var_cardinality = var_cardinality #2 for binary variables
            num_ones = belief_repeats*(2*(var_cardinality**max_factor_state_dimensions)+var_cardinality)
            mlp_size =  msg_passing_iters*num_ones
#             self.final_mlp = Seq(Linear(mlp_size, mlp_size), ReLU(), Linear(mlp_size, 1))
            self.linear1 = Linear(mlp_size, mlp_size)
            self.linear2 = Linear(mlp_size, 1)
            if initialize_to_exact_bethe:  
#                 print("self.linear1.weight:", self.linear1.weight)
#                 print("self.linear1.bias:", self.linear1.bias)
                
#                 self.linear1.weight *= .001
#                 print("self.linear1.weight:", self.linear1.weight)
#                 sleep(alsfdjlksadjflks)

                self.linear1.weight = torch.nn.Parameter(torch.eye(mlp_size))
                self.linear1.bias = torch.nn.Parameter(torch.zeros(self.linear1.bias.shape))
                weight_initialization = torch.zeros((1,mlp_size))
                weight_initialization[0,-num_ones:] = 1.0/belief_repeats
#             print("self.linear2.weight:", self.linear2.weight)
#             print("self.linear2.weight.shape:", self.linear2.weight.shape)
#             print("weight_initialization:", weight_initialization)
                self.linear2.weight = torch.nn.Parameter(weight_initialization)
                self.linear2.bias = torch.nn.Parameter(torch.zeros(self.linear2.bias.shape)) 
        
            
            if bethe_MLP == 'shifted':
                self.shifted_relu = shift_func(ReLU(), shift=-500)
                self.final_mlp = Seq(self.linear1, self.shifted_relu, self.linear2, self.shifted_relu)  
            elif bethe_MLP == 'standard':
                self.final_mlp = Seq(self.linear1, ReLU(), self.linear2, ReLU())  
            elif bethe_MLP == 'linear':
                self.final_mlp = Seq(self.linear1, self.linear2)  
            else:
                assert(False), "Error: invalid value given for bethe_MLP"



        
        
        
    def forward(self, factor_graph):
#         prv_varToFactor_messages, prv_factorToVar_messages, prv_factor_beliefs, prv_var_beliefs = factor_graph.get_initial_beliefs_and_messages(device=self.device)
        prv_varToFactor_messages = factor_graph.prv_varToFactor_messages
        prv_factorToVar_messages = factor_graph.prv_factorToVar_messages
        prv_factor_beliefs = factor_graph.prv_factor_beliefs
#         print("prv_factor_beliefs.shape:", prv_factor_beliefs.shape)
#         print("prv_factorToVar_messages.shape:", prv_factorToVar_messages.shape)
#         print("factor_graph.facToVar_edge_idx.shape:", factor_graph.facToVar_edge_idx.shape)

        
        pooled_states = []
        
        if self.share_weights:
            # for iter in range(self.msg_passing_iters):
            random_msg_passing_iters = np.random.randint(10, 30)
            # random_msg_passing_iters = 40
            for iter in range(random_msg_passing_iters):
                single_layer = False
                if single_layer:
                    varToFactor_messages, factorToVar_messages, var_beliefs, factor_beliefs =\
                        self.message_passing_layer(factor_graph, prv_varToFactor_messages=prv_varToFactor_messages,
                                            prv_factorToVar_messages=prv_factorToVar_messages, prv_factor_beliefs=prv_factor_beliefs)
                    
                else:
                    tmp_varToFactor_messages = prv_varToFactor_messages
                    tmp_factorToVar_messages = prv_factorToVar_messages
                    tmp_factor_beliefs = prv_factor_beliefs
                    for message_passing_layer in self.message_passing_layers:
                        tmp_varToFactor_messages, tmp_factorToVar_messages, tmp_var_beliefs, tmp_factor_beliefs =\
                            message_passing_layer(factor_graph, prv_varToFactor_messages=tmp_varToFactor_messages,
                                                prv_factorToVar_messages=tmp_factorToVar_messages, prv_factor_beliefs=tmp_factor_beliefs)

                    varToFactor_messages = tmp_varToFactor_messages
                    factorToVar_messages = tmp_factorToVar_messages
                    var_beliefs = tmp_var_beliefs
                    factor_beliefs = tmp_factor_beliefs

                check_convergence = False
                if check_convergence:
            
#                     print("prv_varToFactor_messages.shape:", prv_varToFactor_messages.shape)
#                     print("prv_factorToVar_messages.shape:", prv_factorToVar_messages.shape)
                    message_count = 25 #10 # 10
                    batch_size=50
#                     norm_per_isingmodel_vTOf = torch.norm((varToFactor_messages - prv_varToFactor_messages).view([50, 460*16*2]), dim=1)
#                     norm_per_isingmodel_fTOv = torch.norm((factorToVar_messages - prv_factorToVar_messages).view([50, 460*16*2]), dim=1)
                    norm_per_isingmodel_vTOf = torch.norm((varToFactor_messages - prv_varToFactor_messages).view([batch_size, message_count*self.belief_repeats*2]), dim=1)
                    norm_per_isingmodel_fTOv = torch.norm((factorToVar_messages - prv_factorToVar_messages).view([batch_size, message_count*self.belief_repeats*2]), dim=1)
#                     print("norm_per_isingmodel_vTOf:", norm_per_isingmodel_vTOf)
#                     print("norm_per_isingmodel_fTOv:", norm_per_isingmodel_fTOv) 
#                     print("varToFactor_messages:", varToFactor_messages)
#                     print("factorToVar_messages:", factorToVar_messages)
#                     print("varToFactor_messages - prv_varToFactor_messages:", varToFactor_messages - prv_varToFactor_messages)
#                     print("factorToVar_messages - prv_factorToVar_messages:", factorToVar_messages - prv_factorToVar_messages)

#                     print("torch.max(factorToVar_messages - prv_factorToVar_messages):", torch.max(factorToVar_messages - prv_factorToVar_messages))
#                     print("torch.max(varToFactor_messages - prv_varToFactor_messages):", torch.max(varToFactor_messages - prv_varToFactor_messages))

#                     print("sleeping for for debugging")
#                     print("BPNN iter:", iter)
#                     time.sleep(.05)                    
#                     print()
                    
                prv_prv_varToFactor_messages = prv_varToFactor_messages
                prv_prv_factorToVar_messages = prv_factorToVar_messages

                prv_varToFactor_messages = varToFactor_messages
                prv_factorToVar_messages = factorToVar_messages
                prv_var_beliefs = var_beliefs
                prv_factor_beliefs = factor_beliefs
                
                if self.APPLY_BP_EVERY_ITER:
                    random_BP_iters = np.random.randint(0, 4)
                    # print("applying BP :)!!!!!!!")
                    # random_BP_iters = 300
                    for BP_iter in range(random_BP_iters):
                        varToFactor_messages, factorToVar_messages, var_beliefs, factor_beliefs =\
                            self.fixed_BP_layer(factor_graph, prv_varToFactor_messages=prv_varToFactor_messages,
                                                prv_factorToVar_messages=prv_factorToVar_messages, prv_factor_beliefs=prv_factor_beliefs)
            
                        prv_varToFactor_messages = varToFactor_messages
                        prv_factorToVar_messages = factorToVar_messages
                        prv_var_beliefs = var_beliefs
                        prv_factor_beliefs = factor_beliefs                                   



                if self.bethe_MLP != 'none':
                    cur_pooled_states = self.compute_bethe_free_energy_pooledStates_MLP(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
                    pooled_states.append(cur_pooled_states)  


            if self.APPLY_BP_POST_BPNN:
                #apply BP for a random number of iterations
                #goal is to get consistency between variable and factor beleifs
                # random_fixed_BP_iters = np.random.randint(10, 30)
                random_msg_passing_iters = 5
                for iter in range(random_msg_passing_iters):
                    varToFactor_messages, factorToVar_messages, var_beliefs, factor_beliefs =\
                        self.fixed_BP_layer(factor_graph, prv_varToFactor_messages=prv_varToFactor_messages,
                                            prv_factorToVar_messages=prv_factorToVar_messages, prv_factor_beliefs=prv_factor_beliefs)
        
                    prv_prv_varToFactor_messages = prv_varToFactor_messages
                    prv_prv_factorToVar_messages = prv_factorToVar_messages

                    prv_varToFactor_messages = varToFactor_messages
                    prv_factorToVar_messages = factorToVar_messages
                    prv_var_beliefs = var_beliefs
                    prv_factor_beliefs = factor_beliefs
                        
                    print('iter:', iter)
                    print("from nn_models torch.max(prv_prv_factorToVar_messages - prv_factorToVar_messages):", torch.max(prv_prv_factorToVar_messages - prv_factorToVar_messages))
                    print("from nn_models torch.max(prv_prv_varToFactor_messages - prv_varToFactor_messages):", torch.max(prv_prv_varToFactor_messages - prv_varToFactor_messages))
                


        else:
            for message_passing_layer in self.message_passing_layers:
#                 print("prv_varToFactor_messages:", prv_varToFactor_messages)
#                 print("prv_factorToVar_messages:", prv_factorToVar_messages)
#                 print("prv_factor_beliefs:", prv_factor_beliefs)
#                 print("prv_varToFactor_messages.shape:", prv_varToFactor_messages.shape)
#                 print("prv_factorToVar_messages.shape:", prv_factorToVar_messages.shape)
#                 print("prv_factor_beliefs.shape:", prv_factor_beliefs.shape) 
#                 prv_factor_beliefs[torch.where(prv_factor_beliefs==-np.inf)] = 0

                prv_varToFactor_messages, prv_factorToVar_messages, prv_var_beliefs, prv_factor_beliefs =\
                    message_passing_layer(factor_graph, prv_varToFactor_messages=prv_varToFactor_messages,
                                          prv_factorToVar_messages=prv_factorToVar_messages, prv_factor_beliefs=prv_factor_beliefs)
    #                 message_passing_layer(factor_graph, prv_varToFactor_messages=factor_graph.prv_varToFactor_messages,
    #                                       prv_factorToVar_messages=factor_graph.prv_factorToVar_messages, prv_factor_beliefs=factor_graph.prv_factor_beliefs) 
                if self.bethe_MLP != 'none':
                    cur_pooled_states = self.compute_bethe_free_energy_pooledStates_MLP(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
#                     print("cur_pooled_states:", cur_pooled_states)
#                     print(check_pool)
#                     print("cur_pooled_states.shape:", cur_pooled_states.shape)
                    pooled_states.append(cur_pooled_states)
                        
        if self.bethe_MLP != 'none':
#             print("torch.min(pooled_states):", torch.min(torch.cat(pooled_states, dim=1)))
#             print("torch.max(pooled_states):", torch.max(torch.cat(pooled_states, dim=1)))
#             print("torch.mean(pooled_states):", torch.mean(torch.cat(pooled_states, dim=1)))            
            learned_estimated_ln_partition_function = self.final_mlp(torch.cat(pooled_states, dim=1))
                 
            if self.learn_bethe_residual_weight:
                final_pooled_states = self.compute_bethe_free_energy_pooledStates_MLP(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
                bethe_estimated_ln_partition_function = torch.sum(final_pooled_states, dim=1)
                final_estimate = (1-self.alpha_betheMLP)*learned_estimated_ln_partition_function.squeeze() +\
                                 self.alpha_betheMLP*bethe_estimated_ln_partition_function.squeeze()
                
#                 print("bethe_estimated_ln_partition_function.shape:", bethe_estimated_ln_partition_function.shape)
#                 print("learned_estimated_ln_partition_function.shape:", learned_estimated_ln_partition_function.shape)
#                 print("final_estimate.shape:", final_estimate.shape)

                return final_estimate
            else:
                return learned_estimated_ln_partition_function
        
        else:
            if self.use_old_bethe:
#                 print("prv_factor_beliefs.shape:", prv_factor_beliefs.shape)
#                 print("prv_var_beliefs.shape:", prv_var_beliefs.shape)
#                 print("factor_graph.factor_potentials.shape:", factor_graph.factor_potentials.shape)                
#                 sleep(asdlfkjsdlkf)
                #broken for batch_size > 1
                bethe_free_energy = compute_bethe_free_energy(factor_beliefs=prv_factor_beliefs.squeeze(), var_beliefs=prv_var_beliefs.squeeze(), factor_graph=factor_graph)
                estimated_ln_partition_function = -bethe_free_energy
#                 print("prv_factor_beliefs.squeeze():", prv_factor_beliefs.squeeze())
#                 print("prv_var_beliefs.squeeze():", prv_var_beliefs.squeeze())
                
#                 print("factor_graph.factor_potentials.squeeze():", factor_graph.factor_potentials.squeeze())
#                 print("factor_graph.numVars:", factor_graph.numVars)
#                 print("factor_graph.var_degrees:", factor_graph.var_degrees)
                
#                 print("estimated_ln_partition_function:", estimated_ln_partition_function)
#                 sleep(nn_models_debug_alsfj)
                
                debug=False
                if debug:
                    final_pooled_states = self.compute_bethe_free_energy_pooledStates_MLP(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
                    check_estimated_ln_partition_function = torch.sum(final_pooled_states)
    #                 print("check_estimated_ln_partition_function:", check_estimated_ln_partition_function)
    #                 print("estimated_ln_partition_function:", estimated_ln_partition_function)
    #                 sleep(debug_bethe)
                    assert(torch.allclose(check_estimated_ln_partition_function, estimated_ln_partition_function)), (check_estimated_ln_partition_function, estimated_ln_partition_function)
                return estimated_ln_partition_function
  
            #corrected for batch_size > 1
            final_pooled_states = self.compute_bethe_free_energy_pooledStates_MLP(factor_beliefs=prv_factor_beliefs, var_beliefs=prv_var_beliefs, factor_graph=factor_graph)
            estimated_ln_partition_function = torch.sum(final_pooled_states, dim=1)/self.belief_repeats
            return estimated_ln_partition_function, prv_prv_varToFactor_messages, prv_prv_factorToVar_messages, prv_varToFactor_messages, prv_factorToVar_messages
            # return estimated_ln_partition_function            



    def compute_bethe_average_energy_MLP(self, factor_beliefs, factor_potentials, batch_factors, debug=False):
        '''
        Equation (37) in:
        https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf        
        '''
        assert(factor_potentials.shape == factor_beliefs.shape), (factor_potentials.shape, factor_beliefs.shape)
        if debug:
            print()
            print('!!!!!!!')
            print("debugging compute_bethe_average_energy")
            print("torch.exp(factor_beliefs):", torch.exp(factor_beliefs))
            print("neg_inf_to_zero(factor_potentials):", neg_inf_to_zero(factor_potentials))
        
        pooled_fac_beleifPotentials = global_add_pool(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials), batch_factors)
        #keep 1st dimension for # of factors, but flatten remaining dimensions for belief_repeats and each factor        
        pooled_fac_beleifPotentials = pooled_fac_beleifPotentials.view(pooled_fac_beleifPotentials.shape[0], -1)
        if debug:
            factor_beliefs_shape = factor_beliefs.shape
            pooled_fac_beleifPotentials_orig = torch.sum((torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials)).view(factor_beliefs_shape[0], -1), dim=0)
            print("original pooled_fac_beleifPotentials_orig:", pooled_fac_beleifPotentials_orig)
            print("pooled_fac_beleifPotentials:", pooled_fac_beleifPotentials)
            print("factor_beliefs.shape:", factor_beliefs.shape)
            print("pooled_fac_beleifPotentials_orig.shape:", pooled_fac_beleifPotentials_orig.shape)
            print("pooled_fac_beleifPotentials.shape:", pooled_fac_beleifPotentials.shape)
            print("(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials)).view(factor_beliefs_shape[0], -1).shape:", (torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials)).view(factor_beliefs_shape[0], -1).shape)
        return pooled_fac_beleifPotentials #negate and sum to get average bethe energy


    def compute_bethe_entropy_MLP(self, factor_beliefs, var_beliefs, numVars, var_degrees, batch_factors, batch_vars, debug=False):
        '''
        Equation (38) in:
        https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf        
        '''

        pooled_fac_beliefs = -global_add_pool(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_beliefs), batch_factors)
        #keep 1st dimension for # of factors, but flatten remaining dimensions for belief_repeats and each factor        
        pooled_fac_beliefs = pooled_fac_beliefs.view(pooled_fac_beliefs.shape[0], -1)        
        if debug:
            factor_beliefs_shape = factor_beliefs.shape
            pooled_fac_beliefs_orig = -torch.sum((torch.exp(factor_beliefs)*neg_inf_to_zero(factor_beliefs)).view(factor_beliefs_shape[0], -1), dim=0)
            print("pooled_fac_beliefs_orig:", pooled_fac_beliefs_orig)
            print("pooled_fac_beliefs:", pooled_fac_beliefs)
        
        

        var_beliefs_shape = var_beliefs.shape
        assert(var_beliefs_shape[0] == var_degrees.shape[0])
        pooled_var_beliefs = global_add_pool(torch.exp(var_beliefs)*neg_inf_to_zero(var_beliefs)*(var_degrees.float() - 1).view(var_degrees.shape[0], 1, 1), batch_vars)
        #keep 1st dimension for # of factors, but flatten remaining dimensions for belief_repeats and variable states        
        pooled_var_beliefs = pooled_var_beliefs.view(pooled_var_beliefs.shape[0], -1)
    
        
        if debug:
            pooled_var_beliefs_orig = torch.sum(torch.exp(var_beliefs)*neg_inf_to_zero(var_beliefs)*(var_degrees.float() - 1).view(var_beliefs_shape[0], -1), dim=0)            
            print("pooled_var_beliefs_orig:", pooled_var_beliefs_orig)
            print("pooled_var_beliefs:", pooled_var_beliefs)        
    #         sleep(SHAPECHECK)

        return pooled_fac_beliefs, pooled_var_beliefs

    def compute_bethe_free_energy_pooledStates_MLP(self, factor_beliefs, var_beliefs, factor_graph):
        '''
        Compute the Bethe approximation of the free energy.
        - free energy = -ln(Z)
          where Z is the partition function
        - (Bethe approximation of the free energy) = (Bethe average energy) - (Bethe entropy)

        For more details, see page 11 of:
        https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf
        '''
        
#         print("var_beliefs.shape:", var_beliefs.shape)
#         print("factor_beliefs.shape:", factor_beliefs.shape)
        
        
                
        #switch to Temp=False/remove me after generalized to handle repeated beliefs!   
        TEMP=False
        if TEMP:
            var_beliefs = torch.mean(var_beliefs, dim=1) #remove me after generalized to handle repeated beliefs!
            factor_beliefs = torch.mean(factor_beliefs, dim=1) #remove me after generalized to handle repeated beliefs!
            
            normalized_var_beliefs = var_beliefs - logsumexp_multipleDim(var_beliefs, dim_to_keep=[0])#normalize variable beliefs
#             print("normalized_var_beliefs.shape:", normalized_var_beliefs.shape)
            check_normalization = torch.sum(torch.exp(normalized_var_beliefs), dim=[i for i in range(1,len(var_beliefs.shape))])
#             print("check_normalization.shape:", check_normalization.shape)
#             print("check_normalization:", check_normalization)

            assert(torch.max(torch.abs(check_normalization-1)) < .001), (torch.sum(torch.abs(check_normalization-1)), torch.max(torch.abs(check_normalization-1)), check_normalization) 

            normalized_factor_beliefs = factor_beliefs - logsumexp_multipleDim(factor_beliefs, dim_to_keep=[0])#normalize factor beliefs
            check_normalization = torch.sum(torch.exp(normalized_factor_beliefs), dim=[i for i in range(1,len(factor_beliefs.shape))])
#             print("normalized_factor_beliefs.shape:", normalized_factor_beliefs.shape)        
#             print("check_normalization.shape:", check_normalization.shape)
#             print("check_normalization:", check_normalization)
#             print()        
            assert(torch.max(torch.abs(check_normalization-1)) < .00001), (torch.sum(torch.abs(check_normalization-1)), torch.max(torch.abs(check_normalization-1)), check_normalization)             
        else:
            normalized_var_beliefs = var_beliefs - logsumexp_multipleDim(var_beliefs, dim_to_keep=[0,1])#normalize variable beliefs
#             print("normalized_var_beliefs.shape:", normalized_var_beliefs.shape)
            check_normalization = torch.sum(torch.exp(normalized_var_beliefs), dim=[i for i in range(2,len(var_beliefs.shape))])
#             print("check_normalization.shape:", check_normalization.shape)
#             print("check_normalization:", check_normalization)
#             print("var_beliefs[torch.where(torch.abs(check_normalization-1)) >= .00001)]:", var_beliefs[torch.where(torch.abs(check_normalization-1) >= .00001)])
            assert(torch.max(torch.abs(check_normalization-1)) < .01), (torch.sum(torch.abs(check_normalization-1)), torch.max(torch.abs(check_normalization-1)), check_normalization, var_beliefs) 

#             print("factor_beliefs.shape:", factor_beliefs.shape)                    
            normalized_factor_beliefs = factor_beliefs - logsumexp_multipleDim(factor_beliefs, dim_to_keep=[0,1])#normalize factor beliefs
            check_normalization = torch.sum(torch.exp(normalized_factor_beliefs), dim=[i for i in range(2,len(factor_beliefs.shape))])

            CHECK_CONSISTENCY = False
            if CHECK_CONSISTENCY:
                print("normalized_factor_beliefs:", normalized_factor_beliefs)
                print("unary factor beliefs:", normalized_factor_beliefs[:10,:,:,0])
                print("normalized_var_beliefs:", normalized_var_beliefs[:10,::])                
                print("normalized_factor_beliefs.shape:", normalized_factor_beliefs.shape)
                print("normalized_var_beliefs.shape:", normalized_var_beliefs.shape)
                # sleep(temp)
                # assert(normalized_var_beliefs)

#             print("normalized_factor_beliefs.shape:", normalized_factor_beliefs.shape)        
#             print("check_normalization.shape:", check_normalization.shape)
#             print("check_normalization:", check_normalization)
#             print()        
            assert(torch.max(torch.abs(check_normalization-1)) < .01), (torch.sum(torch.abs(check_normalization-1)), torch.max(torch.abs(check_normalization-1)), check_normalization) 
            
#         print("normalized_var_beliefs.shape:", normalized_var_beliefs.shape)
#         print("factor_beliefs.shape:", factor_beliefs.shape)        
#         sleep(salfjlsdkj)
            
        # print("self.compute_bethe_average_energy():", self.compute_bethe_average_energy())
        # print("self.compute_bethe_entropy():", self.compute_bethe_entropy())
        if torch.isnan(normalized_factor_beliefs).any():
            print("values, some should be nan:")
            for val in normalized_factor_beliefs.flatten():
                print(val)
        assert(not torch.isnan(normalized_factor_beliefs).any()), (normalized_factor_beliefs, torch.where(normalized_factor_beliefs == torch.tensor(float('nan'))), torch.where(normalized_var_beliefs == torch.tensor(float('nan'))))
        assert(not torch.isnan(normalized_var_beliefs).any()), normalized_var_beliefs
        
        if TEMP:
            #quick option for not dealing with repeated beliefs
            factor_potentials_quick = factor_graph.factor_potentials[:, 0, ::]
            factor_potentials_check = torch.mean(factor_graph.factor_potentials, dim=1, keepdim=False)
            assert(torch.max(torch.abs(factor_potentials_quick - factor_potentials_check)) < .00001), (factor_potentials_quick, factor_potentials_check, torch.max(torch.abs(factor_potentials_quick - factor_potentials_check)))
        else:
            factor_potentials_quick = factor_graph.factor_potentials
        
        pooled_fac_beleifPotentials = self.compute_bethe_average_energy_MLP(factor_beliefs=normalized_factor_beliefs,\
                                      factor_potentials=factor_potentials_quick, batch_factors=factor_graph.batch_factors)
        pooled_fac_beliefs, pooled_var_beliefs = self.compute_bethe_entropy_MLP(factor_beliefs=normalized_factor_beliefs, var_beliefs=normalized_var_beliefs, numVars=torch.sum(factor_graph.numVars), var_degrees=factor_graph.var_degrees, batch_factors=factor_graph.batch_factors, batch_vars=factor_graph.batch_vars)
        
        
#         print("pooled_fac_beleifPotentials.shape:", pooled_fac_beleifPotentials.shape)
#         print("pooled_fac_beliefs.shape:", pooled_fac_beliefs.shape)
#         print("pooled_var_beliefs.shape:", pooled_var_beliefs.shape)        
#         sleep(laskdjfowin)
        if len(pooled_fac_beleifPotentials.shape) > 1:
            cat_dim = 1
        else:
            cat_dim = 0
                 
        return torch.cat([pooled_fac_beleifPotentials, pooled_fac_beliefs, pooled_var_beliefs], dim=cat_dim)



def compute_bethe_average_energy(factor_beliefs, factor_potentials, debug=False):
    '''
    Equation (37) in:
    https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf        
    '''
    assert(factor_potentials.shape == factor_beliefs.shape)
    if debug:
        print()
        print('!!!!!!!')
        print("debugging compute_bethe_average_energy")
        print("torch.exp(factor_beliefs):", torch.exp(factor_beliefs))
        print("neg_inf_to_zero(factor_potentials):", neg_inf_to_zero(factor_potentials))
    bethe_average_energy = -torch.sum(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_potentials)) #elementwise multiplication, then sum
#     print("bethe_average_energy:", bethe_average_energy)
    return bethe_average_energy

def compute_bethe_entropy(factor_beliefs, var_beliefs, numVars, var_degrees):
    '''
    Equation (38) in:
    https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf        
    '''
    bethe_entropy = -torch.sum(torch.exp(factor_beliefs)*neg_inf_to_zero(factor_beliefs)) #elementwise multiplication, then sum

#     print("numVars:", numVars)
    assert(var_beliefs.shape == torch.Size([numVars, 2])), (var_beliefs.shape, [numVars, 2])
    # sum_{x_i} b_i(x_i)*ln(b_i(x_i))
    inner_sum = torch.einsum('ij,ij->i', [torch.exp(var_beliefs), neg_inf_to_zero(var_beliefs)])
    # sum_{i=1}^N (d_i - 1)*inner_sum
    outer_sum = torch.sum((var_degrees.float() - 1) * inner_sum)
    # outer_sum = torch.einsum('i,i->', [var_degrees - 1, inner_sum])

    bethe_entropy += outer_sum
#     print("bethe_entropy:", bethe_entropy)
    return bethe_entropy

def compute_bethe_free_energy(factor_beliefs, var_beliefs, factor_graph):
    '''
    BROKEN FOR BATCH SIZE > 1
    Compute the Bethe approximation of the free energy.
    - free energy = -ln(Z)
      where Z is the partition function
    - (Bethe approximation of the free energy) = (Bethe average energy) - (Bethe entropy)

    For more details, see page 11 of:
    https://www.cs.princeton.edu/courses/archive/spring06/cos598C/papers/YedidaFreemanWeiss2004.pdf
    '''
    # print("self.compute_bethe_average_energy():", self.compute_bethe_average_energy())
    # print("self.compute_bethe_entropy():", self.compute_bethe_entropy())
    if torch.isnan(factor_beliefs).any():
        print("values, some should be nan:")
        for val in factor_beliefs.flatten():
            print(val)
    assert(not torch.isnan(factor_beliefs).any()), (factor_beliefs, torch.where(factor_beliefs == torch.tensor(float('nan'))), torch.where(var_beliefs == torch.tensor(float('nan'))))
    assert(not torch.isnan(var_beliefs).any()), var_beliefs
    return (compute_bethe_average_energy(factor_beliefs=factor_beliefs, factor_potentials=factor_graph.factor_potentials.squeeze())\
            - compute_bethe_entropy(factor_beliefs=factor_beliefs, var_beliefs=var_beliefs, numVars=torch.sum(factor_graph.numVars), var_degrees=factor_graph.var_degrees))

class GIN_Network_withEdgeFeatures(nn.Module):
    def __init__(self, input_state_size=1, edge_attr_size=1, hidden_size=4, msg_passing_iters=5, feat_all_layers=True, edgedevice=None):
        '''
        Inputs:
        - msg_passing_iters (int): the number of iterations of message passing to run (we have this many
            message passing layers with their own learnable parameters)
        - feat_all_layers (bool): if True, use concatenation of sum of all node features after every layer as input to final MLP
        '''
        super().__init__()
        layers = [GINConv_withEdgeFeatures(nn1=Seq(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)),
                                          nn2=Seq(Linear(input_state_size + edge_attr_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)))] + \
                 [GINConv_withEdgeFeatures(nn1=Seq(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)),
                                          nn2=Seq(Linear(hidden_size + edge_attr_size, hidden_size), ReLU(), Linear(hidden_size, hidden_size)))\
                                          for i in range(msg_passing_iters - 1)]
        self.message_passing_layers = nn.ModuleList(layers)
        self.feat_all_layers = feat_all_layers
        if self.feat_all_layers:
            self.final_mlp = Seq(Linear(msg_passing_iters*hidden_size, msg_passing_iters*hidden_size), ReLU(), Linear(msg_passing_iters*hidden_size, 1))            
        else:
            self.final_mlp = Seq(Linear(hidden_size, hidden_size), ReLU(), Linear(hidden_size, 1))
        
    def forward(self, x, edge_index, edge_attr, batch):
        if self.feat_all_layers:
            summed_node_features_all_layers = []
            for message_passing_layer in self.message_passing_layers:
                x = message_passing_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)
                summed_node_features_all_layers.append(global_add_pool(x, batch))

#             print("torch.cat(summed_node_features_all_layers, dim=0).shape:", torch.cat(summed_node_features_all_layers, dim=1).shape)
#             print("global_add_pool(x, batch).shape:", global_add_pool(x, batch).shape)
            return self.final_mlp(torch.cat(summed_node_features_all_layers, dim=1))
        else:
            for message_passing_layer in self.message_passing_layers:
                x = message_passing_layer(x=x, edge_index=edge_index, edge_attr=edge_attr)

            return self.final_mlp(global_add_pool(x, batch))          
    
class GINConv_withEdgeFeatures(MessagePassing):
    r"""Modification of the graph isomorphism operator from the `"How Powerful are
    Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper, which uses
    edge features.

    .. math::
        \mathbf{x}^{\prime}_i = \text{MLP}_1 \left( (1 + \epsilon) \cdot
        \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \text{MLP}_2 \left( \mathbf{x}_j, \mathbf{e}_{i,j} \right) \right),

    Args:
        nn1 (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
            shape :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        nn2 (torch.nn.Module): A neural network mapping shape [-1, in_channels + edge_features]
            to [-1, in_channels]
        eps (float, optional): (Initial) :math:`\epsilon` value.
            (default: :obj:`0`)
        train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
            will be a trainable parameter. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, nn1, nn2, eps=0, train_eps=False, **kwargs):
        super(GINConv_withEdgeFeatures, self).__init__(aggr='add', **kwargs)
        self.nn1 = nn1
        self.nn2 = nn2
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn1)
        self.eps.data.fill_(self.initial_eps)


    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, _ = remove_self_loops(edge_index)
        out = self.nn1((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out


    def message(self, x_j, edge_attr):
        # x_i has shape [E, in_channels]
        # edge_attr has shape [E, edge_features]
        tmp = torch.cat([x_j, edge_attr], dim=1)  # tmp has shape [E, in_channels + edge_features]
        return self.nn2(tmp)        
