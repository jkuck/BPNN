import mrftools
import numpy as np
from .parameters import MRFTOOLS_LBP_ITERS


def build_single_node_factor(mn, fixed_variables, var_idx, f):
    '''
    copied from https://github.com/jkuck/mrf_nesting_ub/blob/master/sparse_matrix_ising_model.py
    add single node factor to the markov net

    Inputs:
    - mn: (mrftools.MarkovNet)
    - fixed_variables: (dictionary)
        key: (int) 0 to N-1 variable index
        value: (int) -1 or 1, the value the variable is fixed to                 
    - var_idx: (int) variable index, 0 to N-1, for this factor's node
    - f: (float) local field at this node

    Outputs:
    - None
    '''
    if var_idx in fixed_variables:
        # factor = np.array(np.exp(fixed_variables[var_idx]*f))
        factor = np.array([fixed_variables[var_idx]*f])
        mn.set_unary_factor(var_idx, factor)
    else:
        # factor = np.array([np.exp(-f), np.exp(f)])
        factor = np.array([-f, f])
        mn.set_unary_factor(var_idx, factor)


def build_pairwise_factor(mn, fixed_variables, var_idx1, var_idx2, c):
    '''
    copied from https://github.com/jkuck/mrf_nesting_ub/blob/master/sparse_matrix_ising_model.py
    add a pairwise factor to the markov net

    Inputs:
    - mn: (mrftools.MarkovNet)
    - fixed_variables: (dictionary)
        key: (int) 0 to N-1 variable index
        value: (int) -1 or 1, the value the variable is fixed to                 
    - var_idx1: (int) variable index, 0 to N-1, for the first node in this factor
    - var_idx2: (int) variable index, 0 to N-1, for the second node in this factor
    - c: (float) coupling strength for this factor

    Outputs:
    - None
    '''

    #this 'pairwise' factor is over two fixed variables and has only 1 state
    if (var_idx1 in fixed_variables) and (var_idx2 in fixed_variables):
        # factor = np.array(np.exp(fixed_variables[var_idx1]*fixed_variables[var_idx2]*c))
        factor = np.array([[fixed_variables[var_idx1]*fixed_variables[var_idx2]*c]])
        mn.set_edge_factor((var_idx1, var_idx2), factor)
    #this 'pairwise' factor is over one fixed variable and one binary variable and has 2 states
    elif (var_idx1 in fixed_variables) and (var_idx2 not in fixed_variables):
        # factor = np.array([np.exp(-fixed_variables[var_idx1]*c),
        #                    np.exp(fixed_variables[var_idx1]*c)])
        factor = np.array([[-fixed_variables[var_idx1]*c,
                           fixed_variables[var_idx1]*c]])
        mn.set_edge_factor((var_idx1, var_idx2), factor)
    #this 'pairwise' factor is over one fixed variable and one binary variable and has 2 states
    elif (var_idx1 not in fixed_variables) and (var_idx2 in fixed_variables):
        # factor = np.array([np.exp(-fixed_variables[var_idx2]*c),
        #                    np.exp(fixed_variables[var_idx2]*c)])
        factor = np.array([[-fixed_variables[var_idx2]*c],
                           [fixed_variables[var_idx2]*c]])
        mn.set_edge_factor((var_idx1, var_idx2), factor)

    #this pairwise factor is over two binary variables and has 4 states
    elif (var_idx1 not in fixed_variables) and (var_idx2 not in fixed_variables):
        # factor = np.array([[np.exp(c), np.exp(-c)],
        #                    [np.exp(-c), np.exp(c)]])
        factor = np.array([[c, -c],
                           [-c, c]])
        mn.set_edge_factor((var_idx1, var_idx2), factor)
    else:
        assert(False), "This shouldn't happen!!?"

def build_MarkovNet_from_SpinGlassModel(sg_model, fixed_variables={}):
    '''
    copied from https://github.com/jkuck/mrf_nesting_ub/blob/master/sparse_matrix_ising_model.py
    Build an mrftools representation of a SpinGlassModel

    Inputs:
    - sg_model: (SpinGlassModel)
    - fixed_variables: (dictionary)
        key: (int) 0 to N-1 variable index
        value: (int) -1 or 1, the value the variable is fixed to

    Outputs:
    - sg_FactorGraph (mrftools.MarkovNet):
    '''
    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])

    mn = mrftools.MarkovNet()
    # Define factors for each single variable factor
    for var_idx in range(N**2):
        r = var_idx//N
        c = var_idx%N
        build_single_node_factor(mn, fixed_variables, var_idx, f=sg_model.lcl_fld_params[r,c])

    # Define pairwise factors
    for var_idx in range(N**2):
        r = var_idx//N
        c = var_idx%N
        if r < N-1:
            build_pairwise_factor(mn, fixed_variables, var_idx1=var_idx, var_idx2=var_idx+N, c=sg_model.cpl_params_v[r,c])
        if c < N-1:
            build_pairwise_factor(mn, fixed_variables, var_idx1=var_idx, var_idx2=var_idx+1, c=sg_model.cpl_params_h[r,c])

    mn.create_matrices()
    return mn

def brute_force(sg_model):
    '''
    Brute force calculate the partition function of a spin glass model using mrftools
    Inputs:
    - sg_model (SpinGlassModel)

    Outputs:
    - exact_z: the exact partition function
    '''
    sg_as_MarkoveNet = build_MarkovNet_from_SpinGlassModel(sg_model)
    bf = mrftools.BruteForce(sg_as_MarkoveNet)
    exact_z = bf.compute_z()
    return exact_z

def run_LBP(sg_model, max_iter=MRFTOOLS_LBP_ITERS):
    sg_as_MarkoveNet = build_MarkovNet_from_SpinGlassModel(sg_model)


    bp = mrftools.BeliefPropagator(sg_as_MarkoveNet)
    bp.set_max_iter(max_iter)
    # bp.infer(display='full')

    bp.compute_pairwise_beliefs()

    # print("Bethe energy functional: %f" % bp.compute_energy_functional())

    lbp_estimate = bp.compute_energy_functional() 
    return lbp_estimate