import sys
import numpy as np

# Atlas
# LIBDAI_SWIG_DIRECTORY = '/atlas/u/jkuck/libdai2/swig/'
LIBDAI_SWIG_DIRECTORY = '/atlas/u/htang/libdai/swig/'
# LIBDAI_SWIG_DIRECTORY = '/workspace/libdai/swig/'
# LIBDAI_SWIG_DIRECTORY = '/atlas/u/jkuck/libdai_reproduce2/swig/'



# Local
# LIBDAI_SWIG_DIRECTORY = '/Users/jkuck/research/libdai/swig'
sys.path.insert(0, LIBDAI_SWIG_DIRECTORY)
import dai
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from parameters import LIBDAI_LBP_ITERS
from parameters import LIBDAI_MEAN_FIELD_ITERS
# from ..parameters import LIBDAI_LBP_ITERS

def build_single_node_factor(variables, fixed_variables, var_idx, f):
    '''
    copied from https://github.com/jkuck/mrf_nesting_ub/blob/master/Factor_Graphs/libdai_ising_model.py

    Inputs:
    - variables: (list of dai.Var) available variables.
    - fixed_variables: (dictionary)
        key: (int) 0 to N-1 variable index
        value: (int) -1 or 1, the value the variable is fixed to
    - var_idx: (int) variable index, 0 to N-1, for this factor's node
    - f: (float) local field at this node

    Outputs:
    - factor: (dai.Factor)
    '''

    clause_variables = dai.VarSet(variables[var_idx])
    factor = dai.Factor(clause_variables)
    if var_idx in fixed_variables:
        factor[0] = np.exp(fixed_variables[var_idx]*f) #the single fixed factor value
    else:
        factor[0] = np.exp(-f) #corresponding to the node taking value -1
        factor[1] = np.exp(f) #corresponding to the node taking value 1
    return factor

def build_pairwise_factor(variables, fixed_variables, var_idx1, var_idx2, c):
    '''
    copied from https://github.com/jkuck/mrf_nesting_ub/blob/master/Factor_Graphs/libdai_ising_model.py

    Inputs:
    - variables: (list of dai.Var) available variables.
    - fixed_variables: (dictionary)
        key: (int) 0 to N-1 variable index
        value: (int) -1 or 1, the value the variable is fixed to
    - var_idx1: (int) variable index, 0 to N-1, for the first node in this factor
    - var_idx2: (int) variable index, 0 to N-1, for the second node in this factor
    - c: (float) coupling strength for this factor

    Outputs:
    - factor: (dai.Factor)
    '''

    clause_variables = dai.VarSet(variables[var_idx1], variables[var_idx2])
    factor = dai.Factor(clause_variables)
    #this 'pairwise' factor is over two fixed variables and has only 1 state
    if (var_idx1 in fixed_variables) and (var_idx2 in fixed_variables):
        factor[0] = np.exp(fixed_variables[var_idx1]*fixed_variables[var_idx2]*c)
    #this 'pairwise' factor is over one fixed variable and one binary variable and has 2 states
    elif (var_idx1 in fixed_variables) and (var_idx2 not in fixed_variables):
        factor[0] = np.exp(-fixed_variables[var_idx1]*c) # V2 = -1
        factor[1] = np.exp(fixed_variables[var_idx1]*c) # V2 = -1
    #this 'pairwise' factor is over one fixed variable and one binary variable and has 2 states
    elif (var_idx1 not in fixed_variables) and (var_idx2 in fixed_variables):
        factor[0] = np.exp(-fixed_variables[var_idx2]*c) # V1 = -1
        factor[1] = np.exp(fixed_variables[var_idx2]*c) # V1 = -1
    #this pairwise factor is over two binary variables and has 4 states
    elif (var_idx1 not in fixed_variables) and (var_idx2 not in fixed_variables):
        factor[0] = np.exp(c)  # V1 = -1, V2 = -1
        factor[1] = np.exp(-c) # V1 = -1, V2 =  1
        factor[2] = np.exp(-c) # V1 =  1, V2 = -1
        factor[3] = np.exp(c)  # V1 =  1, V2 =  1
    else:
        assert(False), "This shouldn't happen!!?"
    return factor




def build_higher_order_factor(variables, fixed_variables, var_idx1, var_idx2, c):
    '''
    NOT IMPLEMENTED
    Inputs:
    - variables: (list of dai.Var) available variables.
    - fixed_variables: (dictionary)
        key: (int) 0 to N-1 variable index
        value: (int) -1 or 1, the value the variable is fixed to
    - var_idx1: (int) variable index, 0 to N-1, for the first node in this factor
    - var_idx2: (int) variable index, 0 to N-1, for the second node in this factor
    - c: (float) coupling strength for this factor

    Outputs:
    - factor: (dai.Factor)
    '''

    pass

def build_libdaiFactorGraph_from_SpinGlassModel(sg_model, fixed_variables={}):
    '''
    copied from https://github.com/jkuck/mrf_nesting_ub/blob/master/Factor_Graphs/libdai_ising_model.py

    Inputs:
    - sg_model: (SG_model)
    - fixed_variables: (dictionary)
        key: (int) 0 to N-1 variable index
        value: (int) -1 or 1, the value the variable is fixed to

    Outputs:
    - sg_FactorGraph: (dai.FactorGraph)
    '''
    N = sg_model.lcl_fld_params.shape[0]
    assert(N == sg_model.lcl_fld_params.shape[1])
    # accumulator_var_idx = None
    # if len(fixed_variables) > 0:
    #     assert(len(fixed_variables) < N**2)
    #     for var_idx in range(N**2):
    #         if var_idx not in fixed_variables:
    #             accumulator_var_idx = var_idx #this variable gets all the fixed factors multiplied into its single node factor
    #             break
    #     assert(accumulator_var_idx is not None)


    # Define binary variables in the factor graph
    variables = []
    for var_idx in range(N**2):
        if var_idx in fixed_variables:
            variables.append(dai.Var(var_idx, 1)) #variable can take 1 values
        if var_idx not in fixed_variables:
            variables.append(dai.Var(var_idx, 2)) #variable can take 2 values

    factors = []
    # Define factors for each single variable factor
    for var_idx in range(N**2):
        r = var_idx//N
        c = var_idx%N
        factors.append(build_single_node_factor(variables, fixed_variables, var_idx, f=sg_model.lcl_fld_params[r,c]))

    # Define pairwise factors
    for var_idx in range(N**2):
        r = var_idx//N
        c = var_idx%N
        if r < N-1:
            factors.append(build_pairwise_factor(variables, fixed_variables, var_idx1=var_idx, var_idx2=var_idx+N, c=sg_model.cpl_params_v[r,c]))
        if c < N-1:
            factors.append(build_pairwise_factor(variables, fixed_variables, var_idx1=var_idx, var_idx2=var_idx+1, c=sg_model.cpl_params_h[r,c]))

    #Define higher order factors
    if sg_model.contains_higher_order_potentials:
        # Add higher order factors
        for potential_idx in range(sg_model.ho_potential_count):
            factor_potentials_list.append(sg_model.higher_order_potentials[potential_idx])
            masks_list.append(torch.zeros_like(sg_model.higher_order_potentials[potential_idx]))

    assert(len(factors) == N**2 + 2*N*(N-1))

    # Build factor graph
    sg_Factors = dai.VecFactor()
    for factor in factors:
        sg_Factors.append(factor)
    sg_FactorGraph = dai.FactorGraph(sg_Factors)

    return sg_FactorGraph

def junction_tree(sg_model, verbose=False, map_flag=False):
    '''
    Calculate the exact partition function of a spin glass model using the junction tree algorithm
    Inputs:
    - sg_model (SpinGlassModel)

    Outputs:
    - ln_Z: natural logarithm of the exact partition function
    '''
    sg_FactorGraph = build_libdaiFactorGraph_from_SpinGlassModel(sg_model, fixed_variables={})
    # sg_FactorGraph = build_graph_from_clique_ising_model(sg_model, fixed_variables={})

    # Write factorgraph to a file
    sg_FactorGraph.WriteToFile('sg_temp.fg')
    if verbose:
        print( 'spin glass factor graph written to sg_temp.fg')

    # Output some information about the factorgraph
    if verbose:
        print( sg_FactorGraph.nrVars(), 'variables')
        print( sg_FactorGraph.nrFactors(), 'factors')

    # Set some constants
    maxiter = 10000
    tol = 1e-9
    verb = 0
    # Store the constants in a PropertySet object
    opts = dai.PropertySet()
    opts["maxiter"] = str(maxiter)   # Maximum number of iterations
    opts["tol"] = str(tol)           # Tolerance for convergence
    opts["verbose"] = str(verb)      # Verbosity (amount of output generated)bpopts["updates"] = "SEQRND"

    ##################### Run Junction Tree Algorithm #####################
    # Construct a JTree (junction tree) object from the FactorGraph sg_FactorGraph
    # using the parameters specified by opts and an additional property
    # that specifies the type of updates the JTree algorithm should perform
    jtopts = opts
    jtopts["updates"] = "HUGIN"
    jt = dai.JTree( sg_FactorGraph, jtopts )
    # Initialize junction tree algorithm
    jt.init()
    # Run junction tree algorithm
    jt.run()

    # Construct another JTree (junction tree) object that is used to calculate
    # the joint configuration of variables that has maximum probability (MAP state)
    jtmapopts = opts
    jtmapopts["updates"] = "HUGIN"
    jtmapopts["inference"] = "MAXPROD"
    jtmap = dai.JTree( sg_FactorGraph, jtmapopts )
    # Initialize junction tree algorithm
    jtmap.init()
    # Run junction tree algorithm
    jtmap.run()
    # Calculate joint state of all variables that has maximum probability
    jtmapstate = jtmap.findMaximum()

    ln_Z = sg_FactorGraph.logScore(jtmapstate) if map_flag else jt.logZ()
    # Report log partition sum (normalizing constant) of sg_FactorGraph, calculated by the junction tree algorithm
    if verbose:
        print()
        print('-'*80)
        print('Exact log partition sum:', ln_Z)
    return(ln_Z)

def marginal_junction_tree(sg_model, verbose=False, map_flag=True, classification_flag=True):
    '''
    Calculate the exact marginal function of a spin glass model using the junction tree algorithm
    Inputs:
    - sg_model (SpinGlassModel)

    Features:
    - log_marginals: of shape [N, D], where N is the number of variables, and D is
    the state dimensionality. marginal[n,d] denotes the max/sum marginal of variable n
    with state d. (Here, for spinGlass model, D is always equal to 2)

    Output:
    - if classification_flag:
        return the fake probability given the marginals
      else:
        return the difference of log-marginals
    '''
    if not map_flag:
        raise NotImplementedError('Sum marginal hasn\'t been implemented yet in marginal_junction_tree')

    # Set some constants
    maxiter = 10000
    tol = 1e-9
    verb = 0
    # Store the constants in a PropertySet object
    opts = dai.PropertySet()
    opts["maxiter"] = str(maxiter)   # Maximum number of iterations
    opts["tol"] = str(tol)           # Tolerance for convergence
    opts["verbose"] = str(verb)      # Verbosity (amount of output generated)bpopts["updates"] = "SEQRND"
    opts["updates"] = "HUGIN"
    opts["inference"] = "MAXPROD" if map_flag else "SUMPROD"

    N = sg_model.lcl_fld_params.shape[0]
    log_marginals = np.zeros([N*N, 2])

    sg_FactorGraph = build_libdaiFactorGraph_from_SpinGlassModel(sg_model, fixed_variables={})
    jt = dai.JTree( sg_FactorGraph, opts )
    jt.init()
    jt.run()
    state = jt.findMaximum()
    score = sg_FactorGraph.logScore(state)
    log_marginals[np.arange(N*N), np.array(state)] = score

    for vi, s in enumerate(state):
        sg_FactorGraph = build_libdaiFactorGraph_from_SpinGlassModel(sg_model, fixed_variables={vi:(1 if s==0 else -1)})
        jt = dai.JTree( sg_FactorGraph, opts )
        jt.init()
        jt.run()
        cur_state = jt.findMaximum()
        cur_score = sg_FactorGraph.logScore(cur_state)
        log_marginals[vi,1-s] = cur_score

    normalized_log_marginals = log_marginals-log_marginals[:,-1:]
    if classification_flag:
        marginals = np.exp(normalized_log_marginals)
        probability = marginals / np.sum(marginals, axis=-1, keepdims=True)
        return probability.tolist()
    else:
        return normalized_log_marginals[:,0:-1].tolist()

def run_marginal_loopyBP(sg_model, maxiter=None, updates="SEQRND", damping=None,
                         map_flag=True, classification_flag=True):
    if maxiter is None:
        maxiter=LIBDAI_LBP_ITERS


    # Set some constants
    maxiter = maxiter
    tol = 1e-9
    verb = 1
    # Store the constants in a PropertySet object
    opts = dai.PropertySet()
    opts["maxiter"] = str(maxiter)   # Maximum number of iterations
    opts["tol"] = str(tol)           # Tolerance for convergence
    opts["verbose"] = str(verb)      # Verbosity (amount of output generated)bpopts["updates"] = "SEQRND"
    opts["updates"] = updates
    opts["logdomain"] = "1"
    if damping is not None:
        opts["damping"] = str(damping)
    opts['inference'] = ('MAXPROD' if map_flag else 'SUMPROD')

    sg_FactorGraph = build_libdaiFactorGraph_from_SpinGlassModel(sg_model, fixed_variables={})
    bp = dai.BP( sg_FactorGraph, opts )
    bp.init()
    bp.run()

    N = sg_model.lcl_fld_params.shape[0]
    marginals = np.zeros([N*N, 2])
    for vi in range(N*N):
        factor = bp.beliefV(vi)
        marginals[vi, 0] = factor[0]
        marginals[vi, 1] = factor[1]

    if classification_flag:
        probability = marginals / np.sum(marginals, axis=-1, keepdims=True)
        return probability.tolist()
    else:
        log_marginals = np.log(marginals)
        normalized_log_marginals = log_marginals[:, 0:-1]-log_marginals[:, -1:]
        return normalized_log_marginals.tolist()

def run_loopyBP(sg_model, maxiter, updates="SEQRND", damping=None, map_flag=False):
    if maxiter is None:
        maxiter=LIBDAI_LBP_ITERS

    sg_FactorGraph = build_libdaiFactorGraph_from_SpinGlassModel(sg_model, fixed_variables={})
    # sg_FactorGraph = build_graph_from_clique_ising_model(sg_model, fixed_variables={})

    # Write factorgraph to a file
    sg_FactorGraph.WriteToFile('sg_temp.fg')

    # Set some constants
    # maxiter = 10000
    maxiter = maxiter
    # maxiter = 4
    tol = 1e-9
    verb = 1
    # Store the constants in a PropertySet object
    opts = dai.PropertySet()
    opts["maxiter"] = str(maxiter)   # Maximum number of iterations
    opts["tol"] = str(tol)           # Tolerance for convergence
    opts["verbose"] = str(verb)      # Verbosity (amount of output generated)bpopts["updates"] = "SEQRND"


    ##################### Run Loopy Belief Propagation #####################
    # Construct a BP (belief propagation) object from the FactorGraph sg_FactorGraph
    # using the parameters specified by opts and two additional properties,
    # specifying the type of updates the BP algorithm should perform and
    # whether they should be done in the real or in the logdomain
    bpopts = opts
#     bpopts["updates"] = "SEQRND"
#     bpopts["updates"] = "PARALL"
    bpopts["updates"] = updates
    bpopts["logdomain"] = "1"
    if damping is not None:
        bpopts["damping"] = str(damping)
    if map_flag:
        bpopts['inference'] = 'MAXPROD'
    else:
        bpopts['inference'] = 'SUMPROD'

    bp = dai.BP( sg_FactorGraph, bpopts )
    # Initialize belief propagation algorithm
    bp.init()
    # Run belief propagation algorithm
    bp.run()

    # Report log partition sum of sg_FactorGraph, approximated by the belief propagation algorithm
    if map_flag:
        return sg_FactorGraph.logScore(bp.findMaximum())
    else:
        return bp.logZ()

#     print(type(bp.belief(sg_FactorGraph.var(0))))
#     print(bp.belief(sg_FactorGraph.var(0))[0])
#     sleep(aslfkdj)


def run_mean_field(sg_model, maxiter):
    if maxiter is None:
        maxiter=LIBDAI_MEAN_FIELD_ITERS

    sg_FactorGraph = build_libdaiFactorGraph_from_SpinGlassModel(sg_model, fixed_variables={})
    # sg_FactorGraph = build_graph_from_clique_ising_model(sg_model, fixed_variables={})

    # Write factorgraph to a file
    sg_FactorGraph.WriteToFile('sg_temp.fg')

    # Set some constants
    # maxiter = 10000
    maxiter = maxiter
    # maxiter = 4
    tol = 1e-9
    verb = 1
    # Store the constants in a PropertySet object
    opts = dai.PropertySet()
    opts["maxiter"] = str(maxiter)   # Maximum number of iterations
    opts["tol"] = str(tol)           # Tolerance for convergence
    opts["verbose"] = str(verb)      # Verbosity (amount of output generated)


    ##################### Run Loopy Belief Propagation #####################
    # Construct an MF (mean field) object from the FactorGraph sg_FactorGraph
    # using the parameters specified by opts and two additional properties,

    mfopts = opts
#     mfopts["damping"] = ".5"

    mf = dai.MF( sg_FactorGraph, mfopts )
    # Initialize mean field algorithm
    mf.init()
    # Run mean field algorithm
    mf.run()

    # Report log partition sum of sg_FactorGraph, approximated by the mean field algorithm
    ln_z_estimate = mf.logZ()
    return ln_z_estimate



def run_inference(sg_model):
    sg_FactorGraph = build_libdaiFactorGraph_from_SpinGlassModel(sg_model, fixed_variables={})
    # sg_FactorGraph = build_graph_from_clique_ising_model(sg_model, fixed_variables={})

    # Write factorgraph to a file
    sg_FactorGraph.WriteToFile('sg_temp.fg')
    print('spin glass factor graph written to sg_temp.fg')

    # Output some information about the factorgraph
    print( sg_FactorGraph.nrVars(), 'variables')
    print( sg_FactorGraph.nrFactors(), 'factors')

    # Set some constants
    maxiter = 10000
    tol = 1e-9
    verb = 1
    # Store the constants in a PropertySet object
    opts = dai.PropertySet()
    opts["maxiter"] = str(maxiter)   # Maximum number of iterations
    opts["tol"] = str(tol)           # Tolerance for convergence
    opts["verbose"] = str(verb)      # Verbosity (amount of output generated)bpopts["updates"] = "SEQRND"


    ##################### Run Loopy Belief Propagation #####################
    print()
    print( '-'*80        )
    # Construct a BP (belief propagation) object from the FactorGraph sg_FactorGraph
    # using the parameters specified by opts and two additional properties,
    # specifying the type of updates the BP algorithm should perform and
    # whether they should be done in the real or in the logdomain
    bpopts = opts
    bpopts["updates"] = "SEQRND"
    bpopts["logdomain"] = "1"

    bp = dai.BP( sg_FactorGraph, bpopts )
    # Initialize belief propagation algorithm
    bp.init()
    # Run belief propagation algorithm
    bp.run()

    # Report log partition sum of sg_FactorGraph, approximated by the belief propagation algorithm
    print( 'Approximate (loopy belief propagation) log partition sum:', bp.logZ())


    ##################### Run Tree Re-weighted Belief Propagation #####################
    print()
    print( '-'*80        )
    # Construct a BP (belief propagation) object from the FactorGraph sg_FactorGraph
    # using the parameters specified by opts and two additional properties,
    # specifying the type of updates the BP algorithm should perform and
    # whether they should be done in the real or in the logdomain
    trwbp_opts = opts
    trwbp_opts["updates"] = "SEQRND"
    trwbp_opts["nrtrees"] = "10"
    trwbp_opts["logdomain"] = "1"

    trwbp = dai.TRWBP( sg_FactorGraph, trwbp_opts )
    # trwbp = dai.FBP( sg_FactorGraph, trwbp_opts )


    # Initialize belief propagation algorithm
    trwbp.init()
    # Run belief propagation algorithm
    t0 = time.time()
    trwbp.run()
    t1 = time.time()

    # Report log partition sum of sg_FactorGraph, approximated by the belief propagation algorithm
    print( 'Approximate (tree re-weighted belief propagation) log partition sum:', trwbp.logZ())
    print( 'time =', t1-t0)

    ##################### Run Junction Tree Algorithm #####################
    print()
    print( '-'*80        )
    # Construct a JTree (junction tree) object from the FactorGraph sg_FactorGraph
    # using the parameters specified by opts and an additional property
    # that specifies the type of updates the JTree algorithm should perform
    jtopts = opts
    jtopts["updates"] = "HUGIN"
    jt = dai.JTree( sg_FactorGraph, jtopts )
    # Initialize junction tree algorithm
    jt.init()
    # Run junction tree algorithm
    jt.run()

    # Construct another JTree (junction tree) object that is used to calculate
    # the joint configuration of variables that has maximum probability (MAP state)
    jtmapopts = opts
    jtmapopts["updates"] = "HUGIN"
    jtmapopts["inference"] = "MAXPROD"
    jtmap = dai.JTree( sg_FactorGraph, jtmapopts )
    # Initialize junction tree algorithm
    jtmap.init()
    # Run junction tree algorithm
    jtmap.run()
    # Calculate joint state of all variables that has maximum probability
    jtmapstate = jtmap.findMaximum()
    # Report log partition sum (normalizing constant) of sg_FactorGraph, calculated by the junction tree algorithm
    print()
    print( '-'*80    )
    print( 'Exact log partition sum:', jt.logZ())


#==================================Testing===================================
from . import spin_glass_model
def _test_marginal_junction_tree():
    for _ in range(10):
        N = 10
        f,c = np.random.rand(2)
        classification_flag = bool(np.random.choice(2))
        print(N, f, c, classification_flag)

        output =  marginal_junction_tree(spin_glass_model.SpinGlassModel(N, f, c),
                                         classification_flag=classification_flag,)
        assert(type(output) == list)
        output = np.array(output)
        assert(output.shape[0] == N*N)
        assert(output.shape[1] == 2 if classification_flag else 1)
        if classification_flag:
            assert(np.all(np.sum(output, axis=-1)-1 < 1e-7))
def _test_run_marginal_BP():
    for _ in range(10):
        N = 10
        f,c = np.random.rand(2)
        classification_flag = bool(np.random.choice(2))
        print(N, f, c, classification_flag)

        output =  run_marginal_loopyBP(spin_glass_model.SpinGlassModel(N, f, c),
                                       classification_flag=classification_flag,)
        assert(type(output) == list)
        output = np.array(output)
        assert(output.shape[0] == N*N)
        assert(output.shape[1] == 2 if classification_flag else 1)
        if classification_flag:
            assert(np.all(np.sum(output, axis=-1)-1 < 1e-7))

if __name__ == '__main__':
    functions = {k:v for k,v in globals().items() if 'test_' == k[:5]}
    for k,v in functions.items():
        print()
        print('=============Running %s=============='%k)
        v()
