import sys
import numpy as np

# Atlas
LIBDAI_SWIG_DIRECTORY = '/atlas/u/jkuck/libdai2/swig/'
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

def junction_tree(sg_model, verbose=False):
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
#doesn't work    
#     print(type(bp.belief(sg_FactorGraph.var(0))))
#     print(jtmap.belief(sg_FactorGraph.var(0))[0])
#     sleep(aslfkdj)
    
    ln_Z = jt.logZ()
    # Report log partition sum (normalizing constant) of sg_FactorGraph, calculated by the junction tree algorithm
    if verbose:
        print()
        print('-'*80)
        print('Exact log partition sum:', ln_Z)
    return(ln_Z)

def run_loopyBP(sg_model, maxiter, updates="SEQRND", damping=None):
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
        bpopts["damping"] = damping

    bp = dai.BP( sg_FactorGraph, bpopts )
    # Initialize belief propagation algorithm
    bp.init()
    # Run belief propagation algorithm
    bp.run()

    # Report log partition sum of sg_FactorGraph, approximated by the belief propagation algorithm
    ln_z_estimate = bp.logZ()
    
#     print(type(bp.belief(sg_FactorGraph.var(0))))
#     print(bp.belief(sg_FactorGraph.var(0))[0])
#     sleep(aslfkdj)
    
    return ln_z_estimate


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



