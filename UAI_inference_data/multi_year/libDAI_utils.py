import sys
import numpy as np
import time
# Atlas
LIBDAI_SWIG_DIRECTORY = '/atlas/u/jkuck/libdai2/swig/'
# Local
# LIBDAI_SWIG_DIRECTORY = '/Users/jkuck/research/libdai/swig'
sys.path.insert(0, LIBDAI_SWIG_DIRECTORY)
import dai
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from collections import defaultdict

def build_libdaiFactor(factor_potential, factor_variables, all_variables, debug=False):
    '''
    
    Inputs:
    - factor_potential (torch tensor): values of the factor potential
    - factor_variables (list of ints): variables (0 indexed) in the factor
    - all_variables (list of dai.Var): list of all dai variables in the problem.
        variable i (where i can be in [1, N]) is stored at variables[i-1]

    Outputs:
    - factor (dai.Factor)
    '''
    dai_vars_list = [all_variables[variable] for variable in factor_variables]
    if len(factor_variables) == 1:
        dai_vars = dai.VarSet(dai_vars_list[0])
    else:
        dai_vars = dai.VarSet(dai_vars_list[-1], dai_vars_list[-2])
        #have to reverse the order of the variables because of the way libdai
        #stores factor potentials with respect to variable ordering
        # see https://github.com/dbtsai/libDAI/blob/master/swig/example_sprinkler.py
        for var in reversed(dai_vars_list[:-2]):
            dai_vars.append(var)
        
    factor = dai.Factor(dai_vars)
    # print("factor_potential.shape", factor_potential.shape)
    # print("len(factor_variables)", len(factor_variables))
    # print("len(all_variables)", len(all_variables))
    # print("factor_variables", factor_variables)
    # print("all_variables", all_variables)
    for idx, state_val in enumerate(factor_potential.flatten()):
        # print("state_val.item():", state_val.item())
        # print("type(state_val.item()):", type(state_val.item()))
        if debug:
            time.sleep(.1)
        factor[idx] = state_val.item()

    # print("factor_potential:", factor_potential)
    # exit(0)
    return factor

#test me!!
def build_libdaiFactorGraph(factor_potentials, factorToVar_double_list, N, var_cardinality): 
    '''

    Inputs:
    - N: (int) the number of variables
    - factorToVar_double_list (list of list of ints): factorToVar_double_list[i][j] is the 
        index (0-indexed) of the jth variable in the ith factor 
    - factor_potentials (list of torch tensors): factor potentials
    - var_cardinality (int): the number of states a variable can take

    Outputs:
    - sat_FactorGraph: (dai.FactorGraph)
    '''
    # Define binary variables in the factor graph
    all_variables = []
    for var_idx in range(N):
        all_variables.append(dai.Var(var_idx, var_cardinality)) #variable can take var_cardinality values

    assert(len(factor_potentials) == len(factorToVar_double_list)), (len(factor_potentials), len(factorToVar_double_list))
    libdai_factors = dai.VecFactor()
    print("len(factorToVar_double_list):", len(factorToVar_double_list))
    for factor_idx, factor_variables in enumerate(factorToVar_double_list):
        factor_potential = factor_potentials[factor_idx]
        assert(factor_potential.numel() == var_cardinality**len(factor_variables))
        libdai_factor = build_libdaiFactor(factor_potential, factor_variables, all_variables, debug=False)
        # if factor_idx == 5:
        #     exit(0)

        libdai_factors.append(libdai_factor)

        # print("FACTOR APPENDED!")
        # time.sleep(1)
    # Build factor graph
    sat_FactorGraph = dai.FactorGraph(libdai_factors)

    return sat_FactorGraph

def junction_tree(clauses, verbose=False):
    '''
    Calculate the exact partition function of a spin glass model using the junction tree algorithm
    Inputs:
    - clauses: (list of list of ints) variables should be numbered 1 to N with no gaps
    
    Outputs:
    - ln_Z: natural logarithm of the exact partition function
    '''
    sat_FactorGraph = build_libdaiFactorGraph(clauses, N)

    print("FG BUILT!")
    sleep(FG_BUILT)
    # Write factorgraph to a file
    sat_FactorGraph.WriteToFile('sg_temp.fg')
    if verbose:
        print( 'spin glass factor graph written to sg_temp.fg')

    # Output some information about the factorgraph
    if verbose:
        print( sat_FactorGraph.nrVars(), 'variables')
        print( sat_FactorGraph.nrFactors(), 'factors')

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
    # Construct a JTree (junction tree) object from the FactorGraph sat_FactorGraph
    # using the parameters specified by opts and an additional property
    # that specifies the type of updates the JTree algorithm should perform
    jtopts = opts
    jtopts["updates"] = "HUGIN"
    jt = dai.JTree( sat_FactorGraph, jtopts )
    # Initialize junction tree algorithm
    jt.init()
    # Run junction tree algorithm
    jt.run()

    # Construct another JTree (junction tree) object that is used to calculate
    # the joint configuration of variables that has maximum probability (MAP state)
    jtmapopts = opts
    jtmapopts["updates"] = "HUGIN"
    jtmapopts["inference"] = "MAXPROD"
    jtmap = dai.JTree( sat_FactorGraph, jtmapopts )
    # Initialize junction tree algorithm
    jtmap.init()
    # Run junction tree algorithm
    jtmap.run()
    # Calculate joint state of all variables that has maximum probability
    jtmapstate = jtmap.findMaximum()
    ln_Z = jt.logZ()
    # Report log partition sum (normalizing constant) of sat_FactorGraph, calculated by the junction tree algorithm
    if verbose:
        print()
        print('-'*80)
        print('Exact log partition sum:', ln_Z)
    return(ln_Z)

def run_loopyBP(factor_potentials, factorToVar_double_list, N, var_cardinality, maxiter=1000, updates="SEQRND", damping=None):
    sat_FactorGraph = build_libdaiFactorGraph(factor_potentials, factorToVar_double_list, N, var_cardinality)

    # Write factorgraph to a file
    sat_FactorGraph.WriteToFile('sg_temp.fg')

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
    # Construct a BP (belief propagation) object from the FactorGraph sat_FactorGraph
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

    bp = dai.BP( sat_FactorGraph, bpopts )
    # Initialize belief propagation algorithm
    bp.init()
    # Run belief propagation algorithm
    bp.run()

    # Report log partition sum of sat_FactorGraph, approximated by the belief propagation algorithm
    ln_z_estimate = bp.logZ()
    return ln_z_estimate

    # ##################### Run Junction Tree Algorithm #####################
    # print()
    # print( '-'*80        )
    # # Construct a JTree (junction tree) object from the FactorGraph sat_FactorGraph
    # # using the parameters specified by opts and an additional property
    # # that specifies the type of updates the JTree algorithm should perform
    # jtopts = opts
    # jtopts["updates"] = "HUGIN"
    # jt = dai.JTree( sat_FactorGraph, jtopts )
    # # Initialize junction tree algorithm
    # jt.init()
    # # Run junction tree algorithm
    # jt.run()

    # # Construct another JTree (junction tree) object that is used to calculate
    # # the joint configuration of variables that has maximum probability (MAP state)
    # jtmapopts = opts
    # jtmapopts["updates"] = "HUGIN"
    # jtmapopts["inference"] = "MAXPROD"
    # jtmap = dai.JTree( sat_FactorGraph, jtmapopts )
    # # Initialize junction tree algorithm
    # jtmap.init()
    # # Run junction tree algorithm
    # jtmap.run()
    # # Calculate joint state of all variables that has maximum probability
    # jtmapstate = jtmap.findMaximum()
    # # Report log partition sum (normalizing constant) of sat_FactorGraph, calculated by the junction tree algorithm
    # print()
    # print( '-'*80    )
    # print( 'Exact log partition sum:', jt.logZ())

    # return ln_z_estimate
    
def run_mean_field(clauses, maxiter=1000):
    
    sat_FactorGraph = build_libdaiFactorGraph(clauses, N)
    # sat_FactorGraph = build_graph_from_clique_ising_model(clauses)

    # Write factorgraph to a file
    sat_FactorGraph.WriteToFile('sg_temp.fg')

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
    # Construct an MF (mean field) object from the FactorGraph sat_FactorGraph
    # using the parameters specified by opts and two additional properties,

    mfopts = opts
#     mfopts["damping"] = ".5"

    mf = dai.MF( sat_FactorGraph, mfopts )
    # Initialize mean field algorithm
    mf.init()
    # Run mean field algorithm
    mf.run()

    # Report log partition sum of sat_FactorGraph, approximated by the mean field algorithm
    ln_z_estimate = mf.logZ()
    return ln_z_estimate



def run_inference(clauses):
    sat_FactorGraph = build_libdaiFactorGraph(clauses, N)
    # sat_FactorGraph = build_graph_from_clique_ising_model(clauses)

    # Write factorgraph to a file
    sat_FactorGraph.WriteToFile('sg_temp.fg')
    print('spin glass factor graph written to sg_temp.fg')

    # Output some information about the factorgraph
    print( sat_FactorGraph.nrVars(), 'variables')
    print( sat_FactorGraph.nrFactors(), 'factors')

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
    # Construct a BP (belief propagation) object from the FactorGraph sat_FactorGraph
    # using the parameters specified by opts and two additional properties,
    # specifying the type of updates the BP algorithm should perform and
    # whether they should be done in the real or in the logdomain
    bpopts = opts
    bpopts["updates"] = "SEQRND"
    bpopts["logdomain"] = "1"

    bp = dai.BP( sat_FactorGraph, bpopts )
    # Initialize belief propagation algorithm
    bp.init()
    # Run belief propagation algorithm
    bp.run()

    # Report log partition sum of sat_FactorGraph, approximated by the belief propagation algorithm
    print( 'Approximate (loopy belief propagation) log partition sum:', bp.logZ())


    ##################### Run Tree Re-weighted Belief Propagation #####################
    print()
    print( '-'*80        )
    # Construct a BP (belief propagation) object from the FactorGraph sat_FactorGraph
    # using the parameters specified by opts and two additional properties,
    # specifying the type of updates the BP algorithm should perform and
    # whether they should be done in the real or in the logdomain
    trwbp_opts = opts
    trwbp_opts["updates"] = "SEQRND"
    trwbp_opts["nrtrees"] = "10"
    trwbp_opts["logdomain"] = "1"

    trwbp = dai.TRWBP( sat_FactorGraph, trwbp_opts )
    # trwbp = dai.FBP( sat_FactorGraph, trwbp_opts )


    # Initialize belief propagation algorithm
    trwbp.init()
    # Run belief propagation algorithm
    t0 = time.time()
    trwbp.run()
    t1 = time.time()

    # Report log partition sum of sat_FactorGraph, approximated by the belief propagation algorithm
    print( 'Approximate (tree re-weighted belief propagation) log partition sum:', trwbp.logZ())
    print( 'time =', t1-t0)

    ##################### Run Junction Tree Algorithm #####################
    print()
    print( '-'*80        )
    # Construct a JTree (junction tree) object from the FactorGraph sat_FactorGraph
    # using the parameters specified by opts and an additional property
    # that specifies the type of updates the JTree algorithm should perform
    jtopts = opts
    jtopts["updates"] = "HUGIN"
    jt = dai.JTree( sat_FactorGraph, jtopts )
    # Initialize junction tree algorithm
    jt.init()
    # Run junction tree algorithm
    jt.run()

    # Construct another JTree (junction tree) object that is used to calculate
    # the joint configuration of variables that has maximum probability (MAP state)
    jtmapopts = opts
    jtmapopts["updates"] = "HUGIN"
    jtmapopts["inference"] = "MAXPROD"
    jtmap = dai.JTree( sat_FactorGraph, jtmapopts )
    # Initialize junction tree algorithm
    jtmap.init()
    # Run junction tree algorithm
    jtmap.run()
    # Calculate joint state of all variables that has maximum probability
    jtmapstate = jtmap.findMaximum()
    # Report log partition sum (normalizing constant) of sat_FactorGraph, calculated by the junction tree algorithm
    print()
    print( '-'*80    )
    print( 'Exact log partition sum:', jt.logZ())




