import sys
import numpy as np

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
from parameters import LIBDAI_LBP_ITERS
from parameters import LIBDAI_MEAN_FIELD_ITERS
# from ..parameters import LIBDAI_LBP_ITERS
from collections import defaultdict


def parse_dimacs(filename, verbose=False):
    clauses = []
    dictionary_of_vars = defaultdict(int)
    # print("parse_dimacs, filename:", filename)  
    with open(filename, 'r') as f:    
        for line in f:
            line_as_list = line.strip().split()
            if len(line_as_list) == 0:
                continue
            # print("line_as_list[:-1]:")
            # print(line_as_list[:-1])
            # print()            
            if line_as_list[0] == "p":
                n_vars = int(line_as_list[2])
                n_clauses = int(line_as_list[3])
            elif line_as_list[0] == "c":
                continue
            else:
                cur_clause = [int(s) for s in line_as_list[:-1]]
                for var in cur_clause:
                    dictionary_of_vars[int(abs(var))] += 1
                clauses.append(cur_clause)
    # assert(n_clauses == len(clauses)), (n_clauses, len(clauses), filename)
    if(n_clauses != len(clauses)):
        if verbose:
            print("actual clause count doesn't match expected clause count!!")
    
    #make sure that all variables are named something in [1,...,n_vars]
    for var_name, var_degree in dictionary_of_vars.items():
        assert(var_name <= n_vars and var_name >= 1)

    #create dummy clauses for variables that don't explcitly appear, i.e. if 
    #variable 8 never appears explicitly in a clause this is equivalent to having
    #the additional clause (-8 8 0)
    for var_name in range(1, n_vars+1):
        if var_name not in dictionary_of_vars:
            clauses.append([-var_name, var_name])
            dictionary_of_vars[var_name] = 2
            # print("appended clause:", [-var_name, var_name])
            
    # if (len(dictionary_of_vars) == n_vars):
    if True: #missing variables imply an always true clause, e.g. (-8 8 0) if 8 is missing
        if verbose:
            print("variable count checks succeeded")
        load_successful = True
    else:
        if verbose:
            print("variable count check failed")
        load_successful = False
        print("load failed for:", filename)
        print("len(dictionary_of_vars):", len(dictionary_of_vars))
        print("n_vars:", n_vars)
        for i in range(1, n_vars+1):
            if i not in dictionary_of_vars:
                print(i, "missing from dictionary_of_vars")
        print()
    return n_vars, clauses, load_successful

def build_libdaiFactor_fromClause(clause, variables):
    '''
    
    Inputs:
    - clause (list of integers): literals in the clause
    - variables (list of dai.Var): list of all dai variables in the sat problem.
        variable i (where i can be in [1, N]) is stored at variables[i-1]

    Outputs:
    - factor (dai.Factor)
    '''
    dai_vars_list = [variables[abs(literal)-1] for literal in clause]
    if len(clause) == 1:
        dai_vars = dai.VarSet(dai_vars_list[0])
    else:
        dai_vars = dai.VarSet(dai_vars_list[-1], dai_vars_list[-2])
        #have to reverse the order of the variables because of the way libdai
        #stores factor potentials with respect to variable ordering
        # see https://github.com/dbtsai/libDAI/blob/master/swig/example_sprinkler.py
        for var in reversed(dai_vars_list[:-2]):
            dai_vars.append(var)
        
        
    factor = dai.Factor(dai_vars)

    #Create a tensor for the 2^state_dimensions states
    state = np.zeros([2 for i in range(len(clause))])
    #Iterate over the 2^state_dimensions variable assignments and set those to 1 that satisfy the clause
    for indices in np.ndindex(state.shape):
        set_to_1 = False
        for dimension, index_val in enumerate(indices):
            if clause[dimension] > 0 and index_val == 1:
                set_to_1 = True
            elif clause[dimension] < 0 and index_val == 0:
                set_to_1 = True
        if set_to_1:
            state[indices] = 1
        else:
            state[indices] = 0

    for idx, state_val in enumerate(state.flatten()):
        factor[idx] = state_val
    return factor

#test me!!
def build_libdaiFactorGraph_from_SATproblem(clauses, N): 
    '''

    Inputs:
    - clauses: (list of list of ints) variables should be numbered 1 to N with no gaps
    - N: (int) the number of variables in the sat problem.  This should be equal 
        the largest variable name (otherwise could have problems with implicit variables,
        e.g. a missing variable x_i is equivalent to the clause (x_i or not x_i), doubling
        the solution count)

    Outputs:
    - sat_FactorGraph: (dai.FactorGraph)
    '''
    #make sure variables are numbered 1 to N with no gaps
    all_literals = {}
    max_literal = 0
    for clause in clauses:
        for var in clause:
            lit = abs(var)
            all_literals[lit] = True
            if lit > max_literal:
                max_literal = lit
    assert(len(all_literals) == max_literal)
    assert(max_literal == N)
    for i in range (1,max_literal + 1):
        assert(i in all_literals)

    # Define binary variables in the factor graph
    variables = []
    for var_idx in range(max_literal):
        variables.append(dai.Var(var_idx, 2)) #variable can take 2 values

    factors = []
    for clause in clauses:
        factors.append(build_libdaiFactor_fromClause(clause, variables))

    assert(len(factors) == len(clauses))
    assert(len(variables) == max_literal)

    # Build factor graph
    sat_Factors = dai.VecFactor()
    for factor in factors:
        sat_Factors.append(factor)
    sat_FactorGraph = dai.FactorGraph(sat_Factors)

    return sat_FactorGraph

def junction_tree(clauses, verbose=False):
    '''
    Calculate the exact partition function of a spin glass model using the junction tree algorithm
    Inputs:
    - clauses: (list of list of ints) variables should be numbered 1 to N with no gaps
    
    Outputs:
    - ln_Z: natural logarithm of the exact partition function
    '''
    sat_FactorGraph = build_libdaiFactorGraph_from_SATproblem(clauses, N)

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

def run_loopyBP(clauses, n_vars, maxiter, updates="SEQRND", damping=None):
    if maxiter is None:
        maxiter=LIBDAI_LBP_ITERS
    
    sat_FactorGraph = build_libdaiFactorGraph_from_SATproblem(clauses, N=n_vars)
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


def run_mean_field(clauses, maxiter):
    if maxiter is None:
        maxiter=LIBDAI_MEAN_FIELD_ITERS
    
    sat_FactorGraph = build_libdaiFactorGraph_from_SATproblem(clauses, N)
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
    sat_FactorGraph = build_libdaiFactorGraph_from_SATproblem(clauses, N)
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




