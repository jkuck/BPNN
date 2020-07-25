import sys
import numpy as np

LIBDAI_SWIG_DIRECTORY = '/atlas/u/htang/libdai/swig/'
# LIBDAI_SWIG_DIRECTORY = '/workspace/libdai/swig/'

sys.path.insert(0, LIBDAI_SWIG_DIRECTORY)
import dai
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from parameters import LIBDAI_LBP_ITERS
from parameters import LIBDAI_MEAN_FIELD_ITERS


def build_libdaiFactorGraph_from_UAI2012Model(sg_model):
    '''
    The model is actually a list of strings.
    The formats are specified in https://www.cs.huji.ac.il/project/PASCAL/fileFormat.php
    '''
    contents = sg_model[1:] # the first line is always MARKOV
    N, contents = int(contents[0].strip()), contents[1:]
    var_cards, contents = [int(card.strip()) for card in contents[0].split()], contents[1:]
    assert(N == len(var_cards))
    variables = [dai.Var(idx, card) for idx, card in enumerate(var_cards)]

    factors = []
    factor_num, contents = int(contents[0].strip()), contents[1:]
    for fi in range(factor_num):
        factor_contents = contents[fi].strip().split()
        factor_dim = int(factor_contents[0])
        factor_var_idx = [int(vidx.strip()) for vidx in factor_contents[1:]]
        assert(factor_dim == len(factor_var_idx))
        if factor_dim == 1:
            clause_variables = dai.VarSet(variables[factor_var_idx[0]])
        else:
            #have to reverse the order of the variables because of the way libdai
            #stores factor potentials with respect to variable ordering
            # see https://github.com/dbtsai/libDAI/blob/master/swig/example_sprinkler.py
            clause_variables = dai.VarSet(variables[factor_var_idx[-1]], variables[factor_var_idx[-2]])
            for vidx in factor_var_idx[::-1][2:]:
                clause_variables.append(variables[vidx])
        factor = dai.Factor(clause_variables)
        factors.append(factor)
    contents = contents[factor_num:]
    for fidx, factor in enumerate(factors):
        potential_num = int(contents[fidx*2].strip())
        potentials = [float(p.strip()) for p in contents[fidx*2+1].strip().split()]
        assert(potential_num == len(potentials))
        for pi, p in enumerate(potentials):
            factor[pi] = p
    contents = contents[2*factor_num:]
    assert(len(contents) == 0)

    # Build factor graph
    sg_Factors = dai.VecFactor()
    for factor in factors:
        sg_Factors.append(factor)
    sg_FactorGraph = dai.FactorGraph(sg_Factors)

    return sg_FactorGraph

def junction_tree(sg_FactorGraph, verbose=False, map_flag=False):
    '''
    Calculate the exact partition function of a spin glass model using the junction tree algorithm
    Inputs:
    - sg_model (SpinGlassModel)

    Outputs:
    - ln_Z: natural logarithm of the exact partition function
    '''
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

def map_junction_tree(sg_FactorGraph, verbose=False, map_flag=True):
    assert(map_flag)
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
    return jtmapstate

def marginal_junction_tree(sg_FactorGraph, verbose=False, map_flag=True, classification_flag=True):
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

    N = sg_FactorGraph.nrVars()
    log_marginals = np.zeros([N*N, 2])

    if not map_flag:
        opts["inference"] = "SUMPROD"
        jt = dai.JTree( sg_FactorGraph, opts )
        jt.init()
        jt.run()
        logZ = jt.logZ()
        for vi in range(N*N):
            jt = dai.JTree( sg_FactorGraph, opts )
            jt.init()
            jt.run()
            log_marginals[vi,0] = jt.logZ()
        probability = np.exp(log_marginals-logZ)
        probability[:,1] = 1-probability[:,0]
        if classification_flag:
            return probability.tolist()
        else:
            log_marginals = np.log(probability)
            normalized_log_marginals = log_marginals - log_marginals[:,-1:]
            return normalized_log_marginals[:,0:-1].tolist()
    else:
        opts["inference"] = "MAXPROD"
        jt = dai.JTree( sg_FactorGraph, opts )
        jt.init()
        jt.run()
        state = jt.findMaximum()
        score = sg_FactorGraph.logScore(state)
        log_marginals[np.arange(N*N), np.array(state)] = score

        for vi, s in enumerate(state):
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

def map_marginal_junction_tree(sg_FactorGraph, verbose=False):
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

    N = sg_FactorGraph.nrVars()
    max_states =  [[None]*2]*(N*N)

    opts["inference"] = "MAXPROD"
    jt = dai.JTree( sg_FactorGraph, opts )
    jt.init()
    jt.run()
    state = jt.findMaximum()
    # score = sg_FactorGraph.logScore(state)
    for i, s in zip(range(N*N), state):
        max_states[i][s] = state

    for vi, s in enumerate(state):
        jt = dai.JTree( sg_FactorGraph, opts )
        jt.init()
        jt.run()
        cur_state = jt.findMaximum()
        assert(cur_state[vi] == 0)
        cur_state = list(cur_state)
        cur_state[vi] = 1-s
        cur_state = tuple(cur_state)
        # cur_score = sg_FactorGraph.logScore(cur_state)
        # log_marginals[vi,1-s] = cur_score
        max_states[vi][1-s] = cur_state

    return max_states


def logScore(sg_FactorGraph, state):
    # Write factorgraph to a file
    sg_FactorGraph.WriteToFile('sg_temp.fg')
    return sg_FactorGraph.logScore(state)

def run_marginal_loopyBP(sg_FactorGraph, maxiter=None, updates="SEQRND", damping=None,
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

    bp = dai.BP( sg_FactorGraph, opts )
    bp.init()
    bp.run()

    N = sg_FactorGraph.nrVars()
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

def run_map_loopyBP(sg_FactorGraph, maxiter=None, updates="SEQRND", damping=None, map_flag=False):
    assert(map_flag)
    if maxiter is None:
        maxiter=LIBDAI_LBP_ITERS

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

    return bp.findMaximum()

def run_loopyBP(sg_FactorGraph, maxiter=None, updates="SEQRND", damping=None, map_flag=False):
    if maxiter is None:
        maxiter=LIBDAI_LBP_ITERS

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

def run_mean_field(sg_FactorGraph, maxiter):
    if maxiter is None:
        maxiter=LIBDAI_MEAN_FIELD_ITERS

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

def run_inference(sg_FactorGraph,):
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
if __name__ == '__main__':
    from uai2012_models import UAI2012Dataset
    from tqdm import tqdm
    dataset = UAI2012Dataset()[0]

    print('Building Libdai Graphs...')
    libdai_graphs = [build_libdaiFactorGraph_from_UAI2012Model(data) for data in tqdm(dataset)]
    print('Performing BP...')
    map_results = [run_loopyBP(g, map_flag=True) for g in tqdm(libdai_graphs)]

