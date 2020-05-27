import sys
import numpy as np
LIBDAI_SWIG_DIRECTORY = '/atlas/u/shuvamc/libdai2/swig/'
sys.path.insert(0, LIBDAI_SWIG_DIRECTORY)
import dai
import time
import torch
from parameters_sbm import LN_ZERO

def build_unary_libdai_factor(sbm_model, variables, var_idx, fixed_var, fixed_val):
    clause_variables = dai.VarSet(variables[var_idx])
    factor = dai.Factor(clause_variables)
    prob_lst = sbm_model.community_probs
    #if var_idx == fixed_var:
        #factor[fixed_val] = sbm_model.community_probs[fixed_val]
        #return factor
    #    prob_lst = np.ones_like(prob_lst) *np.exp(LN_ZERO)
    #    prob_lst[fixed_val] = sbm_model.community_probs[fixed_val]
    for i in range(sbm_model.C):
        factor[i] = prob_lst[i]
    return factor

def build_binary_libdai_factor(sbm_model, variables, var_1, var_2, edge, fixed_var, fixed_val):
    var_1 = int(var_1)
    var_2 = int(var_2)
    clause_variables = dai.VarSet(variables[var_1], variables[var_2])
    factor = dai.Factor(clause_variables)
    prob_1 = sbm_model.p if edge else 1 - sbm_model.p
    prob_2 = sbm_model.q if edge else 1 - sbm_model.q
    #denom = sbm_model.C*prob_1 + (sbm_model.C ** 2 - sbm_model.C)*prob_2
    #if fixed_var > -1 and fixed_val > -1:
    #    denom = prob_1 + (sbm_model.C - 1) * prob_2
    same_prob = prob_1 #/ denom
    diff_prob = prob_2 #/ denom
    for i in range(sbm_model.C ** 2):
        #if var_1 == fixed_var and int(i / sbm_model.C) != fixed_val:
            #continue
        #    factor[i] = np.exp(LN_ZERO)
        #elif var_2 == fixed_var and i % sbm_model.C != fixed_val:
            #continue
        #    factor[i] = np.exp(LN_ZERO)
        if i % (sbm_model.C + 1) == 0:
            factor[i] = same_prob
        else:
            factor[i] = diff_prob
    return factor

def build_libdaiFactorGraph_from_SBM(sbm_model, fixed_var=-1, fixed_val=-1):
    variables = []
    for i in range(sbm_model.N):
        variables.append(dai.Var(i, sbm_model.C))
    factors = []

    for i in range(sbm_model.N):
        fac = build_unary_libdai_factor(sbm_model, variables, i, fixed_var, fixed_val)
        factors.append(fac)

    for i in range(sbm_model.edge_index.shape[1]):
        fac = build_binary_libdai_factor(sbm_model, variables, sbm_model.edge_index[1][i], sbm_model.edge_index[0][i], True, fixed_var, fixed_val)
        factors.append(fac)

    for i in range(sbm_model.noedge_index.shape[1]):
        fac = build_binary_libdai_factor(sbm_model, variables, sbm_model.noedge_index[1][i], sbm_model.noedge_index[0][i], False, fixed_var, fixed_val)
        factors.append(fac)

    sbm_factors = dai.VecFactor()
    for factor in factors:
        sbm_factors.append(factor)

    sbm_FactorGraph = dai.FactorGraph(sbm_factors)
    if fixed_var >= 0 and fixed_var < sbm_model.N and fixed_val >= 0 and fixed_val < sbm_model.C:
        sbm_FactorGraph.clamp(fixed_var, fixed_val)
    return sbm_FactorGraph

def getBeliefs(sbm_model, fg, bp_object):
    beliefs = []
    for i in range(sbm_model.N):
        b = []
        belief = bp_object.belief(dai.VarSet(fg.var(i)))
        for j in range(sbm_model.C):
            b.append(belief[j])
        beliefs.append(b)
    return beliefs

def runLBPLibdai(sbm_fg, sbm_model, maxiter = 5000, tol = 1e-9, verbose = 1, updates = 'PARALL', logdomain = 1, damping = .5):
    opts = dai.PropertySet()
    opts["maxiter"] = str(maxiter)
    opts["tol"] = str(tol)
    opts["verbose"] = str(verbose)
    opts["updates"] = updates
    opts["logdomain"] = str(logdomain)
    opts["damping"] = str(damping)
    #sbm_fg = build_libdaiFactorGraph_from_SBM(sbm_model, fixed_var, fixed_val)
    bp = dai.BP(sbm_fg, opts)
    bp.init()
    bp.run()
    beliefs = torch.Tensor(getBeliefs(sbm_model, sbm_fg, bp))
    #print(beliefs)
    return beliefs, bp.logZ()
    
def runJT(sbm_fg, sbm_model, maxiter = 10000, tol = 1e-9, verbose = 1, updates = "HUGIN"):
    opts = dai.PropertySet()
    opts["maxiter"] = str(maxiter)
    opts["tol"] = str(tol)
    opts["verbose"] = str(verbose)
    opts["updates"] = updates
    opts["updates"] = "HUGIN"
    #sbm_fg = build_libdaiFactorGraph_from_SBM(sbm_model, fixed_var, fixed_val)
    jt = dai.JTree(sbm_fg, opts)
    # Initialize junction tree algorithm
    jt.init()
    # Run junction tree algorithm
    jt.run()
    beliefs = torch.Tensor(getBeliefs(sbm_model, sbm_fg, jt))
    return jt.logZ(), beliefs

 
