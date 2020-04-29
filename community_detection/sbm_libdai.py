import sys
import numpy as np
LIBDAI_SWIG_DIRECTORY = '/atlas/u/shuvamc/libdai2/swig/'
sys.path.insert(0, LIBDAI_SWIG_DIRECTORY)
import dai
import time
import torch

def build_unary_libdai_factor(sbm_model, variables, var_idx):
    clause_variables = dai.VarSet(variables[var_idx])
    factor = dai.Factor(clause_variables)
    for i in range(sbm_model.C):
        factor[i] = sbm_model.community_probs[i]
    return factor

def build_binary_libdai_factor(sbm_model, variables, var_1, var_2, edge):
    clause_variables = dai.VarSet(variables[var_1], variables[var_2])
    factor = dai.Factor(clause_variables)
    prob_1 = sbm_model.p if edge else 1 - sbm_model.p
    prob_2 = sbm_model.q if edge else 1 - sbm_model.q
    denom = 2*prob_1 + 2*prob_2
    same_prob = prob_1 / denom
    diff_prob = prob_2 / denom
    for i in range(sbm_model.C ** 2):
        if i % (sbm_model.C + 1) == 0:
            factor[i] = same_prob
        else:
            factor[i] = diff_prob
    return factor

def build_libdaiFactorGraph_from_SBM(sbm_model):
    variables = []
    for i in range(sbm_model.N):
        variables.append(dai.Var(i, sbm_model.C))
    factors = []

    for i in range(sbm_model.N):
        fac = build_unary_libdai_factor(sbm_model, variables, i)
        factors.append(fac)

    for i in range(sbm_model.edge_index.shape[1]):
        fac = build_binary_libdai_factor(sbm_model, variables, sbm_model.edge_index[1][i], sbm_model.edge_index[0][i], True)
        factors.append(fac)

    for i in range(sbm_model.noedge_index.shape[1]):
        fac = build_binary_libdai_factor(sbm_model, variables, sbm_model.noedge_index[1][i], sbm_model.noedge_index[0][i], False)
        factors.append(fac)

    sbm_factors = dai.VecFactor()
    for factor in factors:
        sbm_factors.append(factor)

    sbm_FactorGraph = dai.FactorGraph(sbm_factors)
    return sbm_FactorGraph

def getBeliefs(sbm_model, fg, bp_object):
    beliefs = []
    for i in range(sbm_model.N):
        b = []
        belief = bp_object.belief(fg.var(i))
        for j in range(sbm_model.C):
            b.append(belief[j])
        beliefs.append(b)
    return beliefs

def runLBPLibdai(sbm_model, maxiter = 5000, tol = 1e-9, verbose = 1, updates = 'PARALL', logdomain = 1, damping = .8):
    opts = dai.PropertySet()
    opts["maxiter"] = str(maxiter)
    opts["tol"] = str(tol)
    opts["verbose"] = str(verbose)
    opts["updates"] = updates
    opts["logdomain"] = str(logdomain)
    opts["damping"] = str(damping)
    sbm_fg = build_libdaiFactorGraph_from_SBM(sbm_model)
    bp = dai.BP(sbm_fg, opts)
    bp.init()
    bp.run()
    beliefs = torch.Tensor(getBeliefs(sbm_model, sbm_fg, bp))
    #print(beliefs)
    return beliefs
    
