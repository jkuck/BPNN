#!/usr/bin/env python
# coding=utf-8

from uai2012_models import _UAI2012Dataset
from libdai_utils import build_libdaiFactorGraph_from_UAI2012Model, logScore, junction_tree, map_junction_tree, run_loopyBP

import sys, os, os.path as osp, pickle as pkl, numpy as np

if __name__ == '__main__':
    curdir = osp.dirname(osp.abspath(__file__))
    output_dirname = osp.join(curdir, 'MAP_results')

    indexes = [int(idx) for idx in sys.argv[1:]]
    for idx in indexes:
        dataset, paths = _UAI2012Dataset(idx)
        print(idx, 'Building Libdai Graphs...')
        libdai_graphs = [build_libdaiFactorGraph_from_UAI2012Model(data) for data in dataset]
        print(libdai_graphs[0].nrVars())
        # print(idx, 'Performing BP...')
        # logZ = [run_loopyBP(g, map_flag=False) for g in libdai_graphs]
        print(idx, 'Performing Junction Tree...')
        logZ = [junction_tree(g, map_flag=False) for g in libdai_graphs]
        print(idx, 'Performing MAP Junction Tree...')
        max_states = [map_junction_tree(g, map_flag=True) for g in libdai_graphs]
        print(idx, 'Calculate logScores...')
        max_scores = [logScore(g, s) for g,s in zip(libdai_graphs, max_states)]

        results = {p:(state, score, Z, np.exp(score-Z)) for p,Z,state,score in zip(paths, logZ, max_states, max_scores)}
        with open(osp.join(output_dirname, '%d.pkl'%idx)) as f:
            pkl.dump(results, f)
