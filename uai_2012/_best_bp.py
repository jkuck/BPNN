#!/usr/bin/env python
# coding=utf-8

from uai2012_models import _UAI2012Dataset
from libdai_utils import build_libdaiFactorGraph_from_UAI2012Model, logScore, run_map_loopyBP

import sys, os, os.path as osp, pickle as pkl, numpy as np

def best_bp(fg):
    bp_states = [
        run_map_loopyBP(fg, map_flag=True, updates=up, damping=dp)
        for up in ['SEQMAX', 'PARALL', 'SEQRND']
        for dp in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
    ]
    bp_logscores = [logScore(fg, state) for state in bp_states]
    bp_index = np.argmax(bp_logscores)
    return bp_states[bp_index]


if __name__ == '__main__':
    curdir = osp.dirname(osp.abspath(__file__))
    output_dirname = osp.join(curdir, 'BP_MAP_results')

    indexes = [int(idx) for idx in sys.argv[1:]]
    for idx in indexes:
        dataset, paths = _UAI2012Dataset(idx)
        print(idx, 'Building Libdai Graphs...')
        libdai_graphs = [build_libdaiFactorGraph_from_UAI2012Model(data) for data in dataset]
        print(idx, 'Performing BP...')
        max_states = [best_bp(g) for g in libdai_graphs]

        results = {p:state for p,state in zip(paths, max_states)}
        with open(osp.join(output_dirname, '%d.pkl'%idx), 'wb') as f:
            pkl.dump(results, f)
