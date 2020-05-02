#!/bin/bash

import random, subprocess as sp

PROCESS_NUM = 25

jobs = [
    'python learn_bp_map_spinglass.py --no_training_flag %s %s %s %s --alpha %f --layer_num %d'%(
        dflag, aflag, mflag, bflag,
        damping, layer_num,
    )
    for dflag in ['--lr_decay_flag', '']
    for aflag in ['--no_attractive_flag', '']
    for mflag in ['--model_map_flag', '']
    for bflag in ['--bethe_flag', '']
    for damping in [0.999, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]
    for layer_num in [5, 10, 20, 30, 50, 100, 200, 500]
]

random.shuffle(jobs)

start_index = lambda i,n,d: min(int((n*i)/d), n)
indexes = [start_index(i,len(jobs),PROCESS_NUM) for i in range(PROCESS_NUM+1)]
for i in range(PROCESS_NUM):
    fname = 'bpnn_logz_notrain_jobs_%d.sh'%i
    with open(fname, 'w') as f:
        cur_jobs = jobs[indexes[i]:indexes[i+1]]
        f.write('\n'.join(cur_jobs))
    sp.call(['python', 'run_cpu.py', 'python', 'batch/run_scripts.py', fname])
