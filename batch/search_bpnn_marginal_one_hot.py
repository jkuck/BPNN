#!/bin/bash

import random, subprocess as sp

PROCESS_NUM = 40

jobs = [
    'python learn_bp_map_marginal_spinglass.py %s %s %s %s %s %s %s %s -lr %f --alpha %f --layer_num %d --one_hot_ratio %s'%(
        dflag, aflag, mflag, cflag, sflag, bflag, tflag, lname,
        10**(random.random()*5-7), damping, layer_num, one_hot_ratio
    )
    for dflag in ['']
    for aflag in ['']
    for mflag in ['--model_map_flag', '']
    for cflag in ['--classification_flag']
    for lname in ['--loss_name one_hot_cross_entropy']
    for sflag in ['--share_weights_flag', '']
    for bflag in ['--bethe_flag', '']
    for tflag in ['']
    for _ in range(10) #slots for learing rate
    for damping in [0.999, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]
    for layer_num in [5, 10, 20, 30, 50, 100, 500]
    for one_hot_ratio in [0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01]
]

random.shuffle(jobs)

start_index = lambda i,n,d: min(int((n*i)/d), n)
indexes = [start_index(i,len(jobs),PROCESS_NUM) for i in range(PROCESS_NUM+1)]
for i in range(PROCESS_NUM):
    fname = 'bpnn_marginal_one_hot_jobs_%d.sh'%i
    with open(fname, 'w') as f:
        cur_jobs = jobs[indexes[i]:indexes[i+1]]
        f.write('\n'.join(cur_jobs))
    sp.call(['python', 'run_cpu.py', 'python', 'batch/run_scripts.py', fname])
