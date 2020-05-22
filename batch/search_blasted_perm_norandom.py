#!/bin/bash

import random, subprocess as sp

PROCESS_NUM = 25

jobs = [
    'python learn_bp_sat.py --problem_category_train blasted_problems \
    --batch_size 10 --use_MLP1 True --use_MLP2 True --use_MLP3 False --use_MLP4 False\
    --perm_invariant_flag --sample_perm_number %d --random_flag False\
    %s --SHARE_WEIGHTS %s --bethe_mlp %s --learning_rate %f \
    --alpha_damping_FtoV %f --alpha_damping_VtoF %f --msg_passing_iters %d'%(
        perm_num, dflag, sflag, bflag,
        10**(random.random()*4-6), damping, damping, layer_num,
    )
    for perm_num in [1, 5, 10, 15]
    for dflag in ['--lr_decay_flag', '',]
    for sflag in ['True', 'False',]
    for bflag in ['shifted', 'standard', 'linear',]
    for _ in range(10) #slots for learing rate
    for damping in [1., 0.999, 0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]
    for layer_num in [5, 10, 20, 30, 50, 100, 200, 500]
]

random.shuffle(jobs)

start_index = lambda i,n,d: min(int((n*i)/d), n)
indexes = [start_index(i,len(jobs),PROCESS_NUM) for i in range(PROCESS_NUM+1)]
for i in range(PROCESS_NUM):
    fname = 'sat_jobs_%d.sh'%i
    with open(fname, 'w') as f:
        cur_jobs = jobs[indexes[i]:indexes[i+1]]
        f.write('\n'.join(cur_jobs))
    sp.call(['python', 'run_cpu.py', 'python', 'batch/run_scripts.py', fname])
