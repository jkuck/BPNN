#!/bin/bash

import random, subprocess as sp

PROCESS_NUM = 20

jobs = []
for updates in ['SEQFIX', 'SEQRND', 'SEQMAX', 'PARALL']:
    for damping in [0.0, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999]:
        for mflag in ['--model_map_flag', '']:
            for aflag in ['--attractive_flag', '']:
                for miter in [10000, 5, 10, 20]:
                    jobs.append('python bp_map_marginal_spinglass.py %s %s --updates %s --damping %f --maxiter %d'%(aflag, mflag, updates, damping, miter))

random.shuffle(jobs)

start_index = lambda i,n,d: min(int((n*i)/d), n)
indexes = [start_index(i,len(jobs),PROCESS_NUM) for i in range(PROCESS_NUM+1)]
for i in range(PROCESS_NUM):
    fname = 'bp_marginal_jobs_%d.sh'%i
    with open(fname, 'w') as f:
        cur_jobs = jobs[indexes[i]:indexes[i+1]]
        f.write('\n'.join(cur_jobs))
    sp.call(['python', 'run_cpu.py', 'python', 'batch/run_scripts.py', fname])
