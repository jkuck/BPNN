#!/usr/bin/env python
# coding=utf-8

import sys, subprocess

args =sys.argv[1:]
command = ' '.join(args)
name = ''.join([a.strip().strip('-')[0] for a in args])

context = '''#!/bin/bash
#SBATCH --partition=atlas
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

#SBATCH --job-name="%s"
#SBATCH --output=%s-%%j.out

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# source /sailhome/htang/anaconda3/etc/profile.d/conda.sh
# conda activate
bash
source /sailhome/htang/.bashrc
cd /atlas/u/htang/learn_BP/
pip show torch_geometric
%s

# done
echo "Done"'''%(name, name, command)

filename = 'tmp.sh'
with open(filename, 'w') as f:
    f.write(context)
subprocess.call(['sbatch', 'tmp.sh'])
