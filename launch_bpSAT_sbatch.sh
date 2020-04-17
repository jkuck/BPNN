#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# only use the following on partition with GPUs


#SBATCH --job-name="learnBP_SAT"
#SBATCH --output=slurm-%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
NPROCS=`srun --nodes=${SLURM_NNODES} bash -c 'hostname' |wc -l`
echo NPROCS=$NPROCS

cd /atlas/u/jkuck/learn_BP
echo 'hi'
source /sailhome/jkuck/miniconda2/etc/profile.d/conda.sh
conda activate /atlas/u/jkuck/learn_BP/venv35

python learn_BP_SAT.py --max_factor_state_dimensions $1 --msg_passing_iters $2 --belief_repeats $3 --random_seed $4 --problem_category_train $5 --train_val_split $6 --batch_size $7

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# run this file with e.g.:
# sbatch launch_bpSAT_sbatch.sh 5 5 2 1 or_50_problems random_shuffle

# done
echo "Done"
