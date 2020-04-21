#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G


#SBATCH --job-name="learnBP_SAT"
#SBATCH --output=slurm-%j.out

# only use the following on partition with GPUs

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
echo $1
echo $2
echo $3
echo $4
echo $5

source /sailhome/jkuck/miniconda2/etc/profile.d/conda.sh
conda activate /atlas/u/jkuck/learn_BP/venv35

python learn_BP_spinGlass.py --use_MLP1 $1 --use_MLP2 $1 --use_MLP3 $2 --use_MLP4 $2 --SHARE_WEIGHTS $3 --subtract_prv_messages $4 --bethe_mlp $5

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"
