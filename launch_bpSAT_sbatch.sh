#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
####SBATCH --cpus-per-task=1
#SBATCH --mem=8G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:1
#SBATCH --exclude=atlas19,atlas20,atlas21,atlas22,atlas3,atlas8,atlas6,atlas7,atlas5,atlas12,atlas13,atlas1,atlas4

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

# python learn_BP_SAT.py --max_factor_state_dimensions $1 --msg_passing_iters $2 --belief_repeats $3 --random_seed $4 --problem_category_train $5 --train_val_split $6 --batch_size $7

# echo $1
# echo $2
# echo $3
# echo $4
# echo $5
# echo $6
# echo $7
# echo $8
# echo $9
# echo ${10}
# echo ${11}
# echo ${12}
# echo ${13}
# python learn_BP_SAT.py --max_factor_state_dimensions $1 --msg_passing_iters $2 --belief_repeats $3 --random_seed $4 --problem_category_train $5 --train_val_split $6 --batch_size $7 --lne_mlp $8 --alpha_damping_FtoV $9 --alpha_damping_VtoF ${10} --use_MLP1 ${11} --use_MLP2 ${11} --use_MLP3 ${12} --use_MLP4 ${12} --bethe_mlp ${13} --SHARE_WEIGHTS ${14}

python learn_BP_SAT.py --problem_category_train $1 #--subtract_prv_messages $2 --train_val_split $3 --USE_MLP_DAMPING_FtoV $4 --bethe_mlp $5 --learning_rate $6 --factor_graph_representation_invariant $7
# python learn_BP_SAT.py


# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# run this file with e.g.:
# sbatch launch_bpSAT_sbatch.sh 5 5 2 1 or_50_problems random_shuffle

# done
echo "Done"
