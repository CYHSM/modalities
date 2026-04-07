#!/bin/bash
# SLURM SUBMIT SCRIPT
#SBATCH --exclusive
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --job-name=B1_B2_B3-L2_B4
#SBATCH --output=/leonardo_scratch/large/userexternal/mali0000/curriculum_loop/logs/1B/1B_base_B1_B2_B3-L2_B4_%j.out
#SBATCH --error=/leonardo_scratch/large/userexternal/mali0000/curriculum_loop/logs/1B/1B_base_B1_B2_B3-L2_B4_%j.err
#SBATCH --time=23:59:00
#SBATCH --ntasks-per-node=1 
#SBATCH --nodes=16
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0       

#### Environment variables ####
export CXX=g++
export CC=gcc

# force crashing on nccl issues like hanging broadcast
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=10



set -x -e
echo "START TIME: $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export WANDB_ROOT=/leonardo_scratch/large/userexternal/mali0000/curriculum_loop/wandb/1B

export WANDB_DIR=$WANDB_ROOT/runs
export WANDB_CACHE_DIR=$WANDB_ROOT/cache
export WANDB_CONFIG_DIR=$WANDB_ROOT/config


source /leonardo_scratch/large/userexternal/mali0000/curriculum_loop/virtual_envs/looped_model_torch_2_8/bin/activate

##### Network parameters #####
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))

echo "START TIME: $(date)"

srun torchrun   --node_rank=$SLURM_PROCID   --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT    --nnodes $SLURM_JOB_NUM_NODES   --nproc_per_node 4   --rdzv_backend c10d    $(which modalities) run   --config_file_path /leonardo_scratch/large/userexternal/mali0000/curriculum_loop/git_repos/looped-model_experiments/leonardo/configs/1B/B3_Loop/bucket_B1_B2_B3-L2_B4/1B_base.yml

echo "END TIME: $(date)"
echo "=== FINISHED ==
 