#!/bin/bash
#SBATCH --job-name=modalities_run
#SBATCH --account=a0164
#SBATCH --partition=normal
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=11:00:00
#SBATCH --output=/dev/null
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --uenv=pytorch/v2.9.1:/user-environment
#SBATCH --view=default
#SBATCH -C thp_never&nvidia_vboost_enabled

set -x -e
ulimit -c 0

# ============================================================================
# 1. ARRAY SETUP
# ============================================================================
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Error: SLURM_ARRAY_TASK_ID is not set. Submit with --array"
    exit 1
fi

CONFIG_LIST="beatiso.txt"
CONFIG_FILE_PATH=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$CONFIG_LIST")

if [ -z "$CONFIG_FILE_PATH" ]; then
    echo "Error: No config found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# ============================================================================
# 2. PATH CONFIGURATION
# ============================================================================
MODALITIES_DIR="/users/markusfrey/Github/modalities"
EXPERIMENT_ROOT="/capstor/scratch/cscs/markusfrey/experiments"

# ============================================================================
# 3. CSCS SLINGSHOT & NCCL CONFIGURATION
# ============================================================================
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_PROTO=^LL128
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd

export NCCL_DEBUG=WARN               

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export MPICH_GPU_SUPPORT_ENABLED=0
export CUDA_CACHE_DISABLE=1
export TRITON_HOME=/dev/shm/
export TRITON_CACHE_DIR=/dev/shm/.triton_cache_\${SLURM_PROCID}
export OMP_NUM_THREADS=8
export WANDB_MODE=offline

# ============================================================================
# 4. RUN TRAINING
# ============================================================================
echo "=========================================="
echo "Job:       ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Nodes:     ${SLURM_JOB_NUM_NODES}  (${SLURM_JOB_NODELIST})"
echo "Config:    ${CONFIG_FILE_PATH}"
echo "START:     $(date)"
echo "=========================================="

cd "${MODALITIES_DIR}"

# Define these BEFORE the srun command
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(( 20000 + (SLURM_ARRAY_JOB_ID % 10000) + SLURM_ARRAY_TASK_ID ))

# Now inject them into the compute node's subshell
srun -ul bash -c "
    # Activate the venv INSIDE the srun task on the compute node
    source ${MODALITIES_DIR}/.venv/bin/activate
    export PYTHONPATH=${MODALITIES_DIR}/src:\${PYTHONPATH}
    
    # Explicitly set all 5 variables PyTorch needs for env:// rendezvous
    export MASTER_ADDR=${MASTER_ADDR}
    export MASTER_PORT=${MASTER_PORT}
    export RANK=\${SLURM_PROCID}
    export LOCAL_RANK=\${SLURM_LOCALID}
    export WORLD_SIZE=\${SLURM_NTASKS}
    
    python src/modalities/__main__.py run \
        --config_file_path '${CONFIG_FILE_PATH}' \
        --experiments_root_path '${EXPERIMENT_ROOT}'
"

echo "END: $(date)"