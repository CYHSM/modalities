#!/bin/bash

# ============================================================================
# Leonardo SLURM Job Script
# Purpose: Run NCCL tests followed by Llama 8B pretraining with Modalities
# ============================================================================

#SBATCH --job-name=llama3B
#SBATCH --account=euhpc_e05_119
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_bprod #normal boost_qos_bprod boost_qos_dbg
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=23:58:00
#SBATCH --output=../experiments/modalities/%x-%j/outputs/%x-%j.out
#SBATCH --error=../experiments/modalities/%x-%j/outputs/%x-%j.err
#SBATCH --exclusive

set -eux
# Load environment variables from .env file
. ../.env

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

module purge


# Define base paths
EXPERIMENT_ROOT="/leonardo_scratch/large/userexternal/mluebber/experiments/modalities-${SLURM_JOB_NAME}-${SLURM_JOB_ID}" # "$SLURM_SUBMIT_DIR/../../training_runs/modalities-${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
last_checkpoint_info_file_path="/leonardo_work/EUHPC_E05_119/mluebber/experiments/modalities-llama8B-34720625/train/checkpoints/last_checkpoint_info.json"
NCCL_TESTS_CONTAINER_EXPERIMENT_ROOT="${EXPERIMENT_ROOT}/nccl_tests"
TRAIN_EXPERIMENT_ROOT="${EXPERIMENT_ROOT}/train"

# mkdir -p "${NCCL_TESTS_CONTAINER_EXPERIMENT_ROOT}"
# mkdir -p "${TRAIN_EXPERIMENT_ROOT}"

# Container paths
CONTAINER_IMAGE="/leonardo_scratch/fast/EUHPC_D21_101/max_lue/repositories/working/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"

# NCCL test configuration
NCCL_TESTS_CONTAINER_TIME_LIMIT="00:10:00"
NUM_RANKS=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))

# NCCL environment variables
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH
# export NCCL_IB_RETURN_ASYNC_EVENTS=1
# unset NCCL_DEBUG
# unset NCCL_DEBUG_SUBSYS
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=GRAPH,INIT,COLL




# Modalities (environment) variables
export WANDB_MODE=offline
modalities_config_file="configs/soofi_llama3_8b_warmstart.yaml"

# cluster env vars
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-${SLURM_NTASKS_PER_NODE}}

# Create experiment directory
mkdir -p "${EXPERIMENT_ROOT}"

# ============================================================================
# NCCL TESTS (inside Modalities container)
# ============================================================================

echo "=========================================="
echo "Starting NCCL Tests"
echo "=========================================="

TEST_NAME="all_reduce_perf"
TEST_CMD="/nccl-tests/build/${TEST_NAME}"
TEST_ARGS="-b 8 -e 8G -f 2 -g 1 -d float"

NCCL_TEST_OUT_LOG_FILE_PATH="${NCCL_TESTS_CONTAINER_EXPERIMENT_ROOT}/${TEST_NAME}_N${SLURM_JOB_NUM_NODES}n${NUM_RANKS}_container.out"
NCCL_TEST_ERR_LOG_FILE_PATH="${NCCL_TESTS_CONTAINER_EXPERIMENT_ROOT}/${TEST_NAME}_N${SLURM_JOB_NUM_NODES}n${NUM_RANKS}_container.err"

mkdir -p "${NCCL_TESTS_CONTAINER_EXPERIMENT_ROOT}"

# srun --time=$NCCL_TESTS_CONTAINER_TIME_LIMIT \
#      --output="${NCCL_TEST_OUT_LOG_FILE_PATH}" \
#      --error="${NCCL_TEST_ERR_LOG_FILE_PATH}" \
#      --mpi=pmix_v3 \
#      --cpu-bind=cores \
#      singularity exec --nv --bind "${CONTAINER_BIND_SRC}:${CONTAINER_BIND_DST}" \
#      --bind "${DATA_BIND_SRC}:${DATA_BIND_DST}" \
#      "${CONTAINER_IMAGE}" ${TEST_CMD} ${TEST_ARGS}

echo "✅ COMPLETED NCCL Container TEST ${TEST_NAME}"

# # ============================================================================
# # LLAMA 3B PRETRAINING
# # ============================================================================

echo "=========================================="
echo "Starting Soofi Llama 3B Pretraining"
echo "=========================================="
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
export MASTER_PORT=$(( 10000 + (SLURM_JOB_ID % 50000) + (RANDOM % 1000) ))
RDZV_ID="${SLURM_JOB_ID}"
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NODES=${SLURM_JOB_NUM_NODES} GPUS_PER_NODE=${GPUS_PER_NODE} NUM_RANKS=${NUM_RANKS}"


srun -N "${SLURM_JOB_NUM_NODES}" -n "${SLURM_JOB_NUM_NODES}" \
     --ntasks-per-node=1 \
     --gpus-per-node=4 \
     --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
     --cpu-bind=cores \
     --gpu-bind=closest \
     --kill-on-bad-exit=1 \
     --output="${TRAIN_EXPERIMENT_ROOT}/logs/${SLURM_JOB_ID}.out" \
     --error="${TRAIN_EXPERIMENT_ROOT}/logs/${SLURM_JOB_ID}.err" \
     singularity exec --nv --bind "${CONTAINER_BIND_SRC}:${CONTAINER_BIND_DST}" \
     --bind "${DATA_BIND_SRC}:${DATA_BIND_DST}" \
     --bind "${LARGE_SCRATCH_ROOT}:${LARGE_SCRATCH_ROOT}" \
     "${CONTAINER_IMAGE}" bash -lc "
       set -eo pipefail
       export NCCL_GRAPH_DUMP_FILE=${TRAIN_EXPERIMENT_ROOT}/logs/nccl_graph_rank_${SLURM_PROCID}.txt
       export NCCL_TOPO_DUMP_FILE=${TRAIN_EXPERIMENT_ROOT}/logs/nccl_topo_rank_${SLURM_PROCID}.xml
       torchrun \
         --rdzv-endpoint ${MASTER_ADDR}:${MASTER_PORT} \
         --nnodes ${SLURM_JOB_NUM_NODES} \
         --nproc_per_node 4 \
         --node_rank ${SLURM_NODEID} \
         --rdzv_id '${RDZV_ID}' \
         --rdzv_backend c10d \
         --max_restarts 0 \
         --module modalities warmstart \
         --config_file_path '${modalities_config_file}' \
         --last_checkpoint_info_file_path '${last_checkpoint_info_file_path}' \
         --experiments_root_path '${TRAIN_EXPERIMENT_ROOT}' \
         --error_log_folder '${TRAIN_EXPERIMENT_ROOT}'
     "


echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
