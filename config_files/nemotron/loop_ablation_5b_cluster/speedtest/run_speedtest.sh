#!/bin/bash
#SBATCH --job-name=nemo_speedtest
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/speedtest-%j.out
#SBATCH --error=logs/speedtest-%j.err

# Measures A2_loop_moe (slowest arm) wall-clock throughput at dp_degree 4 (1 node) and dp_degree 16
# (4 nodes), pure-replicate FSDP2 (data_parallel_replicate_degree = dp, shard_degree = 1), global
# batch held at 65,536 tokens/step throughout. Picks the node count for the 5B-token wave -- see
# docs/components/nemotron_loops_research_plan.md section 9.3 and ../run_wave_5b.sh.
#
# Bare-metal (no singularity): the shared container image's mamba_ssm was ABI-incompatible with
# its own torch build (undefined symbol at import, on any backend). Rebuilt mamba-ssm/causal-conv1d
# straight into /leonardo_work/EUHPC_D21_101/mfrey/modalities/.venv instead -- see the note in
# docs/components/nemotron_loops_research_plan.md section 9 (added 2026-08-11).
#
# Submit from the repository root:  sbatch config_files/nemotron/loop_ablation_5b_cluster/speedtest/run_speedtest.sh

set -x -e

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
RESULTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/nemo_speedtest"
mkdir -p "${RESULTS_ROOT}" "${REPO}/logs"

module purge
module load cuda/12.6 gcc/12.2.0
export CC=$(which gcc)
export CXX=$(which g++)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=14
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO}/src:${PYTHONPATH}"

ALL_NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
echo "allocated nodes: ${ALL_NODES[*]}"

run_variant () {
    local NAME=$1
    local NNODES=$2
    local CONFIG=$3
    local NODELIST_STR
    NODELIST_STR=$(IFS=,; echo "${ALL_NODES[*]:0:${NNODES}}")
    local MASTER_ADDR=${ALL_NODES[0]}
    local MASTER_PORT=$((29500 + RANDOM % 1000))
    local EXP_ROOT="${RESULTS_ROOT}"

    echo "=========================================="
    echo "variant=${NAME} nnodes=${NNODES} nodelist=${NODELIST_STR} config=${CONFIG}"
    echo "START: $(date +%s) ($(date))"
    echo "=========================================="

    srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
        --nodelist="${NODELIST_STR}" \
        bash -c "
            ${REPO}/.venv/bin/torchrun \
                --node_rank=\${SLURM_PROCID} \
                --nnodes=${NNODES} \
                --nproc_per_node=4 \
                --rdzv_backend=c10d \
                --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
                ${REPO}/.venv/bin/modalities run \
                --config_file_path '${CONFIG}' \
                --experiments_root_path '${EXP_ROOT}' \
                --experiment_id '${NAME}'
        "

    echo "END: $(date +%s) ($(date))"
}

cd "${REPO}"
run_variant dp4_replicate 1 "${REPO}/config_files/nemotron/loop_ablation_5b_cluster/speedtest/dp4_replicate.yaml"
run_variant dp16_replicate 4 "${REPO}/config_files/nemotron/loop_ablation_5b_cluster/speedtest/dp16_replicate.yaml"

echo "=== SPEEDTEST FINISHED ==="
