#!/bin/bash
#SBATCH --job-name=janus-bundle
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=boost_qos_bprod
#SBATCH --partition=boost_usr_prod
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=128
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.err
#
# Run several 16-node trainings side by side in one allocation. boost_qos_bprod has a
# higher priority than normal but needs >= 65 nodes, so single 16-node runs cannot use
# it; a bundle of 5-16 runs can. Each run gets its own 16 nodes, its own torchrun
# rendezvous and its own log (logs/janus-bundle-<job>_<i>.log). Per run, everything is
# identical to run_array_16nodes.sh (same env, same torchrun call, same experiment root).
# The allocation is held until the longest run ends, so bundle runs of similar length.
# The job asks for all 32 cores per node because boost_qos_bprod also has a minimum of 2080
# CPUs; each training step still gets STEP_CPUS=8 per node, as in run_array_16nodes.sh.
#
#   CONFIG_LIST=<file> \
#       sbatch --nodes=$((16 * N)) config_files/janus/run_bundle_bprod.sh
#
# CONFIG_LIST lines: "<config> [<old job id> [<old chain job ids, comma-separated>]]", where the
# old job is the same run still queued elsewhere (normal QOS) and the chain its warmstart/eval
# jobs. If the old job is already running at start, the config is skipped here. Otherwise the
# old job and its chain stay queued as a fallback until this run has trained CANCEL_AFTER
# seconds (default 5400) and logged step >= MIN_STEP (default 200); only then are they cancelled (PENDING only,
# never a running job). A bundle that dies early therefore leaves the fallback untouched.
# Test on the debug QOS with GROUP_NODES=1 or 2 and EXPERIMENT_ROOT=<test root>.
set -x
MY_ROOT="/leonardo_work/AIFAC_S07_154/mfrey"
REPO_DIR="${MY_ROOT}/modalities"
VENV_DIR="${REPO_DIR}/.venv"
GROUP_NODES="${GROUP_NODES:-16}"
STEP_CPUS="${STEP_CPUS:-8}"
CANCEL_AFTER="${CANCEL_AFTER:-5400}"
MIN_STEP="${MIN_STEP:-200}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-/leonardo_scratch/large/userexternal/mfrey000/experiments_budapest}"
[ -f "${CONFIG_LIST}" ] || { echo "CONFIG_LIST not found: ${CONFIG_LIST}"; exit 1; }
mkdir -p "${EXPERIMENT_ROOT}" "${REPO_DIR}/logs"
cd "${REPO_DIR}"

mapfile -t LINES < <(grep . "${CONFIG_LIST}")
CONFIGS=(); OLDJOBS=(); CHAINS=()
for l in "${LINES[@]}"; do
    CONFIGS+=("$(echo "$l" | awk '{print $1}')"); OLDJOBS+=("$(echo "$l" | awk '{print $2}')")
    CHAINS+=("$(echo "$l" | awk '{print $3}' | tr ',' ' ')")
done
mapfile -t HOSTS < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
N=${#CONFIGS[@]}
if [ "${#HOSTS[@]}" -lt $((N * GROUP_NODES)) ]; then
    echo "need $((N * GROUP_NODES)) nodes for ${N} runs, have ${#HOSTS[@]}"; exit 1
fi

module purge
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export UCX_RC_TIMEOUT=4s
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_IB_RETRY_CNT=14
export NCCL_IB_HCA=mlx5
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export JANUS_EVAL_ALL_STEPS="${JANUS_EVAL_ALL_STEPS:-0}"
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
export REPO_DIR VENV_DIR EXPERIMENT_ROOT GPUS_PER_NODE GROUP_NODES JANUS_EVAL_ALL_STEPS

# Resources a plain `srun` gets in run_array_16nodes.sh, for comparison with the steps below.
srun --nodes=1 --ntasks=1 --cpus-per-task="${STEP_CPUS}" --nodelist="${HOSTS[0]}" bash -c 'echo "probe(plain srun): $(hostname) nproc=$(nproc) gpus=$(nvidia-smi -L | wc -l)"'

PIDS=(); RUNIDX=()
for i in $(seq 0 $((N - 1))); do
    CFG="${CONFIGS[$i]}"
    [[ "${CFG}" != /* ]] && CFG="${REPO_DIR}/${CFG}"
    OLD="${OLDJOBS[$i]}"
    if [ -n "${OLD}" ]; then
        if [ -n "$(squeue -h -t R -j "${OLD}" 2>/dev/null)" ]; then
            echo "skip $(basename "${CFG}"): already running as job ${OLD}; its chain is kept"; continue
        fi
    fi
    GROUP=("${HOSTS[@]:$((i * GROUP_NODES)):${GROUP_NODES}}")
    NODELIST=$(IFS=,; echo "${GROUP[*]}")
    echo "run $((i + 1)): ${CFG} on ${NODELIST}"
    srun --nodes="${GROUP_NODES}" --ntasks="${GROUP_NODES}" --ntasks-per-node=1 \
         --cpus-per-task="${STEP_CPUS}" --gpus-per-node="${GPUS_PER_NODE}" \
         --nodelist="${NODELIST}" \
         --output="logs/janus-bundle-${SLURM_JOB_ID}_$((i + 1)).log" \
         --error="logs/janus-bundle-${SLURM_JOB_ID}_$((i + 1)).err" \
         --export=ALL,CONFIG_FILE_PATH="${CFG}",MASTER_ADDR="${GROUP[0]}",MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000 + i)) \
         bash -c '
  [ "${SLURM_PROCID}" = 0 ] && echo "probe(step): $(hostname) nproc=$(nproc) gpus=$(nvidia-smi -L | wc -l) config=${CONFIG_FILE_PATH}"
  source "${VENV_DIR}/bin/activate"
  cd "${REPO_DIR}"
  export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
  torchrun \
      --node_rank="${SLURM_PROCID}" \
      --nnodes="${GROUP_NODES}" \
      --nproc_per_node="${GPUS_PER_NODE}" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
      "$(which modalities)" run \
      --config_file_path "${CONFIG_FILE_PATH}" \
      --experiments_root_path "${EXPERIMENT_ROOT}"
' &
    PIDS+=($!); RUNIDX+=($i)
    sleep 5   # distinct experiment-id timestamps
done

# Cancel the queued duplicates of runs that are healthy after CANCEL_AFTER seconds.
cancel_duplicates() {
    sleep "${CANCEL_AFTER}"
    for k in "${!PIDS[@]}"; do
        i=${RUNIDX[$k]}; OLD="${OLDJOBS[$i]}"
        [ -z "${OLD}" ] && continue
        LOG="logs/janus-bundle-${SLURM_JOB_ID}_$((i + 1)).log"
        STEP=$(grep -oE "step: [0-9]+ \|" "${LOG}" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
        if kill -0 "${PIDS[$k]}" 2>/dev/null && [ "${STEP:-0}" -ge "${MIN_STEP}" ]; then
            if [ -n "$(squeue -h -t R -j "${OLD}" 2>/dev/null)" ]; then
                echo "$(date) run $((i + 1)) healthy (step ${STEP}), but ${OLD} is running too: left alone"
            else
                scancel --state=PENDING "${OLD}" ${CHAINS[$i]}
                echo "$(date) run $((i + 1)) healthy (step ${STEP}): cancelled queued ${OLD} and chain ${CHAINS[$i]:-none}"
            fi
        else
            echo "$(date) run $((i + 1)) not healthy (step ${STEP:-none}): fallback ${OLD} kept"
        fi
    done
}
cancel_duplicates &
CANCELLER=$!

RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=1; done
kill "${CANCELLER}" 2>/dev/null
echo "END: $(date) rc=${RC}"
exit ${RC}
