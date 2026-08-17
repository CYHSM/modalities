#!/bin/bash
#SBATCH --job-name=mn_probe
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/mn_probe-%j.out
#SBATCH --error=logs/mn_probe-%j.err
#
# Measures K=12 throughput at 1, 2 and 4 nodes. This decides where the K=12 refinement arms can run.
#
# The constraint. K=12 executes 67 layers per token and measures 2,632 steps/h on one node, so the
# 76,250-step budget needs ~29h. Partition boost_usr_prod has MaxTime=1-00:00:00, so the `normal`
# QOS cannot run a 29h job at ANY node count -- the only single-node option is boost_qos_lprod
# (4-day wall), whose 8-node/32-GPU cap is shared across the whole project account and is already
# largely consumed by other users. Scaling to more nodes is the alternative: 2 nodes only needs to
# reach 76,250/24 = 3,177 steps/h to fit `normal`, i.e. a 1.21x speedup over one node.
#
# Whether that speedup exists is genuinely unknown here. The Wave 2 base config records dp_degree 16
# (4 nodes) measuring ~38 samples/s against 58-84 on one node and warns not to raise dp_degree
# without re-measuring -- but that was for MoE-heavy arms at K=3. Two things differ at K=12:
# gradient all-reduce volume is set by PARAMETER count (1.025B, identical at every K) while compute
# per step scales with executed depth, so there is ~3x more compute hiding the same communication;
# and this arm executes 60 Mamba layers to 5 MoE, so the thin-routing-batch term barely applies.
#
# The cost of scaling out is comparability: world size sets the resumable sampler's order, so
# multi-node K=12 arms would not share Wave 2's data stream. They would still be internally
# consistent (all five at the same node count), which is the comparison that matters for "which
# refinement helps at K=12" -- but the K=6 -> K=12 depth comparison would then confound depth with
# data order.
#
# The global batch is held at 65,536 tokens/step in every configuration; only its split across GPUs
# changes. All three run inside ONE 4-node allocation so they see identical hardware.
#
# Configs are read from an immutable SNAPSHOT on SHARED storage. Two earlier attempts at this probe
# died: one read the wave's config directory while the generator was re-running it, the other put
# the snapshot on /scratch_local, which is node-local and therefore invisible to the compute nodes.
#
# Submit from the repository root:
#   sbatch config_files/nemotron/loop_refinements/run_multinode_probe.sh

set -x

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
SCRATCH="/leonardo_scratch/large/userexternal/mfrey000/mn_probe_configs"
EXPERIMENTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/experiments_nemotron_mn_probe"
mkdir -p "${EXPERIMENTS_ROOT}" "${REPO}/logs"

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

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
PORT=29900
WINDOW=420   # ~2 min build + several steady-state log lines (training_log_interval_in_steps is 5)

cd "${REPO}"

for NODES in 1 2 4; do
    CONFIG="${SCRATCH}/config_mn_${NODES}node.yaml"
    if [ ! -f "${CONFIG}" ]; then
        echo "Error: missing snapshot ${CONFIG}. Regenerate it before submitting." >&2
        exit 1
    fi
    echo "=============================================================="
    echo "MULTINODE PROBE: ${NODES} node(s), dp_degree=$((4 * NODES))   $(date)"
    echo "=============================================================="
    PORT=$((PORT + 1))
    # --input none: srun otherwise consumes this shell's stdin and the loop runs only once.
    srun --input none --nodes="${NODES}" --ntasks="${NODES}" --ntasks-per-node=1 bash -c "
        timeout --signal=INT --kill-after=30 ${WINDOW} \
        ${REPO}/.venv/bin/torchrun \
            --node_rank=\${SLURM_PROCID} \
            --nnodes=${NODES} \
            --nproc_per_node=4 \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${PORT} \
            ${REPO}/.venv/bin/modalities run \
                --config_file_path ${CONFIG} \
                --experiments_root_path ${EXPERIMENTS_ROOT}
    "
    echo "--- ${NODES} node(s) done (timeout exit 124 is expected) ---"
done

echo "=============================================================="
echo "PROBE RESULTS  (a step is 32 samples at every node count; 5B budget = 76,250 steps)"
echo "   fits 24h 'normal' if >= 3,177 steps/h"
echo "=============================================================="
awk '/^MULTINODE PROBE:/{n=$3}
     /train samples\/s:/{c=split($0,f,/ *\| */); for(i=1;i<=c;i++){split(f[i],kv,": "); if(kv[1]=="train samples/s")s=kv[2]}; last[n]=s}
     END{for(k in last){h=last[k]/32*3600; printf "  %s node(s): %6.1f samples/s -> %6.0f steps/h -> %5.1f h   %s\n", k, last[k], h, 76250/h, (h>=3177 ? "FITS normal" : "needs lprod")}}' \
    "${REPO}/logs/mn_probe-${SLURM_JOB_ID}.out" | sort
echo "=== MULTINODE PROBE FINISHED ==="
