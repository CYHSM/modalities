#!/bin/bash
#SBATCH --job-name=nemo_tying
#SBATCH --account=AIFAC_S07_154
#SBATCH --exclusive
#SBATCH --qos=boost_qos_dbg
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --output=logs/tying_test-%j.out
#SBATCH --error=logs/tying_test-%j.err

# Pins WHERE `use_weight_tying: true` stops holding, and whether a warmstart load re-establishes it.
#
# Motivation (2026-08-14): the 5B-wave checkpoints show transformer.wte.weight and
# transformer.lm_head.weight bit-identical after `modalities warmstart` but genuinely different
# after `modalities run` (44% relative RMS apart, cosine 0.913, zero identical elements), and the
# two paths reach measurably different quality (final train loss 2.76 vs 2.70, grad norm 0.38 vs
# 0.48 on A4). NemotronLLM ties the two by Python identity, and named_parameters() deduplicates a
# tied tensor -- which is why the optimizer tracks it once under the `wte` name with no Adam state
# for lm_head.weight in EITHER path. So the separation happens between construction and saving, and
# checkpoint forensics alone cannot say where. This runs the real component graph and reports `is`
# identity at each stage.
#
# Phases (all on the 4-GPU layout training uses -- FSDP2 needs a real process group, and the whole
# question is whether FSDP2's DTensor conversion is what breaks the identity):
#   1. stage-by-stage identity on the RUN config      (model_raw -> AC -> FSDP2 -> initialized)
#   2. `modalities run` for 10 steps, writing one checkpoint
#   3. inspect that checkpoint's two tensors on disk
#   4. stage-by-stage identity on the WARMSTART config, PLUS identity after the checkpoint is
#      actually loaded into app_state -- the decisive one for "does warmstart re-tie?"
#
# Submit from the repository root:  sbatch config_files/nemotron/loop_ablation_5b_cluster/speedtest/run_tying_test.sh

set -x

REPO="/leonardo_work/EUHPC_D21_101/mfrey/modalities"
RESULTS_ROOT="/leonardo_scratch/large/userexternal/mfrey000/nemo_tying_test"
CONFIG_DIR="${REPO}/config_files/nemotron/loop_ablation_5b_cluster/speedtest"
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

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

launch () {  # launch <port> <script-or-entrypoint> <args...>
    local PORT=$1; shift
    srun --nodes=1 --ntasks=1 --ntasks-per-node=1 bash -c "
        ${REPO}/.venv/bin/torchrun \
            --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${PORT} \
            --nnodes=1 --nproc_per_node=4 $*
    "
}

cd "${REPO}"

echo "=== [1/4] stage-by-stage tie identity, RUN config ==="
launch 29801 "${CONFIG_DIR}/inspect_weight_tying.py" \
    --config_file_path "${CONFIG_DIR}/warmstart_test_train.yaml" \
    --experiments_root_path "${RESULTS_ROOT}"

echo "=== [2/4] modalities run, 10 steps -> one checkpoint ==="
launch 29802 "${REPO}/.venv/bin/modalities run" \
    --config_file_path "${CONFIG_DIR}/warmstart_test_train.yaml" \
    --experiments_root_path "${RESULTS_ROOT}"

LAST_CKPT="${RESULTS_ROOT}/A0_baseline/checkpoints/last_checkpoint_info.json"
if [[ ! -f "${LAST_CKPT}" ]]; then
    echo "FAIL: no checkpoint written at ${LAST_CKPT}" >&2
    exit 1
fi
cat "${LAST_CKPT}"

echo "=== [3/4] the two tensors as written to disk by the RUN path ==="
"${REPO}/.venv/bin/python" - "${LAST_CKPT}" <<'PY'
import json, sys, torch, torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner, _load_state_dict
p = json.load(open(sys.argv[1]))["checkpoint_folder_path"]
want = {"app.model.transformer.wte.weight", "app.model.transformer.lm_head.weight"}
sd = {}
_load_state_dict(sd, storage_reader=dcp.FileSystemReader(p), planner=_EmptyStateDictLoadPlanner(keys=want), no_dist=True)
m = sd["app"]["model"]
w, l = m["transformer.wte.weight"].float(), m["transformer.lm_head.weight"].float()
d = w - l
print(f"  bit-identical            : {torch.equal(w, l)}")
print(f"  fraction elements equal  : {(d == 0).float().mean():.6f}")
print(f"  relative RMS difference  : {(d.pow(2).mean().sqrt() / w.pow(2).mean().sqrt()):.6e}")
print(f"  cosine similarity        : {torch.nn.functional.cosine_similarity(w.flatten(), l.flatten(), dim=0):.8f}")
ks = list(dcp.FileSystemReader(p).read_metadata().state_dict_metadata.keys())
print(f"  lm_head has Adam state   : {any('lm_head.weight.exp_avg' in k for k in ks)}")
print(f"  wte has Adam state       : {any('wte.weight.exp_avg' in k for k in ks)}")
PY

echo "=== [4/4] stage-by-stage + POST-LOAD identity, WARMSTART config ==="
launch 29803 "${CONFIG_DIR}/inspect_weight_tying.py" \
    --config_file_path "${CONFIG_DIR}/warmstart_test_resume.yaml" \
    --experiments_root_path "${RESULTS_ROOT}" \
    --last_checkpoint_info_file_path "${LAST_CKPT}"

echo "=== TYING TEST FINISHED ==="
