#!/bin/bash
#SBATCH --job-name=review-abl
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%A_%a.log
#SBATCH --error=logs/%x-%A_%a.err
#
# Review-round inference interventions on the 18 checkpoints behind Table 4 (one array task each):
#   ratio_shuffle / ratio_mean   change g_d:g_w per token, keep g_d+g_w     (GkkM W2/Q2, YXfZ W2)
#   sum_shuffle / sum_mean       change g_d+g_w, keep the ratio             (converse)
#   *_mask_...+RANDOMMATCHED     random control matched in perturbation size (YXfZ Q3b)
# plus the original learned / class / RANDOM runs if their result file is missing.
#
#   sbatch --array=0-17 scripts/analysis/revision_analysis/run_review_ablations.sh
set -u

REPO_DIR="/leonardo_work/AIFAC_S07_154/mfrey/modalities"
cd "${REPO_DIR}"
source .venv/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

BASE_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
NAMES=(80M_alpha25_K2 80M_alpha50_K2 80M_alpha75_K2 80M_alpha25_K3 80M_alpha50_K3 80M_alpha75_K3
       80M_alpha25_K4 80M_alpha50_K4 80M_alpha75_K4 160M_alpha25_K2 160M_alpha50_K2 160M_alpha75_K2
       160M_alpha25_K3 160M_alpha50_K3 160M_alpha75_K3 160M_alpha25_K4 160M_alpha50_K4 160M_alpha75_K4)
DIRS=(2026-05-18__13-49-25_46661c3a0490b08d 2026-05-13__02-05-13_fa51525556e2cd3b 2026-05-18__19-28-55_0d6659323c9236d8
      2026-05-18__12-12-23_207cde6cd1273411 2026-05-13__02-21-25_03070170b2aa73fa 2026-05-18__19-28-58_635662eb51947f62
      2026-05-18__23-27-56_3af89f56dea89107 2026-05-13__05-25-03_e3f7d3d2ad461681 2026-05-18__19-28-56_159b57f1f3f20442
      2026-05-18__12-12-23_59f7fd9274a1eb2c 2026-05-12__06-09-41_7254f5ea8908ea68 2026-05-18__19-28-56_7d6f149ca64eea51
      2026-05-18__12-12-23_2b5348bcf89b949f 2026-05-14__08-07-59_50122419e575e16d 2026-05-18__19-28-59_b5263d4c7baf76af
      2026-05-18__12-12-23_355c95b231e4474c 2026-05-12__06-22-54_3719e6b84f11c9c1 2026-05-18__19-28-56_55916ccdd76d9adc)

i="${SLURM_ARRAY_TASK_ID}"
NAME="${NAMES[$i]}"
CKPT=$(ls -d "${BASE_DIR}/${DIRS[$i]}"/eid_*-seen_steps_18360-*/hf_checkpoint | head -n 1)
OUT_DIR="scripts/analysis/revision_analysis/results/${NAME}"
mkdir -p "${OUT_DIR}"
echo "model ${NAME}: ${CKPT}"

# "<ablation spec>|<result file name>"
RUNS=(
  "learned|learned"
  "g_deep=0,g_wide=1,mask_tags=NUM+SYM|gd0p0_gw1p0_mask_NUM+SYM"
  "g_deep=1,g_wide=0,mask_tags=ADV+PART+PRON+ADJ+VERB|gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+VERB"
  "g_deep=0,g_wide=1,mask_tags=RANDOM|gd0p0_gw1p0_mask_RANDOM"
  "g_deep=1,g_wide=0,mask_tags=RANDOM+ADV+PART+PRON+ADJ+VERB|gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+RANDOM+VERB"
  "ratio_shuffle|ratio_shuffle"
  "ratio_mean|ratio_mean"
  "sum_shuffle|sum_shuffle"
  "sum_mean|sum_mean"
  "g_deep=0,g_wide=1,mask_tags=NUM+SYM+RANDOMMATCHED|gd0p0_gw1p0_mask_NUM+RANDOMMATCHED+SYM"
  "g_deep=1,g_wide=0,mask_tags=ADV+PART+PRON+ADJ+VERB+RANDOMMATCHED|gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+RANDOMMATCHED+VERB"
)
for run in "${RUNS[@]}"; do
    SPEC="${run%%|*}"; FILE="${run##*|}"
    if [ -f "${OUT_DIR}/${FILE}.json" ]; then echo "  have ${FILE}"; continue; fi
    echo "  -> ${SPEC}"
    python scripts/analysis/revision_analysis/ablation_eval.py \
        --ckpt "${CKPT}" --paloma-dir "${CKPT}/paloma_diagnostics" \
        --ablation "${SPEC}" --sources gsm8k wikitext_103 triviaqa --out-dir "${OUT_DIR}"
done
echo "END $(date)"
