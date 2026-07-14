#!/bin/bash
#SBATCH --job-name=token_ablation
#SBATCH --account=AIFAC_S07_154
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/token_ablation-%j.out

# 1. Activate your virtual environment and set up python path
cd /leonardo_work/EUHPC_D21_101/mfrey/modalities
source .venv/bin/activate
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# 2. Experiment Directories
BASE_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"

# Arrays of the NOCROSS directories (80M and 160M | alpha=25, 50, 75 | K=2, 3, 4)
declare -A EXPERIMENTS
EXPERIMENTS["80M_alpha25_K2"]="2026-05-18__13-49-25_46661c3a0490b08d"
EXPERIMENTS["80M_alpha50_K2"]="2026-05-13__02-05-13_fa51525556e2cd3b"
EXPERIMENTS["80M_alpha75_K2"]="2026-05-18__19-28-55_0d6659323c9236d8"
EXPERIMENTS["80M_alpha25_K3"]="2026-05-18__12-12-23_207cde6cd1273411"
EXPERIMENTS["80M_alpha50_K3"]="2026-05-13__02-21-25_03070170b2aa73fa"
EXPERIMENTS["80M_alpha75_K3"]="2026-05-18__19-28-58_635662eb51947f62"
EXPERIMENTS["80M_alpha25_K4"]="2026-05-18__23-27-56_3af89f56dea89107"
EXPERIMENTS["80M_alpha50_K4"]="2026-05-13__05-25-03_e3f7d3d2ad461681"
EXPERIMENTS["80M_alpha75_K4"]="2026-05-18__19-28-56_159b57f1f3f20442"

EXPERIMENTS["160M_alpha25_K2"]="2026-05-18__12-12-23_59f7fd9274a1eb2c"
EXPERIMENTS["160M_alpha50_K2"]="2026-05-12__06-09-41_7254f5ea8908ea68"
EXPERIMENTS["160M_alpha75_K2"]="2026-05-18__19-28-56_7d6f149ca64eea51"
EXPERIMENTS["160M_alpha25_K3"]="2026-05-18__12-12-23_2b5348bcf89b949f"
EXPERIMENTS["160M_alpha50_K3"]="2026-05-14__08-07-59_50122419e575e16d"
EXPERIMENTS["160M_alpha75_K3"]="2026-05-18__19-28-59_b5263d4c7baf76af"
EXPERIMENTS["160M_alpha25_K4"]="2026-05-18__12-12-23_355c95b231e4474c"
EXPERIMENTS["160M_alpha50_K4"]="2026-05-12__06-22-54_3719e6b84f11c9c1"
EXPERIMENTS["160M_alpha75_K4"]="2026-05-18__19-28-56_55916ccdd76d9adc"


SOURCES="gsm8k wikitext_103 triviaqa"
RESULTS_BASE="config_files/emnlp/revision_analysis/results"

for MODEL_NAME in "${!EXPERIMENTS[@]}"; do
    EXP_DIR="${BASE_DIR}/${EXPERIMENTS[$MODEL_NAME]}"
    
    # Automatically find the final checkpoint folder (assuming seen_steps_18360)
    CKPT=$(ls -d ${EXP_DIR}/eid_*-seen_steps_18360-*/hf_checkpoint 2>/dev/null | head -n 1)
    
    if [ -z "$CKPT" ]; then
        echo "WARNING: Could not find final checkpoint for $MODEL_NAME in $EXP_DIR"
        continue
    fi

    echo "=========================================================="
    echo "Evaluating Model: $MODEL_NAME"
    echo "Checkpoint: $CKPT"
    echo "=========================================================="
    
    PALOMA_DIR="${CKPT}/paloma_diagnostics"
    OUT_DIR="${RESULTS_BASE}/${MODEL_NAME}"
    mkdir -p "$OUT_DIR"
    
    # 1. Baseline
    if [ ! -f "$OUT_DIR/learned.json" ]; then
        echo "  -> Running Baseline"
        python config_files/emnlp/revision_analysis/ablation_eval.py \
            --ckpt "$CKPT" \
            --paloma-dir "$PALOMA_DIR" \
            --ablation "learned" \
            --sources $SOURCES \
            --out-dir "$OUT_DIR"
    fi

    # 2. Arithmetic wide
    if [ ! -f "$OUT_DIR/gd0p0_gw1p0_mask_NUM+SYM.json" ]; then
        echo "  -> Running Arithmetic Wide"
        python config_files/emnlp/revision_analysis/ablation_eval.py \
            --ckpt "$CKPT" \
            --paloma-dir "$PALOMA_DIR" \
            --ablation "g_deep=0,g_wide=1,mask_tags=NUM+SYM" \
            --sources $SOURCES \
            --out-dir "$OUT_DIR"
    fi

    # 3. Function words deep
    if [ ! -f "$OUT_DIR/gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+VERB.json" ]; then
        echo "  -> Running Function Words Deep"
        python config_files/emnlp/revision_analysis/ablation_eval.py \
            --ckpt "$CKPT" \
            --paloma-dir "$PALOMA_DIR" \
            --ablation "g_deep=1,g_wide=0,mask_tags=ADV+PART+PRON+ADJ+VERB" \
            --sources $SOURCES \
            --out-dir "$OUT_DIR"
    fi

    # 4. Random matched wide
    if [ ! -f "$OUT_DIR/gd0p0_gw1p0_mask_RANDOM.json" ]; then
        echo "  -> Running Random Control Wide"
        python config_files/emnlp/revision_analysis/ablation_eval.py \
            --ckpt "$CKPT" \
            --paloma-dir "$PALOMA_DIR" \
            --ablation "g_deep=0,g_wide=1,mask_tags=RANDOM" \
            --sources $SOURCES \
            --out-dir "$OUT_DIR"
    fi

    # 5. Random matched deep
    if [ ! -f "$OUT_DIR/gd1p0_gw0p0_mask_ADJ+ADV+PART+PRON+RANDOM+VERB.json" ]; then
        echo "  -> Running Random Control Deep"
        python config_files/emnlp/revision_analysis/ablation_eval.py \
            --ckpt "$CKPT" \
            --paloma-dir "$PALOMA_DIR" \
            --ablation "g_deep=1,g_wide=0,mask_tags=RANDOM+ADV+PART+PRON+ADJ+VERB" \
            --sources $SOURCES \
            --out-dir "$OUT_DIR"
    fi
done

echo "All evaluations finished. Run python config_files/emnlp/revision_analysis/aggregate_results.py to generate LaTeX table."
