#!/bin/bash
# RUN THIS ON THE LOGIN NODE (Leonardo HPC)
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
EXPERIMENTS_DIR="/leonardo_scratch/large/userexternal/mfrey000/experiments_emnlp"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
HF_CACHE="${EXPERIMENTS_DIR}/../hf_cache"

HF_ACCESS_TOKEN=""

mkdir -p "$HF_CACHE"

export OLMES_TASKS="minerva_math_algebra:bpb::olmes \
minerva_math_counting_and_probability:bpb::olmes \
minerva_math_geometry:bpb::olmes \
minerva_math_intermediate_algebra:bpb::olmes \
minerva_math_number_theory:bpb::olmes \
minerva_math_prealgebra:bpb::olmes \
minerva_math_precalculus:bpb::olmes \
arc_challenge:rc::olmes:full \
arc_easy:rc::olmes:full \
hellaswag:rc::olmes:full \
winogrande:rc::olmes:full \
socialiqa:rc::olmes:full \
piqa:rc::olmes:full \
qasper_yesno:rc::olmes \
lambada \
arc_challenge:rc:bpb::olmes:full \
arc_easy:rc:bpb::olmes:full \
hellaswag:rc:bpb::olmes:full \
winogrande:rc:bpb::olmes:full \
socialiqa:rc:bpb::olmes:full \
piqa:rc:bpb::olmes:full \
qasper_yesno:rc:bpb::olmes \
lambada:bpb \
gsm8k::olmes \
paloma::paloma"

echo "=== Starting cache download inside Singularity ==="
singularity exec \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    "$CONTAINER" bash -c "
    set -e -u
    echo 'Activating OLMES environment...'
    source ${OLMES_VENV}/bin/activate
    export PYTHONPATH=${OLMES_VENV}/lib/python3.12/site-packages:\$PYTHONPATH
    export HF_DATASETS_CACHE=${HF_CACHE}
    export HF_HOME=${HF_CACHE}
    export HF_TOKEN=\"${HF_ACCESS_TOKEN}\"

    # ---- Part 1: OLMES tasks ----
    echo '--- Caching OLMES tasks ---'
    python <<PYEOF
import os, copy
from oe_eval.configs.tasks import TASK_CONFIGS
from oe_eval.launch import resolve_task_suite
from oe_eval.run_eval import load_task

tasks_input = os.environ['OLMES_TASKS'].split()
all_tasks = []
for t in tasks_input:
    try:
        all_tasks += resolve_task_suite(t, {})
    except Exception as e:
        print(f'!! could not resolve {t}: {e}')

print(f'\nWill download {len(all_tasks)} tasks to {os.environ[\"HF_HOME\"]}')
for task_name in all_tasks:
    if task_name not in TASK_CONFIGS:
        print(f'?? not in TASK_CONFIGS: {task_name}')
        continue
    try:
        cfg = copy.deepcopy(TASK_CONFIGS[task_name])
        task = load_task(cfg, '.')
        print(f'-> downloading {task_name}')
        task.download()
        print(f'   done')
    except Exception as e:
        print(f'!! failed {task_name}: {e}')
PYEOF

    # ---- Part 2: pseudo-sources used by paloma_diagnostics.py ----
    echo ''
    echo '--- Caching diagnostics pseudo-source datasets (gsm8k, trivia_qa) ---'
    python <<PYEOF
from datasets import load_dataset

specs = [
    ('gsm8k',     'main',         'test'),
    ('trivia_qa', 'rc.nocontext', 'validation'),
]
for path, config, split in specs:
    try:
        print(f'-> downloading {path} ({config}, {split})')
        load_dataset(path, config, split=split)
        print(f'   done')
    except Exception as e:
        print(f'!! failed {path}: {e}')
PYEOF
"

echo "=== Precaching complete ==="