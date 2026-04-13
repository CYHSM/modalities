#!/bin/bash
MY_ROOT="/leonardo_work/EUHPC_D21_101/mfrey"
SCRATCH_ROOT="/leonardo_scratch/large/userexternal/mfrey000"
CONTAINER="${MY_ROOT}/containers/image_34c40a6bbdb8dcbb6d674d06caaa93af68d5692fb744eeb0e28908eea6158b13.sif"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
HF_CACHE="${SCRATCH_ROOT}/hf_cache"

mkdir -p "$HF_CACHE"

export OLMES_TASKS="arc:rc::olmes:full hellaswag:rc::olmes:full winogrande:rc::olmes:full socialiqa:rc::olmes:full piqa:rc::olmes:full qasper_yesno:rc::olmes lambada arc:rc:bpb::olmes:full hellaswag:rc:bpb::olmes:full winogrande:rc:bpb::olmes:full socialiqa:rc:bpb::olmes:full piqa:rc:bpb::olmes:full qasper_yesno:rc:bpb::olmes lambada:bpb minerva_math_algebra:bpb::olmes minerva_math_counting_and_probability:bpb::olmes minerva_math_geometry:bpb::olmes minerva_math_intermediate_algebra:bpb::olmes minerva_math_number_theory:bpb::olmes minerva_math_prealgebra:bpb::olmes minerva_math_precalculus:bpb::olmes gsm8k::olmes minerva_math::olmes"

singularity exec \
    --bind "${MY_ROOT}:${MY_ROOT}" \
    --bind "${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    --env OLMES_TASKS="${OLMES_TASKS}" \
    "$CONTAINER" bash -c '
    source '"${OLMES_VENV}"'/bin/activate
    export PYTHONPATH='"${OLMES_VENV}"'/lib/python3.12/site-packages:$PYTHONPATH
    export TIKTOKEN_CACHE_DIR='"${MY_ROOT}"'/tiktoken_cache
    export HF_DATASETS_CACHE='"${HF_CACHE}"'
    export HF_HOME='"${HF_CACHE}"'

    python <<PYEOF
import os, copy
from oe_eval.configs.tasks import TASK_CONFIGS
from oe_eval.launch import resolve_task_suite
from oe_eval.run_eval import load_task

tasks_input = os.environ["OLMES_TASKS"].split()
all_tasks = []
for t in tasks_input:
    try:
        all_tasks += resolve_task_suite(t, {})
    except Exception as e:
        print(f"!! could not resolve {t}: {e}")

print(f"Will download {len(all_tasks)} tasks")
for task_name in all_tasks:
    if task_name not in TASK_CONFIGS:
        print(f"?? not in TASK_CONFIGS: {task_name}")
        continue
    try:
        cfg = copy.deepcopy(TASK_CONFIGS[task_name])
        task = load_task(cfg, ".")
        print(f"-> downloading {task_name}")
        task.download()
        print(f"   done")
    except Exception as e:
        print(f"!! failed {task_name}: {e}")
PYEOF
'