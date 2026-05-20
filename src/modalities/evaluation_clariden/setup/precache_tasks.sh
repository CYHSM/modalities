#!/bin/bash
# RUN THIS ON THE LOGIN NODE

MY_ROOT="/users/markusfrey"
EXPERIMENTS_DIR="/capstor/scratch/cscs/markusfrey/experiments"
OLMES_VENV="${MY_ROOT}/venvs/olmes"
HF_CACHE="${EXPERIMENTS_DIR}/../hf_cache"

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
gsm8k::olmes"

# Use uenv run to ensure oe_eval has all the right dependencies to load the configs
uenv run pytorch/v2.9.1 --view=default -- bash -c "
    set -e -u
    source ${OLMES_VENV}/bin/activate
    
    # Notice we DO NOT set the offline flags here
    export HF_DATASETS_CACHE=${HF_CACHE}
    export HF_HOME=${HF_CACHE}

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

print(f'Will download {len(all_tasks)} tasks')
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
"