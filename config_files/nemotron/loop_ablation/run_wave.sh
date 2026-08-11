#!/usr/bin/env bash
# Launches a wave of Nemotron layer-loop ablation arms, several per GPU, detached from this shell.
#
#   ./config_files/nemotron/loop_ablation/run_wave.sh <WAVE_NAME> <HOURS> <ARM:GPU> [<ARM:GPU> ...]
#
# Example -- eight arms, two per GPU, on GPUs 4-7 for six hours each:
#
#   ./config_files/nemotron/loop_ablation/run_wave.sh wave4 6 \
#       A0_baseline:4 A4_loop_mamba_moe:4 \
#       A6_loop_attention_moe:5 N4_anchor_attention_moe:5 \
#       A6a_loop_attention_moe_per_iteration_norm:6 A6b_loop_attention_moe_input_injection:6 \
#       A6c_loop_attention_moe_norm_and_injection:7 A5_loop_mamba_attention:7
#
# Two properties this has that a bare `for ... run_arm.sh &` loop does not:
#
# * **The runs survive this shell.** Each is started with `setsid`, so it is in its own session and
#   process group. Wave 3 died at step ~935 of a planned 2 hours because every run took a SIGTERM
#   at the same second -- the launching shell went away and took the whole process group with it.
# * **Ports do not collide.** run_arm.sh defaults its rendezvous port to 29500 + GPU id, which two
#   arms on the same GPU would share. Ports here are probed for availability, so a wave launched
#   while an earlier one is still running gets its own range. Override the start with
#   RDZV_BASE_PORT.
#
# Putting more than one arm on a GPU is a throughput trade, not a correctness one: the arms share
# SMs, so each runs at roughly 1/n speed. Keep the number of arms per GPU *equal* across the wave,
# or "loss at equal wall-clock" stops being comparable between them (loss at equal steps still is).
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <WAVE_NAME> <HOURS> <ARM:GPU> [<ARM:GPU> ...]" >&2
    exit 1
fi

WAVE_NAME=$1
HOURS=$2
shift 2

REPOSITORY_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
RESULTS_DIRECTORY="${REPOSITORY_ROOT}/results"
SECONDS_PER_ARM=$(python3 -c "print(int(float('${HOURS}') * 3600))")
NEXT_PORT=${RDZV_BASE_PORT:-29600}

mkdir -p "${RESULTS_DIRECTORY}"

# Ports must be free, not merely distinct within this wave. A second wave launched while the first
# is still running would otherwise reuse 29600 upwards and every arm would die instantly with
# EADDRINUSE -- which is exactly what happened when the dense LR bracket was started alongside the
# dense baselines.
claim_free_port() {
    while ss -ltnH "sport = :${NEXT_PORT}" 2> /dev/null | grep -q .; do
        NEXT_PORT=$((NEXT_PORT + 1))
    done
    echo "${NEXT_PORT}"
    NEXT_PORT=$((NEXT_PORT + 1))
}

# Validate everything before launching anything: a typo in the sixth arm should not leave five
# runs already occupying GPUs.
declare -a ARM_NAMES=() GPU_IDS=()
for specification in "$@"; do
    if [[ ${specification} != *:* ]]; then
        echo "malformed specification '${specification}', expected <ARM>:<GPU>" >&2
        exit 1
    fi
    arm_name=${specification%%:*}
    gpu_id=${specification##*:}
    if [[ ! -f "${REPOSITORY_ROOT}/config_files/nemotron/loop_ablation/config_${arm_name}.yaml" ]]; then
        echo "no config for arm '${arm_name}'" >&2
        exit 1
    fi
    ARM_NAMES+=("${arm_name}")
    GPU_IDS+=("${gpu_id}")
done

echo "wave=${WAVE_NAME} hours=${HOURS} (${SECONDS_PER_ARM}s per arm) arms=${#ARM_NAMES[@]}"
for index in "${!ARM_NAMES[@]}"; do
    arm_name=${ARM_NAMES[${index}]}
    gpu_id=${GPU_IDS[${index}]}
    port=$(claim_free_port)

    # Naming the log after the arm alone collides when the same arm is launched more than once,
    # which is exactly what a seed-replicate wave does -- three runs would share one file and
    # interleave their output. Repeats get a _rN suffix; a single occurrence keeps the plain name
    # so existing log paths are unchanged.
    replicate=0
    for other in "${ARM_NAMES[@]}"; do
        [[ ${other} == "${arm_name}" ]] && replicate=$((replicate + 1))
    done
    if [[ ${replicate} -gt 1 ]]; then
        seen=0
        for earlier in $(seq 0 "${index}"); do
            [[ ${ARM_NAMES[${earlier}]} == "${arm_name}" ]] && seen=$((seen + 1))
        done
        log_path="${RESULTS_DIRECTORY}/${WAVE_NAME}_${arm_name}_r${seen}.log"
    else
        log_path="${RESULTS_DIRECTORY}/${WAVE_NAME}_${arm_name}.log"
    fi

    setsid nohup timeout "${SECONDS_PER_ARM}" \
        "${REPOSITORY_ROOT}/config_files/nemotron/loop_ablation/run_arm.sh" \
        "${arm_name}" "${gpu_id}" "${port}" > "${log_path}" 2>&1 &
    disown || true

    echo "  gpu ${gpu_id}  port ${port}  ${arm_name}  -> ${log_path#"${REPOSITORY_ROOT}/"}"
    # Stagger the starts: eight simultaneous CPU-side model builds thrash, and the rendezvous
    # servers come up more reliably when they are not all binding at once.
    sleep 3
done

echo
echo "launched. follow with:  tail -f ${RESULTS_DIRECTORY}/${WAVE_NAME}_*.log"
echo "stop the wave with:     pkill -f 'loop_ablation/config_'"
