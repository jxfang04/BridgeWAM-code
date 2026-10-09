#!/bin/bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
BRIDGEWAM_PYTHON=${BRIDGEWAM_PYTHON:-"$(command -v python)"}

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/LIBERO-plus:$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-"$ROOT_DIR/checkpoints"}
export DIFFSYNTH_SKIP_DOWNLOAD=${DIFFSYNTH_SKIP_DOWNLOAD:-true}
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-all}
export LIBERO_PLUS_REPO=${LIBERO_PLUS_REPO:-"$ROOT_DIR/LIBERO-plus"}

BRIDGEWAM_IMPORT_PATH=$("$BRIDGEWAM_PYTHON" -c 'import bridgewam; print(bridgewam.__file__)')
case "$BRIDGEWAM_IMPORT_PATH" in
    "$ROOT_DIR"/src/bridgewam/*) ;;
    *)
        echo "Error: LIBERO-Plus stage 1 imported BridgeWAM from the wrong checkout."
        echo "Expected: $ROOT_DIR/src/bridgewam/..."
        echo "Actual:   $BRIDGEWAM_IMPORT_PATH"
        exit 2
        ;;
esac
echo "[LIBERO-Plus stage 1] BridgeWAM import: $BRIDGEWAM_IMPORT_PATH"

task_name="libero_uncond_2cam224_lbqs_only_2layer_alternating_cross_self_fullfinetune_1e-4"
output_override=""
sample_cap=""
expected_override=""
for arg in "$@"; do
    case "$arg" in
        task=*) task_name=${arg#task=} ;;
        EVALUATION.output_dir=*) output_override=${arg#EVALUATION.output_dir=} ;;
        LIBERO_PLUS.max_tasks_per_cell=*) sample_cap=${arg#LIBERO_PLUS.max_tasks_per_cell=} ;;
        LIBERO_PLUS.expected_num_tasks=*) expected_override=${arg#LIBERO_PLUS.expected_num_tasks=} ;;
    esac
done

manager_args=("$@")
if [ "${ALLOW_PARTIAL_STAGE1:-false}" = "true" ]; then
    stage1_scope="partial"
else
    stage1_scope="full"
    if [ -n "$sample_cap" ] && [ "$sample_cap" != "null" ]; then
        echo "Error: run_stage1.sh defaults to the full 10,030-task LIBERO-Plus benchmark."
        echo "Remove LIBERO_PLUS.max_tasks_per_cell=$sample_cap, or set ALLOW_PARTIAL_STAGE1=true for a sampled diagnostic run."
        exit 2
    fi
    if [ -z "$expected_override" ]; then
        manager_args+=("LIBERO_PLUS.expected_num_tasks=10030")
    fi
fi

if [ -n "$output_override" ]; then
    STAGE1_OUTPUT_DIR=$output_override
else
    STAGE1_OUTPUT_DIR=${STAGE1_OUTPUT_DIR:-"$ROOT_DIR/runs/eval/libero-plus/$task_name/stage1_${stage1_scope}_$(date +%Y%m%d_%H%M%S)"}
    manager_args+=("EVALUATION.output_dir=$STAGE1_OUTPUT_DIR")
fi

if [ "${LIBERO_PLUS_STAGE1_IN_TMUX:-false}" != "true" ] \
    && [ "${ALLOW_EXISTING_STAGE1_OUTPUT:-false}" != "true" ] \
    && [ -d "$STAGE1_OUTPUT_DIR" ] \
    && [ -n "$(find "$STAGE1_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "Error: stage-one output directory is not empty: $STAGE1_OUTPUT_DIR"
    echo "Unset STAGE1_OUTPUT_DIR or choose a new path. Set ALLOW_EXISTING_STAGE1_OUTPUT=true only for an intentional resume."
    exit 2
fi

echo "[LIBERO-Plus stage 1] Evaluation scope: $stage1_scope"
echo "[LIBERO-Plus stage 1] Output directory: $STAGE1_OUTPUT_DIR"

# Keep the stage-one manager alive when a browser IDE or SSH connection drops.
# The manager itself creates a second tmux session for GPU workers; this outer
# session owns the manager/scheduler process and captures its complete output.
if [ "${LIBERO_PLUS_DETACH:-true}" = "true" ] \
    && [ "${LIBERO_PLUS_STAGE1_IN_TMUX:-false}" != "true" ]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "Error: tmux is required for detached LIBERO-Plus stage-one runs."
        echo "Set LIBERO_PLUS_DETACH=false to run in the foreground."
        exit 1
    fi

    mkdir -p "$STAGE1_OUTPUT_DIR"
    manager_log="$STAGE1_OUTPUT_DIR/stage1_manager.log"
    session_name=${LIBERO_PLUS_STAGE1_SESSION_NAME:-"libero_plus_stage1_$(date +%Y%m%d_%H%M%S)_$$"}
    session_name=$(printf '%s' "$session_name" | tr -c '[:alnum:]_.-' '_')
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "Error: tmux session already exists: $session_name"
        exit 2
    fi

    quoted_args=""
    for arg in "$@"; do
        printf -v quoted_arg '%q' "$arg"
        quoted_args+=" $quoted_arg"
    done

    printf -v inner_command \
        'set -o pipefail
cd %q
export ROOT_DIR=%q
export BRIDGEWAM_PYTHON=%q
export STAGE1_OUTPUT_DIR=%q
export LIBERO_PLUS_STAGE1_IN_TMUX=true
export LIBERO_PLUS_DETACH=false
export ALLOW_EXISTING_STAGE1_OUTPUT=true
export ALLOW_PARTIAL_STAGE1=%q
export DIFFSYNTH_MODEL_BASE_PATH=%q
export DIFFSYNTH_SKIP_DOWNLOAD=%q
export MUJOCO_GL=%q
export PYOPENGL_PLATFORM=%q
export NVIDIA_DRIVER_CAPABILITIES=%q
export LIBERO_PLUS_REPO=%q
export EXP_NAME=%q
export CUDA_VISIBLE_DEVICES=%q
bash experiments/libero-plus/run_stage1.sh%s 2>&1 | tee -a %q
rc=${PIPESTATUS[0]}
echo "[LIBERO-Plus stage 1] manager exited with code ${rc} at $(date -Iseconds)" | tee -a %q
exit "${rc}"' \
        "$ROOT_DIR" \
        "$ROOT_DIR" \
        "$BRIDGEWAM_PYTHON" \
        "$STAGE1_OUTPUT_DIR" \
        "${ALLOW_PARTIAL_STAGE1:-false}" \
        "$DIFFSYNTH_MODEL_BASE_PATH" \
        "$DIFFSYNTH_SKIP_DOWNLOAD" \
        "$MUJOCO_GL" \
        "$PYOPENGL_PLATFORM" \
        "$NVIDIA_DRIVER_CAPABILITIES" \
        "$LIBERO_PLUS_REPO" \
        "${EXP_NAME:-}" \
        "${CUDA_VISIBLE_DEVICES:-}" \
        "$quoted_args" \
        "$manager_log" \
        "$manager_log"
    printf -v tmux_command 'bash -lc %q' "$inner_command"

    printf 'tmux_session=%s\noutput_dir=%s\nmanager_log=%s\nstarted_at=%s\n' \
        "$session_name" \
        "$STAGE1_OUTPUT_DIR" \
        "$manager_log" \
        "$(date -Iseconds)" \
        > "$STAGE1_OUTPUT_DIR/stage1_launcher_info.txt"

    tmux new-session -d -s "$session_name" bash
    tmux set-option -t "$session_name" remain-on-exit on >/dev/null
    tmux send-keys -t "$session_name:0.0" "$tmux_command" C-m

    echo "[LIBERO-Plus stage 1] Detached manager started."
    echo "- tmux session: $session_name"
    echo "- manager log: $manager_log"
    echo "- attach: tmux attach -t $session_name"
    echo "- follow log: tail -f $manager_log"
    echo "- stop: tmux send-keys -t $session_name:0.0 C-c"
    exit 0
fi

cd "$ROOT_DIR"
exec "$BRIDGEWAM_PYTHON" experiments/libero-plus/run_libero_plus_manager.py "${manager_args[@]}"
