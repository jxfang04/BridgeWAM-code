#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
DEFAULT_TRAIN_OUTPUT_BASE="${PROJECT_ROOT}/runs/train"

is_true() {
  case "$1" in
    1|true|TRUE|True|yes|YES|Yes|on|ON|On) return 0 ;;
    *) return 1 ;;
  esac
}

sanitize_tmux_name() {
  local value="$1"
  value="$(printf '%s' "${value}" | tr -c '[:alnum:]_.-' '_')"
  printf '%s' "${value:0:120}"
}

launch_in_tmux() {
  if (( $# < 1 )); then
    echo "Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]" >&2
    return 2
  fi
  if ! command -v tmux >/dev/null 2>&1; then
    echo "Error: tmux is required but not installed." >&2
    return 1
  fi

  local task_basename="train"
  local output_dir_override=""
  local arg
  for arg in "${@:2}"; do
    case "${arg}" in
      task=*)
        task_basename="${arg#task=}"
        task_basename="${task_basename%.yaml}"
        ;;
      --config-name=task/*)
        task_basename="${arg#--config-name=task/}"
        task_basename="${task_basename%.yaml}"
        ;;
      output_dir=*)
        output_dir_override="${arg#output_dir=}"
        ;;
    esac
  done

  local session_name
  session_name="${BRIDGEWAM_TMUX_SESSION_NAME:-bridgewam_${task_basename}_$(date +%Y%m%d_%H%M%S)_$$}"
  session_name="$(sanitize_tmux_name "${session_name}")"
  if [[ -z "${session_name}" ]]; then
    echo "Error: BRIDGEWAM_TMUX_SESSION_NAME produced an empty tmux session name." >&2
    return 1
  fi
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Error: tmux session already exists: ${session_name}" >&2
    return 1
  fi

  local train_output_base="${BRIDGEWAM_TRAIN_OUTPUT_BASE:-${DEFAULT_TRAIN_OUTPUT_BASE}}"
  local num_machines="${NNODES:-1}"
  local launch_run_id="${RUN_ID:-}"
  if [[ -z "${launch_run_id}" && "${num_machines}" == "1" ]]; then
    launch_run_id="$(date +%Y-%m-%d_%H-%M-%S)"
  fi

  local run_output_dir=""
  local log_file=""
  if [[ -n "${output_dir_override}" ]]; then
    run_output_dir="${output_dir_override}"
    log_file="${run_output_dir}/train.log"
    mkdir -p "${run_output_dir}"
    touch "${log_file}"
  elif [[ -n "${launch_run_id}" ]]; then
    run_output_dir="${train_output_base}/${task_basename}/${launch_run_id}"
    log_file="${run_output_dir}/train.log"
    mkdir -p "${run_output_dir}"
    touch "${log_file}"
  fi

  # A long-lived tmux server may have a stale environment. Forward the values
  # that affect Python/CUDA/distributed training explicitly to the inner job.
  local -a inner_command=(env "BRIDGEWAM_TMUX_INNER=1")
  if [[ -n "${launch_run_id}" ]]; then
    inner_command+=("RUN_ID=${launch_run_id}")
  fi
  local env_name
  local -a forwarded_env_names=(
    PATH PYTHONPATH LD_LIBRARY_PATH CONDA_PREFIX VIRTUAL_ENV CUDA_HOME
    CUDA_VISIBLE_DEVICES
    DIFFSYNTH_MODEL_BASE_PATH DIFFSYNTH_SKIP_DOWNLOAD
    ACTION_DIT_PRETRAINED_PATH BRIDGEWAM_TRAIN_OUTPUT_BASE
    NNODES NODE_RANK MASTER_ADDR MASTER_PORT
    RUN_ID_SYNC_TIMEOUT RUN_ID_SYNC_PORT
    WANDB_MODE WANDB_PROJECT WANDB_ENTITY WANDB_DIR
  )
  while IFS= read -r env_name; do
    case "${env_name}" in
      NCCL_*|TORCH_NCCL_*) forwarded_env_names+=("${env_name}") ;;
    esac
  done < <(compgen -e)

  for env_name in "${forwarded_env_names[@]}"; do
    if declare -p "${env_name}" >/dev/null 2>&1; then
      inner_command+=("${env_name}=${!env_name}")
    fi
  done
  inner_command+=(bash "${SCRIPT_PATH}" "$@")

  local quoted_inner_command
  printf -v quoted_inner_command '%q ' "${inner_command[@]}"
  quoted_inner_command+="; rc=\$?; echo; echo \"[tmux] training exited with code \$rc at \$(date '+%Y-%m-%d %H:%M:%S')\"; exit \$rc"

  if ! tmux new-session -d -s "${session_name}" -c "${PROJECT_ROOT}"; then
    echo "Error: failed to create tmux session: ${session_name}" >&2
    return 1
  fi
  if ! tmux set-option -t "${session_name}" remain-on-exit on \
    || ! tmux respawn-pane -k -t "${session_name}:0.0" "${quoted_inner_command}"; then
    tmux kill-session -t "${session_name}" 2>/dev/null || true
    echo "Error: failed to start training in tmux session: ${session_name}" >&2
    return 1
  fi

  echo "[tmux] training launched in detached session."
  echo "[tmux] session: ${session_name}"
  if [[ -n "${log_file}" ]]; then
    echo "[tmux] output directory: ${run_output_dir}"
    echo "[tmux] log: ${log_file}"
    echo "[tmux] follow log: tail -f ${log_file}"
  else
    echo "[tmux] log: ${train_output_base}/${task_basename}/<synchronized RUN_ID>/train.log"
    echo "[tmux] note: multi-machine RUN_ID will be synchronized by the inner training job."
  fi
  echo "[tmux] attach: tmux attach -t ${session_name}"
  echo "[tmux] stop: tmux kill-session -t ${session_name}"
}

# The user-facing invocation creates a detached tmux session. The marker is
# only set for the command running inside that session, preventing recursion.
# BRIDGEWAM_TMUX_DISABLED=1 keeps the original foreground behavior when needed.
if [[ "${BRIDGEWAM_TMUX_INNER:-0}" != "1" ]] \
  && ! is_true "${BRIDGEWAM_TMUX_DISABLED:-0}"; then
  launch_in_tmux "$@"
  exit $?
fi

NPROC_PER_NODE="${1:?Usage: bash scripts/train_zero1.sh <nproc_per_node> [hydra_overrides...]}"
shift

MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${PROJECT_ROOT}/checkpoints}"
ACTION_DIT_PRETRAINED_PATH="${ACTION_DIT_PRETRAINED_PATH:-${MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
TRAIN_OUTPUT_BASE="${BRIDGEWAM_TRAIN_OUTPUT_BASE:-${DEFAULT_TRAIN_OUTPUT_BASE}}"
WAN_MODEL_DIR="${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"
TOKENIZER_DIR="${MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
COMMON_MODEL_DIR="${MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors"

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"

EXTRA_ARGS=("$@")
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"

is_integer() {
  [[ "${1}" =~ ^[0-9]+$ ]]
}

if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

extract_task_basename() {
  local cfg="$1"
  if [[ "${cfg}" == task/* ]]; then
    local name="${cfg#task/}"
    name="${name%.yaml}"
    echo "${name}"
    return 0
  fi
  return 1
}

TASK_BASENAME="train"
OUTPUT_DIR_OVERRIDE=""
LOAD_TEXT_ENCODER="false"
SKIP_DIT_LOAD_FROM_PRETRAIN="false"
REDIRECT_COMMON_FILES="true"
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
  arg="${EXTRA_ARGS[$i]}"
  case "${arg}" in
    --config-name)
      if ((i + 1 < ${#EXTRA_ARGS[@]})); then
        next="${EXTRA_ARGS[$((i + 1))]}"
        if parsed="$(extract_task_basename "${next}")"; then
          TASK_BASENAME="${parsed}"
        fi
      fi
      ;;
    --config-name=*)
      cfg="${arg#--config-name=}"
      if parsed="$(extract_task_basename "${cfg}")"; then
        TASK_BASENAME="${parsed}"
      fi
      ;;
    task=*)
      cfg="${arg#task=}"
      cfg="${cfg%.yaml}"
      TASK_BASENAME="${cfg}"
      ;;
    output_dir=*)
      OUTPUT_DIR_OVERRIDE="${arg#output_dir=}"
      ;;
    model.load_text_encoder=*)
      LOAD_TEXT_ENCODER="${arg#model.load_text_encoder=}"
      ;;
    model.skip_dit_load_from_pretrain=*)
      SKIP_DIT_LOAD_FROM_PRETRAIN="${arg#model.skip_dit_load_from_pretrain=}"
      ;;
    model.redirect_common_files=*)
      REDIRECT_COMMON_FILES="${arg#model.redirect_common_files=}"
      ;;
  esac
done

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"

    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT
    export RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"

    RUN_ID="$(
      python - <<'PY'
import datetime
import os
from datetime import timedelta

import torch.distributed as dist

host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")

store = dist.TCPStore(
    host_name=host,
    port=port,
    world_size=num_machines,
    is_master=(machine_rank == 0),
    timeout=timedelta(seconds=timeout_s),
)
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
run_id = store.get(key).decode("utf-8")
print(run_id)
PY
    )"

    echo "[run_id_sync] mode=tcpstore host=${RUN_ID_SYNC_HOST} port=${RUN_ID_SYNC_PORT} timeout_s=${RUN_ID_SYNC_TIMEOUT} run_id=${RUN_ID}"
  fi
fi

if [[ -n "${OUTPUT_DIR_OVERRIDE}" ]]; then
  RUN_OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE}"
else
  mkdir -p "${TRAIN_OUTPUT_BASE}"
  RUN_OUTPUT_DIR="${TRAIN_OUTPUT_BASE}/${TASK_BASENAME}/${RUN_ID}"
fi
mkdir -p "${RUN_OUTPUT_DIR}"
TRAIN_LOG_FILE="${RUN_OUTPUT_DIR}/train.log"

# Keep the live tmux output while also storing this run's complete training
# stream next to its checkpoints. Appending preserves logs when a RUN_ID is
# intentionally reused for a resumed launch.
exec > >(tee -a "${TRAIN_LOG_FILE}") 2>&1

if [[ ! -d "${MODEL_BASE_PATH}" ]]; then
  echo "Error: base model directory does not exist: ${MODEL_BASE_PATH}" >&2
  exit 1
fi

require_nonempty_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "${path}" ]]; then
    echo "Error: ${label} is missing or empty: ${path}" >&2
    exit 1
  fi
}

if [[ "${REDIRECT_COMMON_FILES,,}" == "true" ]]; then
  VAE_CHECKPOINT="${COMMON_MODEL_DIR}/Wan2.2_VAE.safetensors"
  TEXT_ENCODER_CHECKPOINT="${COMMON_MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.safetensors"
else
  VAE_CHECKPOINT="${WAN_MODEL_DIR}/Wan2.2_VAE.pth"
  TEXT_ENCODER_CHECKPOINT="${WAN_MODEL_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
fi

require_nonempty_file "Wan2.2 VAE checkpoint" "${VAE_CHECKPOINT}"

VIDEO_DIT_FILE_COUNT=0
if [[ "${SKIP_DIT_LOAD_FROM_PRETRAIN,,}" != "true" ]]; then
  shopt -s nullglob
  video_dit_files=("${WAN_MODEL_DIR}"/diffusion_pytorch_model*.safetensors)
  shopt -u nullglob
  if (( ${#video_dit_files[@]} == 0 )); then
    echo "Error: no pretrained Video DiT checkpoint matched:" >&2
    echo "  ${WAN_MODEL_DIR}/diffusion_pytorch_model*.safetensors" >&2
    exit 1
  fi
  for video_dit_file in "${video_dit_files[@]}"; do
    require_nonempty_file "Video DiT checkpoint shard" "${video_dit_file}"
  done
  VIDEO_DIT_FILE_COUNT="${#video_dit_files[@]}"
  require_nonempty_file "ActionDiT checkpoint" "${ACTION_DIT_PRETRAINED_PATH}"
fi

if [[ "${LOAD_TEXT_ENCODER,,}" == "true" ]]; then
  require_nonempty_file "T5 text-encoder checkpoint" "${TEXT_ENCODER_CHECKPOINT}"
  if [[ ! -d "${TOKENIZER_DIR}" ]] \
      || [[ -z "$(find "${TOKENIZER_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Error: tokenizer directory is missing or empty: ${TOKENIZER_DIR}" >&2
    exit 1
  fi
fi

echo "[launch] nproc_per_node=${NPROC_PER_NODE} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} run_id=${RUN_ID}"
echo "[launch] model_base_path=${MODEL_BASE_PATH}"
echo "[launch] diffsynth_skip_download=${DIFFSYNTH_SKIP_DOWNLOAD}"
echo "[launch] redirect_common_files=${REDIRECT_COMMON_FILES}"
echo "[launch] vae_checkpoint=${VAE_CHECKPOINT}"
if [[ "${SKIP_DIT_LOAD_FROM_PRETRAIN,,}" == "true" ]]; then
  echo "[launch] pretrained_dit_load=skipped"
else
  echo "[launch] video_dit_dir=${WAN_MODEL_DIR} matched_files=${VIDEO_DIT_FILE_COUNT}"
  echo "[launch] action_dit_pretrained_path=${ACTION_DIT_PRETRAINED_PATH}"
fi
if [[ "${LOAD_TEXT_ENCODER,,}" == "true" ]]; then
  echo "[launch] text_encoder_checkpoint=${TEXT_ENCODER_CHECKPOINT}"
  echo "[launch] tokenizer_dir=${TOKENIZER_DIR}"
else
  echo "[launch] text_encoder_load=disabled (using cached text embeddings)"
fi
echo "[launch] output_dir=${RUN_OUTPUT_DIR}"
echo "[launch] train_log=${TRAIN_LOG_FILE}"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes "${NPROC_PER_NODE}" \
  scripts/train.py \
  "output_dir=${RUN_OUTPUT_DIR}" \
  "wandb.name=${TASK_BASENAME}" \
  "model.action_dit_pretrained_path=${ACTION_DIT_PRETRAINED_PATH}" \
  "${EXTRA_ARGS[@]}"
