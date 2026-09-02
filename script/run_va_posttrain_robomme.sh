#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Keep caches and temporary runtime/build files off AutoDL's small root disk.
LINGBOT_CACHE_ROOT="${LINGBOT_CACHE_ROOT:-${REPO_ROOT}/.cache}"
LINGBOT_TMP_ROOT="${LINGBOT_TMP_ROOT:-${REPO_ROOT}/.tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LINGBOT_CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${LINGBOT_CACHE_ROOT}/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export TORCH_HOME="${TORCH_HOME:-${LINGBOT_CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LINGBOT_CACHE_ROOT}/triton}"
export TMPDIR="${TMPDIR:-${LINGBOT_TMP_ROOT}}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
mkdir -p \
  "${HF_DATASETS_CACHE}" \
  "${HF_HUB_CACHE}" \
  "${HF_LEROBOT_HOME}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TMPDIR}"

: "${DATASET_PATH:?Set DATASET_PATH=/path/to/prepared/robomme_data_lingbot}"
: "${PRETRAINED_MODEL:?Set PRETRAINED_MODEL=/path/to/lingbot-va-base}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NORM_STAT_PATH="${NORM_STAT_PATH:-${DATASET_PATH}/meta/lingbot_va_robomme_norm_stats.json}"
RUN_DATA_CHECK="${RUN_DATA_CHECK:-0}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-1600}"
EXPECTED_SEGMENTS="${EXPECTED_SEGMENTS:-1600}"
EXPECTED_FRAMES="${EXPECTED_FRAMES:-768897}"
EXPECTED_TASKS="${EXPECTED_TASKS:-116}"
EXPECTED_LATENT_FILES="${EXPECTED_LATENT_FILES:-3200}"

if [[ "${PRETRAINED_MODEL}" == /path/to/* ]]; then
  echo "PRETRAINED_MODEL is still a placeholder: ${PRETRAINED_MODEL}" >&2
  echo "Set it to a real LingBot-VA model directory containing transformer/config.json." >&2
  exit 2
fi

if [[ ! -f "${PRETRAINED_MODEL}/transformer/config.json" ]]; then
  echo "Missing ${PRETRAINED_MODEL}/transformer/config.json" >&2
  echo "PRETRAINED_MODEL must point to the LingBot-VA base directory, not transformer/ itself." >&2
  exit 2
fi

if [[ ! -f "${NORM_STAT_PATH}" ]]; then
  echo "Missing RoboMME normalization stats: ${NORM_STAT_PATH}" >&2
  exit 2
fi

"${PYTHON_BIN}" tools/check_wanva_env.py --strict
if [[ "${RUN_DATA_CHECK}" == "1" ]]; then
  "${PYTHON_BIN}" tools/check_robomme_lingbot.py \
    --dataset-root "${DATASET_PATH}" \
    --cam-keys image,wrist_image \
    --require-norm-stats \
    --require-latents \
    --check-actions \
    --verify-norm-stats \
    --expect-binary-gripper \
    --expect-episodes "${EXPECTED_EPISODES}" \
    --expect-segments "${EXPECTED_SEGMENTS}" \
    --expect-frames "${EXPECTED_FRAMES}" \
    --expect-tasks "${EXPECTED_TASKS}" \
    --expect-latent-files "${EXPECTED_LATENT_FILES}" \
    --require-full-latent-frames
fi

NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG_RANK="${LOG_RANK:-0}"
CONFIG_NAME="${CONFIG_NAME:-robomme_train}"
TRAIN_MODE="${TRAIN_MODE:-lora}"
SAVE_ROOT="${SAVE_ROOT:-./train_out/robomme_${TRAIN_MODE}}"

export TOKENIZERS_PARALLELISM=false

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
PYTORCH_ALLOC_CONF="expandable_segments:True" \
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --local-ranks-filter="${LOG_RANK}" \
  --master_port "${MASTER_PORT}" \
  --tee 3 \
  -m wan_va.train \
  --config-name "${CONFIG_NAME}" \
  --dataset-path "${DATASET_PATH}" \
  --pretrained-model "${PRETRAINED_MODEL}" \
  --norm-stat-path "${NORM_STAT_PATH}" \
  --save-root "${SAVE_ROOT}" \
  --train-mode "${TRAIN_MODE}" \
  --disable-wandb \
  "$@"
