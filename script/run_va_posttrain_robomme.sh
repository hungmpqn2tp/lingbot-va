#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET_PATH:?Set DATASET_PATH=/path/to/prepared/robomme_data_lingbot}"
: "${PRETRAINED_MODEL:?Set PRETRAINED_MODEL=/path/to/lingbot-va-base}"
PYTHON_BIN="${PYTHON_BIN:-python}"

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

"${PYTHON_BIN}" tools/check_wanva_env.py

NGPU="${NGPU:-1}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG_RANK="${LOG_RANK:-0}"
CONFIG_NAME="${CONFIG_NAME:-robomme_train}"
TRAIN_MODE="${TRAIN_MODE:-full}"
SAVE_ROOT="${SAVE_ROOT:-./train_out/robomme_${TRAIN_MODE}}"

export TOKENIZERS_PARALLELISM=false

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --local-ranks-filter="${LOG_RANK}" \
  --master_port "${MASTER_PORT}" \
  --tee 3 \
  -m wan_va.train \
  --config-name "${CONFIG_NAME}" \
  --dataset-path "${DATASET_PATH}" \
  --pretrained-model "${PRETRAINED_MODEL}" \
  --save-root "${SAVE_ROOT}" \
  --train-mode "${TRAIN_MODE}" \
  --disable-wandb \
  "$@"
