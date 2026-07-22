#!/usr/bin/env bash
set -euo pipefail
set -x

umask 007

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
PORT="${PORT:-29536}"
LOG_RANK="${LOG_RANK:-0}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
CONFIG_NAME="${CONFIG_NAME:-robotwin}"
PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" tools/check_wanva_env.py --strict

server_args=(
  --config-name "${CONFIG_NAME}"
  --port "${PORT}"
)
server_args+=("$@")

export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
"${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NGPU}" \
    --local-ranks-filter="${LOG_RANK}" \
    --master_port "${MASTER_PORT}" \
    --tee 3 \
    -m wan_va.wan_va_server "${server_args[@]}"
