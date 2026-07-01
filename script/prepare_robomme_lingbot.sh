#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

SOURCE_ROOT="${SOURCE_ROOT:-../robomme_data_lerobot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-../robomme_data_lingbot}"
SEGMENT_SOURCE="${SEGMENT_SOURCE:-task}"
MIN_SEGMENT_FRAMES="${MIN_SEGMENT_FRAMES:-8}"
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

args=(
  --source-root "${SOURCE_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --segment-source "${SEGMENT_SOURCE}"
  --min-segment-frames "${MIN_SEGMENT_FRAMES}"
  --overwrite
)

if [[ "${COMPUTE_NORM_STATS}" == "1" ]]; then
  args+=(--compute-norm-stats)
fi

"${PYTHON_BIN}" tools/prepare_robomme_lingbot.py "${args[@]}"
"${PYTHON_BIN}" tools/check_robomme_lingbot.py \
  --dataset-root "${OUTPUT_ROOT}" \
  --require-norm-stats
