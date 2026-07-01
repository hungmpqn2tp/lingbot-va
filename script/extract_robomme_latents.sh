#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${DATASET_PATH:?Set DATASET_PATH=/path/to/prepared/robomme_data_lingbot}"
: "${PRETRAINED_MODEL:?Set PRETRAINED_MODEL=/path/to/lingbot-va-base}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${PRETRAINED_MODEL}" == /path/to/* ]]; then
  echo "PRETRAINED_MODEL is still a placeholder: ${PRETRAINED_MODEL}" >&2
  echo "Set it to a real LingBot-VA model directory containing vae/, text_encoder/, tokenizer/, and transformer/." >&2
  exit 2
fi

for subdir in vae text_encoder tokenizer transformer; do
  if [[ ! -d "${PRETRAINED_MODEL}/${subdir}" ]]; then
    echo "Missing ${PRETRAINED_MODEL}/${subdir}" >&2
    echo "PRETRAINED_MODEL must point to the LingBot-VA base directory, not the subdirectory itself." >&2
    exit 2
  fi
done

if [[ ! -f "${PRETRAINED_MODEL}/vae/config.json" || ! -f "${PRETRAINED_MODEL}/transformer/config.json" ]]; then
  echo "Missing model config files under ${PRETRAINED_MODEL}." >&2
  echo "Expected at least vae/config.json and transformer/config.json." >&2
  exit 2
fi

"${PYTHON_BIN}" tools/check_wanva_env.py

DEVICE="${DEVICE:-cuda:0}"
TEXT_DEVICE="${TEXT_DEVICE:-cpu}"
DTYPE="${DTYPE:-bf16}"
TARGET_FPS="${TARGET_FPS:-10}"
VAE_FRAME_BATCH="${VAE_FRAME_BATCH:-81}"
CAM_KEYS="${CAM_KEYS:-image,wrist_image}"

args=(
  --dataset-root "${DATASET_PATH}"
  --pretrained-model "${PRETRAINED_MODEL}"
  --device "${DEVICE}"
  --text-device "${TEXT_DEVICE}"
  --dtype "${DTYPE}"
  --target-fps "${TARGET_FPS}"
  --vae-frame-batch "${VAE_FRAME_BATCH}"
  --cam-keys "${CAM_KEYS}"
)

if [[ -n "${EPISODE_START:-}" ]]; then
  args+=(--episode-start "${EPISODE_START}")
fi
if [[ -n "${EPISODE_LIMIT:-}" ]]; then
  args+=(--episode-limit "${EPISODE_LIMIT}")
fi
if [[ -n "${MAX_SEGMENTS:-}" ]]; then
  args+=(--max-segments "${MAX_SEGMENTS}")
fi
if [[ -n "${MAX_VIDEO_FRAMES:-}" ]]; then
  args+=(--max-video-frames "${MAX_VIDEO_FRAMES}")
fi
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  args+=(--overwrite)
fi

"${PYTHON_BIN}" tools/extract_robomme_latents.py "${args[@]}"
check_args=(
  --dataset-root "${DATASET_PATH}"
  --cam-keys "${CAM_KEYS}"
  --require-norm-stats
)

if [[ -z "${EPISODE_START:-}" && -z "${EPISODE_LIMIT:-}" && -z "${MAX_SEGMENTS:-}" ]]; then
  check_args+=(--require-latents)
fi

"${PYTHON_BIN}" tools/check_robomme_lingbot.py "${check_args[@]}"
