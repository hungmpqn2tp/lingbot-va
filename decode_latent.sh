#!/bin/bash
export PYTHON_BIN=$(which python)
export BASE_MODEL=/root/autodl-tmp/lingbot-va-base
export DEBUG_DIR=/root/autodl-tmp/lingbot-va/debug_imagination_run/real
export OUT_DIR=/root/autodl-tmp/lingbot-va/imagination_videos

find "$DEBUG_DIR" -type f -name 'latents_*.pt' -printf '%h\0' | sort -zu | head -zn 5 | \
while IFS= read -r -d '' session_dir; do
  session_name=$(basename "$session_dir")
  "$PYTHON_BIN" tools/decode_robomme_imagination.py \
    --model "$BASE_MODEL" \
    --latent-dir "$session_dir" \
    --output-dir "$OUT_DIR/$session_name" \
    --device cuda:0 \
    --dtype bfloat16 \
    --fps 10 \
    --drop-initial-frame
done