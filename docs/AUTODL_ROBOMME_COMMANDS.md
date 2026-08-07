# LingBot-VA on RoboMME: AutoDL Commands

This runbook assumes the following AutoDL layout:

```text
/root/autodl-tmp/lingbot-va
/root/autodl-tmp/lingbot-va-base
/root/autodl-tmp/robomme_benchmark
/root/autodl-tmp/robomme_data_lerobot
/root/autodl-tmp/robomme_data_lingbot
```

LingBot uses `/root/miniconda3/envs/wanva/bin/python`. RoboMME uses its own
`uv` environment. Run long commands in `tmux`, and keep data, caches, and
outputs under `/root/autodl-tmp`.

## 1. Prepare RoboMME Data

Skip these commands when the prepared dataset already has complete `latents/`
and `meta/lingbot_va_robomme_norm_stats.json`.

```bash
cd /root/autodl-tmp/lingbot-va
export PYTHON_BIN=/root/miniconda3/envs/wanva/bin/python

SOURCE_ROOT=/root/autodl-tmp/robomme_data_lerobot \
OUTPUT_ROOT=/root/autodl-tmp/robomme_data_lingbot \
PYTHON_BIN="$PYTHON_BIN" \
bash script/prepare_robomme_lingbot.sh

DATASET_PATH=/root/autodl-tmp/robomme_data_lingbot \
PRETRAINED_MODEL=/root/autodl-tmp/lingbot-va-base \
PYTHON_BIN="$PYTHON_BIN" DEVICE=cuda:0 \
bash script/extract_robomme_latents.sh
```

Extraction is restartable when `OVERWRITE=1` is not set.

## 2. Common Training Variables

```bash
cd /root/autodl-tmp/lingbot-va
export PYTHON_BIN=/root/miniconda3/envs/wanva/bin/python
export DATASET_PATH=/root/autodl-tmp/robomme_data_lingbot
export PRETRAINED_MODEL=/root/autodl-tmp/lingbot-va-base
export NORM_STAT_PATH="$DATASET_PATH/meta/lingbot_va_robomme_norm_stats.json"
export NGPU=8
export MASTER_PORT=29501
```

Optional strict check before a full run:

```bash
"$PYTHON_BIN" tools/check_wanva_env.py --strict
"$PYTHON_BIN" tools/check_robomme_lingbot.py \
  --dataset-root "$DATASET_PATH" \
  --cam-keys image,wrist_image \
  --require-norm-stats --require-latents \
  --check-actions --verify-norm-stats \
  --expect-binary-gripper \
  --expect-episodes 1600 --expect-segments 1600 \
  --expect-frames 768897 --expect-tasks 116 \
  --expect-latent-files 3200 --require-full-latent-frames
```

## 3. Train LoRA

```bash
export TRAIN_MODE=lora
export SAVE_ROOT=/root/autodl-tmp/lingbot-va/train_out/robomme_lora_r16_a32

bash script/run_va_posttrain_robomme.sh \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0 \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-steps 3000 \
  --save-interval 500 \
  --load-worker 2
```

LoRA checkpoints are saved under
`$SAVE_ROOT/checkpoints/checkpoint_step_N/lora_adapter`.

## 4. Fine-Tune the Full Model

```bash
export TRAIN_MODE=full
export SAVE_ROOT=/root/autodl-tmp/lingbot-va/train_out/robomme_full

bash script/run_va_posttrain_robomme.sh \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-steps 3000 \
  --save-interval 500 \
  --load-worker 2
```

The effective global batch is
`NGPU * batch-size * gradient-accumulation-steps`. Full training requires much
more VRAM and checkpoint space than LoRA.

## 5. Evaluate a Checkpoint

Evaluation uses three terminals.

### Terminal 1: LingBot server for LoRA

```bash
cd /root/autodl-tmp/lingbot-va
export PYTHON_BIN=/root/miniconda3/envs/wanva/bin/python
export PRETRAINED_MODEL=/root/autodl-tmp/lingbot-va-base
export NORM_STAT_PATH=/root/autodl-tmp/robomme_data_lingbot/meta/lingbot_va_robomme_norm_stats.json
export CHECKPOINT_ROOT=/root/autodl-tmp/lingbot-va/train_out/robomme_lora_r16_a32/checkpoints/checkpoint_step_500

CONFIG_NAME=robomme NGPU=8 MASTER_PORT=29502 PORT=29536 \
bash script/run_launch_va_server_sync.sh \
  --pretrained-model "$PRETRAINED_MODEL" \
  --norm-stat-path "$NORM_STAT_PATH" \
  --lora-adapter "$CHECKPOINT_ROOT" \
  --require-lora \
  --action-num-inference-steps 50 \
  --save-root /root/autodl-tmp/lingbot-va/train_out/robomme_eval_s0500
```

For a full-model checkpoint, link the unchanged base components once and serve
the checkpoint without LoRA flags:

```bash
cd /root/autodl-tmp/lingbot-va
export PYTHON_BIN=/root/miniconda3/envs/wanva/bin/python
export BASE_MODEL=/root/autodl-tmp/lingbot-va-base
export FULL_CHECKPOINT=/root/autodl-tmp/lingbot-va/train_out/robomme_full/checkpoints/checkpoint_step_500
export NORM_STAT_PATH=/root/autodl-tmp/robomme_data_lingbot/meta/lingbot_va_robomme_norm_stats.json

ln -sfn "$BASE_MODEL/vae" "$FULL_CHECKPOINT/vae"
ln -sfn "$BASE_MODEL/text_encoder" "$FULL_CHECKPOINT/text_encoder"
ln -sfn "$BASE_MODEL/tokenizer" "$FULL_CHECKPOINT/tokenizer"

CONFIG_NAME=robomme NGPU=8 MASTER_PORT=29502 PORT=29536 \
bash script/run_launch_va_server_sync.sh \
  --pretrained-model "$FULL_CHECKPOINT" \
  --norm-stat-path "$NORM_STAT_PATH" \
  --action-num-inference-steps 50 \
  --save-root /root/autodl-tmp/lingbot-va/train_out/robomme_full_eval_s0500
```

### Terminal 2: RoboMME policy adapter

```bash
cd /root/autodl-tmp/robomme_benchmark
uv sync --group server
uv run python -m challenge_interface.scripts.deploy_lingbot \
  --transport websocket \
  --host 0.0.0.0 \
  --port 8001 \
  --lingbot-host 127.0.0.1 \
  --lingbot-port 29536
```

### Terminal 3: evaluator

Use a new `team_id` for every checkpoint. Reusing one resumes old results.

```bash
cd /root/autodl-tmp/robomme_benchmark

# One episode per task: smoke evaluation
uv run python -m challenge_interface.scripts.phase1_eval \
  --transport websocket \
  --host 127.0.0.1 \
  --port 8001 \
  --action_space joint_angle \
  --max_steps 300 \
  --num_episodes 1 \
  --team_id lingbot_s0500_smoke

# Ten episodes per task: full evaluation
uv run python -m challenge_interface.scripts.phase1_eval \
  --transport websocket \
  --host 127.0.0.1 \
  --port 8001 \
  --action_space joint_angle \
  --max_steps 1500 \
  --num_episodes 10 \
  --team_id lingbot_s0500_full
```

Use `0.0.0.0` only for server binding. Clients use `127.0.0.1` on the same
machine or the server's reachable IP on another machine.

## 6. Decode Imagination Videos

The server saves `latents_*.pt` only with `--save-debug-artifacts`. Add that
flag to the Terminal 1 server command for a short smoke run. Do not enable it
for a full evaluation because the artifacts are large.

```bash
# Add these arguments to the LingBot server command:
--save-debug-artifacts \
--save-root /root/autodl-tmp/lingbot-va/train_out/robomme_debug_s0500
```

After one evaluation episode:

```bash
cd /root/autodl-tmp/lingbot-va
export PYTHON_BIN=/root/miniconda3/envs/wanva/bin/python

find /root/autodl-tmp/lingbot-va/train_out/robomme_debug_s0500/real \
  -type f -name 'latents_*.pt' | head

export SESSION_DIR=/root/autodl-tmp/lingbot-va/train_out/robomme_debug_s0500/real/SESSION_NAME
export VIDEO_DIR="$SESSION_DIR/decoded"

"$PYTHON_BIN" tools/decode_robomme_imagination.py \
  --model /root/autodl-tmp/lingbot-va-base \
  --latent-dir "$SESSION_DIR" \
  --output-dir "$VIDEO_DIR" \
  --device cuda:0 \
  --dtype bfloat16 \
  --fps 10 \
  --drop-initial-frame
```

Replace `SESSION_NAME` with the directory found above. Outputs include
`imagination_combined.mp4`, `imagination_front.mp4`, and
`imagination_wrist.mp4`, plus one MP4 per chunk. Add
`--no-split-cameras` when the decoded frames are not a side-by-side
front/wrist layout.

## AutoDL Checks

```bash
nvidia-smi
df -h /root/autodl-tmp
ss -ltnp | grep 29536
ss -ltnp | grep 8001
```

Use a different `MASTER_PORT` for simultaneous distributed jobs. Also ensure
ports 29536 and 8001 are free and reachable when components run on different
hosts.
