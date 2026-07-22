# RoboMME Fine-Tuning Workflow

This repo supports RoboMME LeRobot data through the `robomme_train` config.
RoboMME keeps its 8-D action column as `actions`; LingBot maps it into the
model's 30-D action layout as channels `0..6` plus gripper channel `28`.

There are two intended workflows:

- Local machine: smoke-test only with a tiny latent subset.
- Training server: extract all latents and run the real fine-tune.

## Local Smoke Test

Run these from the LingBot-VA repo root:

```bash
cd /mnt/ssd1tb/hung/projects/MemWAM/lingbot-va
```

If your shell already shows `(wanva)`, plain `python` should be fine. If a
script accidentally uses system Python, prefix commands with:

```bash
PYTHON_BIN=/home/hung/miniconda3/envs/wanva/bin/python
```

### 1. Prepare Metadata

This is fast and does not copy the 242GB dataset. It symlinks `data/`, adds
LingBot `action_config`, and computes action normalization stats.

```bash
SOURCE_ROOT=../robomme_data_lerobot \
OUTPUT_ROOT=../robomme_data_lingbot \
bash script/prepare_robomme_lingbot.sh
```

### 2. Extract Tiny Smoke Latents

Do not extract the full dataset locally. This command extracts only two action
segments and caps each segment to 17 video frames, enough to test the data path.

```bash
DATASET_PATH=../robomme_data_lingbot \
PRETRAINED_MODEL=../lingbot-va-base \
EPISODE_LIMIT=2 \
MAX_SEGMENTS=2 \
MAX_VIDEO_FRAMES=17 \
OVERWRITE=1 \
bash script/extract_robomme_latents.sh
```

Expected result:

```text
Latent extraction complete under .../robomme_data_lingbot/latents
RoboMME LingBot check passed
```

The training dataset loader automatically skips episodes whose latent files are
missing, so this partial latent folder is enough for smoke training.

### 3. Smoke Fine-Tune

LoRA smoke test:

```bash
DATASET_PATH=../robomme_data_lingbot \
PRETRAINED_MODEL=../lingbot-va-base \
TRAIN_MODE=lora \
NGPU=1 \
SAVE_ROOT=./train_out/robomme_smoke_lora \
bash script/run_va_posttrain_robomme.sh \
  --num-steps 1 \
  --save-interval 1 \
  --load-worker 0 \
  --batch-size 1
```

Full fine-tune smoke test:

```bash
DATASET_PATH=../robomme_data_lingbot \
PRETRAINED_MODEL=../lingbot-va-base \
TRAIN_MODE=full \
NGPU=1 \
SAVE_ROOT=./train_out/robomme_smoke_full \
bash script/run_va_posttrain_robomme.sh \
  --num-steps 1 \
  --save-interval 1 \
  --load-worker 0 \
  --batch-size 1
```

If this OOMs, check `nvidia-smi`. The local GPU may already be occupied; the
smoke data is tiny, but the LingBot transformer itself still needs VRAM.

## Training Server Full Run

Old checkpoints must not be reused: the corrected pipeline changes the
non-executed initial action condition to normalized/model-space zero in both
training and inference. Start a clean run from `lingbot-va-base`; LoRA resume
is intentionally rejected because adapter checkpoints do not contain optimizer
state.

Keep two environments separate:

- LingBot: Python 3.10, torch 2.9.x, torchvision 0.24.x, torchaudio 2.9.x,
  diffusers 0.36.x, and transformers 4.55.x.
- RoboMME benchmark: its Python 3.11+ `uv` environment.

### 1. Full Data Check

Run this once from the LingBot repo before training. It checks every Parquet
action, recomputes q01/q99, loads every camera latent, checks tensor finiteness
and temporal metadata, and requires the official full-dataset counts.

```bash
cd /path/to/lingbot-va
export PYTHON_BIN=/path/to/conda/envs/wanva/bin/python
export DATASET_PATH=/path/to/robomme_data_lingbot

"$PYTHON_BIN" tools/check_wanva_env.py --strict

"$PYTHON_BIN" tools/check_robomme_lingbot.py \
  --dataset-root "$DATASET_PATH" \
  --cam-keys image,wrist_image \
  --require-norm-stats \
  --require-latents \
  --check-actions \
  --verify-norm-stats \
  --expect-binary-gripper \
  --expect-episodes 1600 \
  --expect-segments 1600 \
  --expect-frames 768897 \
  --expect-tasks 116 \
  --expect-latent-files 3200 \
  --require-full-latent-frames
```

The final lines must report 1,600 episodes, 1,600 segments, 768,897 action
rows, verified normalization stats, and `RoboMME LingBot check passed`.
Do not use the strict latent/count flags on the intentionally partial local
smoke dataset.

### 2. Recommended Clean LoRA Run

Rank 16 / alpha 32 is the recommended first run. Rank 128 / alpha 256 is much
larger than needed for the initial experiment and does not fix a broken
conditioning or streaming path. This command has effective global batch size
`8 * 1 * 4 = 32` and saves candidates every 500 optimizer steps.

```bash
cd /path/to/lingbot-va
export PYTHON_BIN=/path/to/conda/envs/wanva/bin/python
export DATASET_PATH=/path/to/robomme_data_lingbot
export PRETRAINED_MODEL=/path/to/lingbot-va-base
export NORM_STAT_PATH="${DATASET_PATH}/meta/lingbot_va_robomme_norm_stats.json"
export TRAIN_MODE=lora
export NGPU=8
export MASTER_PORT=29501
export SAVE_ROOT=/path/to/output/robomme_lora_r16_a32

bash script/run_va_posttrain_robomme.sh \
  --lora-rank 16 \
  --lora-alpha 32 \
  --lora-dropout 0 \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --num-steps 3000 \
  --save-interval 500 \
  --load-worker 8
```

Set `RUN_DATA_CHECK=1` on this command only if the explicit full check was not
run separately. Evaluate steps 500, 1000, 1500, 2000, 2500, and 3000; choose by
closed-loop success rather than training loss alone.

### 3. Serve One Adapter Directly

The checkpoint root contains `lora_adapter/`; it is not a complete base
model. The server now validates and merges that adapter into the unsharded base
model before FSDP. `--require-lora` prevents silently evaluating the base.

```bash
cd /path/to/lingbot-va
export PYTHON_BIN=/path/to/conda/envs/wanva/bin/python
export PRETRAINED_MODEL=/path/to/lingbot-va-base
export NORM_STAT_PATH=/path/to/robomme_data_lingbot/meta/lingbot_va_robomme_norm_stats.json
export CHECKPOINT_ROOT=/path/to/output/robomme_lora_r16_a32/checkpoints/checkpoint_step_500
export CONFIG_NAME=robomme
export NGPU=8
export MASTER_PORT=29502
export PORT=29536

bash script/run_launch_va_server_sync.sh \
  --pretrained-model "$PRETRAINED_MODEL" \
  --norm-stat-path "$NORM_STAT_PATH" \
  --lora-adapter "$CHECKPOINT_ROOT" \
  --require-lora \
  --action-num-inference-steps 50 \
  --save-root /path/to/output/robomme_eval_server
```

The adapter argument may point either at `checkpoint_step_N/` or directly at
`checkpoint_step_N/lora_adapter/`. Do not enable
`--save-debug-artifacts` for a full evaluation; the per-chunk tensors are
large. Fifty action denoising steps is the quality-first setting; compare with
10 only if latency is a problem.

### 4. Start the RoboMME Adapter

In a second terminal, use the benchmark environment. The adapter now streams
the complete demonstration video, caches one observation per executed action,
and forwards the same thresholded gripper commands that were executed.

```bash
cd /path/to/robomme_benchmark
uv sync --group server
uv run python -m challenge_interface.scripts.deploy_lingbot \
  --transport websocket \
  --host 0.0.0.0 \
  --port 8001 \
  --lingbot-host 127.0.0.1 \
  --lingbot-port 29536
```

Use the model server's reachable IP instead of `127.0.0.1` if it is on a
different host.

### 5. Evaluate Again

Use a new `team_id` for every checkpoint and rerun. The evaluator resumes and
skips completed entries under `challenge_results/<team_id>`, so reusing an old
ID can silently reuse old results.

```bash
# Smoke test: one episode for each benchmark task.
cd /path/to/robomme_benchmark
uv run python -m challenge_interface.scripts.phase1_eval \
  --transport websocket \
  --host 127.0.0.1 \
  --port 8001 \
  --action_space joint_angle \
  --max_steps 300 \
  --num_episodes 1 \
  --team_id lingbot_lora_s0500_smoke_20260722

# Full evaluation for the selected checkpoint.
uv run python -m challenge_interface.scripts.phase1_eval \
  --transport websocket \
  --host 127.0.0.1 \
  --port 8001 \
  --action_space joint_angle \
  --max_steps 1500 \
  --num_episodes 10 \
  --team_id lingbot_lora_s0500_full_20260722
```

Use `127.0.0.1` or a real destination IP for clients; `0.0.0.0` is only a
server bind address.
