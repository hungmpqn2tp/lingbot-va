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

On the server, download both RoboMME and `lingbot-va-base`, then repeat the same
metadata prep. Do not set `MAX_SEGMENTS`, `EPISODE_LIMIT`, or
`MAX_VIDEO_FRAMES` for the full latent extraction.

```bash
cd /path/to/lingbot-va

SOURCE_ROOT=/path/to/robomme_data_lerobot \
OUTPUT_ROOT=/path/to/robomme_data_lingbot \
bash script/prepare_robomme_lingbot.sh

DATASET_PATH=/path/to/robomme_data_lingbot \
PRETRAINED_MODEL=/path/to/lingbot-va-base \
bash script/extract_robomme_latents.sh
```

Then launch real LoRA training:

```bash
DATASET_PATH=/path/to/robomme_data_lingbot \
PRETRAINED_MODEL=/path/to/lingbot-va-base \
TRAIN_MODE=lora \
NGPU=8 \
SAVE_ROOT=/path/to/output/robomme_lora \
bash script/run_va_posttrain_robomme.sh
```

For full fine-tuning:

```bash
DATASET_PATH=/path/to/robomme_data_lingbot \
PRETRAINED_MODEL=/path/to/lingbot-va-base \
TRAIN_MODE=full \
NGPU=8 \
SAVE_ROOT=/path/to/output/robomme_full \
bash script/run_va_posttrain_robomme.sh \
  --batch-size 1 \
  --gradient-accumulation-steps 4
```

## Checks

Metadata and norm stats:

```bash
python tools/check_robomme_lingbot.py \
  --dataset-root ../robomme_data_lingbot \
  --require-norm-stats
```

Full latent coverage, for the server after full extraction only:

```bash
python tools/check_robomme_lingbot.py \
  --dataset-root /path/to/robomme_data_lingbot \
  --require-norm-stats \
  --require-latents
```

Do not use `--require-latents` for local smoke data, because only two segments
are intentionally extracted.
