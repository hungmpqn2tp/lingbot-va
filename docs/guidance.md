# Final finetune guidelines

This picks up from a LoRA checkpoint that already carries the original 20k-step
full fine-tune plus a 20k-step latent-only phase (merged into one standalone
model root: `lingbot-va-base-merged-s20000-latent20k`). This guide sets up a
**new** server from scratch and runs `--train-mode full` for 30k more steps,
then evaluates the result.

All paths below are relative to one anchor variable, `$WORKDIR`, so this
guide ports to the next server by changing one line.

```bash
export WORKDIR=~/robomme_workspace
mkdir -p "$WORKDIR"
```

Everything else in this doc assumes `cd "$WORKDIR"` has already happened
once per shell, and paths are written relative to it.

## 0. Layout this guide assumes

```text
$WORKDIR/lingbot-va                                        # this repo
$WORKDIR/robomme_benchmark                                 # eval harness repo
$WORKDIR/data/robomme_data_lingbot                          # prepared dataset (latents + norm stats)
$WORKDIR/checkpoints/lingbot-va-base-merged-s20000-latent20k # input model for this phase
$WORKDIR/train_out/robomme_phase2_full                      # this phase's checkpoints
```

## 1. One-time environment setup on the new server

```bash
cd "$WORKDIR"
git clone https://github.com/hungmpqn2tp/lingbot-va.git lingbot-va
git clone https://github.com/hungmpqn2tp/robomme_benchmark.git robomme_benchmark
```

Follow `lingbot-va/INSTALL.md` to create the conda environment (this repo's
docs elsewhere call it `wanva` or `lingbot` depending on how you name it --
name doesn't matter, just be consistent). Confirm the interpreter path before
using it anywhere else:

```bash
conda activate <your-env-name>
which python
```

Set that path once:

```bash
export PYTHON_BIN="$(which python)"
```

Sanity-check the install:

```bash
cd "$WORKDIR/lingbot-va"
"$PYTHON_BIN" tools/check_wanva_env.py --strict
```

## 2. Place the required inputs

Two things need to land on this box: the merged checkpoint and the prepared
dataset. The dataset still has to come over via scp/rsync (it's not on
ModelScope), but the checkpoint was pushed to ModelScope earlier specifically
so pulling it here is a single command instead of a multi-GB scp.

### Pull the checkpoint from ModelScope

```bash
pip install modelscope   # if not already present in this env
export MODELSCOPE_ENDPOINT=https://www.modelscope.ai   
mkdir -p "$WORKDIR/checkpoints"
```

```bash
python -c "
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    model_id='lonewolf134/lingbot-va-base-merged-s20000-latent20k',
    local_dir='$WORKDIR/checkpoints/lingbot-va-base-merged-s20000-latent20k',
    token=<your_model_scope_token>, 
)
"
```


Verify it actually landed complete before moving on -- `transformer/`,
`vae/`, `tokenizer/`, `text_encoder/` should all be present:

```bash
ls "$WORKDIR/checkpoints/lingbot-va-base-merged-s20000-latent20k"
```

### Get the dataset 

**1. Pull the raw RoboMME LeRobot dataset from Hugging Face:**

```bash
pip install -U huggingface_hub
huggingface-cli download Yinpei/robomme_data_lerobot \
  --repo-type dataset \
  --local-dir "$WORKDIR/data/robomme_data_lerobot"
```

**2. Convert LeRobot -> LingBot format** with `tools/prepare_robomme_lingbot.py`,
via its `script/prepare_robomme_lingbot.sh` wrapper (which also runs the
post-conversion check):

```bash
cd "$WORKDIR/lingbot-va"
export SOURCE_ROOT="$WORKDIR/data/robomme_data_lerobot"
export OUTPUT_ROOT="$WORKDIR/data/robomme_data_lingbot"
export PYTHON_BIN="$PYTHON_BIN"
bash script/prepare_robomme_lingbot.sh
```


**3. Extract latents.** This needs a LingBot-VA base model with a `vae/`, this is the one place in this whole doc where you *do* need
`lingbot-va-base` on this machine, even though the note below says training
itself doesn't need it. Pull it from Hugging Face if it isn't already here:

```bash
huggingface-cli download lerobot/lingbot_va_base \
  --local-dir "$WORKDIR/checkpoints/lingbot-va-base"
```

```bash
export BASE_MODEL="$WORKDIR/checkpoints/lingbot-va-base"
DATASET_PATH="$OUTPUT_ROOT" \
PRETRAINED_MODEL="$BASE_MODEL" \
PYTHON_BIN="$PYTHON_BIN" DEVICE=cuda:0 \
bash script/extract_robomme_latents.sh
```

Extraction is restartable when `OVERWRITE=1` is not set. Once it finishes,
`$WORKDIR/data/robomme_data_lingbot` should satisfy the same
`latents/` + `meta/lingbot_va_robomme_norm_stats.json` requirement as
Option A, and the full data check below should pass.

---

Run the full data check once:

```bash
cd "$WORKDIR/lingbot-va"
"$PYTHON_BIN" tools/check_robomme_lingbot.py \
  --dataset-root "$WORKDIR/data/robomme_data_lingbot" \
  --cam-keys image,wrist_image \
  --require-norm-stats --require-latents \
  --check-actions --verify-norm-stats \
  --expect-binary-gripper \
  --expect-episodes 1600 --expect-segments 1600 \
  --expect-frames 768897 --expect-tasks 116 \
  --expect-latent-files 3200 --require-full-latent-frames
```

## 4. Launch the 30k-step run

Use the `--batch-size` suitable for your setup; pick
`--gradient-accumulation-steps` to hit whatever global batch you want
(`global_batch = NGPU * batch_size * grad_accum`, i.e. `4 * batch_size * grad_accum`
here).

```bash
tmux new-session -d -s phase2_train "
cd $WORKDIR/lingbot-va
export PYTHON_BIN=$PYTHON_BIN
export DATASET_PATH=$WORKDIR/data/robomme_data_lingbot
export PRETRAINED_MODEL=$WORKDIR/checkpoints/lingbot-va-base-merged-s20000-latent20k
export NORM_STAT_PATH=\$DATASET_PATH/meta/lingbot_va_robomme_norm_stats.json
export TRAIN_MODE=full
export NGPU=4
export SAVE_ROOT=$WORKDIR/train_out/robomme_phase2_full
bash script/run_va_posttrain_robomme.sh \
  --learning-rate 1e-5 \
  --batch-size <your value> \
  --gradient-accumulation-steps <your value> \
  --num-steps 30000 \
  --save-interval 2000 \
  --load-worker 2 \
  2>&1 | tee $WORKDIR/train_out/phase2_train_log.txt
"
```

## 5. Watch the loss

```bash
cd "$WORKDIR/lingbot-va"
"$PYTHON_BIN" tools/plot_train_loss.py \
  "$WORKDIR/train_out/phase2_train_log.txt" \
  --metrics latent_loss,action_loss \
  --smooth 20 \
  --out "$WORKDIR/train_out/phase2_loss.png"
```

## 6. Prepare the finished checkpoint for evaluation


```bash
export FULL_CHECKPOINT="$WORKDIR/train_out/robomme_phase2_full/checkpoints/checkpoint_step_30000"
export BASE_MODEL="$WORKDIR/checkpoints/lingbot-va-base-merged-s20000-latent20k"

ln -sfn "$BASE_MODEL/vae" "$FULL_CHECKPOINT/vae"
ln -sfn "$BASE_MODEL/text_encoder" "$FULL_CHECKPOINT/text_encoder"
ln -sfn "$BASE_MODEL/tokenizer" "$FULL_CHECKPOINT/tokenizer"
```

`$FULL_CHECKPOINT` is now a complete, servable model root.


## 7. Evaluate the checkpoint

Three separate tmux sessions: the LingBot server, the RoboMME<->LingBot
bridge, and the evaluator.

```bash
tmux new-session -d -s eval_server "
cd $WORKDIR/lingbot-va
export PYTHON_BIN=$PYTHON_BIN
export PRETRAINED_MODEL=$FULL_CHECKPOINT
export NORM_STAT_PATH=$WORKDIR/data/robomme_data_lingbot/meta/lingbot_va_robomme_norm_stats.json
CONFIG_NAME=robomme NGPU=4 MASTER_PORT=29502 PORT=29536 \
bash script/run_launch_va_server_sync.sh \
  --pretrained-model \"\$PRETRAINED_MODEL\" \
  --norm-stat-path \"\$NORM_STAT_PATH\" \
  --action-num-inference-steps 50 \
  --save-root $WORKDIR/train_out/eval_run
"
```

`CONFIG_NAME=robomme` is set inline on purpose, training used
`robomme_train` (which has optimizer settings serving doesn't need), and this
inline prefix wins over anything left exported in the shell from an earlier
training command.

```bash
tmux new-session -d -s eval_bridge "
cd $WORKDIR/robomme_benchmark
uv sync --group server
uv run python -m challenge_interface.scripts.deploy_lingbot \
  --transport websocket --host 0.0.0.0 --port 8001 \
  --lingbot-host 127.0.0.1 --lingbot-port 29536
"

tmux new-session -d -s eval_run -c "$WORKDIR/robomme_benchmark"
```

Wait for `eval_server` to finish loading the model and `eval_bridge` to
connect, then attach to `eval_run` and fire the evaluator (always a fresh
`--team_id`, reusing one resumes/skips old results instead of starting over):

```bash
tmux attach -t eval_run
```
```bash
# smoke: 1 episode/task
uv run python -m challenge_interface.scripts.phase1_eval \
  --transport websocket --host 127.0.0.1 --port 8001 \
  --action_space joint_angle --max_steps 300 --num_episodes 1 \
  --team_id phase2_full_s30000_smoke

# full: 10 episodes/task
uv run python -m challenge_interface.scripts.phase1_eval \
  --transport websocket --host 127.0.0.1 --port 8001 \
  --action_space joint_angle --max_steps 1500 --num_episodes 10 \
  --team_id phase2_full_s30000_full
```

Results land in `$WORKDIR/robomme_benchmark/challenge_results/<team_id>/`.

## 8. Cleanup

```bash
tmux kill-session -t eval_server
tmux kill-session -t eval_bridge
tmux kill-session -t eval_run
```
