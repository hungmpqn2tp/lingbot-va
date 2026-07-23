# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from easydict import EasyDict

from .va_robomme_cfg import va_robomme_cfg

va_robomme_train_cfg = EasyDict(__name__='Config: VA RoboMME train')
va_robomme_train_cfg.update(va_robomme_cfg)

va_robomme_train_cfg.dataset_path = os.environ.get(
    "LINGBOT_ROBOMME_DATASET", "/path/to/your/robomme_lingbot_dataset"
)
va_robomme_train_cfg.empty_emb_path = os.environ.get(
    "LINGBOT_ROBOMME_EMPTY_EMB",
    os.path.join(va_robomme_train_cfg.dataset_path, "empty_emb.pt"),
)

va_robomme_train_cfg.enable_wandb = False
# This value is per rank; 2 means 16 DataLoader workers on an 8-GPU run.
va_robomme_train_cfg.load_worker = 2
va_robomme_train_cfg.save_interval = 1000
va_robomme_train_cfg.gc_interval = 50
va_robomme_train_cfg.cfg_prob = 0.1

# Training parameters
va_robomme_train_cfg.learning_rate = 1e-5
va_robomme_train_cfg.beta1 = 0.9
va_robomme_train_cfg.beta2 = 0.95
va_robomme_train_cfg.weight_decay = 0.1
va_robomme_train_cfg.warmup_steps = 10
va_robomme_train_cfg.batch_size = 1
va_robomme_train_cfg.gradient_accumulation_steps = 1
va_robomme_train_cfg.num_steps = 50000

# Fine-tuning mode. Use "full" for standard FSDP fine-tuning or "lora" for
# trainable low-rank adapters on top of the frozen transformer.
va_robomme_train_cfg.train_mode = os.environ.get("LINGBOT_TRAIN_MODE", "full")
va_robomme_train_cfg.lora_rank = int(os.environ.get("LINGBOT_LORA_RANK", "16"))
va_robomme_train_cfg.lora_alpha = float(os.environ.get("LINGBOT_LORA_ALPHA", "32"))
va_robomme_train_cfg.lora_dropout = float(os.environ.get("LINGBOT_LORA_DROPOUT", "0.0"))
va_robomme_train_cfg.lora_target_modules = [
    "blocks.*.attn1.to_q",
    "blocks.*.attn1.to_k",
    "blocks.*.attn1.to_v",
    "blocks.*.attn1.to_out.0",
    "blocks.*.attn2.to_q",
    "blocks.*.attn2.to_k",
    "blocks.*.attn2.to_v",
    "blocks.*.attn2.to_out.0",
    "action_embedder",
    "action_proj_out",
]
va_robomme_train_cfg.lora_save_merged = False
