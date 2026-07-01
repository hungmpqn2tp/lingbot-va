# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os
from easydict import EasyDict

from .shared_config import va_shared_cfg


def _robomme_inverse_used_action_channel_ids():
    """Map LingBot's 30 action channels back to RoboMME's 8-D action vector."""
    used_action_channel_ids = list(range(0, 7)) + [28]
    inverse = [8] * 30
    for source_idx, model_channel in enumerate(used_action_channel_ids):
        inverse[model_channel] = source_idx
    return used_action_channel_ids, inverse


def _model_channel_norm(action_q01, action_q99):
    q01 = [0.0] * 30
    q99 = [1.0] * 30
    source_to_model = list(range(0, 7)) + [28]
    for source_idx, model_channel in enumerate(source_to_model):
        q01[model_channel] = float(action_q01[source_idx])
        q99[model_channel] = float(action_q99[source_idx])
    return {"q01": q01, "q99": q99}


def _load_norm_stat(dataset_path):
    norm_stat_path = os.environ.get(
        "LINGBOT_ROBOMME_NORM_STAT",
        os.path.join(dataset_path, "meta", "lingbot_va_robomme_norm_stats.json"),
    )
    if not os.path.exists(norm_stat_path):
        # Placeholder values keep the config importable. Run
        # tools/prepare_robomme_lingbot.py --compute-norm-stats before training.
        return _model_channel_norm([-1.0] * 8, [1.0] * 8)

    with open(norm_stat_path, "r") as f:
        payload = json.load(f)

    if "model_q01" in payload and "model_q99" in payload:
        return {"q01": payload["model_q01"], "q99": payload["model_q99"]}
    if "q01" in payload and "q99" in payload and len(payload["q01"]) == 30:
        return {"q01": payload["q01"], "q99": payload["q99"]}
    if "action_q01" in payload and "action_q99" in payload:
        return _model_channel_norm(payload["action_q01"], payload["action_q99"])
    raise ValueError(f"Unsupported RoboMME norm-stat format: {norm_stat_path}")


va_robomme_cfg = EasyDict(__name__='Config: VA RoboMME')
va_robomme_cfg.update(va_shared_cfg)

va_robomme_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "LINGBOT_VA_PRETRAINED", "/path/to/pretrained/model"
)

va_robomme_cfg.attn_window = 30
va_robomme_cfg.frame_chunk_size = 4
va_robomme_cfg.env_type = 'none'

va_robomme_cfg.height = 256
va_robomme_cfg.width = 256
va_robomme_cfg.action_dim = 30
va_robomme_cfg.action_source_dim = 8
va_robomme_cfg.action_per_frame = 4
va_robomme_cfg.action_key = 'actions'
va_robomme_cfg.obs_cam_keys = ['image', 'wrist_image']
va_robomme_cfg.guidance_scale = 5
va_robomme_cfg.action_guidance_scale = 1

va_robomme_cfg.num_inference_steps = 5
va_robomme_cfg.video_exec_step = -1
va_robomme_cfg.action_num_inference_steps = 10

va_robomme_cfg.snr_shift = 5.0
va_robomme_cfg.action_snr_shift = 1.0

(
    va_robomme_cfg.used_action_channel_ids,
    va_robomme_cfg.inverse_used_action_channel_ids,
) = _robomme_inverse_used_action_channel_ids()

va_robomme_cfg.action_norm_method = 'quantiles'
_default_dataset_path = os.environ.get(
    "LINGBOT_ROBOMME_DATASET", "/path/to/your/robomme_lingbot_dataset"
)
va_robomme_cfg.norm_stat = _load_norm_stat(_default_dataset_path)
