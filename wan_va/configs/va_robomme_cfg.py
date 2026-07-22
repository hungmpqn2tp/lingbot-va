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


def validate_robomme_norm_stat(norm_stat, source="<memory>"):
    if not isinstance(norm_stat, dict):
        raise ValueError(f"RoboMME norm stats from {source} must be a dictionary")
    q01 = norm_stat.get("q01")
    q99 = norm_stat.get("q99")
    if not isinstance(q01, list) or not isinstance(q99, list):
        raise ValueError(f"RoboMME norm stats from {source} must contain q01/q99 lists")
    if len(q01) != 30 or len(q99) != 30:
        raise ValueError(
            f"RoboMME norm stats from {source} must have 30 channels, "
            f"got q01={len(q01)}, q99={len(q99)}"
        )
    if not all(float("-inf") < float(value) < float("inf") for value in q01 + q99):
        raise ValueError(f"RoboMME norm stats from {source} contain non-finite values")
    for channel in list(range(7)) + [28]:
        if float(q99[channel]) <= float(q01[channel]):
            raise ValueError(
                f"RoboMME norm stats from {source} have q99 <= q01 at channel {channel}"
            )
    return {"q01": [float(value) for value in q01], "q99": [float(value) for value in q99]}


def load_robomme_norm_stat(norm_stat_path):
    norm_stat_path = os.path.abspath(os.path.expanduser(str(norm_stat_path)))
    if not os.path.isfile(norm_stat_path):
        raise FileNotFoundError(f"Missing RoboMME normalization stats: {norm_stat_path}")

    with open(norm_stat_path, "r") as f:
        payload = json.load(f)

    if "model_q01" in payload and "model_q99" in payload:
        norm_stat = {"q01": payload["model_q01"], "q99": payload["model_q99"]}
    elif "q01" in payload and "q99" in payload:
        norm_stat = {"q01": payload["q01"], "q99": payload["q99"]}
    elif "action_q01" in payload and "action_q99" in payload:
        norm_stat = _model_channel_norm(payload["action_q01"], payload["action_q99"])
    else:
        raise ValueError(f"Unsupported RoboMME norm-stat format: {norm_stat_path}")
    return validate_robomme_norm_stat(norm_stat, source=norm_stat_path)


va_robomme_cfg = EasyDict(__name__='Config: VA RoboMME')
va_robomme_cfg.update(va_shared_cfg)
va_robomme_cfg.infer_mode = 'server'

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
va_robomme_cfg.initial_action_condition = 'model_zero'

(
    va_robomme_cfg.used_action_channel_ids,
    va_robomme_cfg.inverse_used_action_channel_ids,
) = _robomme_inverse_used_action_channel_ids()

va_robomme_cfg.action_norm_method = 'quantiles'
_default_dataset_path = os.environ.get("LINGBOT_ROBOMME_DATASET")
_default_norm_stat_path = os.environ.get("LINGBOT_ROBOMME_NORM_STAT")
if _default_norm_stat_path is None and _default_dataset_path:
    _default_norm_stat_path = os.path.join(
        _default_dataset_path,
        "meta",
        "lingbot_va_robomme_norm_stats.json",
    )
va_robomme_cfg.norm_stat_path = _default_norm_stat_path
# Keep loading lazy so CLI precedence is honored and unrelated configs import.
va_robomme_cfg.norm_stat = None
va_robomme_cfg.lora_adapter_path = os.environ.get("LINGBOT_LORA_ADAPTER")
va_robomme_cfg.require_lora = os.environ.get("LINGBOT_REQUIRE_LORA", "0").lower() in {
    "1",
    "true",
    "yes",
}
va_robomme_cfg.save_debug_artifacts = os.environ.get(
    "LINGBOT_SAVE_DEBUG_ARTIFACTS", "0"
).lower() in {
    "1", "true", "yes",
}
