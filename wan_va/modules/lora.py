# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import fnmatch
import json
import math
from pathlib import Path

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_layer, rank=16, alpha=32.0, dropout=0.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear modules")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")

        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base_layer.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base_layer.out_features, bias=False)

        for param in self.base_layer.parameters():
            param.requires_grad_(False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_input = self.dropout(x).to(self.lora_A.weight.dtype)
        lora_out = self.lora_B(self.lora_A(lora_input)) * self.scaling
        return base_out + lora_out.to(base_out.dtype)

    def merged_weight(self):
        delta = self.lora_B.weight @ self.lora_A.weight
        delta = delta * self.scaling
        return self.base_layer.weight.detach().float() + delta.detach().float()


def _get_parent_module(model, module_name):
    parent = model
    parts = module_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _matches(name, patterns):
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def inject_lora(model, target_patterns, rank=16, alpha=32.0, dropout=0.0):
    replaced = []
    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if not name or not isinstance(module, nn.Linear):
            continue
        if not _matches(name, target_patterns):
            continue
        parent, child_name = _get_parent_module(model, name)
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        if child_name.isdigit():
            parent[int(child_name)] = wrapped
        else:
            setattr(parent, child_name, wrapped)
        replaced.append(name)
    if not replaced:
        raise ValueError(
            "No Linear modules matched LoRA target patterns: "
            + ", ".join(target_patterns)
        )
    return replaced


def mark_only_lora_as_trainable(model):
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.weight.requires_grad_(True)
            module.lora_B.weight.requires_grad_(True)


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def resolve_lora_adapter_dir(path):
    path = Path(path).expanduser().resolve()
    for candidate in (path, path / "lora_adapter"):
        if (
            (candidate / "adapter_config.json").is_file()
            and (candidate / "adapter_model.safetensors").is_file()
        ):
            return candidate
    raise FileNotFoundError(
        "Could not find adapter_config.json and adapter_model.safetensors in "
        f"{path} or {path / 'lora_adapter'}"
    )


@torch.no_grad()
def merge_lora_adapter_into_model(model, adapter_path):
    """Merge a native LingBot-VA LoRA checkpoint into an unsharded model."""
    from safetensors.torch import load_file

    if getattr(model, "_lingbot_lora_merged", False):
        raise RuntimeError("A LoRA adapter has already been merged into this model")

    adapter_dir = resolve_lora_adapter_dir(adapter_path)
    with open(adapter_dir / "adapter_config.json", "r") as f:
        adapter_config = json.load(f)

    if adapter_config.get("format") != "lingbot-va-lora":
        raise ValueError(
            f"Unsupported LoRA format in {adapter_dir}: "
            f"{adapter_config.get('format')!r}"
        )

    adapter_state = load_file(
        adapter_dir / "adapter_model.safetensors",
        device="cpu",
    )
    base_model_name_or_path = adapter_config.get("base_model_name_or_path")
    if not isinstance(base_model_name_or_path, str) or not base_model_name_or_path:
        raise ValueError(
            f"LoRA adapter in {adapter_dir} has no base_model_name_or_path"
        )
    configured_modules = adapter_config.get("matched_modules")
    if not isinstance(configured_modules, list) or not configured_modules:
        raise ValueError(
            f"LoRA matched_modules in {adapter_dir} must be a non-empty list"
        )
    matched_modules = list(configured_modules)
    target_patterns = adapter_config.get("target_modules")
    if (
        not isinstance(target_patterns, list)
        or not target_patterns
        or any(not isinstance(pattern, str) or not pattern for pattern in target_patterns)
    ):
        raise ValueError(
            f"LoRA target_modules in {adapter_dir} must be a non-empty string list"
        )

    expected_keys = {
        key
        for name in matched_modules
        for key in (f"{name}.lora_A.weight", f"{name}.lora_B.weight")
    }
    actual_keys = set(adapter_state)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "LoRA adapter state is incomplete or contains unexpected tensors. "
            f"Missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    modules = dict(model.named_modules())
    expanded_targets = sorted(
        name
        for name, module in modules.items()
        if name and isinstance(module, nn.Linear) and _matches(name, target_patterns)
    )
    if set(expanded_targets) != set(matched_modules):
        missing = sorted(set(expanded_targets) - set(matched_modules))
        unexpected = sorted(set(matched_modules) - set(expanded_targets))
        raise ValueError(
            "LoRA adapter does not cover exactly the configured base targets. "
            f"Missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    try:
        alpha = float(adapter_config["alpha"])
        configured_rank = int(adapter_config["rank"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid LoRA rank/alpha in {adapter_dir}") from exc
    if configured_rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError(
            f"LoRA rank and alpha must be positive and alpha finite, got "
            f"rank={configured_rank}, alpha={alpha}"
        )
    if any(not isinstance(name, str) or not name for name in matched_modules):
        raise ValueError("LoRA matched_modules must contain non-empty strings")
    if len(set(matched_modules)) != len(matched_modules):
        raise ValueError("LoRA matched_modules contains duplicate entries")

    # Validate the entire adapter before changing any base-model weight. This
    # prevents a malformed late tensor from leaving a partially merged model.
    merge_plan = []
    for name in matched_modules:
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(
                f"LoRA target {name!r} is missing or is not nn.Linear in the base model"
            )

        a_weight = adapter_state[f"{name}.lora_A.weight"]
        b_weight = adapter_state[f"{name}.lora_B.weight"]
        if a_weight.ndim != 2 or b_weight.ndim != 2:
            raise ValueError(
                f"LoRA tensors for {name} must be matrices, got "
                f"A={tuple(a_weight.shape)}, B={tuple(b_weight.shape)}"
            )
        if not torch.isfinite(a_weight).all() or not torch.isfinite(b_weight).all():
            raise ValueError(f"LoRA tensors for {name} contain non-finite values")
        rank = int(a_weight.shape[0])
        if rank != configured_rank or b_weight.shape[1] != rank:
            raise ValueError(
                f"LoRA rank mismatch for {name}: config={configured_rank}, "
                f"A={tuple(a_weight.shape)}, B={tuple(b_weight.shape)}"
            )
        if a_weight.shape[1] != module.in_features or b_weight.shape[0] != module.out_features:
            raise ValueError(
                f"LoRA/base shape mismatch for {name}: A={tuple(a_weight.shape)}, "z
                f"B={tuple(b_weight.shape)}, base={tuple(module.weight.shape)}"
            )

        merge_plan.append((module, a_weight, b_weight))

    scale = alpha / configured_rank
    delta_l2 = 0.0
    for module, a_weight, b_weight in merge_plan:
        device = module.weight.device
        delta = b_weight.to(device=device, dtype=torch.float32) @ a_weight.to(
            device=device,
            dtype=torch.float32,
        )
        scaled_delta = delta * scale
        merged = module.weight.detach().float() + scaled_delta
        module.weight.copy_(merged.to(dtype=module.weight.dtype))
        delta_l2 += scaled_delta.square().sum().item()

    model._lingbot_lora_merged = True

    return {
        "adapter_dir": str(adapter_dir),
        "rank": configured_rank,
        "alpha": alpha,
        "scale": scale,
        "delta_l2": math.sqrt(delta_l2),
        "base_model_name_or_path": base_model_name_or_path,
        "matched_modules": matched_modules,
    }


def lora_state_dict(full_state_dict):
    return {
        key: value
        for key, value in full_state_dict.items()
        if ".lora_A." in key or ".lora_B." in key
    }


def merged_lora_state_dict(full_state_dict, alpha_by_prefix=None):
    alpha_by_prefix = alpha_by_prefix or {}
    merged = {}
    consumed = set()
    for key, value in full_state_dict.items():
        if key.endswith(".base_layer.weight"):
            prefix = key[: -len(".base_layer.weight")]
            a_key = f"{prefix}.lora_A.weight"
            b_key = f"{prefix}.lora_B.weight"
            if a_key in full_state_dict and b_key in full_state_dict:
                rank = full_state_dict[a_key].shape[0]
                alpha = alpha_by_prefix.get(prefix, rank)
                delta = full_state_dict[b_key].float() @ full_state_dict[a_key].float()
                merged[f"{prefix}.weight"] = (value.float() + delta * (alpha / rank)).to(
                    value.dtype
                )
                consumed.update({key, a_key, b_key})
                bias_key = f"{prefix}.base_layer.bias"
                if bias_key in full_state_dict:
                    merged[f"{prefix}.bias"] = full_state_dict[bias_key]
                    consumed.add(bias_key)
                continue

        if ".base_layer." in key:
            merged[key.replace(".base_layer.", ".")] = value
            consumed.add(key)
        elif ".lora_A." not in key and ".lora_B." not in key and key not in consumed:
            merged[key] = value
    return merged


def save_lora_checkpoint(
    full_state_dict,
    output_dir,
    config,
    target_modules,
    matched_modules,
):
    from safetensors.torch import save_file

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_state = lora_state_dict(full_state_dict)
    expected_keys = {
        key
        for name in matched_modules
        for key in (f"{name}.lora_A.weight", f"{name}.lora_B.weight")
    }
    if set(adapter_state) != expected_keys:
        missing = sorted(expected_keys - set(adapter_state))
        unexpected = sorted(set(adapter_state) - expected_keys)
        raise ValueError(
            "Gathered LoRA state does not match injected modules. "
            f"Missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    nonfinite = [
        key
        for key, value in adapter_state.items()
        if not torch.isfinite(value).all()
    ]
    if nonfinite:
        raise ValueError(f"Cannot save non-finite LoRA tensors: {nonfinite[:10]}")
    save_file(adapter_state, output_dir / "adapter_model.safetensors")
    adapter_config = {
        "format": "lingbot-va-lora",
        "base_model_name_or_path": config.wan22_pretrained_model_name_or_path,
        "rank": int(config.lora_rank),
        "alpha": float(config.lora_alpha),
        "dropout": float(config.lora_dropout),
        "target_modules": list(target_modules),
        "matched_modules": list(matched_modules),
    }
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)
