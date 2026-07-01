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
