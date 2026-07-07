#!/usr/bin/env python3
"""Merge a LingBot-VA LoRA adapter into a standalone eval model directory.

The training script saves LoRA checkpoints as:

    checkpoint_step_N/lora_adapter/{adapter_config.json,adapter_model.safetensors}

LingBot-VA inference loads a regular model root containing tokenizer/,
text_encoder/, vae/, and transformer/. This helper applies the adapter delta to
the base transformer's safetensors weights and writes such a model root.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


TRANSFORMER_WEIGHT = "diffusion_pytorch_model.safetensors"
TRANSFORMER_INDEX = "diffusion_pytorch_model.safetensors.index.json"
COMPONENT_DIRS = ("tokenizer", "text_encoder", "vae")


def _resolve_adapter_dir(path: Path) -> Path:
    if (path / "adapter_model.safetensors").exists():
        return path
    nested = path / "lora_adapter"
    if (nested / "adapter_model.safetensors").exists():
        return nested
    raise FileNotFoundError(
        "Could not find adapter_model.safetensors in "
        f"{path} or {nested}"
    )


def _weight_files(transformer_dir: Path) -> list[Path]:
    index_path = transformer_dir / TRANSFORMER_INDEX
    if index_path.exists():
        with index_path.open("r") as f:
            index = json.load(f)
        names = sorted(set(index["weight_map"].values()))
        return [transformer_dir / name for name in names]

    weight_path = transformer_dir / TRANSFORMER_WEIGHT
    if weight_path.exists():
        return [weight_path]

    raise FileNotFoundError(
        f"No {TRANSFORMER_WEIGHT} or {TRANSFORMER_INDEX} under {transformer_dir}"
    )


def _load_transformer_state(transformer_dir: Path) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for weight_file in _weight_files(transformer_dir):
        shard = load_file(weight_file, device="cpu")
        overlap = set(state).intersection(shard)
        if overlap:
            sample = ", ".join(sorted(overlap)[:5])
            raise ValueError(f"Duplicate keys while reading {weight_file}: {sample}")
        state.update(shard)
    return state


def _find_base_key(state: dict[str, torch.Tensor], prefix: str) -> str:
    candidates = [
        f"{prefix}.weight",
        f"{prefix}.base_layer.weight",
    ]
    for key in candidates:
        if key in state:
            return key
    raise KeyError(
        f"Could not find base weight for LoRA prefix {prefix!r}. "
        f"Tried: {', '.join(candidates)}"
    )


def _merge_lora(
    base_state: dict[str, torch.Tensor],
    adapter_state: dict[str, torch.Tensor],
    alpha: float,
) -> int:
    merged = 0
    for a_key, a_weight in sorted(adapter_state.items()):
        if not a_key.endswith(".lora_A.weight"):
            continue

        prefix = a_key[: -len(".lora_A.weight")]
        b_key = f"{prefix}.lora_B.weight"
        if b_key not in adapter_state:
            raise KeyError(f"Missing matching LoRA B weight for {a_key}")

        base_key = _find_base_key(base_state, prefix)
        base_weight = base_state[base_key]
        b_weight = adapter_state[b_key]

        rank = int(a_weight.shape[0])
        if rank <= 0:
            raise ValueError(f"Invalid LoRA rank for {a_key}: {rank}")
        if b_weight.shape[1] != rank:
            raise ValueError(
                f"LoRA shape mismatch for {prefix}: "
                f"A={tuple(a_weight.shape)}, B={tuple(b_weight.shape)}"
            )

        delta = b_weight.float() @ a_weight.float()
        delta = delta * (float(alpha) / rank)
        if delta.shape != base_weight.shape:
            raise ValueError(
                f"Merged delta shape mismatch for {prefix}: "
                f"delta={tuple(delta.shape)}, base={tuple(base_weight.shape)}"
            )

        base_state[base_key] = (base_weight.float() + delta).to(base_weight.dtype)
        merged += 1

    return merged


def _copy_or_link_component(src: Path, dst: Path, copy_components: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing base model component: {src}")
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if copy_components:
        shutil.copytree(src, dst)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=True)


def _write_config(base_transformer: Path, out_transformer: Path, attn_mode: str) -> None:
    config_src = base_transformer / "config.json"
    if not config_src.exists():
        raise FileNotFoundError(f"Missing transformer config: {config_src}")
    with config_src.open("r") as f:
        config = json.load(f)
    if "attn_mode" in config:
        config["attn_mode"] = attn_mode
    with (out_transformer / "config.json").open("w") as f:
        json.dump(config, f, indent=2)


def merge_adapter(args: argparse.Namespace) -> None:
    base_model = Path(args.base_model).expanduser().resolve()
    adapter_dir = _resolve_adapter_dir(Path(args.adapter).expanduser().resolve())
    output_model = Path(args.output_model).expanduser().resolve()

    base_transformer = base_model / "transformer"
    out_transformer = output_model / "transformer"

    if output_model.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_model} already exists. Pass --overwrite to replace it."
        )
    if output_model.exists():
        shutil.rmtree(output_model)

    output_model.mkdir(parents=True)
    out_transformer.mkdir(parents=True)

    with (adapter_dir / "adapter_config.json").open("r") as f:
        adapter_config = json.load(f)

    alpha = float(adapter_config["alpha"])
    print(f"Loading base transformer from {base_transformer}")
    base_state = _load_transformer_state(base_transformer)
    print(f"Loading LoRA adapter from {adapter_dir}")
    adapter_state = load_file(adapter_dir / "adapter_model.safetensors", device="cpu")

    merged_count = _merge_lora(base_state, adapter_state, alpha=alpha)
    if merged_count == 0:
        raise ValueError(f"No LoRA A/B weights found in {adapter_dir}")

    print(f"Merged {merged_count} LoRA modules")
    merged_path = out_transformer / TRANSFORMER_WEIGHT
    save_file(
        {key: value.contiguous() for key, value in base_state.items()},
        merged_path,
    )
    _write_config(base_transformer, out_transformer, args.attn_mode)

    for component in COMPONENT_DIRS:
        _copy_or_link_component(
            base_model / component,
            output_model / component,
            copy_components=args.copy_components,
        )

    with (output_model / "merged_lora_metadata.json").open("w") as f:
        json.dump(
            {
                "base_model": str(base_model),
                "adapter": str(adapter_dir),
                "alpha": alpha,
                "merged_modules": merged_count,
                "attn_mode": args.attn_mode,
            },
            f,
            indent=2,
        )

    print(f"Wrote merged eval model to {output_model}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a LingBot-VA LoRA adapter into an eval model root."
    )
    parser.add_argument(
        "--base-model",
        required=True,
        help="Base LingBot-VA model root containing transformer/, tokenizer/, text_encoder/, vae/.",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help="LoRA adapter dir, or checkpoint_step_N dir containing lora_adapter/.",
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="Output model root to create for inference/evaluation.",
    )
    parser.add_argument(
        "--attn-mode",
        default="torch",
        choices=("torch", "flashattn"),
        help="Inference attention mode to write into transformer/config.json.",
    )
    parser.add_argument(
        "--copy-components",
        action="store_true",
        help="Copy tokenizer/text_encoder/vae instead of symlinking them from the base model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output-model if it already exists.",
    )
    merge_adapter(parser.parse_args())


if __name__ == "__main__":
    main()
