#!/usr/bin/env python3
"""Decode RoboMME imagination latents saved by the LingBot-VA server."""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
WAN_VA_ROOT = REPO_ROOT / "wan_va"
sys.path.insert(0, str(WAN_VA_ROOT))

from modules.utils import load_vae  # noqa: E402


LATENT_FILE_RE = re.compile(r"latents_(\d+)\.pt$")


def _frame_start(path: Path) -> int:
    match = LATENT_FILE_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected latent filename: {path.name}")
    return int(match.group(1))


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("Use --dtype float32 when decoding on CPU")
    return dtype


def _load_latent(path: Path) -> torch.Tensor:
    latent = torch.load(path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(latent):
        raise TypeError(f"{path} does not contain a tensor")
    if latent.ndim != 5 or latent.shape[0] != 1:
        raise ValueError(
            f"Expected {path} to have shape [1, C, F, H, W], got {tuple(latent.shape)}"
        )
    if not torch.isfinite(latent).all():
        raise ValueError(f"{path} contains non-finite values")
    return latent


@torch.inference_mode()
def _decode_latent(vae, processor: VideoProcessor, latent: torch.Tensor) -> np.ndarray:
    device = next(vae.parameters()).device
    latent = latent.to(device=device, dtype=vae.dtype)
    mean = torch.as_tensor(
        vae.config.latents_mean, device=device, dtype=vae.dtype
    ).view(1, vae.config.z_dim, 1, 1, 1)
    inverse_std = (
        1.0
        / torch.as_tensor(vae.config.latents_std, device=device, dtype=vae.dtype)
    ).view(1, vae.config.z_dim, 1, 1, 1)
    latent = latent / inverse_std + mean
    video = vae.decode(latent, return_dict=False)[0]
    frames = processor.postprocess_video(video, output_type="np")[0]
    return np.asarray(frames)


def _export(frames: np.ndarray, path: Path, fps: int) -> None:
    if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
        raise ValueError(f"Cannot export frames with shape {frames.shape} to {path}")
    export_to_video(frames, str(path), fps=fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode latents_*.pt files saved by RoboMME LingBot inference."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="LingBot-VA base model directory containing vae/.",
    )
    parser.add_argument(
        "--latent-dir",
        required=True,
        type=Path,
        help="Session directory containing latents_*.pt.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--device", default="auto", help="Torch device, for example auto, cuda, cuda:1, or cpu."
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--drop-initial-frame",
        action="store_true",
        help="Remove the first decoded RGB frame (the initial F0 observation condition).",
    )
    parser.add_argument(
        "--no-split-cameras",
        action="store_true",
        help="Do not split the combined front/wrist video into equal-width views.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    vae_path = args.model.expanduser().resolve() / "vae"
    latent_dir = args.latent_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not vae_path.is_dir():
        raise FileNotFoundError(f"Missing VAE directory: {vae_path}")
    if not latent_dir.is_dir():
        raise FileNotFoundError(f"Missing latent directory: {latent_dir}")

    latent_files = sorted(latent_dir.glob("latents_*.pt"), key=_frame_start)
    if not latent_files:
        raise FileNotFoundError(f"No latents_*.pt files found in {latent_dir}")

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading VAE from {vae_path} on {device} with {dtype}")
    vae = load_vae(str(vae_path), torch_dtype=dtype, torch_device=device)
    vae.eval()
    processor = VideoProcessor(vae_scale_factor=1)

    decoded_chunks = []
    for index, latent_file in enumerate(latent_files):
        print(f"Decoding {latent_file.name}")
        frames = _decode_latent(vae, processor, _load_latent(latent_file))
        if index == 0 and args.drop_initial_frame:
            if len(frames) < 2:
                raise ValueError("The first chunk has no future frame after dropping F0")
            frames = frames[1:]
        decoded_chunks.append(frames)
        _export(frames, output_dir / f"{latent_file.stem}.mp4", args.fps)

    combined = np.concatenate(decoded_chunks, axis=0)
    _export(combined, output_dir / "imagination_combined.mp4", args.fps)

    if not args.no_split_cameras:
        width = combined.shape[2]
        if width % 2:
            raise ValueError(
                f"Decoded width {width} is not divisible by two; use --no-split-cameras"
            )
        midpoint = width // 2
        _export(combined[:, :, :midpoint], output_dir / "imagination_front.mp4", args.fps)
        _export(combined[:, :, midpoint:], output_dir / "imagination_wrist.mp4", args.fps)

    print(f"Decoded {len(latent_files)} chunks ({len(combined)} RGB frames) into {output_dir}")


if __name__ == "__main__":
    main()
