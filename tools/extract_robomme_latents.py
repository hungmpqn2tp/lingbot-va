#!/usr/bin/env python3
"""Extract LingBot-VA latents from a prepared RoboMME LeRobot dataset."""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "wan_va"))

try:
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
except Exception:  # pragma: no cover - older diffusers fallback
    def prompt_clean(text):
        return text


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read RoboMME parquet files. Install the "
            "post-training dependencies first, e.g. `pip install pyarrow datasets lerobot==0.3.3`."
        ) from exc
    return pq


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_file(dataset_root, info, episode_index):
    chunk = episode_index // int(info["chunks_size"])
    rel = info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    return Path(dataset_root) / rel


def episode_chunk(info, episode_index):
    return episode_index // int(info["chunks_size"])


def torch_dtype(name):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def decode_image(value, dataset_root):
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            value = value["bytes"]
        elif value.get("path") is not None:
            with Image.open(Path(dataset_root) / value["path"]) as image:
                return np.asarray(image.convert("RGB"))
        else:
            raise ValueError(f"Unsupported image dict keys: {list(value)}")
    if isinstance(value, (bytes, bytearray, memoryview)):
        with Image.open(io.BytesIO(bytes(value))) as image:
            return np.asarray(image.convert("RGB"))
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"))
    array = np.asarray(value)
    if array.ndim == 3:
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return array
    raise ValueError(f"Unsupported image value type/shape: {type(value)} {array.shape}")


def sample_frame_ids(start_frame, end_frame, ori_fps, target_fps):
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")
    stride = max(1, int(round(float(ori_fps) / float(target_fps))))
    frame_ids = list(range(start_frame, end_frame, stride))
    if len(frame_ids) < 2 and end_frame - start_frame >= 2:
        frame_ids = [start_frame, end_frame - 1]
    if len(frame_ids) < 2:
        raise ValueError(
            f"Segment {start_frame}:{end_frame} is too short after sampling; "
            "use longer action_config segments"
        )
    return frame_ids, stride


def align_wan_frame_ids(frame_ids, max_video_frames=None):
    if max_video_frames is not None:
        frame_ids = frame_ids[:max_video_frames]
    aligned_len = ((len(frame_ids) - 1) // 4) * 4 + 1
    frame_ids = frame_ids[:aligned_len]
    if len(frame_ids) < 5:
        raise ValueError(
            "Wan VAE latent extraction needs at least 5 sampled frames "
            f"after 4n+1 alignment, got {len(frame_ids)}"
        )
    return frame_ids


def load_frames(dataset_root, parquet_path, cam_key, frame_ids):
    pq = _require_pyarrow()
    column = pq.read_table(parquet_path, columns=[cam_key]).column(cam_key).to_pylist()
    return [decode_image(column[idx], dataset_root) for idx in frame_ids]


@torch.no_grad()
def encode_text(text, tokenizer, text_encoder, device, dtype, max_sequence_length=512):
    text = prompt_clean(text or "")
    text_inputs = tokenizer(
        [text],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids, mask).last_hidden_state.to(dtype=dtype)
    prompt_embeds = [emb[:seq_len] for emb, seq_len in zip(prompt_embeds, seq_lens)]
    prompt_embeds = torch.stack(
        [
            torch.cat(
                [emb, emb.new_zeros(max_sequence_length - emb.size(0), emb.size(1))]
            )
            for emb in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds[0].detach().cpu()


def normalize_latents(latents, vae):
    latents_mean = torch.tensor(vae.config.latents_mean, device=latents.device)
    latents_std = torch.tensor(vae.config.latents_std, device=latents.device)
    latents_mean = latents_mean.view(1, -1, 1, 1, 1)
    latents_std = latents_std.view(1, -1, 1, 1, 1)
    return ((latents.float() - latents_mean) / latents_std).to(latents.dtype)


def frames_to_video_tensor(frames, height, width):
    video = torch.from_numpy(np.stack(frames)).float().permute(3, 0, 1, 2)
    video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
    return video.unsqueeze(0) / 255.0 * 2.0 - 1.0


@torch.no_grad()
def encode_frames(frames, vae, device, dtype, height, width):
    source_height, source_width = frames[0].shape[:2]
    video = frames_to_video_tensor(frames, height, width).to(device=device, dtype=dtype)
    latent_dist = vae.encode(video, return_dict=False)[0]
    mu = latent_dist.mode()
    mu_norm = normalize_latents(mu, vae).detach().cpu()
    latent = mu_norm[0].detach().cpu().permute(1, 2, 3, 0).reshape(-1, mu_norm.shape[1])
    return {
        "latent": latent.to(torch.bfloat16),
        "latent_num_frames": int(mu_norm.shape[2]),
        "latent_height": int(mu_norm.shape[3]),
        "latent_width": int(mu_norm.shape[4]),
        "video_num_frames": len(frames),
        "video_height": int(source_height),
        "video_width": int(source_width),
    }


def output_file(dataset_root, info, cam_key, episode_index, start_frame, end_frame):
    chunk = episode_chunk(info, episode_index)
    return (
        Path(dataset_root)
        / "latents"
        / f"chunk-{chunk:03d}"
        / cam_key
        / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
    )


def iter_segments(episodes, episode_start=0, episode_limit=None, max_segments=None):
    yielded = 0
    selected = episodes[episode_start:]
    if episode_limit is not None:
        selected = selected[:episode_limit]
    for episode in selected:
        for segment in episode.get("action_config", []):
            yield episode, segment
            yielded += 1
            if max_segments is not None and yielded >= max_segments:
                return


def extract(args):
    from modules.utils import load_text_encoder, load_tokenizer, load_vae

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    pretrained_model = Path(args.pretrained_model).expanduser().resolve()
    device = torch.device(args.device)
    text_device = torch.device(args.text_device)
    dtype = torch_dtype(args.dtype)
    cam_keys = [item.strip() for item in args.cam_keys.split(",") if item.strip()]

    with open(dataset_root / "meta" / "info.json", "r") as f:
        info = json.load(f)
    episodes = read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    ori_fps = int(info.get("fps", args.target_fps))

    vae = load_vae(pretrained_model / "vae", torch_dtype=dtype, torch_device=device)
    vae.eval()
    tokenizer = load_tokenizer(pretrained_model / "tokenizer")
    text_encoder = load_text_encoder(
        pretrained_model / "text_encoder",
        torch_dtype=dtype,
        torch_device=text_device,
    )
    text_encoder.eval()

    empty_emb_path = dataset_root / "empty_emb.pt"
    if args.overwrite or not empty_emb_path.exists():
        empty_emb = encode_text("", tokenizer, text_encoder, text_device, dtype)
        torch.save(empty_emb, empty_emb_path)

    text_cache = {"": torch.load(empty_emb_path, weights_only=False)}
    segments = list(
        iter_segments(
            episodes,
            episode_start=args.episode_start,
            episode_limit=args.episode_limit,
            max_segments=args.max_segments,
        )
    )
    progress = tqdm(segments, desc="Extract RoboMME latents")
    for episode, segment in progress:
        episode_index = int(episode["episode_index"])
        start_frame = int(segment["start_frame"])
        end_frame = int(segment["end_frame"])
        action_text = segment.get("action_text") or (episode.get("tasks") or [""])[0]
        frame_ids, _ = sample_frame_ids(start_frame, end_frame, ori_fps, args.target_fps)
        frame_ids = align_wan_frame_ids(frame_ids, max_video_frames=args.max_video_frames)
        parquet_path = episode_file(dataset_root, info, episode_index)
        if action_text not in text_cache:
            text_cache[action_text] = encode_text(action_text, tokenizer, text_encoder, text_device, dtype)
        text_emb = text_cache[action_text]

        for cam_key in cam_keys:
            out_file = output_file(
                dataset_root,
                info,
                cam_key,
                episode_index,
                start_frame,
                end_frame,
            )
            if out_file.exists() and not args.overwrite:
                continue
            out_file.parent.mkdir(parents=True, exist_ok=True)
            frames = load_frames(dataset_root, parquet_path, cam_key, frame_ids)
            payload = encode_frames(
                frames,
                vae,
                device,
                dtype,
                args.height,
                args.width,
            )
            payload.update(
                {
                    "text_emb": text_emb.to(torch.bfloat16),
                    "text": action_text,
                    "frame_ids": frame_ids,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "fps": int(args.target_fps),
                    "ori_fps": int(ori_fps),
                }
            )
            torch.save(payload, out_file)

    print(f"Latent extraction complete under {dataset_root / 'latents'}")
    print(f"Empty CFG embedding: {empty_emb_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pretrained-model", required=True)
    parser.add_argument("--cam-keys", default="image,wrist_image")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--target-fps", type=int, default=10)
    parser.add_argument("--vae-frame-batch", type=int, default=81, help="Deprecated; kept for wrapper compatibility")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--text-device", default="cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--episode-limit", type=int, default=None)
    parser.add_argument("--max-segments", type=int, default=None)
    parser.add_argument("--max-video-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    extract(args)


if __name__ == "__main__":
    main()
