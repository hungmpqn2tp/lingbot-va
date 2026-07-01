#!/usr/bin/env python3
"""Check a RoboMME dataset prepared for LingBot-VA."""
import argparse
import json
from pathlib import Path

import torch


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


def latent_file(dataset_root, info, cam_key, episode_index, start_frame, end_frame):
    chunk = episode_index // int(info["chunks_size"])
    return (
        Path(dataset_root)
        / "latents"
        / f"chunk-{chunk:03d}"
        / cam_key
        / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
    )


def check(args):
    root = Path(args.dataset_root).expanduser().resolve()
    cam_keys = [item.strip() for item in args.cam_keys.split(",") if item.strip()]
    with open(root / "meta" / "info.json", "r") as f:
        info = json.load(f)
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")

    problems = []
    if len(episodes) != int(info["total_episodes"]):
        problems.append(f"episode count mismatch: {len(episodes)} vs {info['total_episodes']}")

    segment_count = 0
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        if not episode_file(root, info, episode_index).exists():
            problems.append(f"missing parquet for episode {episode_index}")
        action_config = episode.get("action_config")
        if not action_config:
            problems.append(f"episode {episode_index} has no action_config")
            continue
        for segment in action_config:
            segment_count += 1
            start = int(segment["start_frame"])
            end = int(segment["end_frame"])
            if start < 0 or end <= start or end > int(episode["length"]):
                problems.append(f"bad segment {episode_index}: {start}:{end}")
            if args.require_latents:
                for cam_key in cam_keys:
                    path = latent_file(root, info, cam_key, episode_index, start, end)
                    if not path.exists():
                        problems.append(f"missing latent: {path}")
                        continue
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    for key in (
                        "latent",
                        "latent_num_frames",
                        "latent_height",
                        "latent_width",
                        "text_emb",
                        "frame_ids",
                        "start_frame",
                        "end_frame",
                    ):
                        if key not in payload:
                            problems.append(f"{path} missing key {key}")
                    if payload.get("start_frame") != start or payload.get("end_frame") != end:
                        problems.append(f"{path} segment metadata mismatch")

    norm_path = root / "meta" / "lingbot_va_robomme_norm_stats.json"
    if args.require_norm_stats and not norm_path.exists():
        problems.append(f"missing norm stats: {norm_path}")
    empty_emb_path = root / "empty_emb.pt"
    if args.require_latents and not empty_emb_path.exists():
        problems.append(f"missing empty embedding: {empty_emb_path}")

    if problems:
        print("RoboMME LingBot check failed:")
        for problem in problems[:200]:
            print(f"  - {problem}")
        if len(problems) > 200:
            print(f"  - ... {len(problems) - 200} more")
        raise SystemExit(1)

    print(f"Dataset root: {root}")
    print(f"Episodes: {len(episodes)}")
    print(f"Action segments: {segment_count}")
    print(f"Latents required: {args.require_latents}")
    print("RoboMME LingBot check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cam-keys", default="image,wrist_image")
    parser.add_argument("--require-latents", action="store_true")
    parser.add_argument("--require-norm-stats", action="store_true")
    args = parser.parse_args()
    check(args)


if __name__ == "__main__":
    main()
