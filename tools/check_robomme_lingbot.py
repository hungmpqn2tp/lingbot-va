#!/usr/bin/env python3
"""Check a RoboMME dataset prepared for LingBot-VA."""
import argparse
import json
from pathlib import Path

import numpy as np
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


def norm_action_quantiles(payload):
    if "action_q01" in payload and "action_q99" in payload:
        return np.asarray(payload["action_q01"]), np.asarray(payload["action_q99"])
    channel_ids = np.asarray(payload.get("source_to_model_channel", list(range(7)) + [28]))
    if "model_q01" in payload and "model_q99" in payload:
        return (
            np.asarray(payload["model_q01"])[channel_ids],
            np.asarray(payload["model_q99"])[channel_ids],
        )
    if "q01" in payload and "q99" in payload:
        q01 = np.asarray(payload["q01"])
        q99 = np.asarray(payload["q99"])
        if len(q01) == 30 and len(q99) == 30:
            return q01[channel_ids], q99[channel_ids]
        return q01, q99
    raise ValueError("normalization file has no supported q01/q99 fields")


def finite_tensor(payload, key):
    value = payload.get(key)
    if not torch.is_tensor(value):
        return False
    return bool(torch.isfinite(value).all())


def check(args):
    root = Path(args.dataset_root).expanduser().resolve()
    cam_keys = [item.strip() for item in args.cam_keys.split(",") if item.strip()]
    with open(root / "meta" / "info.json", "r") as f:
        info = json.load(f)
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")

    problems = []
    if args.expect_episodes is not None and len(episodes) != args.expect_episodes:
        problems.append(
            f"expected {args.expect_episodes} episodes, found {len(episodes)}"
        )
    episode_indices = [int(episode["episode_index"]) for episode in episodes]
    if len(set(episode_indices)) != len(episode_indices):
        problems.append("episode indices are not unique")
    if sorted(episode_indices) != list(range(len(episodes))):
        problems.append("episode indices are not contiguous from zero")
    episode_frame_count = sum(int(episode["length"]) for episode in episodes)
    if episode_frame_count != int(info["total_frames"]):
        problems.append(
            f"episode length sum mismatch: {episode_frame_count} "
            f"vs {info['total_frames']}"
        )
    if args.expect_frames is not None and episode_frame_count != args.expect_frames:
        problems.append(
            f"expected {args.expect_frames} frames, found {episode_frame_count}"
        )
    if args.expect_tasks is not None and int(info["total_tasks"]) != args.expect_tasks:
        problems.append(
            f"expected {args.expect_tasks} tasks, found {info['total_tasks']}"
        )
    if len(episodes) != int(info["total_episodes"]):
        problems.append(f"episode count mismatch: {len(episodes)} vs {info['total_episodes']}")

    parquet = None
    if args.check_actions:
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("--check-actions requires pyarrow") from exc

    segment_count = 0
    total_action_rows = 0
    all_actions = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        data_path = episode_file(root, info, episode_index)
        if not data_path.exists():
            problems.append(f"missing parquet for episode {episode_index}")
        elif args.check_actions:
            try:
                table = parquet.read_table(data_path, columns=[args.action_key])
                action = np.asarray(
                    table[args.action_key].to_pylist(),
                    dtype=np.float32,
                )
                expected_length = int(episode["length"])
                if action.shape != (expected_length, args.action_dim):
                    problems.append(
                        f"{data_path} action shape {action.shape}, "
                        f"expected {(expected_length, args.action_dim)}"
                    )
                elif not np.isfinite(action).all():
                    problems.append(f"{data_path} contains non-finite actions")
                else:
                    total_action_rows += len(action)
                    all_actions.append(action)
                    if args.expect_binary_gripper and not np.isin(
                        action[:, -1], (-1.0, 1.0)
                    ).all():
                        problems.append(
                            f"{data_path} gripper is not entirely -1/+1"
                        )
            except Exception as exc:
                problems.append(
                    f"failed to validate actions in {data_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
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
                segment_signature = None
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
                    if not finite_tensor(payload, "latent"):
                        problems.append(f"{path} latent is missing, invalid, or non-finite")
                    if not finite_tensor(payload, "text_emb"):
                        problems.append(f"{path} text_emb is missing, invalid, or non-finite")
                    if payload.get("start_frame") != start or payload.get("end_frame") != end:
                        problems.append(f"{path} segment metadata mismatch")
                    try:
                        latent_num_frames = int(payload["latent_num_frames"])
                        latent_height = int(payload["latent_height"])
                        latent_width = int(payload["latent_width"])
                        frame_ids = np.asarray(payload["frame_ids"], dtype=np.int64)
                        latent = payload["latent"]
                        if frame_ids.ndim != 1 or len(frame_ids) == 0:
                            raise ValueError(f"invalid frame_ids shape {frame_ids.shape}")
                        if np.any(np.diff(frame_ids) <= 0):
                            raise ValueError("frame_ids are not strictly increasing")
                        if frame_ids[0] < start or frame_ids[-1] >= end:
                            raise ValueError(
                                f"frame_ids range {frame_ids[0]}:{frame_ids[-1]} "
                                f"is outside {start}:{end}"
                            )
                        if args.require_full_latent_frames:
                            ori_fps = int(payload.get("ori_fps", info["fps"]))
                            latent_fps = int(payload.get("fps", info["fps"]))
                            stride = max(1, int(round(ori_fps / latent_fps)))
                            sampled_count = len(range(start, end, stride))
                            expected_count = ((sampled_count - 1) // 4) * 4 + 1
                            expected_last = start + (expected_count - 1) * stride
                            if (
                                len(frame_ids) != expected_count
                                or frame_ids[0] != start
                                or frame_ids[-1] != expected_last
                            ):
                                raise ValueError(
                                    f"latent covers {len(frame_ids)} aligned raw frames, "
                                    f"expected {expected_count}"
                                )
                        if int(payload.get("video_num_frames", len(frame_ids))) != len(frame_ids):
                            raise ValueError("video_num_frames does not match frame_ids")
                        expected_latent_frames = (len(frame_ids) - 1) // 4 + 1
                        if latent_num_frames != expected_latent_frames:
                            raise ValueError(
                                f"latent_num_frames={latent_num_frames}, "
                                f"expected {expected_latent_frames}"
                            )
                        expected_tokens = (
                            latent_num_frames * latent_height * latent_width
                        )
                        if latent.ndim != 2 or latent.shape[0] != expected_tokens:
                            raise ValueError(
                                f"latent shape {tuple(latent.shape)}, "
                                f"expected first dimension {expected_tokens}"
                            )
                        signature = (
                            tuple(frame_ids.tolist()),
                            latent_num_frames,
                            latent_height,
                            latent_width,
                        )
                        if segment_signature is None:
                            segment_signature = signature
                        elif signature != segment_signature:
                            raise ValueError(
                                "camera latent metadata does not match the other camera"
                            )
                    except Exception as exc:
                        problems.append(
                            f"{path} invalid latent metadata: "
                            f"{type(exc).__name__}: {exc}"
                        )

    if args.expect_segments is not None and segment_count != args.expect_segments:
        problems.append(
            f"expected {args.expect_segments} action segments, found {segment_count}"
        )
    if args.expect_latent_files is not None:
        actual_latent_files = sum(
            1
            for cam_key in cam_keys
            for _ in (root / "latents").glob(f"chunk-*/{cam_key}/*.pth")
        )
        if actual_latent_files != args.expect_latent_files:
            problems.append(
                f"expected {args.expect_latent_files} latent files, found {actual_latent_files}"
            )
    if args.check_actions and total_action_rows != int(info["total_frames"]):
        problems.append(
            f"action row count mismatch: {total_action_rows} vs {info['total_frames']}"
        )

    norm_path = root / "meta" / "lingbot_va_robomme_norm_stats.json"
    if args.require_norm_stats and not norm_path.exists():
        problems.append(f"missing norm stats: {norm_path}")
    empty_emb_path = root / "empty_emb.pt"
    if args.require_latents and not empty_emb_path.exists():
        problems.append(f"missing empty embedding: {empty_emb_path}")
    elif args.require_latents:
        try:
            empty_emb = torch.load(empty_emb_path, map_location="cpu", weights_only=False)
            if (
                not torch.is_tensor(empty_emb)
                or tuple(empty_emb.shape) != (512, 4096)
                or not torch.isfinite(empty_emb).all()
            ):
                problems.append(
                    f"{empty_emb_path} must be a finite tensor with shape (512, 4096)"
                )
        except Exception as exc:
            problems.append(f"failed to validate {empty_emb_path}: {exc}")

    if args.verify_norm_stats:
        if not all_actions:
            problems.append("cannot verify normalization stats without valid actions")
        elif not norm_path.exists():
            problems.append(f"missing norm stats: {norm_path}")
        else:
            try:
                with open(norm_path, "r") as stream:
                    norm_payload = json.load(stream)
                expected_q01, expected_q99 = norm_action_quantiles(norm_payload)
                actions = np.concatenate(all_actions, axis=0)
                actual_q01 = np.quantile(actions, 0.01, axis=0)
                actual_q99 = np.quantile(actions, 0.99, axis=0)
                if expected_q01.shape != (args.action_dim,) or expected_q99.shape != (
                    args.action_dim,
                ):
                    raise ValueError(
                        f"norm-stat action shapes are "
                        f"{expected_q01.shape}/{expected_q99.shape}"
                    )
                if not np.allclose(actual_q01, expected_q01, rtol=0, atol=1e-6):
                    raise ValueError("stored action_q01 does not match full dataset")
                if not np.allclose(actual_q99, expected_q99, rtol=0, atol=1e-6):
                    raise ValueError("stored action_q99 does not match full dataset")
            except Exception as exc:
                problems.append(
                    f"failed to verify norm stats: {type(exc).__name__}: {exc}"
                )

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
    if args.check_actions:
        actions = np.concatenate(all_actions, axis=0)
        print(f"Action rows: {total_action_rows}")
        print(f"Action min: {actions.min(axis=0).tolist()}")
        print(f"Action max: {actions.max(axis=0).tolist()}")
        print(f"Norm stats verified: {args.verify_norm_stats}")
    print("RoboMME LingBot check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--cam-keys", default="image,wrist_image")
    parser.add_argument("--require-latents", action="store_true")
    parser.add_argument("--require-norm-stats", action="store_true")
    parser.add_argument("--check-actions", action="store_true")
    parser.add_argument("--verify-norm-stats", action="store_true")
    parser.add_argument("--expect-binary-gripper", action="store_true")
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--expect-episodes", type=int, default=None)
    parser.add_argument("--expect-segments", type=int, default=None)
    parser.add_argument("--expect-frames", type=int, default=None)
    parser.add_argument("--expect-tasks", type=int, default=None)
    parser.add_argument("--expect-latent-files", type=int, default=None)
    parser.add_argument("--require-full-latent-frames", action="store_true")
    args = parser.parse_args()
    if args.verify_norm_stats:
        args.check_actions = True
    check(args)


if __name__ == "__main__":
    main()
