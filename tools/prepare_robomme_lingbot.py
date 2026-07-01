#!/usr/bin/env python3
"""Prepare RoboMME LeRobot data for LingBot-VA training.

The prepared directory is intentionally lightweight: by default it symlinks the
large `data/` folder and rewrites only metadata needed by LingBot-VA.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


ROBO_MME_SUBGOAL_COLUMNS = (
    "simple_subgoal",
    "grounded_subgoal",
    "simple_subgoal_online",
    "grounded_subgoal_online",
)


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required for this option. Install the post-training "
            "dependencies first, e.g. `pip install pyarrow datasets lerobot==0.3.3`."
        ) from exc
    return pq


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_file(dataset_root, info, episode_index):
    chunk = episode_index // int(info["chunks_size"])
    rel = info["data_path"].format(
        episode_chunk=chunk,
        episode_index=episode_index,
    )
    return Path(dataset_root) / rel


def _task_text(episode):
    tasks = episode.get("tasks") or []
    return tasks[0] if tasks else "perform the robot manipulation task"


def _merge_short_segments(segments, min_frames):
    if min_frames <= 1:
        return segments
    merged = []
    for segment in segments:
        length = segment["end_frame"] - segment["start_frame"]
        if length >= min_frames or not merged:
            merged.append(segment)
            continue
        merged[-1]["end_frame"] = segment["end_frame"]
        if segment["action_text"] and segment["action_text"] not in merged[-1]["action_text"]:
            merged[-1]["action_text"] = (
                merged[-1]["action_text"].rstrip(".") + ". " + segment["action_text"]
            )
    return merged


def _segments_from_column(dataset_root, info, episode, column, min_frames):
    pq = _require_pyarrow()
    path = episode_file(dataset_root, info, episode["episode_index"])
    values = pq.read_table(path, columns=[column]).column(column).to_pylist()
    default_text = _task_text(episode)
    segments = []
    start = 0
    previous = values[0] if values else default_text
    for idx, value in enumerate(values[1:], start=1):
        if value == previous:
            continue
        text = str(previous or default_text)
        segments.append({"start_frame": start, "end_frame": idx, "action_text": text})
        start = idx
        previous = value
    if values:
        text = str(previous or default_text)
        segments.append({"start_frame": start, "end_frame": len(values), "action_text": text})
    if not segments:
        segments = [
            {
                "start_frame": 0,
                "end_frame": int(episode["length"]),
                "action_text": default_text,
            }
        ]
    return _merge_short_segments(segments, min_frames)


def build_action_config(dataset_root, info, episode, segment_source, min_segment_frames):
    length = int(episode["length"])
    if segment_source == "task":
        return [{"start_frame": 0, "end_frame": length, "action_text": _task_text(episode)}]
    segments = _segments_from_column(
        dataset_root,
        info,
        episode,
        segment_source,
        min_segment_frames,
    )
    out = []
    for segment in segments:
        start = max(0, min(int(segment["start_frame"]), length))
        end = max(start, min(int(segment["end_frame"]), length))
        if end > start:
            out.append(
                {
                    "start_frame": start,
                    "end_frame": end,
                    "action_text": segment["action_text"],
                }
            )
    return out or [{"start_frame": 0, "end_frame": length, "action_text": _task_text(episode)}]


def link_or_copy_data(source_root, output_root, copy_data=False):
    source_data = source_root / "data"
    output_data = output_root / "data"
    if output_data.exists() or output_data.is_symlink():
        if output_data.is_symlink():
            output_data.unlink()
        else:
            raise FileExistsError(f"{output_data} already exists and is not a symlink")
    if copy_data:
        shutil.copytree(source_data, output_data)
    else:
        rel_source = os.path.relpath(source_data, output_root)
        output_data.symlink_to(rel_source, target_is_directory=True)


def copy_metadata(source_root, output_root):
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    for name in ("info.json", "tasks.jsonl", "episodes_stats.jsonl"):
        shutil.copy2(source_root / "meta" / name, output_meta / name)
    for name in ("README.md", ".gitattributes"):
        source_file = source_root / name
        if source_file.exists():
            shutil.copy2(source_file, output_root / name)


def source_to_model_norm(action_q01, action_q99, action_dim=30):
    model_q01 = [0.0] * action_dim
    model_q99 = [1.0] * action_dim
    source_to_model = list(range(0, 7)) + [28]
    for source_idx, model_channel in enumerate(source_to_model):
        model_q01[model_channel] = float(action_q01[source_idx])
        model_q99[model_channel] = float(action_q99[source_idx])
    return model_q01, model_q99


def compute_norm_stats(dataset_root, info, episodes, action_key, action_source_dim):
    pq = _require_pyarrow()
    arrays = []
    for episode in episodes:
        path = episode_file(dataset_root, info, episode["episode_index"])
        table = pq.read_table(path, columns=[action_key])
        action = np.asarray(table.column(action_key).to_pylist(), dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != action_source_dim:
            raise ValueError(
                f"{path} column {action_key!r} has shape {action.shape}, "
                f"expected [frames, {action_source_dim}]"
            )
        arrays.append(action)
    all_actions = np.concatenate(arrays, axis=0)
    action_q01 = np.quantile(all_actions, 0.01, axis=0).astype(float).tolist()
    action_q99 = np.quantile(all_actions, 0.99, axis=0).astype(float).tolist()
    model_q01, model_q99 = source_to_model_norm(action_q01, action_q99)
    return {
        "source_action_key": action_key,
        "source_action_dim": action_source_dim,
        "model_action_dim": 30,
        "source_to_model_channel": list(range(0, 7)) + [28],
        "action_q01": action_q01,
        "action_q99": action_q99,
        "model_q01": model_q01,
        "model_q99": model_q99,
    }


def prepare(args):
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not (source_root / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Missing LeRobot metadata: {source_root / 'meta/info.json'}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} exists. Pass --overwrite to reuse it.")
    output_root.mkdir(parents=True, exist_ok=True)
    copy_metadata(source_root, output_root)
    link_or_copy_data(source_root, output_root, copy_data=args.copy_data)

    with open(source_root / "meta" / "info.json", "r") as f:
        info = json.load(f)
    episodes = read_jsonl(source_root / "meta" / "episodes.jsonl")

    prepared = []
    for episode in episodes:
        row = dict(episode)
        row["action_config"] = build_action_config(
            source_root,
            info,
            episode,
            args.segment_source,
            args.min_segment_frames,
        )
        prepared.append(row)
    write_jsonl(output_root / "meta" / "episodes.jsonl", prepared)

    if args.compute_norm_stats:
        stats = compute_norm_stats(
            source_root,
            info,
            episodes,
            args.action_key,
            args.action_source_dim,
        )
        with open(output_root / "meta" / "lingbot_va_robomme_norm_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "segment_source": args.segment_source,
        "min_segment_frames": args.min_segment_frames,
        "action_key": args.action_key,
        "data_is_symlink": not args.copy_data,
        "episodes": len(prepared),
    }
    with open(output_root / "meta" / "lingbot_va_robomme_prep.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared RoboMME LingBot dataset at {output_root}")
    print(f"Episodes: {len(prepared)}")
    if args.compute_norm_stats:
        print("Wrote meta/lingbot_va_robomme_norm_stats.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Original RoboMME LeRobot root")
    parser.add_argument("--output-root", required=True, help="Prepared LingBot-compatible root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-data", action="store_true", help="Copy data instead of symlinking it")
    parser.add_argument(
        "--segment-source",
        choices=("task",) + ROBO_MME_SUBGOAL_COLUMNS,
        default="task",
        help="Where action_config text/segments should come from",
    )
    parser.add_argument("--min-segment-frames", type=int, default=8)
    parser.add_argument("--action-key", default="actions")
    parser.add_argument("--action-source-dim", type=int, default=8)
    parser.add_argument("--compute-norm-stats", action="store_true")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
