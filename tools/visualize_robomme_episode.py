#!/usr/bin/env python3
"""Render front/wrist frames and action traces from a RoboMME parquet episode."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


def _episode_path(root, episode_index):
    with (root / "meta" / "info.json").open() as f:
        info = json.load(f)
    chunk = episode_index // int(info["chunks_size"])
    return root / info["data_path"].format(
        episode_chunk=chunk, episode_index=episode_index
    )


def _decode(cell):
    payload = cell.as_py()
    if payload.get("bytes") is not None:
        return Image.open(io.BytesIO(payload["bytes"])).convert("RGB")
    return Image.open(payload["path"]).convert("RGB")


def _sample_indices(length, count, boundary):
    ids = set(np.linspace(0, length - 1, count, dtype=int).tolist())
    ids.update(i for i in (boundary - 1, boundary, boundary + 1) if 0 <= i < length)
    return sorted(ids)


def render(root, episode_index, output, samples):
    table = pq.read_table(_episode_path(root, episode_index))
    length = table.num_rows
    boundary = int(table["exec_start_idx"][0].as_py())
    indices = _sample_indices(length, samples, boundary)
    thumb_w, thumb_h = 256, 256
    label_h, left_w = 54, 110
    canvas = Image.new("RGB", (left_w + len(indices) * thumb_w, label_h + 2 * thumb_h + 250), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    task = table["simple_subgoal"][0].as_py()
    draw.text((8, 8), f"episode {episode_index} | rows={length} | exec_start={boundary} | first subgoal: {task}", fill="black", font=font)
    draw.text((8, label_h + 120), "front", fill="black", font=font)
    draw.text((8, label_h + thumb_h + 120), "wrist", fill="black", font=font)

    for col, frame_id in enumerate(indices):
        x = left_w + col * thumb_w
        phase = "DEMO" if bool(table["is_demo"][frame_id].as_py()) else "EXEC"
        color = "#1565c0" if phase == "DEMO" else "#c62828"
        draw.text((x + 4, 30), f"f={frame_id} {phase}", fill=color, font=font)
        for row, key in enumerate(("image", "wrist_image")):
            image = _decode(table[key][frame_id]).resize((thumb_w, thumb_h))
            canvas.paste(image, (x, label_h + row * thumb_h))
        if frame_id in (boundary - 1, boundary):
            draw.rectangle((x, label_h, x + thumb_w - 1, label_h + 2 * thumb_h - 1), outline="#ffb300", width=5)

    actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
    plot_y, plot_h = label_h + 2 * thumb_h + 36, 190
    draw.text((8, plot_y - 25), "raw action channels (vertical line = execution start)", fill="black", font=font)
    colors = ("#d32f2f", "#1976d2", "#388e3c", "#7b1fa2", "#f57c00", "#00796b", "#5d4037", "#455a64")
    lo, hi = np.quantile(actions, [0.01, 0.99])
    span = max(float(hi - lo), 1e-6)
    x0, x1 = left_w, canvas.width - 10
    for channel in range(actions.shape[1]):
        points = []
        for i, value in enumerate(actions[:, channel]):
            x = x0 + i * (x1 - x0) / max(length - 1, 1)
            y = plot_y + plot_h - np.clip((value - lo) / span, 0, 1) * plot_h
            points.append((x, y))
        draw.line(points, fill=colors[channel], width=2)
        draw.text((8 + channel * 42, plot_y + plot_h + 8), f"a{channel}", fill=colors[channel], font=font)
    bx = x0 + boundary * (x1 - x0) / max(length - 1, 1)
    draw.line((bx, plot_y, bx, plot_y + plot_h), fill="#ffb300", width=4)

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()
    render(args.dataset_root.resolve(), args.episode, args.output.resolve(), args.samples)


if __name__ == "__main__":
    main()
