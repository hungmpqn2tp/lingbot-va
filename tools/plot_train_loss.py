#!/usr/bin/env python3
"""Plot latent_loss / action_loss over training steps from a run_va_posttrain_robomme.sh log.

Training always runs with --disable-wandb (see script/run_va_posttrain_robomme.sh),
so the tqdm progress-bar postfix in the redirected stdout log is the only record
of the loss curve. Each line looks like:

    [default0]:Training:  10%|... [..., latent_loss=0.0735, action_loss=0.0009,
    step=1996, grad_norm=0.01, lr=1.00e-05]

This parses every key=value pair off those lines and plots the requested metrics
against `step`.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on a training server
import matplotlib.pyplot as plt

KV_PATTERN = re.compile(r"(\w+)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def parse_log(path: Path) -> dict[str, list[float]]:
    columns: dict[str, list[float]] = {}
    text = path.read_text(errors="ignore")
    for line in re.split(r"[\r\n]+", text):
        if "latent_loss=" not in line or "step=" not in line:
            continue
        fields = dict(KV_PATTERN.findall(line))
        if "step" not in fields:
            continue
        for key, value in fields.items():
            columns.setdefault(key, []).append(float(value))
    return columns


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_file", type=Path, help="Path to the redirected training stdout log.")
    parser.add_argument(
        "--metrics",
        default="latent_loss,action_loss",
        help="Comma-separated field names to plot (must exist as key=value in the log).",
    )
    parser.add_argument("--smooth", type=int, default=20, help="Moving-average window (1 disables smoothing).")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <log_file>.png).")
    parser.add_argument("--title", default=None, help="Plot title (default: the log file name).")
    args = parser.parse_args()

    columns = parse_log(args.log_file)
    if "step" not in columns:
        raise SystemExit(f"No lines with step=... found in {args.log_file}")

    steps = columns["step"]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    missing = [m for m in metrics if m not in columns]
    if missing:
        raise SystemExit(f"Metric(s) not found in log: {missing}. Available: {sorted(columns)}")

    fig, axes = plt.subplots(len(metrics), 1, figsize=(9, 3.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = columns[metric]
        n = min(len(steps), len(values))
        ax.plot(steps[:n], values[:n], color="#9ca3af", linewidth=0.8, alpha=0.6, label="raw")
        if args.smooth > 1:
            ax.plot(
                steps[:n],
                moving_average(values[:n], args.smooth),
                color="#2563eb",
                linewidth=1.6,
                label=f"moving avg ({args.smooth})",
            )
        ax.set_ylabel(metric)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("step")
    fig.suptitle(args.title or args.log_file.name)
    fig.tight_layout()

    out_path = args.out or args.log_file.with_suffix(".png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path} ({len(steps)} logged points)")


if __name__ == "__main__":
    main()
