#!/usr/bin/env python3
"""Backfill BridgeWAM training metrics from a text log into a W&B run."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
TRAIN_LINE_RE = re.compile(
    rf"\[train\]\s+epoch=(?P<epoch>\d+)\s+"
    rf"step=(?P<step>\d+)/(?P<max_steps>\d+)\s+"
    rf"loss=(?P<loss>{NUMBER})(?P<tail>.*)"
)
KEY_VALUE_RE = re.compile(rf"(?P<key>[A-Za-z_][\w./-]*)=(?P<value>{NUMBER})")
SPEED_RE = re.compile(
    rf"speed=(?P<steps_per_sec>{NUMBER})\s+step/s,\s*"
    rf"(?P<samples_per_sec>{NUMBER})\s+samples/s"
)
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_train_line(line: str) -> tuple[int, dict[str, float | int]] | None:
    line = ANSI_ESCAPE_RE.sub("", line)
    match = TRAIN_LINE_RE.search(line)
    if match is None:
        return None

    step = int(match.group("step"))
    payload: dict[str, float | int] = {
        "global_step": step,
        "train/epoch": int(match.group("epoch")),
        "train/loss": float(match.group("loss")),
        "train/max_steps": int(match.group("max_steps")),
    }

    tail = match.group("tail")
    for item in KEY_VALUE_RE.finditer(tail):
        key = item.group("key")
        value = float(item.group("value"))
        if key.startswith("loss_"):
            payload[f"train/{key}"] = value
        elif key == "lr":
            payload["train/lr"] = value

    speed = SPEED_RE.search(tail)
    if speed is not None:
        payload["performance/steps_per_sec"] = float(speed.group("steps_per_sec"))
        payload["performance/samples_per_sec"] = float(speed.group("samples_per_sec"))

    numeric_values = [value for value in payload.values() if isinstance(value, float)]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError(f"non-finite metric at step {step}: {line.rstrip()}")
    return step, payload


def load_metrics(log_path: Path) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    by_step: dict[int, dict[str, Any]] = {}
    matched_lines = 0
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = parse_train_line(line)
            if parsed is None:
                continue
            matched_lines += 1
            step, payload = parsed
            # A resumed training job can repeat steps. The latest log entry wins.
            by_step[step] = payload
    return sorted(by_step.items()), matched_lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parse BridgeWAM [train] lines and backfill their metrics to W&B."
    )
    parser.add_argument("log_path", type=Path, help="Path to train.log")
    parser.add_argument(
        "--entity",
        default=None,
        help="W&B entity/workspace (defaults to your configured W&B account)",
    )
    parser.add_argument(
        "--project",
        default="bridgewam",
        help="W&B project name",
    )
    parser.add_argument("--name", default="train-log-backfill", help="Name of the new W&B run")
    parser.add_argument(
        "--run-id",
        help="Existing run ID to resume. Omit this to create a separate backfill run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse and summarize the log; do not contact W&B.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_path = args.log_path.expanduser().resolve()
    if not log_path.is_file():
        raise SystemExit(f"Log file does not exist: {log_path}")

    metrics, matched_lines = load_metrics(log_path)
    if not metrics:
        raise SystemExit(
            "No BridgeWAM training lines matched. Expected text like "
            "'[train] epoch=0 step=10/1000 loss=0.1234'."
        )

    first_step, first_payload = metrics[0]
    last_step, last_payload = metrics[-1]
    duplicate_count = matched_lines - len(metrics)
    print(
        f"Parsed {len(metrics)} unique points from {matched_lines} training lines "
        f"(steps {first_step}..{last_step}, duplicates replaced: {duplicate_count})."
    )
    print(f"First: {first_payload}")
    print(f"Last:  {last_payload}")
    print("Note: metric precision is limited to the values printed in train.log.")
    if args.dry_run:
        print("Dry run complete; nothing was uploaded.")
        return

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit("wandb is not installed. Install the project dependencies first.") from exc

    init_kwargs: dict[str, Any] = {
        "entity": args.entity,
        "project": args.project,
        "name": args.name,
        "job_type": "metrics-backfill",
        "tags": ["backfill", "train.log"],
        "config": {
            "backfill/source_log": str(log_path),
            "backfill/unique_points": len(metrics),
            "backfill/duplicate_lines_replaced": duplicate_count,
            "backfill/precision": "printed log values",
        },
    }
    if args.run_id:
        init_kwargs.update(id=args.run_id, resume="must")

    with wandb.init(**init_kwargs) as run:
        for step, payload in metrics:
            run.log(payload, step=step)
        run.summary["backfill/first_step"] = first_step
        run.summary["backfill/last_step"] = last_step
        run.summary["backfill/points_uploaded"] = len(metrics)
        run_url = run.url

    print(f"Uploaded {len(metrics)} metric points to {run_url}")


if __name__ == "__main__":
    main()


#   python scripts/backfill_wandb_from_log.py train.log \
#     --name exp1-drop00-05-loss-backfill
