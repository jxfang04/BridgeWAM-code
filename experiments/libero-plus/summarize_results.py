"""Summarize LIBERO-Plus results by perturbation, difficulty, and suite."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from libero_plus_metadata import PlusTask, read_manifest


SUITE_ORDER = [
    "libero_10",
    "libero_goal",
    "libero_spatial",
    "libero_object",
]

# Keep the primary CSV stable and easy to compare across runs. The first item
# is the human-readable column title and the second is the metadata slug.
PERTURBATION_COLUMNS = [
    ("Background Textures", "background_textures"),
    ("Camera Viewpoints", "camera_viewpoints"),
    ("Language Instructions", "language_instructions"),
    ("Light Conditions", "light_conditions"),
    ("Objects Layout", "objects_layout"),
    ("Robot Initial States", "robot_initial_states"),
    ("Sensor Noise", "sensor_noise"),
]


def _new_stats() -> dict:
    return {
        "completed_tasks": 0,
        "total_episodes": 0,
        "successes": 0,
        "duration_seconds": 0.0,
    }


def _finalize(stats: dict) -> dict:
    value = dict(stats)
    episodes = value["total_episodes"]
    value["success_rate"] = 100.0 * value["successes"] / episodes if episodes else None
    value["average_task_duration_seconds"] = (
        value["duration_seconds"] / value["completed_tasks"]
        if value["completed_tasks"]
        else None
    )
    return value


def _add(stats: dict, result: dict) -> None:
    stats["completed_tasks"] += 1
    stats["total_episodes"] += int(result["total_episodes"])
    stats["successes"] += int(result["successes"])
    stats["duration_seconds"] += float(result.get("duration", 0.0))


def _find_result(output_dir: Path, task: PlusTask) -> Path | None:
    task_dir = output_dir / task.suite
    matches = []
    shard_result = task_dir / f"task{task.task_id}_results.json"
    if shard_result.is_file():
        matches.append(shard_result)
    matches.extend(sorted(task_dir.glob(f"gpu*_task{task.task_id}_results.json")))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple result files found for {task.suite} task {task.task_id}: {matches}")
    return matches[0] if matches else None


def summarize(output_dir: Path, manifest_path: Path) -> dict:
    tasks = read_manifest(manifest_path)
    aggregate = {
        "overall": _new_stats(),
        "by_category": defaultdict(_new_stats),
        "by_category_difficulty": defaultdict(_new_stats),
        "by_suite": defaultdict(_new_stats),
        "by_suite_category": defaultdict(_new_stats),
    }
    details = []
    missing = []

    for task in tasks:
        result_path = _find_result(output_dir, task)
        if result_path is None:
            missing.append(task.to_dict())
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        _add(aggregate["overall"], result)
        _add(aggregate["by_category"][task.category_slug], result)
        _add(
            aggregate["by_category_difficulty"][(task.category_slug, task.difficulty_label)],
            result,
        )
        _add(aggregate["by_suite"][task.suite], result)
        _add(aggregate["by_suite_category"][(task.suite, task.category_slug)], result)
        details.append(
            {
                **task.to_dict(),
                "successes": int(result["successes"]),
                "total_episodes": int(result["total_episodes"]),
                "success_rate": 100.0 * int(result["successes"]) / int(result["total_episodes"]),
                "duration_seconds": float(result.get("duration", 0.0)),
                "task_description": result.get("task_description", ""),
                "result_file": str(result_path.relative_to(output_dir)),
            }
        )

    summary = {
        "run_id": output_dir.name,
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "expected_tasks": len(tasks),
        "completed_tasks": len(details),
        "missing_tasks": len(missing),
        "is_complete": not missing,
        "overall": _finalize(aggregate["overall"]),
        "by_category": {
            key: _finalize(value) for key, value in sorted(aggregate["by_category"].items())
        },
        "by_category_difficulty": {
            f"{category}|{difficulty}": _finalize(value)
            for (category, difficulty), value in sorted(
                aggregate["by_category_difficulty"].items()
            )
        },
        "by_suite": {
            key: _finalize(value) for key, value in sorted(aggregate["by_suite"].items())
        },
        "by_suite_category": {
            f"{suite}|{category}": _finalize(value)
            for (suite, category), value in sorted(aggregate["by_suite_category"].items())
        },
        "missing_task_records": missing,
    }

    (output_dir / "libero_plus_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    _write_details(output_dir / "libero_plus_task_results.csv", details)
    _write_success_rate_matrix(output_dir / "libero_plus_summary.csv", summary)
    _write_category_difficulty_summary(
        output_dir / "libero_plus_summary_by_difficulty.csv", summary
    )
    _print_summary(summary)
    return summary


def _write_details(path: Path, rows: list[dict]) -> None:
    fields = [
        "suite",
        "task_id",
        "official_id",
        "name",
        "category",
        "category_slug",
        "difficulty_level",
        "difficulty_label",
        "successes",
        "total_episodes",
        "success_rate",
        "duration_seconds",
        "task_description",
        "result_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_success_rate(stats: dict | None) -> str:
    if not stats or stats.get("success_rate") is None:
        return "N/A"
    return f"{float(stats['success_rate']):.2f}%"


def _success_rate_matrix_rows(summary: dict) -> list[dict[str, str]]:
    rows = []
    for suite in SUITE_ORDER:
        row = {"Suite": suite}
        for title, category_slug in PERTURBATION_COLUMNS:
            stats = summary["by_suite_category"].get(f"{suite}|{category_slug}")
            row[title] = _format_success_rate(stats)
        row["Total"] = _format_success_rate(summary["by_suite"].get(suite))
        rows.append(row)
    return rows


def _write_success_rate_matrix(path: Path, summary: dict) -> None:
    """Write the primary human-readable 4-suite x 7-perturbation matrix."""
    fields = ["Suite", *[title for title, _ in PERTURBATION_COLUMNS], "Total"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_success_rate_matrix_rows(summary))


def _write_category_difficulty_summary(path: Path, summary: dict) -> None:
    """Keep the original long-form breakdown for detailed analysis."""
    fields = [
        "category",
        "difficulty",
        "completed_tasks",
        "total_episodes",
        "successes",
        "success_rate",
        "duration_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for key, stats in summary["by_category_difficulty"].items():
            category, difficulty = key.split("|", 1)
            writer.writerow({"category": category, "difficulty": difficulty, **stats})


def _print_summary(summary: dict) -> None:
    print("\n=== LIBERO-Plus Summary ===")
    print(
        f"Completed: {summary['completed_tasks']}/{summary['expected_tasks']} "
        f"(missing: {summary['missing_tasks']})"
    )
    headers = ["Suite", *[title for title, _ in PERTURBATION_COLUMNS], "Total"]
    rows = _success_rate_matrix_rows(summary)
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }
    print("  ".join(f"{header:<{widths[header]}}" for header in headers))
    for row in rows:
        print("  ".join(f"{row[header]:<{widths[header]}}" for header in headers))
    output_dir = Path(summary["output_dir"])
    print(f"Success-rate CSV: {output_dir / 'libero_plus_summary.csv'}")
    print(
        "Difficulty-detail CSV: "
        f"{output_dir / 'libero_plus_summary_by_difficulty.csv'}"
    )
    print(f"Summary JSON: {output_dir / 'libero_plus_summary.json'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = args.manifest or output_dir / "task_manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"LIBERO-Plus task manifest not found: {manifest}")
    summarize(output_dir, manifest)


if __name__ == "__main__":
    main()
