"""Manifest-based Pro aggregation. Missing/duplicate results never count as success."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def summarize(output: Path) -> dict:
    records = [json.loads(line) for line in (output / "task_manifest.jsonl").read_text().splitlines() if line.strip()]
    keys = [(task["suite"], int(task["task_id"])) for task in records]
    if not records or len(keys) != len(set(keys)):
        raise ValueError("Empty or duplicate task manifest.")
    groups = defaultdict(lambda: dict(successes=0, episodes=0, tasks=0))
    missing = []
    for task in records:
        paths = list((output / task["suite"]).glob(f"gpu*_task{task['task_id']}_results.json"))
        if not paths:
            missing.append([task["suite"], task["task_id"]])
            continue
        if len(paths) != 1:
            raise ValueError(f"Duplicate results for {task['suite']}:{task['task_id']}")
        result = json.loads(paths[0].read_text())
        if (result.get("task_suite"), result.get("task_id"), result.get("task_name"), result.get("benchmark")) != (
            task["suite"], task["task_id"], task["name"], "LIBERO-Pro"
        ):
            raise ValueError(f"Result identity does not match manifest: {paths[0]}")
        successes, episodes = int(result["successes"]), int(result["total_episodes"])
        if episodes <= 0 or not 0 <= successes <= episodes:
            raise ValueError(f"Invalid episode counts: {paths[0]}")
        for group in ("overall", "suite/" + task["base_suite"], "perturbation/" + task["perturbation"],
                      "cell/" + task["suite"]):
            groups[group]["successes"] += successes
            groups[group]["episodes"] += episodes
            groups[group]["tasks"] += 1
    for value in groups.values():
        value["success_rate"] = value["successes"] / value["episodes"]
    summary = dict(benchmark="LIBERO-Pro", complete=not missing, expected_tasks=len(records),
                   missing_tasks=missing, groups=dict(groups))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.output_dir)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["complete"] else 1)
