"""Read the published Pro task map without importing MuJoCo or modifying LIBERO."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
PERTURBATIONS = {
    "object": "object", "position": "swap", "language": "lan",
    "task": "task", "environment": "env",
}
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}


def resolve_repo_path(project_root: Path, configured=None) -> Path:
    value = configured or os.environ.get("LIBERO_PRO_REPO") or project_root / "LIBERO-PRO"
    return Path(os.path.expandvars(str(value))).expanduser().resolve()


def split_suite(suite: str) -> tuple[str, str]:
    for base in BASE_SUITES:
        for perturbation, suffix in PERTURBATIONS.items():
            if suite == f"{base}_{suffix}":
                return base, perturbation
    raise ValueError(f"Unknown LIBERO-Pro suite: {suite!r}")


def load_task_map(repo: Path) -> dict:
    source = repo / "libero/libero/benchmark/libero_suite_task_map.py"
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "libero_task_map" for target in node.targets
        ):
            mapping = ast.literal_eval(node.value)
            if isinstance(mapping, dict):
                return mapping
    raise ValueError(f"Expected a literal official libero_task_map in {source}")


def select_tasks(repo: Path, suites, perturbations, task_ids=None) -> list[dict]:
    suites, perturbations = list(suites), list(perturbations)
    if not suites or not perturbations:
        raise ValueError("Select at least one base suite and perturbation.")
    if len(set(suites)) != len(suites) or len(set(perturbations)) != len(perturbations):
        raise ValueError("Duplicate suites/perturbations would evaluate tasks twice.")
    if set(suites) - set(BASE_SUITES) or set(perturbations) - set(PERTURBATIONS):
        raise ValueError(f"Expected base suites {BASE_SUITES} and perturbations {tuple(PERTURBATIONS)}")
    mapping = load_task_map(repo)
    tasks = []
    for base in suites:
        for perturbation in perturbations:
            suite = f"{base}_{PERTURBATIONS[perturbation]}"
            names = mapping.get(suite)
            if not isinstance(names, list) or not names or len(set(names)) != len(names):
                raise ValueError(f"Missing, empty, or duplicate official task list: {suite}")
            ids = list(range(len(names))) if task_ids is None else [int(i) for i in task_ids]
            if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= len(names) for i in ids):
                raise ValueError(f"Invalid task IDs for {suite}: {ids}")
            for task_id in ids:
                name = names[task_id]
                if not isinstance(name, str) or Path(name).name != name:
                    raise ValueError(f"Invalid task filename: {name!r}")
                tasks.append(dict(suite=suite, base_suite=base, perturbation=perturbation,
                                  task_id=task_id, name=name))
    return tasks


def validate_resources(repo: Path, tasks: list[dict], *, require_assets: bool) -> None:
    root = repo / "libero/libero"
    missing = []
    for task in tasks:
        for folder, suffix in (("bddl_files", ".bddl"), ("init_files", ".pruned_init")):
            path = root / folder / task["suite"] / (task["name"] + suffix)
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if require_assets and not (root / "assets").is_dir():
        missing.append(str(root / "assets"))
    if missing:
        raise FileNotFoundError(
            f"LIBERO-Pro resources missing or empty ({len(missing)} paths). "
            "Install the published Pro bddl_files/init_files and simulator assets.\n"
            + "\n".join(missing[:10])
        )


def write_run_files(output: Path, repo: Path, tasks: list[dict]) -> tuple[Path, str]:
    from omegaconf import OmegaConf

    output.mkdir(parents=True, exist_ok=True)
    manifest = "".join(json.dumps(task, sort_keys=True) + "\n" for task in tasks)
    (output / "task_manifest.jsonl").write_text(manifest, encoding="utf-8")
    digest = hashlib.sha256(manifest.encode()).hexdigest()
    (output / "task_manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    (output / "tasks.txt").write_text(
        "".join(f"{t['suite']},{t['task_id']}\n" for t in tasks), encoding="utf-8"
    )
    root = repo / "libero/libero"
    config_dir = output / "libero_pro_config"
    config_dir.mkdir(exist_ok=True)
    OmegaConf.save(OmegaConf.create({
        "benchmark_root": str(root), "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"), "assets": str(root / "assets"),
        "datasets": str(root.parent / "datasets"),
    }), config_dir / "config.yaml")
    return config_dir, digest
