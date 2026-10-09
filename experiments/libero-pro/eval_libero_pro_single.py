"""LIBERO-Pro adapter around the unchanged BridgeWAM policy/rollout implementation."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

from libero_pro_metadata import (
    MAX_STEPS, resolve_repo_path, select_tasks, split_suite, validate_resources, write_run_files,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_environment(repo: Path, config_dir: Path):
    """Select the Pro package before any LIBERO-dependent model/eval imports."""
    if not (config_dir / "config.yaml").is_file():
        raise FileNotFoundError(config_dir / "config.yaml")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    # Keep environment, policy source and checkout roots independent.
    for path in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "experiments/libero", repo):
        value = str(path)
        sys.path[:] = [p for p in sys.path if p != value]
        sys.path.insert(0, value)
    package = importlib.import_module("libero.libero")
    loaded = Path(package.__file__).resolve()
    if repo / "libero/libero" not in loaded.parents:
        raise RuntimeError(f"Wrong LIBERO package already loaded: {loaded}; expected {repo}")
    for key, folder in (("bddl_files", "bddl_files"), ("init_states", "init_files"), ("assets", "assets")):
        actual = Path(package.get_libero_path(key)).resolve()
        expected = (repo / "libero/libero" / folder).resolve()
        if actual != expected:
            raise RuntimeError(f"Wrong LIBERO-Pro {key} root: {actual}; expected {expected}")


def get_pro_env(task, resolution, seed):
    from experiments.libero.libero_utils import get_libero_env

    env, _ = get_libero_env(task, resolution, seed)
    # Pro filenames can retain the original instruction even for language/task
    # perturbations. The BDDL parsed by the environment is authoritative.
    description = getattr(env, "language_instruction", None)
    if not isinstance(description, str) or not description.strip():
        env.close()
        raise ValueError(f"Missing BDDL language instruction for Pro task {task.name}")
    return env, description.strip()


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_pro")
def main(cfg: DictConfig):
    repo = resolve_repo_path(PROJECT_ROOT, cfg.LIBERO_PRO.repo_path)
    output = Path(os.path.expandvars(str(cfg.EVALUATION.output_dir))).expanduser().resolve()
    base, perturbation = split_suite(str(cfg.EVALUATION.task_suite_name))
    task = select_tasks(repo, [base], [perturbation], [cfg.EVALUATION.task_id])[0]
    validate_resources(repo, [task], require_assets=bool(cfg.LIBERO_PRO.require_assets))
    if cfg.EVALUATION.max_steps is None:
        cfg.EVALUATION.max_steps = MAX_STEPS[base]
    if int(cfg.EVALUATION.num_trials) <= 0:
        raise ValueError("EVALUATION.num_trials must be positive.")
    manifest_path = output / "task_manifest.jsonl"
    if manifest_path.exists():
        records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        if task not in records:
            raise ValueError("Selected Pro task does not match the run manifest.")
        config_dir = output / "libero_pro_config"
    else:
        config_dir, _ = write_run_files(output, repo, [task])
    setup_environment(repo, config_dir)
    from experiments.libero.eval_libero_single import evaluate_single_process

    result = evaluate_single_process(
        cfg, env_factory=get_pro_env, require_full_init_states=True, expected_task_name=task["name"],
    )
    result.update(benchmark="LIBERO-Pro", base_suite=base, perturbation=perturbation,
                  task_name=task["name"], libero_pro_repo=str(repo), max_steps=int(cfg.EVALUATION.max_steps))
    path = output / task["suite"] / f"gpu{cfg.gpu_id}_task{task['task_id']}_results.json"
    temp = path.with_suffix(f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temp.replace(path)
    return result


if __name__ == "__main__":
    main()
