"""Hydra manager for distributed BridgeWAM evaluation on LIBERO-Plus."""

from __future__ import annotations

import os
import shlex
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from libero_plus_metadata import (
    create_balanced_shards,
    load_tasks,
    resolve_repo_path,
    select_tasks,
    write_task_files,
    write_shard_files,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    blocked_exact = {
        "task",
        "ckpt",
        "gpu_id",
        "EVALUATION.task_suite_name",
        "EVALUATION.task_id",
        "EVALUATION.output_dir",
    }
    return key in blocked_exact or key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [
        value
        for value in HydraConfig.get().overrides.task
        if not _is_blocked_override(value)
    ]


def _resolve_worker_task_choice() -> str:
    task_choice = HydraConfig.get().runtime.choices.get("task")
    if task_choice is None or not str(task_choice).strip():
        raise ValueError("Pass a BridgeWAM task config with task=..., for example task=libero_uncond_2cam224_lbqs_only_2layer_alternating_cross_self_fullfinetune_1e-4.")
    return str(task_choice)


def _validate_plus_repo(repo_path: Path, *, require_assets: bool) -> Path:
    benchmark_root = repo_path / "libero" / "libero"
    required = [
        benchmark_root / "benchmark" / "task_classification.json",
        benchmark_root / "bddl_files",
        benchmark_root / "init_files",
    ]
    if require_assets:
        required.append(benchmark_root / "assets")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "LIBERO-Plus is missing required benchmark resources:\n- " + "\n- ".join(missing)
        )
    return benchmark_root


def _write_libero_config(output_dir: Path, benchmark_root: Path) -> Path:
    config_dir = output_dir / "libero_plus_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    OmegaConf.save(OmegaConf.create(config), f=str(config_dir / "config.yaml"))
    return config_dir


def _print_selection(tasks) -> None:
    print("\nLIBERO-Plus selection:")
    print(f"- Total tasks: {len(tasks)}")
    for (category, difficulty), count in sorted(
        Counter((task.category, task.difficulty_label) for task in tasks).items()
    ):
        print(f"- {category}, difficulty={difficulty}: {count}")


def _run_evaluation(
    *,
    task_file: Path,
    task_choice: str,
    ckpt: str,
    num_gpus: int,
    num_trials: int,
    max_tasks_per_gpu: int,
    output_dir: Path,
    extra_overrides: list[str],
    plus_repo: Path,
    libero_config_dir: Path,
) -> None:
    scheduler = PROJECT_ROOT / "experiments" / "libero" / "run_libero_parallel_test.sh"
    if not scheduler.is_file():
        raise FileNotFoundError(f"LIBERO scheduler not found: {scheduler}")

    run_id = f"libero_plus_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    session_name = os.environ.get("LIBERO_TMUX_SESSION_NAME", run_id)
    env = os.environ.copy()
    env.update(
        {
            "CONFIG": task_choice,
            "CKPT": ckpt,
            "NUM_GPUS": str(num_gpus),
            "NUM_TRIALS": str(num_trials),
            "MAX_TASKS_PER_GPU": str(max_tasks_per_gpu),
            "ROOT_DIR": str(PROJECT_ROOT),
            "RUN_ID": run_id,
            "LIBERO_TMUX_SESSION_NAME": session_name,
            "OUTPUT_DIR": str(output_dir),
            "EXTRA_ARGS": shlex.join(extra_overrides) if extra_overrides else "",
            "EXP_NAME": os.environ.get("EXP_NAME", ""),
            "LIBERO_ROOT": str(plus_repo),
            "LIBERO_PLUS_REPO": str(plus_repo),
            "LIBERO_CONFIG_PATH": str(libero_config_dir),
            "EVAL_SCRIPT": "experiments/libero-plus/eval_libero_plus_shard.py",
            "SUMMARY_SCRIPT": "experiments/libero-plus/summarize_results.py",
        }
    )

    print("\nStarting LIBERO-Plus evaluation:")
    print(f"- Checkpoint: {ckpt}")
    print(f"- GPUs: {num_gpus}")
    print(f"- Trials per task: {num_trials}")
    print(f"- Output: {output_dir}")
    print(f"- Plus repository: {plus_repo}")
    print(f"- tmux session: {session_name}")
    if env["EXTRA_ARGS"]:
        print(f"- Forwarded overrides: {env['EXTRA_ARGS']}")

    subprocess.run(
        ["bash", str(scheduler), str(task_file)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        text=True,
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus")
def main(cfg: DictConfig) -> None:
    if cfg.ckpt is None:
        raise ValueError("ckpt must not be null.")
    if cfg.EVALUATION.output_dir is None:
        raise ValueError("EVALUATION.output_dir must not be null.")
    if int(cfg.EVALUATION.num_trials) != 1:
        raise ValueError("Official LIBERO-Plus evaluation requires EVALUATION.num_trials=1.")

    output_dir = Path(os.path.expandvars(os.path.expanduser(str(cfg.EVALUATION.output_dir)))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plus_cfg = cfg.LIBERO_PLUS
    repo_path = resolve_repo_path(PROJECT_ROOT, plus_cfg.get("repo_path"))
    create_only = bool(cfg.MULTIRUN.get("create_only", False))
    benchmark_root = _validate_plus_repo(
        repo_path,
        require_assets=bool(plus_cfg.get("require_assets", True)) and not create_only,
    )

    tasks = select_tasks(
        load_tasks(repo_path),
        suites=list(cfg.MULTIRUN.task_suite_names),
        categories=None if plus_cfg.categories is None else list(plus_cfg.categories),
        difficulty_levels=(
            None if plus_cfg.difficulty_levels is None else list(plus_cfg.difficulty_levels)
        ),
        include_unknown_difficulty=bool(plus_cfg.include_unknown_difficulty),
        task_ids=(
            None if cfg.MULTIRUN.task_ids is None else [int(value) for value in cfg.MULTIRUN.task_ids]
        ),
        max_tasks_per_cell=plus_cfg.get("max_tasks_per_cell"),
        sample_seed=int(plus_cfg.sample_seed),
    )
    if not tasks:
        raise ValueError("The LIBERO-Plus filters selected zero tasks.")
    expected_num_tasks = plus_cfg.get("expected_num_tasks")
    if expected_num_tasks is not None and len(tasks) != int(expected_num_tasks):
        raise ValueError(
            "LIBERO-Plus task-count guard failed: "
            f"selected={len(tasks)}, expected={int(expected_num_tasks)}. "
            "Remove sampling/filter overrides for a full run, or explicitly enable a partial-stage workflow."
        )

    task_file = output_dir / "tasks.txt"
    manifest_file = output_dir / "task_manifest.jsonl"
    selected_task_file = output_dir / "selected_tasks.txt"
    write_task_files(tasks, selected_task_file, manifest_file)
    shards = create_balanced_shards(tasks, int(cfg.MULTIRUN.tasks_per_worker))
    write_shard_files(shards, task_file, output_dir / "shards")
    _print_selection(tasks)
    shard_counts = Counter(shard.suite for shard in shards)
    print("\nWorker shard plan:")
    print(f"- Tasks per worker: {int(cfg.MULTIRUN.tasks_per_worker)}")
    print(f"- Total worker processes: {len(shards)}")
    for suite, count in sorted(shard_counts.items()):
        sizes = [len(shard.tasks) for shard in shards if shard.suite == suite]
        print(f"- {suite}: {count} workers, {min(sizes)}-{max(sizes)} tasks per worker")
    libero_config_dir = _write_libero_config(output_dir, benchmark_root)
    OmegaConf.save(cfg, f=str(output_dir / "manager_config.yaml"))
    print(f"- Task file: {task_file}")
    print(f"- Selected task pairs: {selected_task_file}")
    print(f"- Manifest: {manifest_file}")

    if create_only:
        print("MULTIRUN.create_only=true; task generation completed without launching workers.")
        return

    _run_evaluation(
        task_file=task_file,
        task_choice=_resolve_worker_task_choice(),
        ckpt=str(cfg.ckpt),
        num_gpus=int(cfg.MULTIRUN.num_gpus),
        num_trials=int(cfg.EVALUATION.num_trials),
        max_tasks_per_gpu=int(cfg.MULTIRUN.max_tasks_per_gpu),
        output_dir=output_dir,
        extra_overrides=_collect_worker_overrides(),
        plus_repo=repo_path,
        libero_config_dir=libero_config_dir,
    )


if __name__ == "__main__":
    main()
