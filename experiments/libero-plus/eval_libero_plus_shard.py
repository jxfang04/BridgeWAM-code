"""Evaluate one LIBERO-Plus task shard with a single BridgeWAM model load."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepend_once(path: Path) -> None:
    value = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != value]
    sys.path.insert(0, value)


def _setup_libero_plus() -> None:
    plus_repo = Path(
        os.environ.get("LIBERO_PLUS_REPO", str(PROJECT_ROOT / "LIBERO-plus"))
    ).expanduser().resolve()
    config_dir = os.environ.get("LIBERO_CONFIG_PATH")
    if not config_dir:
        raise RuntimeError("LIBERO_CONFIG_PATH must point to the per-run LIBERO-Plus config.")
    config_file = Path(config_dir).expanduser() / "config.yaml"
    if not config_file.is_file():
        raise FileNotFoundError(f"LIBERO-Plus config file not found: {config_file}")

    _prepend_once(PROJECT_ROOT)
    _prepend_once(PROJECT_ROOT / "experiments" / "libero")
    _prepend_once(plus_repo)
    plus_package = importlib.import_module("libero.libero")
    loaded_from = Path(plus_package.__file__).resolve()
    expected_root = (plus_repo / "libero" / "libero").resolve()
    if expected_root not in loaded_from.parents:
        raise RuntimeError(
            f"Loaded the wrong LIBERO package: {loaded_from}; expected it under {expected_root}."
        )


_setup_libero_plus()

import hydra
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig

from experiments.libero import eval_libero_single as base_eval
from bridgewam.datasets.lerobot.processors.bridgewam_processor import BridgeWAMProcessor
from bridgewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from bridgewam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark
from libero_plus_metadata import PlusTask, read_manifest


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=base_eval.NumpyEncoder)
    os.replace(temp_path, path)


def _task_result_path(output_dir: Path, task: PlusTask) -> Path:
    return output_dir / task.suite / f"task{task.task_id}_results.json"


def _read_completed_result(path: Path, task: PlusTask) -> dict | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("task_suite") != task.suite or int(result.get("task_id", -1)) != task.task_id:
        raise ValueError(
            f"Existing task result metadata does not match {task.suite}:{task.task_id}: {path}"
        )
    return result


def _initialize_model_and_processor(cfg: DictConfig):
    partial_state = PartialState()
    partial_state.config = cfg
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be null.")
    base_eval._validate_visualize_future_video_cfg(cfg)
    if int(cfg.EVALUATION.get("env_num", 1)) != 1:
        raise ValueError("LIBERO-Plus shard evaluation requires EVALUATION.env_num=1.")

    model_device = base_eval._resolve_eval_device(cfg)
    model_dtype = base_eval._mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    base_eval._load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = base_eval._resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: BridgeWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon")
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if action_horizon_cfg is None
        else int(action_horizon_cfg)
    )
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")
    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    return model, processor, model_device, action_horizon, int(video_size[1]), int(video_size[0])


def _evaluate_task(
    cfg: DictConfig,
    task_meta: PlusTask,
    task_suite,
    model,
    processor,
    model_device: str,
    action_horizon: int,
    input_w: int,
    input_h: int,
    output_dir: Path,
) -> dict:
    cfg.EVALUATION.task_suite_name = task_meta.suite
    cfg.EVALUATION.task_id = task_meta.task_id
    task = task_suite.get_task(task_meta.task_id)
    if task.name != task_meta.name:
        raise ValueError(
            f"Shard/benchmark task mismatch for {task_meta.suite}:{task_meta.task_id}: "
            f"{task_meta.name!r} != {task.name!r}"
        )
    initial_states = task_suite.get_task_init_states(task_meta.task_id)
    if len(initial_states) < int(cfg.EVALUATION.num_trials):
        raise ValueError(
            f"Task {task_meta.suite}:{task_meta.task_id} has {len(initial_states)} initial states, "
            f"but {int(cfg.EVALUATION.num_trials)} trials were requested."
        )

    task_start = time.time()
    video_dir = output_dir / task_meta.suite / "videos"
    predicted_video_dir = output_dir / task_meta.suite / "predicted_videos"
    if bool(cfg.EVALUATION.get("save_rollout_video", True)):
        video_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "task_suite": task_meta.suite,
        "task_id": task_meta.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0.0,
    }
    task_result = base_eval.run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    result.update(task_result)
    result["duration"] = time.time() - task_start
    result["category"] = task_meta.category
    result["difficulty_level"] = task_meta.difficulty_level
    return result


def _write_shard_marker(
    marker_path: Path,
    shard_id: int,
    suite: str,
    results: list[dict],
    started_at: float,
) -> None:
    _write_json_atomic(
        marker_path,
        {
            "shard_id": shard_id,
            "suite": suite,
            "num_tasks": len(results),
            "total_episodes": sum(int(result["total_episodes"]) for result in results),
            "successes": sum(int(result["successes"]) for result in results),
            "duration": time.time() - started_at,
            "task_ids": [int(result["task_id"]) for result in results],
        },
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_plus")
def eval_shard(cfg: DictConfig) -> None:
    shard_started_at = time.time()
    if cfg.EVALUATION.task_suite_name != "libero_plus_shard":
        raise ValueError(
            "Shard worker expects EVALUATION.task_suite_name=libero_plus_shard, got "
            f"{cfg.EVALUATION.task_suite_name!r}."
        )
    shard_id = int(cfg.EVALUATION.task_id)
    output_dir = Path(cfg.EVALUATION.output_dir).expanduser().resolve()
    shard_file = output_dir / "shards" / f"shard_{shard_id:05d}.jsonl"
    if not shard_file.is_file():
        raise FileNotFoundError(f"Shard manifest not found: {shard_file}")
    tasks = read_manifest(shard_file)
    if not tasks:
        raise ValueError(f"Shard manifest is empty: {shard_file}")
    suites = {task.suite for task in tasks}
    if len(suites) != 1:
        raise ValueError(f"Shard {shard_id} crosses suites: {sorted(suites)}")
    suite_name = next(iter(suites))
    marker_path = output_dir / "libero_plus_shard" / f"gpu{cfg.gpu_id}_task{shard_id}_results.json"

    completed = []
    pending = []
    for task_meta in tasks:
        existing = _read_completed_result(_task_result_path(output_dir, task_meta), task_meta)
        if existing is None:
            pending.append(task_meta)
        else:
            completed.append(existing)
    logging.info(
        "Shard %s (%s): total=%s completed=%s pending=%s",
        shard_id,
        suite_name,
        len(tasks),
        len(completed),
        len(pending),
    )

    if pending:
        model, processor, model_device, action_horizon, input_w, input_h = (
            _initialize_model_and_processor(cfg)
        )
        task_suite = benchmark.get_benchmark_dict()[suite_name]()
        for position, task_meta in enumerate(pending, start=1):
            logging.info(
                "Shard %s task %s/%s: %s task_id=%s",
                shard_id,
                position,
                len(pending),
                task_meta.suite,
                task_meta.task_id,
            )
            result = _evaluate_task(
                cfg=cfg,
                task_meta=task_meta,
                task_suite=task_suite,
                model=model,
                processor=processor,
                model_device=model_device,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                output_dir=output_dir,
            )
            _write_json_atomic(_task_result_path(output_dir, task_meta), result)
            completed.append(result)
            print(
                f"Shard {shard_id}: {task_meta.suite} task {task_meta.task_id} completed "
                f"({result['successes']}/{result['total_episodes']})"
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    completed_by_key = {
        (str(result["task_suite"]), int(result["task_id"])): result for result in completed
    }
    ordered_results = [completed_by_key[(task.suite, task.task_id)] for task in tasks]
    _write_shard_marker(marker_path, shard_id, suite_name, ordered_results, shard_started_at)
    print(
        f"Shard {shard_id} completed: {len(ordered_results)} tasks, "
        f"{sum(int(result['successes']) for result in ordered_results)} successes"
    )


if __name__ == "__main__":
    eval_shard()
