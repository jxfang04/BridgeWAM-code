"""Create a reproducible Pro task manifest and run GPU workers in the foreground."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from libero_pro_metadata import resolve_repo_path, select_tasks, validate_resources, write_run_files
from summarize_results import summarize

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_workers(cfg, tasks, output: Path):
    num_gpus, concurrency = int(cfg.MULTIRUN.num_gpus), int(cfg.MULTIRUN.max_tasks_per_gpu)
    if num_gpus <= 0 or concurrency <= 0:
        raise ValueError("GPU count and max_tasks_per_gpu must be positive.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [x.strip() for x in visible.split(",") if x.strip()] if visible is not None else [str(i) for i in range(num_gpus)]
    if len(devices) < num_gpus or len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain enough distinct GPUs.")
    pending = Queue()
    for task in tasks:
        pending.put(task)
    log_dir = output / "task_logs"
    log_dir.mkdir(exist_ok=True)
    worker = Path(__file__).with_name("eval_libero_pro_single.py")
    python = os.environ.get("BRIDGEWAM_PYTHON") or sys.executable

    def consume(slot):
        gpu_index = slot // concurrency
        outcomes = []
        while True:
            try:
                task = pending.get_nowait()
            except Empty:
                return outcomes
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[gpu_index],
                       LIBERO_CONFIG_PATH=str(output / "libero_pro_config"))
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT), env.get("PYTHONPATH")]))
            command = [python, str(worker), "--config-path", str(output), "--config-name", "worker_config",
                       f"EVALUATION.task_suite_name={task['suite']}", f"EVALUATION.task_id={task['task_id']}",
                       f"gpu_id={gpu_index}"]
            log_path = log_dir / f"{task['suite']}_task{task['task_id']}.log"
            with log_path.open("w") as log:
                completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            outcome = dict(suite=task["suite"], task_id=task["task_id"], gpu=devices[gpu_index],
                           exit_code=completed.returncode, log=str(log_path))
            outcomes.append(outcome)
            print(json.dumps(outcome), flush=True)

    with ThreadPoolExecutor(max_workers=num_gpus * concurrency) as executor:
        return [outcome for batch in executor.map(consume, range(num_gpus * concurrency)) for outcome in batch]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero_pro")
def main(cfg: DictConfig):
    create_only = bool(cfg.MULTIRUN.create_only)
    if not create_only and (cfg.ckpt is None or not Path(str(cfg.ckpt)).expanduser().is_file()):
        raise FileNotFoundError("Pass an existing BridgeWAM checkpoint with ckpt=...")
    if int(cfg.EVALUATION.num_trials) <= 0:
        raise ValueError("EVALUATION.num_trials must be positive.")
    if cfg.MULTIRUN.task_file is not None:
        raise ValueError("Select Pro tasks with task_suite_names, task_ids and LIBERO_PRO.perturbations.")
    output = Path(os.path.expandvars(str(cfg.EVALUATION.output_dir))).expanduser().resolve()
    if (output / "manager_runtime.json").exists() or any(output.glob("*/gpu*_results.json")):
        raise FileExistsError("Choose a new output directory; existing run results must not be mixed.")
    repo = resolve_repo_path(PROJECT_ROOT, cfg.LIBERO_PRO.repo_path)
    tasks = select_tasks(repo, cfg.MULTIRUN.task_suite_names, cfg.LIBERO_PRO.perturbations, cfg.MULTIRUN.task_ids)
    expected = cfg.LIBERO_PRO.expected_num_tasks
    if expected is not None and len(tasks) != int(expected):
        raise ValueError(f"Expected {expected} tasks, selected {len(tasks)}.")
    if not create_only:
        validate_resources(repo, tasks, require_assets=bool(cfg.LIBERO_PRO.require_assets))
    _, digest = write_run_files(output, repo, tasks)
    cfg.EVALUATION.output_dir = str(output)
    cfg.LIBERO_PRO.repo_path = str(repo)
    if cfg.ckpt is not None:
        cfg.ckpt = str(Path(str(cfg.ckpt)).expanduser().resolve())
    OmegaConf.save(cfg, output / "worker_config.yaml")
    print(f"LIBERO-Pro: {len(tasks)} tasks, {cfg.EVALUATION.num_trials} trials/task; manifest SHA256={digest}")
    print(f"Environment: {repo}\nOutput: {output}")
    if create_only:
        print("Manifest/configuration created; simulator resources were not validated and no workers launched.")
        return
    runtime_path = output / "manager_runtime.json"
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, capture_output=True)
    runtime = dict(base_code_commit=revision.stdout.strip() if revision.returncode == 0 else None,
                   task=HydraConfig.get().runtime.choices.get("task"), checkpoint=str(cfg.ckpt),
                   manifest_sha256=digest, started_at=datetime.now(timezone.utc).isoformat(), exit_code=None)
    runtime_path.write_text(json.dumps(runtime, indent=2))
    try:
        runtime["workers"] = run_workers(cfg, tasks, output)
        summary = summarize(output)
        if any(item["exit_code"] != 0 for item in runtime["workers"]) or not summary["complete"]:
            raise RuntimeError("LIBERO-Pro evaluation incomplete; inspect task_logs and summary.json.")
        runtime["exit_code"] = 0
    finally:
        if runtime["exit_code"] is None:
            runtime["exit_code"] = 1
        runtime["ended_at"] = datetime.now(timezone.utc).isoformat()
        runtime_path.write_text(json.dumps(runtime, indent=2))


if __name__ == "__main__":
    main()
