import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2
TMUX_INNER_ENV = "BRIDGEWAM_ROBOTWIN_MANAGER_IN_TMUX"
TMUX_DISABLE_ENV = "BRIDGEWAM_ROBOTWIN_MANAGER_NO_TMUX"
TMUX_SESSION_ENV = "ROBOTWIN_TMUX_SESSION_NAME"
TMUX_LOG_FILENAME = "robotwin_manager.log"
TMUX_RUNTIME_FILENAME = "tmux_runtime.txt"
TMUX_ENV_ALLOWLIST = (
    "PATH",
    "PYTHONPATH",
    "LD_LIBRARY_PATH",
    "CUDA_HOME",
    "CONDA_PREFIX",
    "CUDA_VISIBLE_DEVICES",
    "DIFFSYNTH_MODEL_BASE_PATH",
    "DIFFSYNTH_SKIP_DOWNLOAD",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "NVIDIA_DRIVER_CAPABILITIES",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_CACHE",
    "TRANSFORMERS_OFFLINE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
    "WANDB_MODE",
    "WANDB_PROJECT",
)


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sanitize_tmux_session_name(raw_name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("_")
    if sanitized == "":
        sanitized = f"robotwin_{os.getpid()}"
    return sanitized[:120]


def _inner_manager_args(output_dir: Path) -> list[str]:
    args: list[str] = []
    for arg in sys.argv[1:]:
        key = arg.split("=", 1)[0].lstrip("+~")
        if key == "EVALUATION.output_dir":
            continue
        args.append(arg)
    args.append(f"EVALUATION.output_dir={output_dir}")
    return args


def _launch_manager_in_tmux(output_dir: Path) -> None:
    tmux_binary = shutil.which("tmux")
    if tmux_binary is None:
        raise RuntimeError(
            "tmux is required to launch RoboTwin evaluation. "
            f"Set {TMUX_DISABLE_ENV}=1 to run the manager in the foreground."
        )

    session_default = f"robotwin_{output_dir.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    session_name = _sanitize_tmux_session_name(
        os.environ.get(TMUX_SESSION_ENV, session_default)
    )
    session_check = subprocess.run(
        [tmux_binary, "has-session", "-t", session_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if session_check.returncode == 0:
        raise RuntimeError(f"tmux session already exists: {session_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / TMUX_LOG_FILENAME
    runtime_file = output_dir / TMUX_RUNTIME_FILENAME

    inner_cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        *_inner_manager_args(output_dir),
    ]
    shell_lines = [
        "set -o pipefail",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
        f"export {TMUX_INNER_ENV}=1",
    ]
    for name in TMUX_ENV_ALLOWLIST:
        if name in os.environ:
            shell_lines.append(f"export {name}={shlex.quote(os.environ[name])}")
    shell_lines.extend(
        [
            (
                f"printf '\\n[%s] RoboTwin manager started in tmux session "
                f"{shlex.quote(session_name)}\\n' \"$(date '+%Y-%m-%d %H:%M:%S')\" "
                f"| tee -a {shlex.quote(str(log_file))}"
            ),
            f"{shlex.join(inner_cmd)} 2>&1 | tee -a {shlex.quote(str(log_file))}",
            "rc=${PIPESTATUS[0]}",
            (
                f"printf 'ended_at=%s\\nexit_code=%s\\n' "
                f"\"$(date '+%Y-%m-%d %H:%M:%S')\" \"$rc\" "
                f">> {shlex.quote(str(runtime_file))}"
            ),
            'exit "$rc"',
        ]
    )
    shell_command = "\n".join(shell_lines)

    runtime_file.write_text(
        "\n".join(
            [
                f"tmux_session={session_name}",
                f"output_dir={output_dir}",
                f"log_file={log_file}",
                f"launcher_pid={os.getpid()}",
                f"started_at={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"command={shlex.join(inner_cmd)}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    try:
        subprocess.run(
            [
                tmux_binary,
                "new-session",
                "-d",
                "-s",
                session_name,
                "/bin/bash",
                "-lc",
                shell_command,
            ],
            check=True,
        )
    except Exception:
        runtime_file.unlink(missing_ok=True)
        raise

    print(f"RoboTwin evaluation started in tmux session: {session_name}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Combined log: {log_file}", flush=True)
    print(f"Attach: tmux attach -t {shlex.quote(session_name)}", flush=True)
    print(f"Follow log: tail -f {shlex.quote(str(log_file))}", flush=True)


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = list(range(num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_output_dir = output_dir
    run_output_dir.mkdir(parents=True, exist_ok=True)

    if not _env_flag_enabled(TMUX_INNER_ENV) and not _env_flag_enabled(TMUX_DISABLE_ENV):
        _launch_manager_in_tmux(run_output_dir)
        return

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks()
    else:
        tasks = [str(task_name_cfg)]

    extra_overrides = _collect_worker_overrides()

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()
            running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase="clean"))

    def write_outputs() -> None:
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_rates[task]["clean"],
                        task_rates[task]["random"],
                    ]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} output_dir={run_output_dir}"
    )

    # Launch initial tasks for each GPU up to capacity.
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    failure_message = ""

    while len(running_states) > 0:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_rates[state.task_name][state.phase] = success_rate
            log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f}"
            )

            if state.phase == "clean":
                running_states.append(launch_phase(
                    task_name=state.task_name,
                    gpu_id=gpu_id,
                    phase="random",
                ))
                continue

            try_launch_pending(gpu_id)

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # Mark not started tasks when failure happened.
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
