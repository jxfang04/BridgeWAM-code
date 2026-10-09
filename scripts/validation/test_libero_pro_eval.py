"""Pro manifest, adapter and scheduling checks without MuJoCo or model downloads."""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

from omegaconf import OmegaConf
import pytest


@pytest.fixture
def pro(monkeypatch):
    directory = Path(__file__).resolve().parents[2] / "experiments/libero-pro"
    monkeypatch.syspath_prepend(str(directory))
    return SimpleNamespace(
        metadata=importlib.import_module("libero_pro_metadata"),
        worker=importlib.import_module("eval_libero_pro_single"),
        manager=importlib.import_module("run_libero_pro_manager"),
        summary=importlib.import_module("summarize_results"),
    )


@pytest.fixture
def repo(tmp_path, pro):
    repo = tmp_path / "external Pro environment"
    root = repo / "libero/libero"
    (root / "benchmark").mkdir(parents=True)
    (root / "assets").mkdir()
    mapping = {}
    for base in pro.metadata.BASE_SUITES:
        for suffix in pro.metadata.PERTURBATIONS.values():
            suite = f"{base}_{suffix}"
            mapping[suite] = ["old_instruction", "another_instruction"]
            for name in mapping[suite]:
                for folder, extension in (
                    ("bddl_files", ".bddl"),
                    ("init_files", ".pruned_init"),
                ):
                    directory = root / folder / suite
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / (name + extension)).write_text("fixture resource")
    (root / "benchmark/libero_suite_task_map.py").write_text(
        "libero_task_map = " + repr(mapping)
    )
    return repo


def test_pro_selects_published_suffixes_and_checks_resources(pro, repo, tmp_path):
    tasks = pro.metadata.select_tasks(
        repo, pro.metadata.BASE_SUITES, pro.metadata.PERTURBATIONS
    )
    assert len(tasks) == 40
    assert len({task["suite"] for task in tasks}) == 20
    assert {task["perturbation"] for task in tasks} == set(pro.metadata.PERTURBATIONS)
    pro.metadata.validate_resources(repo, tasks, require_assets=True)
    directory, digest = pro.metadata.write_run_files(tmp_path / "run", repo, tasks)
    settings = OmegaConf.load(directory / "config.yaml")
    assert settings.bddl_files == str(repo / "libero/libero/bddl_files")
    assert len(digest) == 64
    assert (tmp_path / "run/tasks.txt").read_text().splitlines()[
        0
    ] == "libero_spatial_object,0"
    (
        repo
        / "libero/libero/init_files/libero_spatial_object/old_instruction.pruned_init"
    ).unlink()
    with pytest.raises(FileNotFoundError, match="resources missing"):
        pro.metadata.validate_resources(repo, tasks, require_assets=True)


@pytest.mark.parametrize("ids", [[-1], [2], [0, 0], []])
def test_pro_rejects_invalid_task_selection(pro, repo, ids):
    with pytest.raises(ValueError, match="Invalid task IDs"):
        pro.metadata.select_tasks(repo, ["libero_goal"], ["language"], ids)


def test_language_adapter_uses_bddl_not_filename(pro, monkeypatch):
    utility = ModuleType("experiments.libero.libero_utils")
    env = SimpleNamespace(
        language_instruction="  grasp the blue cup  ", close=lambda: None
    )
    utility.get_libero_env = lambda *args: (
        env,
        "old instruction derived from filename",
    )
    monkeypatch.setitem(sys.modules, utility.__name__, utility)
    actual, text = pro.worker.get_pro_env(
        SimpleNamespace(name="old_instruction"), 256, 42
    )
    assert actual is env and text == "grasp the blue cup"
    env.language_instruction = ""
    with pytest.raises(ValueError, match="Missing BDDL"):
        pro.worker.get_pro_env(SimpleNamespace(name="old_instruction"), 256, 42)


def test_worker_reuses_shared_evaluator_with_pro_limits(
    pro, repo, tmp_path, monkeypatch
):
    output = tmp_path / "run"
    cfg = OmegaConf.create(
        dict(
            LIBERO_PRO=dict(repo_path=str(repo), require_assets=True),
            gpu_id=0,
            EVALUATION=dict(
                output_dir=str(output),
                task_suite_name="libero_goal_lan",
                task_id=0,
                num_trials=50,
                max_steps=None,
            ),
        )
    )
    calls = []
    base = ModuleType("experiments.libero.eval_libero_single")

    def evaluate(config, **kwargs):
        calls.append((config, kwargs))
        (output / "libero_goal_lan").mkdir(parents=True)
        return dict(
            task_suite="libero_goal_lan", task_id=0, successes=4, total_episodes=50
        )

    base.evaluate_single_process = evaluate
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setattr(pro.worker, "setup_environment", lambda *args: None)
    result = pro.worker.main.__wrapped__(cfg)
    assert result["max_steps"] == 300
    assert calls[0][1] == dict(
        env_factory=pro.worker.get_pro_env,
        require_full_init_states=True,
        expected_task_name="old_instruction",
    )
    assert result["benchmark"] == "LIBERO-Pro" and result["perturbation"] == "language"
    summary = pro.summary.summarize(output)
    assert (
        summary["complete"] and summary["groups"]["overall"]["success_rate"] == 4 / 50
    )


def test_summary_requires_complete_unique_results_and_pools_episodes(
    pro, repo, tmp_path
):
    output = tmp_path / "run"
    tasks = pro.metadata.select_tasks(repo, ["libero_goal"], ["object"])
    pro.metadata.write_run_files(output, repo, tasks)
    directory = output / "libero_goal_object"
    directory.mkdir()

    def write(task_id, successes, episodes):
        result = dict(
            benchmark="LIBERO-Pro",
            task_suite="libero_goal_object",
            task_id=task_id,
            task_name=tasks[task_id]["name"],
            successes=successes,
            total_episodes=episodes,
        )
        path = directory / f"gpu0_task{task_id}_results.json"
        path.write_text(json.dumps(result))
        return path

    first = write(0, 1, 2)
    partial = pro.summary.summarize(output)
    assert not partial["complete"] and len(partial["missing_tasks"]) == 1
    write(1, 1, 8)
    assert pro.summary.summarize(output)["groups"]["overall"]["success_rate"] == 0.2
    (directory / "gpu1_task0_results.json").write_text(first.read_text())
    with pytest.raises(ValueError, match="Duplicate results"):
        pro.summary.summarize(output)


def test_manager_forwards_frozen_config_and_gpu_isolation(
    pro, repo, tmp_path, monkeypatch
):
    cfg = OmegaConf.create(dict(MULTIRUN=dict(num_gpus=2, max_tasks_per_gpu=1)))
    tasks = pro.metadata.select_tasks(repo, ["libero_goal"], ["object"])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    monkeypatch.setenv("BRIDGEWAM_PYTHON", sys.executable)
    monkeypatch.setattr(pro.manager.subprocess, "run", run)
    results = pro.manager.run_workers(cfg, tasks, tmp_path)
    assert len(results) == len(calls) == 2
    for command, kwargs in calls:
        assert command[0] == sys.executable
        assert command[command.index("--config-path") + 1] == str(tmp_path)
        assert "worker_config" in command
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] in {"3", "5"}
        assert kwargs["env"]["LIBERO_CONFIG_PATH"] == str(
            tmp_path / "libero_pro_config"
        )
        assert kwargs.get("shell", False) is False


def test_create_only_cli_needs_no_simulator_and_preserves_model_config(
    pro, repo, tmp_path
):
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "dry run"
    command = [
        sys.executable,
        str(root / "experiments/libero-pro/run_libero_pro_manager.py"),
        "MULTIRUN.create_only=true",
        f"LIBERO_PRO.repo_path={repo}",
        f"EVALUATION.output_dir={output}",
        "LIBERO_PRO.expected_num_tasks=40",
        "model.latent_bridge_queries.num_lbqs=16",
    ]
    subprocess.run(
        command,
        cwd=tmp_path,
        check=True,
        capture_output=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    config = OmegaConf.load(output / "worker_config.yaml")
    assert config.model.latent_bridge_queries.enabled
    assert config.model.latent_bridge_queries.num_lbqs == 16
    assert config.model.action_dit_config.num_layers == 2
    assert not (output / "manager_runtime.json").exists()


def test_environment_bootstrap_selects_external_pro_package(pro, repo, tmp_path):
    tasks = pro.metadata.select_tasks(repo, ["libero_goal"], ["object"])
    directory, _ = pro.metadata.write_run_files(tmp_path / "run", repo, tasks)
    (repo / "libero/__init__.py").write_text("")
    (repo / "libero/libero/__init__.py").write_text("""
import os, yaml
from pathlib import Path
config = yaml.safe_load((Path(os.environ['LIBERO_CONFIG_PATH']) / 'config.yaml').read_text())
def get_libero_path(key): return config[key]
""")
    code = """
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from eval_libero_pro_single import setup_environment
setup_environment(Path(sys.argv[2]), Path(sys.argv[3]))
import libero.libero
assert str(Path(sys.argv[2])) in libero.libero.__file__
"""
    subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(Path(pro.worker.__file__).parent),
            str(repo),
            str(directory),
        ],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        check=True,
        capture_output=True,
    )
