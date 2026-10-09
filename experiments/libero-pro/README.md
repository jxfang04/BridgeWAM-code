# BridgeWAM on LIBERO-Pro

This evaluator reuses the standard LIBERO model loader, image/state processing,
action denormalization, gripper conversion, rollout loop and video writer.
Only the simulator environment comes from the selected external LIBERO-PRO checkout.

## Protocol

Four base suites (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`)
combine with five separate perturbation categories:

| Category | Official suite suffix |
|---|---|
| object | `_object` |
| position | `_swap` |
| language | `_lan` |
| task | `_task` |
| environment | `_env` |

Task IDs and ordering come from the external `benchmark/libero_suite_task_map.py`.
The referenced table has 200 tasks; 50 initial states per task give 10,000 episodes.
Insufficient initial states fail instead of repeating available states.
Instructions use parsed BDDL `:language`, preserving language perturbations.
Default horizons are Spatial 220, Object 280, Goal 300 and LIBERO-10 520, with ten
additional wait steps and the configured ten-step replanning interval. Overrides
are recorded in the run config; keep them fixed when comparing results.
This entry evaluates separate perturbation categories, not runtime combinations.

## Environment

Install the matching [LIBERO-Pro environment](https://github.com/Zxy-MLlab/LIBERO-PRO)
and [resources](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro), including
`libero/libero/bddl_files`, `init_files`, `assets`, and the task map. Model code must
come from this BridgeWAM checkout. No simulator resources are bundled here.

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export LIBERO_PRO_REPO=/path/to/LIBERO-PRO
```

Alternatively pass `LIBERO_PRO.repo_path=/path/to/LIBERO-PRO`. Each run creates an
isolated `libero_pro_config/config.yaml` and verifies the imported environment and
resource paths. This avoids accidentally using standard LIBERO or Plus resources.

## Plan without launching

This checks task selection without requiring a checkpoint, GPU or MuJoCo. It does
not certify resource completeness:

```bash
python experiments/libero-pro/run_libero_pro_manager.py \
  MULTIRUN.create_only=true \
  LIBERO_PRO.expected_num_tasks=200 \
  EVALUATION.output_dir=/path/to/pro-plan
```

## Evaluate

The default model task is the main 32-LBQ, 30-Video-layer, two-Action-layer BridgeWAM
configuration. Supply the checkpoint's matching task for any ablation.

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python experiments/libero-pro/run_libero_pro_manager.py \
  ckpt=/path/to/bridgewam_step.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  EVALUATION.output_dir=/path/to/new-pro-run \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=1 \
  LIBERO_PRO.expected_num_tasks=200
```

The manager stays in the foreground and waits for workers. Use tmux for persistent
background execution. Each worker sees one assigned GPU and loads the model for
one task. `BRIDGEWAM_PYTHON` overrides the current interpreter. Allocate idle GPUs;
the manager does not control unrelated processes.

For a one-task language check, append:

```text
'MULTIRUN.task_suite_names=[libero_goal]'
'MULTIRUN.task_ids=[0]'
'LIBERO_PRO.perturbations=[language]'
MULTIRUN.num_gpus=1
```

Use `eval_libero_pro_single.py` for a direct worker, with
`EVALUATION.task_suite_name=libero_goal_lan` and `EVALUATION.task_id=0`.

## Results

The manager saves the full `worker_config.yaml`, ordered `task_manifest.jsonl`,
its SHA-256, `tasks.txt`, isolated environment config, worker logs, per-task JSON,
`manager_runtime.json` and `summary.json`. Without Git history, the runtime commit
may be null; the root README records the source commit.

Success rates pool successful episodes over evaluated episodes. Missing tasks mark
the run incomplete; duplicate/mismatched task results fail explicitly. Exit zero
requires all workers to succeed and all results to be present. Always use a fresh
output directory for a new checkpoint or experiment.

```bash
python experiments/libero-pro/summarize_results.py --output_dir=/path/to/pro-run
```

Manifest, adapter and scheduler tests live in `scripts/validation`. CPU/mock tests
do not measure real MuJoCo/GPU success rates.
