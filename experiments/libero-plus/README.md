# BridgeWAM LIBERO-Plus evaluation

This pipeline keeps the original `experiments/libero` inference behavior and uses the
official LIBERO-Plus task classification to evaluate robustness by perturbation and
difficulty. LIBERO-Plus uses one trial per task.

For efficiency, selected tasks are grouped within each suite into balanced worker
shards. `MULTIRUN.tasks_per_worker=50` means one worker loads BridgeWAM once and then
evaluates 49-50 different LIBERO-Plus task variants sequentially. The full benchmark
therefore uses 203 model-loading processes instead of 10,030. Rollout MP4 generation
is disabled by default for Plus and can be restored with
`EVALUATION.save_rollout_video=true`.

## External environment

LIBERO-Plus is an external environment; the source distribution does not bundle it. Add
`LIBERO_PLUS.repo_path=/path/to/LIBERO-plus` to each manager/launcher command below.
All model code comes from this checkout. The default task is now the main 32-LBQ
BridgeWAM model, so supply a matching checkpoint rather than a baseline checkpoint.
`BRIDGEWAM_PYTHON` selects the interpreter; no historical environment-variable
alias is supported.

## Prerequisites

1. Place the official downloaded `assets` directory at
   `LIBERO-plus/libero/libero/assets`.
2. Install LIBERO-Plus extra dependencies from `LIBERO-plus/extra_requirements.txt`
   in the BridgeWAM environment. The sensor-noise tasks also require ImageMagick.
3. Run from the BridgeWAM project root. The manager creates a per-run LIBERO path
   config, so it does not overwrite `~/.libero/config.yaml`.

## Stage one

First generate and inspect the complete 10,030-task manifest without launching:

```bash
python experiments/libero-plus/run_libero_plus_manager.py \
  task=libero_uncond_2cam224_lbqs_only_2layer_alternating_cross_self_fullfinetune_1e-4 \
  ckpt=/path/to/bridgewam_step.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.create_only=true
```

Run the full official benchmark:

```bash
bash experiments/libero-plus/run_stage1.sh \
  task=libero_uncond_2cam224_lbqs_only_2layer_alternating_cross_self_fullfinetune_1e-4 \
  ckpt=/path/to/bridgewam_step.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.num_gpus=6 \
  MULTIRUN.max_tasks_per_gpu=1
```

`run_stage1.sh` requires the full 10,030-task selection by default. For a deterministic
141-task diagnostic pilot, explicitly set `ALLOW_PARTIAL_STAGE1=true` and add
`LIBERO_PLUS.max_tasks_per_cell=1`. This selects at most one task from every suite,
perturbation, and difficulty cell. Filters can use:

```text
LIBERO_PLUS.categories=[camera_viewpoints,robot_initial_states]
LIBERO_PLUS.difficulty_levels=[1,2,3]
LIBERO_PLUS.include_unknown_difficulty=false
```

The main outputs are `task_manifest.jsonl`, `libero_plus_task_results.csv`,
`libero_plus_summary.csv`, and `libero_plus_summary.json`. Missing tasks are reported
separately and are never counted as failed episodes.

`run_stage1.sh` launches the stage-one manager in a detached tmux session by
default, so the scheduler and GPU workers survive an SSH or browser-IDE
disconnect. The launcher prints the tmux session and manager-log paths and also
writes them to `<output_dir>/stage1_launcher_info.txt`. Use `tmux attach -t
<session>` to watch the manager or `tail -f <output_dir>/stage1_manager.log` to
follow its persistent log. Set `LIBERO_PLUS_DETACH=false` for foreground
debugging.
