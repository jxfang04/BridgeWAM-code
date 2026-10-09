# BridgeWAM Ablations

All four models use 32 LBQs, preserve their original checkpoint identities, and
freeze the VAE and text encoder. They share the real BridgeOfExperts components;
there is no baseline model parent or separate direct Video K/V cache path.

| Task | Model and Action routing |
|---|---|
| `libero_bridgewam_ablation_frozen_video_2layer_1e-4` | Frozen Video/Proprio; Action cross-attention reads LBQs; all Action and LBQ parameters train |
| `libero_bridgewam_ablation_lbq_kv_mot_2layer_1e-4` | Text+State cross-attention and Action/LBQ mixed self-attention; Video/Action/LBQ/Proprio train |
| `libero_bridgewam_ablation_idm_2layer_1e-4` | Full future Video feeds Action through LBQs; Video/Action/LBQ/Proprio train |
| `libero_bridgewam_ablation_joint_30layer_1e-4` | Synchronous Video/LBQ/Action mixed attention in 30 layers; Action cross-attention reads Text+State |

The Joint experiment still has bidirectional cross-stream edges in its joint
attention mask. It is not the same causal information flow as the main two-layer
BridgeWAM model. IDM and Joint action inference generate/decode future video;
Frozen Video and LBQ-K/V use action-only inference.

## Initialization and training

For Frozen Video, explicitly provide the reference release weights:

```bash
export BRIDGEWAM_RELEASE_CKPT=/path/to/libero_uncond_2cam224.pt
```

Only Video DiT and Proprio weights load from that file. ActionDiT uses the existing
Action backbone, mapping its two layers to source layers 0 and 29. VAE weights come
from the matching Wan model. The historical release identity string is retained
solely as checkpoint metadata; no old model factory or import package is present.

Choose one task from the table and run:

```bash
bash scripts/train_zero1.sh 8 task="$TASK_NAME"
```

Configure data, text cache, pretrained weights and output paths as explained in the
root README. `checkpoint_attention` is the current gradient-checkpointing option.

## Evaluation and regression checks

```bash
python experiments/libero/run_libero_manager.py \
  task="$TASK_NAME" ckpt=/path/to/step.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json \
  MULTIRUN.num_gpus=8
```

Use an architecture- and experiment-matched checkpoint. Identity mismatch is an
error. `scripts/verify_bridgewam_refactor.py` compares all four implementations
against the pinned reference, including loss/gradients, action/joint inference and
optimizer continuation. `scripts/validation` checks real small factory builds and
public API contracts. These tests do not run GPU simulators.
