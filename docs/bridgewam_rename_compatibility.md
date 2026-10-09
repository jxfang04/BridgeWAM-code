# BridgeWAM-only migration and checkpoint contract

This document describes the BridgeWAM-only implementation and its checkpoint
compatibility contract. It supersedes an earlier naming-only migration that
retained baseline implementations and import aliases; the current implementation
removes those interfaces.

## Implementation and supported paths

`src/bridgewam/models/wan22/bridge_of_experts.py` now defines the complete
`BridgeOfExperts(nn.Module)` backbone. It does not inherit from a `MoT` class.
The Video/LBQ attention, action denoising and gradient-checkpointed operations
retain the reference computation for the supported BridgeWAM configurations.
The backbone has three explicit interfaces:

- `forward_bridge_video`: final Video tokens and the configured LBQ readout.
- `prefill_bridge`: fixed LBQ conditions, stopping at the configured readout.
- `forward_bridge_action`: Action denoising conditioned on LBQs.

The main training path is `scripts/train.py` -> `runtime.run_training` ->
`BridgeWAMTrainer` -> `BridgeWAM.training_loss`. The model factory lives in
`bridgewam.models.wan22.factory.create_bridgewam`.

Main action inference prefills Video/LBQ conditions once per action chunk, then
reuses those conditions for iterative Action denoising. `infer_joint` retains
joint video/action inference. Frozen Video, LBQ-K/V, IDM and Joint ablations
retain their own topology and checkpoint identity checks. The `lbq_kv_mot`
experiment identifier is retained; it does not import the removed MoT class.

LIBERO, LIBERO-Plus, LIBERO-Pro and RoboTwin evaluation entrypoints remain in
`experiments/`. Their simulator installations and assets are separate runtime
requirements. Local environments, private research records and generated results
are not included in this source distribution. RoboTwin's legacy policy symlink is replaced
with `policy/bridgewam_policy`, pointing to the canonical adapter.

## Removed and moved interfaces

| Previous interface | Current interface |
|---|---|
| `fastwam.*`, `FastWAM*`, legacy import hooks | Removed; import `bridgewam.*` |
| `mot.py`, direct Video-to-Action K/V routing | Removed; use the LBQ backbone |
| `create_fastwam_baseline`, disabled-LBQ model/task configs | Removed; no baseline fallback |
| `bridgewam.runtime.create_bridgewam` | `bridgewam.models.wan22.factory.create_bridgewam` |
| `mot_checkpoint_mixed_attn` | `checkpoint_attention` |
| `latent_bridge_queries` top-level package | `bridgewam.models.wan22.lbq` |
| `datasets/lerobot/lerobot` | `datasets/lerobot/backend` |
| `ablation_bridgewam/mot_variants.py` | `ablation_bridgewam/backbones.py` |
| `model.mot` as an execution interface | `model.bridge` |
| Unused `_predict_action_noise`, generic `infer` | `infer_action`, `infer_joint` |
| `FASTWAM_*` environment aliases | `BRIDGEWAM_*` |
| Old standalone Joint/IDM and baseline tasks | Explicit retained LBQ ablation tasks |

Old saved Hydra configurations must be updated before instantiation. Select a
current task, preserve its matching dataset/statistics and topology, and pass
the old weight checkpoint explicitly. Old Python full-model pickles and import
paths are not supported. Work that requires a removed architecture needs a
separately supplied reference implementation.

Unsupported CFG and tiled input encoding now raise explicit errors. Use
`text_cfg_scale=1`, an empty `negative_prompt`, and `tiled=false`. This cleanup
does not silently substitute an implementation for an unsupported option.

## Why historical names remain in checkpoint data

The runtime accesses `model.bridge`, but the same module is registered under
`mot` and `dit` to preserve state-dictionary keys and parameter ordering. The
historical `action_video_kv_layer_mask` buffer remains serialized; it does not
enable any direct-Video attention path. This is a data-format contract, not a
remaining MoT implementation.

`checkpoint_compat.py` continues to normalize known `fastwam`, `bridgewam`, `boe`,
`module`, expert and MetaQuery/LBQ weight names. The Frozen Video ablation retains
its historical `video_source` identity. Checkpoint filenames, including filenames
containing old model names, need no renaming. Conflicting aliases, missing or
unexpected parameters, shape mismatches and incompatible architecture metadata
continue to fail explicitly.

Compatibility applies to matching BridgeWAM topology and weight dictionaries.
It does not convert arbitrary baseline or different-network checkpoints into
BridgeWAM. Pretrained Video/Action initialization remains separate from restoring
a complete trained BridgeWAM checkpoint.

- `resume=/path/to/step.pt` restores model weights and starts a fresh optimizer.
- `resume=/path/to/checkpoints/state/step_xxxxxx` uses the existing
  Accelerate/DeepSpeed full-state restoration and its distributed requirements.
- `model.load_checkpoint(path, optimizer=optimizer)` restores a compatible saved
  optimizer when present. Tiny CPU AdamW continuation is tested.

## Validation

Release verification: **109 passed, 10 subtests passed, 1 failed**;
all **12 reference-source numerical cases matched exactly on CPU**. The failure
is a configuration test referencing the absent spectral experiment task YAML.
It is also reproducible in the source checkout; the release preserves the source
configuration set rather than silently restoring an old task. Python and shell
syntax checks passed. The validation environment uses Python 3.11 and CPU
PyTorch 2.7.1. See [source release checks](source_release_check.md) for scope.

Run both the migrated original tests and the source-distribution tests:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/src:$PWD" \
  python -m pytest -q -p no:cacheprovider tests scripts/validation

python scripts/verify_bridgewam_refactor.py \
  --reference-repo=/path/to/reference-repository \
  --reference-ref=YOUR_REFERENCE_REVISION
```

The pinned-source comparison extracts the original `src` tree from Git into a
temporary directory. It runs the reference and current implementations in separate
processes for six paths (main, spectral, Frozen Video, LBQ-K/V, IDM and Joint),
each with gradient checkpointing enabled and disabled. It checks exact CPU
weights, parameter order, loss, gradients, action/joint inference, optimizer
updates and continuation from a checkpoint written by the reference code.

The training smoke test uses synthetic observations, real dataset batching and
input preparation, a tiny VAE substitute, Accelerate backward, AdamW and checkpoint
saving. Other tests cover all seven task factories, strict checkpoint loading,
spectral gradient routing, pruned attention and LIBERO-Pro orchestration.
Tests for the removed baseline, legacy module aliases and pickle class aliases
were retired. The LIBERO-Pro test file is consolidated under `scripts/validation`.

These checks do not establish real full-size checkpoint restoration on CUDA,
eight-GPU training, distributed-state restoration or simulator success rates.
See the root [README](../README.md) for current launch commands and asset paths.
