# Source release checks

## Source fidelity

The model package (`src`, 56 files), configurations (`configs`, 15 files),
evaluation entrypoints (`experiments`, 23 files), tests (7 files), and tracked
third-party entries (1,038 files or links) match the selected source snapshot
byte for byte. No model computation, initialization, optimizer, loss or main-task
hyperparameter was changed during publication preparation.

Two auxiliary scripts have publication-only changes:

- `scripts/backfill_wandb_from_log.py` no longer supplies a personal W&B workspace.
  Pass `--entity` explicitly or use the configured W&B account.
- `scripts/verify_bridgewam_refactor.py` requires `--reference-ref` instead of
  embedding an internal repository revision.

README and documentation references to internal branches, revisions and private
server paths were removed or replaced with generic examples. Upstream license
notices and dependency/data links were retained.

## Publication scope

The release includes training, evaluation, validation and package source code.
It excludes local connection settings, personal experiment notes, generated
figures/results, historical research utilities, external benchmark environments,
model weights, datasets, caches and `README_zh.md`.

The tracked RoboTwin tree is mostly empty placeholders, exactly as in the source
snapshot; it is not a complete simulator installation. Install the external
simulator and its assets as described in the root README before evaluation.

## Verification

- Python and shell syntax checks passed.
- CPU test suite: 109 passed, 10 subtests passed, 1 failed.
- The failing spectral configuration test references
  `libero_uncond_2cam224_lbqs_spectral_2layer_fullfinetune_1e-4`, which is absent
  from the selected source snapshot. The same test fails in that source checkout.
  It was not repaired by introducing a configuration from a different revision.
- All 12 tiny-model reference comparisons matched exactly on CPU, including
  weights, losses, gradients, inference, optimizer updates and checkpoint resume.
- Working-tree scanning found no matching personal identifiers, private absolute
  paths, email addresses, credential/private-key patterns or private tracking URLs.
  Remaining external links refer to upstream software, models or datasets.

These checks do not certify the absence of all unknown identifying information,
and do not cover Git history, hosting-account metadata or previously published
copies. Git metadata was explicitly left untouched. Full CUDA/distributed and
simulator evaluations were not run as part of this source synchronization.
