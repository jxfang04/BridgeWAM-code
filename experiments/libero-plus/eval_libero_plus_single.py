"""Bootstrap the existing BridgeWAM worker against the LIBERO-Plus package."""

from __future__ import annotations

import importlib
import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepend_once(path: Path) -> None:
    value = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != value]
    sys.path.insert(0, value)


def main() -> None:
    plus_repo = Path(
        os.environ.get("LIBERO_PLUS_REPO", str(PROJECT_ROOT / "LIBERO-plus"))
    ).expanduser().resolve()
    config_dir = os.environ.get("LIBERO_CONFIG_PATH")
    if not config_dir:
        raise RuntimeError("LIBERO_CONFIG_PATH must point to the per-run LIBERO-Plus config directory.")
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

    if not any(arg == "--config-name" or arg.startswith("--config-name=") for arg in sys.argv[1:]):
        sys.argv[1:1] = ["--config-name", "sim_libero_plus"]

    worker = PROJECT_ROOT / "experiments" / "libero" / "eval_libero_single.py"
    runpy.run_path(str(worker), run_name="__main__")


if __name__ == "__main__":
    main()
