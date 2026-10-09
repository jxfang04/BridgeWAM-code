# Prefer this checkout even when another BridgeWAM is installed.
import sys
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SOURCE_ROOT))

import hydra
from omegaconf import DictConfig

from bridgewam.runtime import run_training
from bridgewam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
