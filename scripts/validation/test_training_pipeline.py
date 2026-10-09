"""One actual CPU Accelerate training step with synthetic observations and VAE."""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from hydra import compose, initialize_config_dir

from bridgewam.trainer import BridgeWAMTrainer
from bridgewam.utils.config_resolvers import register_default_resolvers
from reference_worker import build_model

ROOT = Path(__file__).resolve().parents[2]


class TinyVAE(nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 16

    def encode(self, videos, device, **kwargs):
        result = []
        for video in videos:
            # Deterministic stand-in for image encoding; real model.build_inputs runs.
            x = video[:2, ::4].unsqueeze(0).to(device)
            result.append(torch.nn.functional.avg_pool3d(x, (1, 16, 16)).squeeze(0))
        return torch.stack(result)


class Observations(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(40 + index)
        return dict(
            video=torch.randn(3, 9, 32, 32, generator=g),
            context=torch.randn(3, 8, generator=g),
            context_mask=torch.ones(3, dtype=torch.bool),
            action=torch.randn(8, 2, generator=g),
            proprio=torch.randn(9, 2, generator=g),
            action_is_pad=torch.zeros(8, dtype=torch.bool),
            image_is_pad=torch.zeros(9, dtype=torch.bool),
        )


def test_cpu_training_runs_real_input_loss_optimizer_and_checkpoint(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    register_default_resolvers()
    torch.set_num_threads(1)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[
                "mixed_precision=no",
                "max_steps=1",
                "batch_size=1",
                "num_workers=0",
                "save_every=1",
                "eval_every=0",
                "log_every=0",
                f"output_dir={tmp_path}",
            ],
        )
    model = build_model("main", True)
    model.vae = TinyVAE()
    before = model.action_expert.blocks[0].cross_attn.q.weight.detach().clone()
    trainer = BridgeWAMTrainer(model, Observations(), cfg=cfg)
    trainer.train()
    assert trainer.global_step == 1
    assert not torch.equal(before, model.action_expert.blocks[0].cross_attn.q.weight)
    weights = list((tmp_path / "checkpoints/weights").glob("*.pt"))
    assert weights
    payload = torch.load(weights[0], weights_only=True)
    assert payload["step"] == 1 and payload["model_name"] == "bridgewam"
    assert payload["latent_bridge_queries"]["num_lbqs"] == 32
