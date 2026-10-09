"""Pipeline contracts for the BridgeWAM-only source tree; no GPU or downloads."""

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from bridgewam.models.wan22.bridge_of_experts import BridgeOfExperts
from bridgewam.models.wan22.wan_video_dit import WanVideoDiT
from bridgewam.utils.config_resolvers import register_default_resolvers
from reference_worker import build_model

ROOT = Path(__file__).resolve().parents[2]
TASKS = sorted(p.stem for p in (ROOT / "configs/task").glob("*.yaml"))
register_default_resolvers()
torch.set_num_threads(1)


def test_no_legacy_model_or_import_entrypoints():
    assert not (ROOT / "src/fastwam").exists()
    assert not list((ROOT / "configs/model").glob("fastwam*"))
    assert not (ROOT / "src/bridgewam/models/wan22/mot.py").exists()
    assert BridgeOfExperts.__bases__ == (nn.Module,)
    model = build_model("main", True)
    assert model.bridge is model.mot is model.dit
    assert not hasattr(model, "_predict_action_noise")
    assert not hasattr(model, "infer")
    with pytest.raises(ValueError, match="enabled LBQs"):
        BridgeOfExperts(model.bridge.mixtures, latent_bridge_queries={"enabled": False})


@pytest.mark.parametrize("task", TASKS)
def test_every_task_factory_builds_its_real_lbq_topology(task, tmp_path):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        config = compose(config_name="train", overrides=["task=" + task])
    assert config.model.latent_bridge_queries.enabled
    assert not config.model.latent_bridge_queries.preserve_direct_video_kv
    config.model.video_dit_config.update(
        hidden_dim=8,
        ffn_dim=16,
        text_dim=8,
        freq_dim=8,
        num_heads=2,
        attn_head_dim=4,
        in_dim=2,
        out_dim=2,
        patch_size=[1, 1, 1],
    )
    config.model.action_dit_config.update(
        hidden_dim=8, ffn_dim=16, text_dim=8, freq_dim=8, num_heads=2, attn_head_dim=4
    )
    config.model.skip_dit_load_from_pretrain = True
    config.model.load_text_encoder = False
    vc = OmegaConf.to_container(config.model.video_dit_config, resolve=True)
    video = WanVideoDiT(**vc)
    parts = SimpleNamespace(
        dit=video,
        vae=nn.Identity(),
        text_encoder=None,
        tokenizer=None,
        dit_path="test-video",
        vae_path="test-vae",
        text_encoder_path=None,
        tokenizer_path=None,
    )
    ablation = config.model.get("ablation")
    if ablation and ablation.type == "frozen_video_action_reader":
        weights = tmp_path / "release.pt"
        torch.save(
            {
                "mot": {
                    "mixtures.video." + k: v for k, v in video.state_dict().items()
                },
                "proprio_encoder": nn.Linear(
                    int(config.model.proprio_dim), 8
                ).state_dict(),
            },
            weights,
        )
        config.model.ablation.release_checkpoint_path = str(weights)
    module = (
        "bridgewam.models.wan22.ablation_bridgewam.factory"
        if ablation
        else "bridgewam.models.wan22.bridgewam"
    )
    with patch(module + ".load_wan22_ti2v_5b_components", return_value=parts):
        model = instantiate(config.model, device="cpu", model_dtype=torch.float32)
    assert model.bridge.latent_bridge_queries.num_lbqs == 32
    assert model.bridge.video_num_layers == 30
    assert model.bridge.action_num_layers == (
        30 if ablation and ablation.type == "joint" else 2
    )
    assert not model.bridge.requires_direct_video_kv_cache
    if ablation:
        model.bridge.configure_lbq_trainable_parameters()


@pytest.mark.parametrize(
    "config_name",
    ["train", "sim_libero", "sim_libero_plus", "sim_libero_pro", "sim_robotwin"],
)
def test_defaults_never_select_the_removed_baseline(config_name):
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        c = compose(config_name=config_name)
    assert c.model.latent_bridge_queries.enabled
    assert c.model.latent_bridge_queries.num_lbqs == 32
    assert c.model.action_dit_config.num_layers == 2
    assert c.model._target_ == "bridgewam.models.wan22.factory.create_bridgewam"


@pytest.mark.parametrize("kind", ["main", "spectral", "frozen", "lbq_kv"])
def test_action_inference_reuses_one_lbq_prefill(kind):
    model = build_model(kind, True)
    first = torch.randn(1, 2, 1, 2, 2)
    with (
        patch.object(model, "_encode_input_image_latents_tensor", return_value=first),
        patch.object(
            model.bridge, "prefill_bridge", wraps=model.bridge.prefill_bridge
        ) as prefill,
        patch.object(
            model.bridge,
            "forward_bridge_action",
            wraps=model.bridge.forward_bridge_action,
        ) as action,
    ):
        out = model.infer_action(
            prompt=None,
            input_image=torch.zeros(1, 3, 32, 32),
            action_horizon=4,
            context=torch.zeros(1, 3, 8),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
            num_inference_steps=3,
            seed=4,
        )
    assert out["action"].shape == (4, 2)
    assert prefill.call_count == 1 and action.call_count == 3


@pytest.mark.parametrize(
    "kwargs",
    [{"text_cfg_scale": 2.0}, {"negative_prompt": "unwanted"}, {"tiled": True}],
)
@pytest.mark.parametrize("kind", ["main", "idm", "joint"])
def test_unimplemented_options_fail_explicitly(kind, kwargs):
    model = build_model(kind, True)
    with pytest.raises(ValueError, match="CFG|tiled=false"):
        model.infer_joint(
            prompt="test",
            input_image=torch.zeros(1, 3, 32, 32),
            num_video_frames=9,
            action_horizon=4,
            **kwargs,
        )


def test_historical_checkpoint_names_are_data_only(tmp_path):
    model = build_model("main", True)
    native = tmp_path / "native.pt"
    legacy = tmp_path / "old_fastwam_named_bridgewam.pt"
    model.save_checkpoint(native, step=7)
    payload = torch.load(native, weights_only=True)
    state = payload.pop("mot")
    payload["fastwam"] = {"module.fastwam.mot." + k: v for k, v in state.items()}
    torch.save(payload, legacy)
    restored = build_model("main", True)
    assert restored.load_checkpoint(legacy)["step"] == 7
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)


def test_old_hydra_model_name_is_rejected():
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        with pytest.raises(Exception, match="fastwam"):
            compose(config_name="train", overrides=["model=fastwam"])


def test_preprocessing_builds_full_depth_artifact_for_two_layer_model(
    tmp_path, monkeypatch
):
    path = ROOT / "scripts/preprocess_action_dit_backbone.py"
    spec = importlib.util.spec_from_file_location("preprocess_for_test", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    video_config = dict(
        hidden_dim=8,
        in_dim=2,
        ffn_dim=16,
        out_dim=2,
        text_dim=8,
        freq_dim=8,
        eps=1e-6,
        patch_size=[1, 1, 1],
        num_heads=2,
        attn_head_dim=4,
        num_layers=3,
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
    )
    action_config = dict(
        action_dim=2,
        hidden_dim=8,
        ffn_dim=16,
        text_dim=8,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=4,
        num_layers=2,
        architecture="alternating_cross_self",
        add_pos_embed=True,
        max_action_horizon=16,
    )
    cfg = tmp_path / "model.yaml"
    out = tmp_path / "action_backbone.pt"
    OmegaConf.save(
        OmegaConf.create(
            {"video_dit_config": video_config, "action_dit_config": action_config}
        ),
        cfg,
    )
    video = WanVideoDiT(**video_config)
    monkeypatch.setattr(
        script, "load_wan22_ti2v_5b_components", lambda **_: SimpleNamespace(dit=video)
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "preprocess",
            "--model-config",
            str(cfg),
            "--output",
            str(out),
            "--device",
            "cpu",
        ],
    )
    script.main()
    payload = torch.load(out, weights_only=True)
    assert payload["meta"]["num_layers"] == 3
    from bridgewam.models.wan22.action_dit import ActionDiT

    head = ActionDiT.from_pretrained(
        action_dit_config=action_config,
        action_dit_pretrained_path=str(out),
        device="cpu",
        torch_dtype=torch.float32,
    )
    assert head.pretrained_source_layers == [0, 2]
