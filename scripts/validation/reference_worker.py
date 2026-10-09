"""Small-model numerical fixtures for the pinned reference and current source.

Run through scripts/verify_bridgewam_refactor.py, which isolates the two imports.
VAE encoding/decoding is substituted with fixed latent tensors: this checks the
real DiT/LBQ/loss/scheduler math, not simulator or image-encoder integration.
"""

import argparse
import importlib
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from bridgewam.models.wan22.action_dit import ActionDiT
from bridgewam.models.wan22.bridgewam import BridgeWAM
from bridgewam.models.wan22.bridge_of_experts import BridgeOfExperts
from bridgewam.models.wan22.wan_video_dit import WanVideoDiT
from bridgewam.models.wan22.ablation_bridgewam import models as ablations
from bridgewam.models.wan22.ablation_bridgewam.factory import _checkpoint_identity


def build_model(kind, current, checkpoint_attention=False):
    torch.manual_seed(1207)
    depth = 3
    joint = kind == "joint"
    video = WanVideoDiT(
        hidden_dim=8,
        in_dim=2,
        ffn_dim=16,
        out_dim=2,
        text_dim=8,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 1, 1),
        num_heads=2,
        attn_head_dim=4,
        num_layers=depth,
        has_image_input=False,
        seperated_timestep=True,
        fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal",
        use_gradient_checkpointing=checkpoint_attention,
    )
    mode = "text_state" if kind in {"joint", "lbq_kv"} else "lbq_only"
    action = ActionDiT(
        action_dim=2,
        hidden_dim=8,
        ffn_dim=16,
        text_dim=8,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=4,
        num_layers=depth if joint else 2,
        lbq_dim=None if joint else 8,
        architecture="full" if joint else "alternating_cross_self",
        conditioning_mode=mode,
        add_pos_embed=not joint,
        max_action_horizon=16,
        use_gradient_checkpointing=checkpoint_attention,
    )
    cfg = dict(
        enabled=True,
        num_lbqs=32,
        start_layer=0,
        readout_layer=-1,
        generation_coupling="none"
        if kind in {"joint", "frozen"}
        else "future_video_reads_lbq",
        injection_mode=mode,
        lbq_attention="bidirectional",
        lbq_rope_mode="identity",
        preserve_direct_video_kv=False,
        freeze_video_expert=kind == "frozen",
        strict_training_scope=kind == "frozen",
        train_action_cross_attention=True,
        action_cross_attention_lr_scale=1.0,
    )
    identity = None
    model_class = BridgeWAM
    backbone_class = BridgeOfExperts
    kw = {}
    if kind in {"frozen", "lbq_kv", "idm", "joint"}:
        variants = importlib.import_module(
            "bridgewam.models.wan22.ablation_bridgewam."
            + ("backbones" if current else "mot_variants")
        )
        names = {
            "frozen": (
                "FrozenVideoBridge",
                "FrozenVideoAblationMoT",
                "FrozenVideoBridgeWAM",
                "frozen_video_action_reader",
            ),
            "lbq_kv": (
                "AblationBridge",
                "TaggedAblationMoT",
                "LBQKVMoTBridgeWAM",
                "lbq_kv_mot",
            ),
            "idm": (
                "FullVideoBridge",
                "FullVideoLBQAblationMoT",
                "BridgeWAMIDMAblation",
                "idm",
            ),
            "joint": (
                "JointAttentionBridge",
                "JointLBQAblationMoT",
                "BridgeWAMJointAblation",
                "joint",
            ),
        }
        canonical, historical, model_name, tag = names[kind]
        backbone_class = getattr(variants, canonical if current else historical)
        model_class = getattr(ablations, model_name)
        identity = _checkpoint_identity(
            ablation_type=tag, ablation={"video_cond_noise_prob": 0.5}
        )
        kw["ablation_type"] = tag
    kw["checkpoint_attention" if current else "mot_checkpoint_mixed_attn"] = (
        checkpoint_attention
    )
    backbone = backbone_class(
        mixtures={"video": video, "action": action}, latent_bridge_queries=cfg, **kw
    )
    if identity is not None:
        backbone.ablation_config = dict(identity)
    model_kw = {"bridge" if current else "mot": backbone}
    if identity is not None:
        model_kw["ablation_config"] = identity
    vae = nn.Identity()
    vae.temporal_downsample_factor = 4
    vae.upsampling_factor = 16
    vae.model = nn.Identity()
    vae.model.z_dim = 2
    model = model_class(
        video_expert=video,
        action_expert=action,
        vae=vae,
        text_dim=8,
        proprio_dim=2,
        device="cpu",
        torch_dtype=torch.float32,
        loss_lambda_lbq_spectral=1e-4 if kind == "spectral" else 0,
        lbq_spectral_diagnostics=kind == "spectral",
        **model_kw,
    )
    backbone.configure_lbq_trainable_parameters()
    if kind == "frozen":
        model.proprio_encoder.requires_grad_(False)
    return model


def exercise(model):
    torch.manual_seed(219)
    latents = torch.randn(1, 2, 3, 2, 2)
    data = dict(
        input_latents=latents,
        first_frame_latents=latents[:, :, :1],
        context=model.proprio_encoder(torch.randn(1, 3, 2)),
        context_mask=torch.ones(1, 3, dtype=torch.bool),
        action=torch.randn(1, 4, 2),
        action_is_pad=None,
        action_dim_is_pad=None,
        image_is_pad=None,
        fuse_vae_embedding_in_latents=True,
    )
    model.train()
    with patch.object(model, "build_inputs", return_value=data):
        loss, metrics = model.training_loss({})
    loss.backward()
    return (
        loss,
        metrics,
        {
            n: None if p.grad is None else p.grad.clone()
            for n, p in model.named_parameters()
        },
    )


def inference(model):
    torch.manual_seed(514)
    first = torch.randn(1, 2, 1, 2, 2)
    context = torch.randn(1, 3, 8)
    mask = torch.ones(1, 3, dtype=torch.bool)
    # A 32x32 image maps to the fixed 2x2 latent grid in this fixture.
    image = torch.zeros(1, 3, 32, 32)
    kwargs = dict(
        prompt=None,
        input_image=image,
        action_horizon=4,
        context=context,
        context_mask=mask,
        num_inference_steps=3,
        seed=991,
        text_cfg_scale=1.0,
    )
    profile = {}
    with (
        patch.object(model, "_encode_input_image_latents_tensor", return_value=first),
        patch.object(
            model, "_decode_latents", side_effect=lambda x, **_: x.detach().clone()
        ),
    ):
        if type(model).__name__ in {"BridgeWAMIDMAblation", "BridgeWAMJointAblation"}:
            action = model.infer_action(**kwargs, num_video_frames=9, profile=profile)[
                "action"
            ]
        else:
            action = model.infer_action(**kwargs, profile=profile)["action"]
        joint = model.infer_joint(
            **kwargs, num_video_frames=9, test_action_with_infer_action=False
        )
    return {"action": action, "joint_action": joint["action"], "video": joint["video"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["reference", "current"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    for kind in ["main", "spectral", "frozen", "lbq_kv", "idm", "joint"]:
        for gc in [False, True]:
            name = f"{kind}-gc{int(gc)}"
            current = args.mode == "current"
            model = build_model(kind, current, gc)
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=1e-4
            )
            seed_path = args.directory / (name + "-seed.pt")
            if current:
                model.load_checkpoint(seed_path, optimizer=optimizer)
            else:
                model.save_checkpoint(seed_path, optimizer=optimizer, step=0)
            restored = {k: v.clone() for k, v in model.state_dict().items()}
            loss, metrics, gradients = exercise(model)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            first_update = {k: v.clone() for k, v in model.state_dict().items()}
            outputs = inference(model)
            resume_path = args.directory / (name + "-resume.pt")
            if not current:
                model.save_checkpoint(resume_path, optimizer=optimizer, step=1)
            # Both implementations continue from the reference optimizer + weights.
            model.load_checkpoint(resume_path, optimizer=optimizer)
            resumed_loss, _, _ = exercise(model)
            optimizer.step()
            actual = dict(
                initial=initial,
                restored=restored,
                loss=loss.detach(),
                metrics=metrics,
                gradients=gradients,
                parameters=list(dict(model.named_parameters())),
                first_update=first_update,
                outputs=outputs,
                resumed_loss=resumed_loss.detach(),
                resumed_update={k: v.clone() for k, v in model.state_dict().items()},
            )
            target = args.directory / (name + "-expected.pt")
            if not current:
                torch.save(actual, target)
            else:
                expected = torch.load(target, weights_only=True)
                assert actual.pop("parameters") == expected.pop("parameters"), (
                    "Parameter registration order changed"
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(
                f"{args.mode} PASS {name}: weights, loss, gradients, parameter order, action/joint inference, AdamW resume",
                flush=True,
            )


if __name__ == "__main__":
    main()
