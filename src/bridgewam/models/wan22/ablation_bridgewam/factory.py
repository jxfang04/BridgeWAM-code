from __future__ import annotations

from typing import Any, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from ..action_dit import ActionDiT
from ..helpers.loader import load_wan22_ti2v_5b_components
from .models import (
    BridgeWAMIDMAblation,
    BridgeWAMJointAblation,
    FrozenVideoBridgeWAM,
    LBQKVMoTBridgeWAM,
)
from .backbones import (
    FrozenVideoBridge,
    FullVideoBridge,
    JointAttentionBridge,
    AblationBridge,
)


ABLATION_TYPES = {
    "frozen_video_action_reader",
    "lbq_kv_mot",
    "idm",
    "joint",
}


def _as_dict(value, *, name: str, required: bool = True) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None:
        if required:
            raise ValueError(f"`{name}` is required.")
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"`{name}` must resolve to a dict, got {type(value)}.")
    return dict(value)


def _validate_topology(
    *,
    ablation_type: str,
    video_config: dict[str, Any],
    action_config: dict[str, Any],
    lbq_config: dict[str, Any],
) -> None:
    if not bool(lbq_config.get("enabled", False)):
        raise ValueError(
            "Every BridgeWAM ablation requires Video Latent Bridge Queries."
        )
    if int(lbq_config.get("num_lbqs", 0)) != 32:
        raise ValueError(
            "These ablations are defined for exactly 32 Latent Bridge Queries tokens."
        )
    if bool(lbq_config.get("preserve_direct_video_kv", True)):
        raise ValueError("Ablations must not use the legacy raw Video K/V cache.")

    injection_mode = str(lbq_config.get("injection_mode"))
    coupling = str(lbq_config.get("generation_coupling"))
    architecture = str(action_config.get("architecture", "full"))
    action_layers = int(action_config.get("num_layers", 0))
    video_layers = int(video_config.get("num_layers", 0))
    if video_layers != 30:
        raise ValueError(
            f"BridgeWAM ablations require the original 30-layer Video DiT, got {video_layers}."
        )
    if int(lbq_config.get("start_layer", -1)) != 0:
        raise ValueError(
            "BridgeWAM ablations require `latent_bridge_queries.start_layer=0`."
        )
    readout_layer = int(lbq_config.get("readout_layer", -1))
    if readout_layer < 0:
        readout_layer += video_layers
    if readout_layer != video_layers - 1:
        raise ValueError(
            "BridgeWAM ablations require Latent Bridge Queries readout from the final Video layer."
        )
    if str(lbq_config.get("lbq_attention")) != "bidirectional":
        raise ValueError("BridgeWAM ablations require bidirectional LBQ attention.")
    if str(lbq_config.get("lbq_rope_mode")) != "identity":
        raise ValueError("BridgeWAM ablations require identity LBQ RoPE.")

    if ablation_type == "frozen_video_action_reader":
        expected = ("lbq_only", "none", "alternating_cross_self", 2)
    elif ablation_type == "lbq_kv_mot":
        expected = (
            "text_state",
            "future_video_reads_lbq",
            "alternating_cross_self",
            2,
        )
    elif ablation_type == "idm":
        expected = (
            "lbq_only",
            "future_video_reads_lbq",
            "alternating_cross_self",
            2,
        )
    else:
        expected = ("text_state", "none", "full", 30)

    got = (injection_mode, coupling, architecture, action_layers)
    if got != expected:
        raise ValueError(
            f"Invalid topology for ablation {ablation_type}: expected {expected}, got {got}."
        )
    if ablation_type == "frozen_video_action_reader":
        if not bool(lbq_config.get("freeze_video_expert", False)):
            raise ValueError("Frozen-Video ablation must freeze the Video expert.")
    elif bool(lbq_config.get("freeze_video_expert", True)):
        raise ValueError(f"Ablation {ablation_type} requires full expert finetuning.")


def _checkpoint_identity(
    *,
    ablation_type: str,
    ablation: dict[str, Any],
) -> dict[str, Any]:
    if ablation_type == "frozen_video_action_reader":
        return {
            "type": ablation_type,
            "video_source": "fastwam_release",
            "gradient_policy": "action_loss_to_action_and_lbq",
            "keep_full_video_loss": True,
        }
    if ablation_type == "lbq_kv_mot":
        return {
            "type": ablation_type,
            "gradient_policy": "total_loss_no_detach",
            "action_route": "text_state_cross_action_lbq_mixed_self",
        }
    if ablation_type == "idm":
        return {
            "type": ablation_type,
            "video_cond_noise_prob": float(ablation.get("video_cond_noise_prob", 0.5)),
            "action_route": "full_video_lbq_only",
        }
    return {
        "type": ablation_type,
        "joint_video_mask": "first_frame_causal",
        "joint_cross_stream_mask": "bidirectional",
        "action_cross_context": "text_state",
        "action_layers": 30,
    }


def create_bridgewam_ablation(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    ablation,
    latent_bridge_queries,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: Optional[int] = None,
    action_dit_config=None,
    action_dit_pretrained_path: Optional[str] = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    checkpoint_attention: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    video_config = _as_dict(video_dit_config, name="video_dit_config")
    action_config = _as_dict(
        action_dit_config, name="action_dit_config", required=False
    )
    lbq_config = _as_dict(latent_bridge_queries, name="latent_bridge_queries")
    ablation_config = _as_dict(ablation, name="ablation")
    video_scheduler = _as_dict(video_scheduler, name="video_scheduler", required=False)
    action_scheduler = _as_dict(action_scheduler, name="action_scheduler")
    loss = _as_dict(loss, name="loss", required=False)
    if float(loss.get("lambda_lbq_spectral", 0.0)) != 0 or loss.get(
        "lbq_spectral_diagnostics", False
    ):
        raise ValueError(
            "LBQ spectral loss/diagnostics currently require the main create_bridgewam factory."
        )

    ablation_type = str(ablation_config.get("type", ""))
    if ablation_type not in ABLATION_TYPES:
        raise ValueError(
            f"`ablation.type` must be one of {sorted(ABLATION_TYPES)}, got {ablation_type!r}."
        )
    _validate_topology(
        ablation_type=ablation_type,
        video_config=video_config,
        action_config=action_config,
        lbq_config=lbq_config,
    )

    if "text_dim" not in video_config:
        raise ValueError("`video_dit_config.text_dim` is required.")
    injection_mode = str(lbq_config["injection_mode"])
    action_config["conditioning_mode"] = injection_mode
    if ablation_type == "joint":
        # Joint LBQ tokens participate directly in mixed attention with Video
        # projections. Action therefore needs Text+State projection, not the
        # sequential BridgeWAM LBQ projector.
        action_config.pop("lbq_dim", None)
    else:
        action_config["lbq_dim"] = int(video_config["hidden_dim"])

    components = load_wan22_ti2v_5b_components(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        redirect_common_files=bool(redirect_common_files),
        dit_config=video_config,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        load_text_encoder=bool(load_text_encoder),
    )
    video_expert = components.dit
    action_expert = ActionDiT.from_pretrained(
        action_dit_config=action_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        device=device,
        torch_dtype=model_dtype,
    )
    if action_expert.num_heads != video_expert.num_heads or (
        action_expert.attn_head_dim != video_expert.attn_head_dim
    ):
        raise ValueError("Video and Action attention head topology must match.")

    common_mot_kwargs = {
        "mixtures": {"video": video_expert, "action": action_expert},
        "checkpoint_attention": bool(checkpoint_attention),
        "latent_bridge_queries": lbq_config,
        "ablation_type": ablation_type,
    }
    if ablation_type == "frozen_video_action_reader":
        mot = FrozenVideoBridge(
            **common_mot_kwargs,
        )
        model_class = FrozenVideoBridgeWAM
    elif ablation_type == "lbq_kv_mot":
        mot = AblationBridge(
            **common_mot_kwargs,
        )
        model_class = LBQKVMoTBridgeWAM
    elif ablation_type == "idm":
        mot = FullVideoBridge(
            **common_mot_kwargs,
        )
        model_class = BridgeWAMIDMAblation
    else:
        mot = JointAttentionBridge(**common_mot_kwargs)
        model_class = BridgeWAMJointAblation

    identity = _checkpoint_identity(
        ablation_type=ablation_type,
        ablation=ablation_config,
    )
    mot.ablation_config = dict(identity)
    model = model_class(
        video_expert=video_expert,
        action_expert=action_expert,
        bridge=mot,
        vae=components.vae,
        text_encoder=components.text_encoder,
        tokenizer=components.tokenizer,
        text_dim=int(video_config["text_dim"]),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        device=device,
        torch_dtype=model_dtype,
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(
            action_scheduler.get("num_train_timesteps", 1000)
        ),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        latent_bridge_queries=lbq_config,
        ablation_config=identity,
    )
    model.model_paths = {
        "video_dit": components.dit_path,
        "vae": components.vae_path,
        "text_encoder": components.text_encoder_path,
        "tokenizer": components.tokenizer_path,
        "action_dit_backbone": (
            "SKIPPED_PRETRAIN"
            if skip_dit_load_from_pretrain
            else action_dit_pretrained_path
        ),
    }

    if ablation_type == "frozen_video_action_reader":
        release_path = ablation_config.get("release_checkpoint_path")
        if not release_path:
            raise ValueError(
                "Frozen-Video ablation requires `ablation.release_checkpoint_path`."
            )
        model.load_release_video_and_proprio(str(release_path))
        model.model_paths["frozen_bridgewam_release"] = str(release_path)
    return model
