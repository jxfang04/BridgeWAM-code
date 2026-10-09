"""Hydra factory for the BridgeWAM LBQ model."""

import torch
from omegaconf import DictConfig, OmegaConf


def create_bridgewam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    checkpoint_attention: bool = True,
    latent_bridge_queries=None,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
):
    from .bridgewam import BridgeWAM

    if isinstance(video_dit_config, DictConfig):
        video_dit_config = OmegaConf.to_container(video_dit_config, resolve=True)
    if not isinstance(video_dit_config, dict):
        raise ValueError(
            f"`video_dit_config` must resolve to a dict, got {type(video_dit_config)}"
        )

    if isinstance(action_dit_config, DictConfig):
        action_dit_config = OmegaConf.to_container(action_dit_config, resolve=True)
    if action_dit_config is None:
        action_dit_config = {}
    if not isinstance(action_dit_config, dict):
        raise ValueError(
            f"`action_dit_config` must resolve to a dict, got {type(action_dit_config)}"
        )

    if isinstance(video_scheduler, DictConfig):
        video_scheduler = OmegaConf.to_container(video_scheduler, resolve=True)
    if video_scheduler is None:
        video_scheduler = {}
    if not isinstance(video_scheduler, dict):
        raise ValueError(
            f"`video_scheduler` must be dict-like, got {type(video_scheduler)}"
        )

    if isinstance(action_scheduler, DictConfig):
        action_scheduler = OmegaConf.to_container(action_scheduler, resolve=True)
    if action_scheduler is None:
        raise ValueError("`action_scheduler` is required for BridgeWAM.")
    if not isinstance(action_scheduler, dict):
        raise ValueError(
            f"`action_scheduler` must be dict-like, got {type(action_scheduler)}"
        )
    required_action_scheduler_keys = {
        "train_shift",
        "infer_shift",
        "num_train_timesteps",
    }
    missing_keys = required_action_scheduler_keys - set(action_scheduler.keys())
    if missing_keys:
        raise ValueError(
            f"`action_scheduler` missing required keys: {sorted(missing_keys)}. "
            "Expected keys: train_shift, infer_shift, num_train_timesteps."
        )

    if isinstance(loss, DictConfig):
        loss = OmegaConf.to_container(loss, resolve=True)
    if loss is None:
        loss = {}
    if not isinstance(loss, dict):
        raise ValueError(f"`loss` must be dict-like, got {type(loss)}")

    if isinstance(latent_bridge_queries, DictConfig):
        latent_bridge_queries = OmegaConf.to_container(
            latent_bridge_queries, resolve=True
        )
    if latent_bridge_queries is not None and not isinstance(
        latent_bridge_queries, dict
    ):
        raise ValueError(
            f"`latent_bridge_queries` must be dict-like or null, got {type(latent_bridge_queries)}"
        )

    return BridgeWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        checkpoint_attention=bool(checkpoint_attention),
        latent_bridge_queries=latent_bridge_queries,
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        loss_lambda_lbq_spectral=float(loss.get("lambda_lbq_spectral", 0.0)),
        lbq_spectral_diagnostics=bool(loss.get("lbq_spectral_diagnostics", False)),
    )
