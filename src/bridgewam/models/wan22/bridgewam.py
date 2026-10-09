from typing import Any, Optional, Sequence, Union
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from bridgewam.utils.logging_config import get_logger
from bridgewam.models.wan22.lbq.spectral_regularization import lbq_spectral_diversity

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .bridge_of_experts import BridgeOfExperts
from .checkpoint_compat import normalize_checkpoint_payload
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


def _profile_sync(device: torch.device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _profile_stage(
    profile: Optional[dict[str, Any]],
    key: str,
    device: torch.device,
    fn,
    *,
    append_ms: bool = False,
):
    if profile is None:
        return fn()

    flops_key = f"{key}_flops"
    if bool(profile.get("measure_flops", False)) and flops_key not in profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.device(device).type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        _profile_sync(device)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_flops=True,
        ) as prof:
            fn()
            _profile_sync(device)
        profile[flops_key] = int(
            sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
        )

    _profile_sync(device)
    t0 = time.perf_counter()
    out = fn()
    _profile_sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    ms_key = f"{key}_ms"
    if append_ms:
        profile.setdefault(ms_key, []).append(elapsed_ms)
    else:
        profile[ms_key] = elapsed_ms
    return out


class BridgeWAM(torch.nn.Module):
    """Video/action world model connected by Latent Bridge Queries."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        bridge: BridgeOfExperts,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        latent_bridge_queries: Optional[dict[str, Any]] = None,
        loss_lambda_lbq_spectral: float = 0.0,
        lbq_spectral_diagnostics: bool = False,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        if (
            not bridge.latent_bridge_queries_enabled
            or bridge.requires_direct_video_kv_cache
        ):
            raise ValueError(
                "BridgeWAM requires an LBQ backbone without direct Video K/V."
            )
        self.mot = bridge
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.bridge

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError(
                    "`text_dim` is required when `text_encoder` is not loaded."
                )
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(
                torch_dtype
            )
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Scheduler aliases used by the training utilities.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)
        self.loss_lambda_lbq_spectral = float(loss_lambda_lbq_spectral)
        self.lbq_spectral_diagnostics = bool(lbq_spectral_diagnostics)
        if (
            not math.isfinite(self.loss_lambda_lbq_spectral)
            or self.loss_lambda_lbq_spectral < 0
        ):
            raise ValueError("loss.lambda_lbq_spectral must be finite and nonnegative.")
        if (
            self.loss_lambda_lbq_spectral > 0 or self.lbq_spectral_diagnostics
        ) and not (self.bridge.latent_bridge_queries_enabled):
            raise ValueError(
                "LBQ spectral regularization/diagnostics require enabled LBQs."
            )

        self.to(self.device)

    @property
    def bridge(self) -> BridgeOfExperts:
        """Live backbone; `mot` remains the registered checkpoint key only."""
        return self.mot

    @staticmethod
    def _validate_inference_options(negative_prompt, text_cfg_scale, tiled):
        if negative_prompt not in (None, "") or float(text_cfg_scale) != 1.0:
            raise ValueError(
                "BridgeWAM supports text_cfg_scale=1 and an empty negative_prompt; CFG is not implemented."
            )
        if tiled:
            raise ValueError(
                "BridgeWAM's Wan2.2 VAE input encoder requires tiled=false."
            )

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        checkpoint_attention: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        latent_bridge_queries: Optional[dict[str, Any]] = None,
        loss_lambda_lbq_spectral: float = 0.0,
        lbq_spectral_diagnostics: bool = False,
    ):
        if video_dit_config is None:
            raise ValueError(
                "`video_dit_config` is required for BridgeWAM.from_wan22_pretrained()."
            )
        if "text_dim" not in video_dit_config:
            raise ValueError(
                "`video_dit_config['text_dim']` is required for BridgeWAM."
            )
        action_dit_config = dict(action_dit_config or {})
        lbq_enabled = bool((latent_bridge_queries or {}).get("enabled", False))
        if not lbq_enabled or (latent_bridge_queries or {}).get(
            "preserve_direct_video_kv", True
        ):
            raise ValueError(
                "BridgeWAM requires enabled LBQs and preserve_direct_video_kv=false."
            )
        injection_mode = str(
            (latent_bridge_queries or {}).get("injection_mode", "lbq_only")
            if lbq_enabled
            else "text_state"
        )
        configured_conditioning_mode = action_dit_config.get("conditioning_mode")
        if (
            configured_conditioning_mode is not None
            and str(configured_conditioning_mode) != injection_mode
        ):
            raise ValueError(
                "ActionDiT `conditioning_mode` is derived from "
                "`latent_bridge_queries.injection_mode`; got conflicting values "
                f"{configured_conditioning_mode!r} and {injection_mode!r}."
            )
        action_dit_config["conditioning_mode"] = injection_mode
        if lbq_enabled:
            video_hidden_dim = int(video_dit_config["hidden_dim"])
            configured_lbq_dim = action_dit_config.get("lbq_dim")
            if (
                configured_lbq_dim is not None
                and int(configured_lbq_dim) != video_hidden_dim
            ):
                raise ValueError(
                    "ActionDiT `lbq_dim` must match Video DiT hidden dim, got "
                    f"{configured_lbq_dim} and {video_hidden_dim}."
                )
            action_dit_config["lbq_dim"] = video_hidden_dim

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError(
                "ActionDiT `num_heads` must match video expert for BridgeWAM shared attention."
            )
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError(
                "ActionDiT `attn_head_dim` must match video expert for BridgeWAM shared attention."
            )
        action_architecture = str(getattr(action_expert, "architecture", "full"))
        allow_separate_action_depth = (
            action_architecture == "alternating_cross_self"
            and lbq_enabled
            and not bool(
                (latent_bridge_queries or {}).get("preserve_direct_video_kv", True)
            )
        )
        if (
            int(len(action_expert.blocks)) != int(len(video_expert.blocks))
            and not allow_separate_action_depth
        ):
            raise ValueError(
                "ActionDiT `num_layers` may differ from Video DiT only for "
                "Latent Bridge Queries alternating_cross_self with "
                "preserve_direct_video_kv=false."
            )

        bridge = BridgeOfExperts(
            mixtures={"video": video_expert, "action": action_expert},
            checkpoint_attention=checkpoint_attention,
            latent_bridge_queries=latent_bridge_queries,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            bridge=bridge,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            latent_bridge_queries=latent_bridge_queries,
            loss_lambda_lbq_spectral=loss_lambda_lbq_spectral,
            lbq_spectral_diagnostics=lbq_spectral_diagnostics,
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
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.bridge.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(
                f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}"
            )
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype)  # [B, 1, D]
        proprio_mask = torch.ones(
            (context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device
        )
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(
        self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(
        self,
        input_image: torch.Tensor,
        tiled=False,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode(
            [image],
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(
        self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)
    ):
        video_tensor = self.vae.decode(
            latents,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "BridgeWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(
                f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}"
            )
        if video.shape[1] != 3:
            raise ValueError(
                f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}"
            )

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(
                f"Video T must be > 1 for action-conditioned training, got T={num_frames}"
            )

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for BridgeWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(
                f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}"
            )
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if (
                action_is_pad.shape[0] != batch_size
                or action_is_pad.shape[1] != action_horizon
            ):
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if (
                image_is_pad.shape[0] != batch_size
                or image_is_pad.shape[1] != num_frames
            ):
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )

        input_video = video.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )
        context_mask = context_mask.to(
            device=self.device, dtype=torch.bool, non_blocking=True
        )
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError(
                    "`sample['proprio']` is required when `proprio_dim` is enabled."
                )
            if proprio.ndim != 3:
                raise ValueError(
                    f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :]  # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(
            device=self.device, dtype=self.torch_dtype, non_blocking=True
        )

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
        }

    def _prepare_action_dit_inputs(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        text_state_context: torch.Tensor,
        text_state_mask: torch.Tensor,
        lbq_hidden: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]]:
        conditioning = self.action_expert.prepare_conditioning(
            text_state_context=text_state_context,
            text_state_mask=text_state_mask,
            lbq_hidden=lbq_hidden,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=conditioning["cross_context"],
            context_mask=conditioning["cross_mask"],
        )
        return (
            action_pre,
            conditioning["self_lbq_context"],
            conditioning["lbq_context"],
        )

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(
            pred_video.float(), target_video.float(), reduction="none"
        ).mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(
                f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}."
            )
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(
            image_is_pad.shape[0], -1, temporal_factor
        ).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(
            device=video_loss_token.device, dtype=video_loss_token.dtype
        )
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def _run_joint_training_branch(
        self,
        *,
        latents_video,
        noisy_action,
        timestep_video,
        timestep_action,
        context,
        context_mask,
        action,
        fuse_vae_embedding_in_latents,
    ):
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            video_seq_len=int(video_pre["tokens"].shape[1]),
            device=video_pre["tokens"].device,
        )
        video_tokens, lbq_tokens = self.bridge.forward_bridge_video(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        )
        action_pre, self_lbq_context, lbq_context = self._prepare_action_dit_inputs(
            action_tokens=noisy_action,
            timestep=timestep_action,
            text_state_context=context,
            text_state_mask=context_mask,
            lbq_hidden=lbq_tokens,
        )
        action_tokens = self.bridge.forward_bridge_action(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            self_lbq_context=self_lbq_context,
        )
        return (
            self.video_expert.post_dit(video_tokens, video_pre),
            self.action_expert.post_dit(action_tokens, action_pre),
            {"lbq_tokens": lbq_tokens, "lbq_context": lbq_context},
        )

    def _compute_weighted_action_loss(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        timestep_action: torch.Tensor,
    ) -> torch.Tensor:
        action_loss_token = F.mse_loss(
            pred_action.float(),
            target_action.float(),
            reduction="none",
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device,
                dtype=action_loss_token.dtype,
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device,
            dtype=action_loss_per_sample.dtype,
        )
        return (action_loss_per_sample * action_weight).mean()

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        first_frame_latents = inputs["first_frame_latents"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        noisy_video = self.train_video_scheduler.add_noise(
            input_latents,
            noise_video,
            timestep_video,
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents,
            noise_video,
            timestep_video,
        )
        if first_frame_latents is not None:
            noisy_video[:, :, 0:1] = first_frame_latents

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(
            action, noise_action, timestep_action
        )
        target_action = self.train_action_scheduler.training_target(
            action, noise_action, timestep_action
        )

        pred_video, pred_action, auxiliary = self._run_joint_training_branch(
            latents_video=noisy_video,
            noisy_action=noisy_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        include_initial_video_step = first_frame_latents is None
        if first_frame_latents is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        loss_action = self._compute_weighted_action_loss(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=action_is_pad,
            timestep_action=timestep_action,
        )

        loss_total = (
            self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if self.bridge.latent_bridge_queries_enabled:
            lbq_tokens = auxiliary["lbq_tokens"]
            lbq_context = auxiliary["lbq_context"]
            loss_dict["lbq_token_rms"] = float(
                lbq_tokens.detach().float().square().mean().sqrt().item()
            )
            loss_dict["lbq_context_rms"] = float(
                lbq_context.detach().float().square().mean().sqrt().item()
            )
            if self.loss_lambda_lbq_spectral > 0 or self.lbq_spectral_diagnostics:
                # A diagnostics-only run must not add even a zero-weight edge
                # to autograd, nor change which parameters receive gradients.
                spectral_input = (
                    lbq_tokens
                    if self.loss_lambda_lbq_spectral > 0
                    else lbq_tokens.detach()
                )
                loss_spectral, readout_metrics = lbq_spectral_diversity(
                    spectral_input, diagnostics=self.lbq_spectral_diagnostics
                )
                loss_dict["loss_lbq_spectral_raw"] = float(
                    loss_spectral.detach().item()
                )
                loss_dict["loss_lbq_spectral"] = (
                    self.loss_lambda_lbq_spectral * loss_dict["loss_lbq_spectral_raw"]
                )
                if self.loss_lambda_lbq_spectral > 0:
                    loss_total = (
                        loss_total + self.loss_lambda_lbq_spectral * loss_spectral
                    )
                if self.lbq_spectral_diagnostics:
                    sources = {
                        "embedding": self.bridge.latent_bridge_queries.lbq_embeddings,
                        "context": lbq_context,
                    }
                    all_metrics = {"readout": readout_metrics}
                    for name, hidden in sources.items():
                        _, all_metrics[name] = lbq_spectral_diversity(
                            hidden.detach(), diagnostics=True
                        )
                    for name, metrics in all_metrics.items():
                        loss_dict.update(
                            {
                                f"lbq_{name}_{key}": value
                                for key, value in metrics.items()
                            }
                        )
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video,
        latents_action,
        timestep_video,
        timestep_action,
        context,
        context_mask,
        fuse_vae_embedding_in_latents,
        gt_action=None,
    ):
        video, action, _ = self._run_joint_training_branch(
            latents_video=latents_video,
            noisy_action=latents_action,
            timestep_video=timestep_video,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        return video, action

    @torch.no_grad()
    def _prefill_video_for_action(
        self,
        first_frame_latents,
        context,
        context_mask,
        fuse_vae_embedding_in_latents,
    ):
        """Read the observed frame once and cache the selected LBQ readout."""
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        lbq_tokens = self.bridge.prefill_bridge(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_mask,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
        )
        return video_pre, lbq_tokens

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        action_cross_context: torch.Tensor,
        action_cross_mask: torch.Tensor,
        self_lbq_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=action_cross_context,
            context_mask=action_cross_mask,
        )
        action_tokens = self.bridge.forward_bridge_action(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            self_lbq_context=self_lbq_context,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[
            torch.Tensor
        ] = None,  # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self._validate_inference_options(negative_prompt, text_cfg_scale, tiled)
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError(
                    "`test_action_with_infer_action=True` requires non-null `seed`."
                )
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if (
                action.ndim != 3
                or action.shape[0] != 1
                or action.shape[1] != action_horizon
            ):
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(
                    f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        action_generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and not use_context:
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must be both provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = (
            self.infer_video_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_video.dtype,
                shift_override=sigma_shift,
            )
        )
        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(
                dtype=latents_video.dtype, device=self.device
            )
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta_video, latents_video
            )
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        profile: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._validate_inference_options(negative_prompt, text_cfg_scale, tiled)
        self.eval()
        if (
            str(getattr(self.video_expert, "video_attention_mask_mode", ""))
            != "first_frame_causal"
        ):
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if (
            input_image.ndim != 4
            or input_image.shape[0] != 1
            or input_image.shape[1] != 3
        ):
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError(
                    "`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled."
                )
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(
                    f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}"
                )
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
                )
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = (
            None
            if seed is None
            else torch.Generator(device=rand_device).manual_seed(seed)
        )
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image, tiled=tiled
        )
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError(
                "`prompt` and `context/context_mask` are mutually exclusive."
            )
        if not use_prompt and not use_context:
            raise ValueError(
                "Either `prompt` or both `context/context_mask` must be provided."
            )

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError(
                    "`context` and `context_mask` must be both provided together."
                )
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(
                device=self.device, dtype=self.torch_dtype, non_blocking=True
            )
            context_mask = context_mask.to(
                device=self.device, dtype=torch.bool, non_blocking=True
            )
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        video_pre, lbq_tokens = _profile_stage(
            profile,
            "video_prefill_once",
            self.device,
            lambda: self._prefill_video_for_action(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
            ),
        )
        if profile is not None:
            profile["video_seq_len"] = int(video_pre["tokens"].shape[1])
            profile["video_tokens_per_frame"] = int(
                video_pre["meta"]["tokens_per_frame"]
            )
            profile["video_grid_size"] = [
                int(x) for x in video_pre["meta"]["grid_size"]
            ]
            profile["action_seq_len"] = int(latents_action.shape[1])
            profile["action_horizon"] = int(action_horizon)
            profile["num_inference_steps"] = int(num_inference_steps)
            if self.bridge.latent_bridge_queries_enabled:
                profile["latent_bridge_queries"] = (
                    self.bridge.latent_bridge_queries_config()
                )

        action_conditioning = _profile_stage(
            profile,
            "action_conditioning_once",
            self.device,
            lambda: self.action_expert.prepare_conditioning(
                text_state_context=context,
                text_state_mask=context_mask,
                lbq_hidden=lbq_tokens,
            ),
        )
        if profile is not None:
            profile["action_context_source"] = self.action_expert.conditioning_mode
            profile["action_cross_context_tokens"] = int(
                action_conditioning["cross_context"].shape[1]
            )
            profile["action_self_lbq_tokens"] = int(
                0
                if action_conditioning["self_lbq_context"] is None
                else action_conditioning["self_lbq_context"].shape[1]
            )

        infer_timesteps_action, infer_deltas_action = (
            self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_inference_steps,
                device=self.device,
                dtype=latents_action.dtype,
                shift_override=sigma_shift,
            )
        )
        for step_t_action, step_delta_action in zip(
            infer_timesteps_action, infer_deltas_action
        ):
            timestep_action = step_t_action.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )

            pred_action_posi = _profile_stage(
                profile,
                "action_dit_cached_step",
                self.device,
                lambda: self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    action_cross_context=action_conditioning["cross_context"],
                    action_cross_mask=action_conditioning["cross_mask"],
                    self_lbq_context=action_conditioning["self_lbq_context"],
                ),
                append_ms=True,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta_action, latents_action
            )

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.bridge.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "checkpoint_format_version": 2,
            "model_name": "bridgewam",
        }
        lbq_config = self.bridge.latent_bridge_queries_config()
        if lbq_config is not None:
            payload["latent_bridge_queries"] = lbq_config
        action_architecture = self.action_expert.architecture_config()
        if self.action_expert.pretrained_source_layers is not None:
            action_architecture["pretrained_source_layers"] = list(
                self.action_expert.pretrained_source_layers
            )
        payload["action_dit_architecture"] = action_architecture
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = normalize_checkpoint_payload(torch.load(path, map_location="cpu"))
        checkpoint_lbq_config = payload.get("latent_bridge_queries")
        current_lbq_config = self.bridge.latent_bridge_queries_config()
        normalized_checkpoint_lbq_config = None
        normalized_current_lbq_config = None
        if current_lbq_config is not None:
            normalized_current_lbq_config = (
                self.bridge.normalize_latent_bridge_queries_checkpoint_config(
                    current_lbq_config,
                    num_layers=self.bridge.num_layers,
                )
            )
        if checkpoint_lbq_config is not None and current_lbq_config is None:
            raise ValueError(
                "Checkpoint contains Latent Bridge Queries weights, but "
                "`model.latent_bridge_queries.enabled=false`."
            )
        if checkpoint_lbq_config is not None and current_lbq_config is not None:
            if not isinstance(checkpoint_lbq_config, dict):
                raise ValueError(
                    "Checkpoint `latent_bridge_queries` metadata must be a dictionary."
                )
            readout = checkpoint_lbq_config.get("readout_layer", -1)
            if isinstance(readout, (list, tuple)) or (
                isinstance(readout, str) and not readout.lstrip("-").isdigit()
            ):
                raise ValueError(
                    "Multi-readout begin8.7 checkpoints require their original "
                    "multi-readout architecture; this 917 model uses one readout layer."
                )
            normalized_checkpoint_lbq_config = (
                self.bridge.normalize_latent_bridge_queries_checkpoint_config(
                    checkpoint_lbq_config,
                    num_layers=self.bridge.num_layers,
                )
            )

        checkpoint_action_architecture = payload.get("action_dit_architecture")
        current_action_architecture = self.action_expert.architecture_config()
        if checkpoint_action_architecture is None:
            if current_action_architecture["architecture"] != "full":
                raise ValueError(
                    "Checkpoint has no ActionDiT architecture metadata, but the "
                    "current model uses a non-default ActionDiT architecture."
                )
        else:
            if not isinstance(checkpoint_action_architecture, dict):
                raise ValueError(
                    "Checkpoint `action_dit_architecture` metadata must be a dictionary."
                )
            checkpoint_action_architecture = dict(checkpoint_action_architecture)
            if checkpoint_action_architecture.get("lbqs_read_action", False):
                raise ValueError(
                    "Checkpoint uses begin8.7 lbqs_read_action=true; the 917 architecture does not implement that route."
                )
            checkpoint_conditioning_mode = (
                "text_state"
                if normalized_checkpoint_lbq_config is None
                else normalized_checkpoint_lbq_config["injection_mode"]
            )
            checkpoint_action_architecture.setdefault(
                "conditioning_mode",
                checkpoint_conditioning_mode,
            )
            checkpoint_action_architecture.setdefault(
                "uses_lbq_self_attention",
                bool(
                    normalized_checkpoint_lbq_config is not None
                    and checkpoint_conditioning_mode == "text_state"
                ),
            )
            action_architecture_keys = (
                "architecture",
                "num_layers",
                "block_types",
                "add_pos_embed",
                "max_action_horizon",
            )
            if not (
                normalized_checkpoint_lbq_config is None
                and normalized_current_lbq_config is not None
            ):
                action_architecture_keys += (
                    "conditioning_mode",
                    "uses_lbq_self_attention",
                )
            action_architecture_mismatches = {
                key: (
                    checkpoint_action_architecture.get(key),
                    current_action_architecture.get(key),
                )
                for key in action_architecture_keys
                if checkpoint_action_architecture.get(key)
                != current_action_architecture.get(key)
            }
            if action_architecture_mismatches:
                raise ValueError(
                    "ActionDiT checkpoint/config mismatch: "
                    f"{action_architecture_mismatches}."
                )
        if normalized_checkpoint_lbq_config is not None:
            architecture_keys = (
                "num_lbqs",
                "start_layer",
                "readout_layer",
                "lbq_attention",
                "lbq_rope_mode",
                "eps",
                "injection_mode",
                "preserve_direct_video_kv",
                "generation_coupling",
                "lbq_embedding_input_dim",
                "lbq_embedding_output_dim",
            )
            mismatches = {
                key: (
                    normalized_checkpoint_lbq_config.get(key),
                    normalized_current_lbq_config.get(key),
                )
                for key in architecture_keys
                if normalized_checkpoint_lbq_config.get(key)
                != normalized_current_lbq_config.get(key)
            }
            if mismatches:
                raise ValueError(f"BridgeWAM checkpoint/config mismatch: {mismatches}.")
        if "mot" in payload:
            mot_state = dict(payload["mot"])
            migrated_legacy_state = False
            if not self.action_expert.uses_text_state_context:
                # Cleanup-era lbq_only checkpoints omitted this unused module.
                # Backfill its initialized state so those checkpoints remain
                # loadable; it is still excluded from the forward route.
                text_prefix = "mixtures.action.text_embedding."
                missing_text_keys = []
                for (
                    key,
                    value,
                ) in self.action_expert.text_embedding.state_dict().items():
                    full_key = f"{text_prefix}{key}"
                    if full_key not in mot_state:
                        mot_state[full_key] = value
                        missing_text_keys.append(full_key)
                migrated_legacy_state = bool(missing_text_keys)
            if normalized_checkpoint_lbq_config is not None:
                obsolete_prefixes = (
                    "latent_bridge_queries.input_norm.",
                    "latent_bridge_queries.input_proj.",
                    "latent_bridge_queries.connector.",
                    "latent_bridge_queries.connector_norm.",
                    "latent_bridge_queries.output_proj.",
                )
                obsolete_keys = [
                    key for key in mot_state if key.startswith(obsolete_prefixes)
                ]
                for key in obsolete_keys:
                    mot_state.pop(key)
                migrated_legacy_state = migrated_legacy_state or bool(obsolete_keys)

            # Check coverage before copying tensors. Renamed/unknown keys must
            # never result in a successful partial baseline restore.
            expected = self.bridge.state_dict()
            allowed_missing = (
                {"action_video_kv_layer_mask"}
                if normalized_checkpoint_lbq_config is None
                else set()
            )
            if (
                normalized_current_lbq_config is not None
                and normalized_checkpoint_lbq_config is None
            ):
                allowed_missing.update(
                    k
                    for k in expected
                    if k.startswith(
                        ("latent_bridge_queries.", "mixtures.action.lbq_embedding.")
                    )
                )
            missing = set(expected) - set(mot_state) - allowed_missing
            unexpected = set(mot_state) - set(expected)
            mismatched = [
                k
                for k in set(expected) & set(mot_state)
                if not torch.is_tensor(mot_state[k])
                or expected[k].shape != mot_state[k].shape
            ]
            if missing or unexpected or mismatched:
                raise ValueError(
                    f"Checkpoint state mismatch: Missing keys: {sorted(missing)}; unexpected keys: {sorted(unexpected)}; shape mismatch: {sorted(mismatched)}."
                )
            incompatible = self.bridge.load_state_dict(mot_state, strict=False)
            if normalized_checkpoint_lbq_config is not None:
                if incompatible.missing_keys or incompatible.unexpected_keys:
                    raise ValueError(
                        "BridgeWAM checkpoint is incomplete. "
                        f"Missing keys: {incompatible.missing_keys}; "
                        f"unexpected keys: {incompatible.unexpected_keys}."
                    )
            if (
                normalized_current_lbq_config is not None
                and normalized_checkpoint_lbq_config is None
                and self.bridge.latent_bridge_queries_strict_training_scope
            ):
                allowed_missing = {"action_video_kv_layer_mask"}
                critical_missing = [
                    key
                    for key in incompatible.missing_keys
                    if not key.startswith("latent_bridge_queries.")
                    and not key.startswith("mixtures.action.lbq_embedding.")
                    and key not in allowed_missing
                ]
                if critical_missing or incompatible.unexpected_keys:
                    raise ValueError(
                        "Baseline checkpoint does not fully initialize frozen "
                        "BridgeWAM parameters. "
                        f"Missing: {critical_missing}; "
                        f"unexpected: {incompatible.unexpected_keys}."
                    )
        elif "dit" in payload:
            if normalized_current_lbq_config is not None:
                raise ValueError(
                    "BridgeWAM training requires a full BridgeWAM checkpoint "
                    "containing `mot`, not a legacy video-only checkpoint."
                )
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(
                    payload["proprio_encoder"], strict=True
                )
            elif (
                normalized_current_lbq_config is not None
                and self.bridge.latent_bridge_queries_strict_training_scope
            ):
                raise ValueError(
                    "BridgeWAM freezes proprio_encoder, but the baseline "
                    "checkpoint does not contain its trained weights."
                )
            else:
                logger.warning(
                    "Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params."
                )
        elif "proprio_encoder" in payload:
            logger.warning(
                "Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring."
            )

        if optimizer is not None and "optimizer" in payload:
            if "mot" in payload and migrated_legacy_state:
                raise ValueError(
                    "Legacy BridgeWAM weights were migrated, but their optimizer "
                    "state is topology-dependent and cannot be resumed. Load the "
                    "checkpoint with `optimizer=None` and start a fresh optimizer."
                )
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
