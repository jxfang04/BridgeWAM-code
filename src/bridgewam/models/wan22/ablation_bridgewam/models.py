from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from bridgewam.utils.logging_config import get_logger

from ..bridgewam import BridgeWAM, _profile_stage


logger = get_logger(__name__)


class AblationCheckpointMixin:
    """Attach a strict experiment identity to ordinary BridgeWAM checkpoints."""

    def __init__(self, *args, ablation_config: dict[str, Any], **kwargs):
        self.ablation_config = dict(ablation_config)
        super().__init__(*args, **kwargs)

    def save_checkpoint(self, path, optimizer=None, step=None):
        # Match BridgeWAM.save_checkpoint exactly while adding the ablation tag in
        # the same write. Reopening a 5B checkpoint just to append metadata can
        # temporarily double CPU memory and filesystem I/O.
        payload = {
            "mot": self.bridge.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
            "checkpoint_format_version": 2,
            "model_name": "bridgewam",
            "bridgewam_ablation": dict(self.ablation_config),
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
        # BridgeWAM performs the architecture checks and returns the already
        # loaded payload. Validate the experiment identity without loading the
        # multi-billion-parameter checkpoint a second time.
        payload = super().load_checkpoint(path, optimizer=optimizer)
        checkpoint_config = payload.get("bridgewam_ablation")
        if checkpoint_config != self.ablation_config:
            raise ValueError(
                "BridgeWAM ablation checkpoint/config mismatch: "
                f"checkpoint={checkpoint_config}, current={self.ablation_config}."
            )
        return payload


class FrozenVideoBridgeWAM(AblationCheckpointMixin, BridgeWAM):
    """LBQs read a frozen LIBERO Video expert; Action remains fully trainable."""

    def load_release_video_and_proprio(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"BridgeWAM release checkpoint does not exist: {checkpoint_path}"
            )
        payload = torch.load(str(path), map_location="cpu")
        from ..checkpoint_compat import normalize_checkpoint_payload

        payload = normalize_checkpoint_payload(payload)
        mot_state = payload.get("mot")
        if not isinstance(mot_state, dict):
            raise ValueError(
                "BridgeWAM release checkpoint must contain a `mot` state dictionary."
            )

        prefix = "mixtures.video."
        video_state = {
            key[len(prefix) :]: value
            for key, value in mot_state.items()
            if key.startswith(prefix)
        }
        expected_video_keys = set(self.video_expert.state_dict())
        if set(video_state) != expected_video_keys:
            missing = sorted(expected_video_keys - set(video_state))
            unexpected = sorted(set(video_state) - expected_video_keys)
            raise ValueError(
                "Release Video DiT state is incomplete: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}."
            )
        self.video_expert.load_state_dict(video_state, strict=True)

        if self.proprio_encoder is not None:
            proprio_state = payload.get("proprio_encoder")
            if not isinstance(proprio_state, dict):
                raise ValueError(
                    "Frozen-Video ablation requires `proprio_encoder` in the "
                    "BridgeWAM release checkpoint."
                )
            self.proprio_encoder.load_state_dict(proprio_state, strict=True)
        logger.info(
            "Initialized frozen Video DiT and proprio encoder from %s; "
            "ActionDiT and Latent Bridge Queries initialization were preserved.",
            checkpoint_path,
        )


class LBQKVMoTBridgeWAM(AblationCheckpointMixin, BridgeWAM):
    """Text+State cross-attention plus Action/LBQ mixed self-attention."""


def _prepare_inference_context(
    model: BridgeWAM,
    *,
    prompt: Optional[str],
    context: Optional[torch.Tensor],
    context_mask: Optional[torch.Tensor],
    proprio: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    use_prompt = prompt is not None
    use_context = context is not None or context_mask is not None
    if use_prompt and use_context:
        raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
    if not use_prompt and not use_context:
        raise ValueError(
            "Either `prompt` or both `context/context_mask` must be provided."
        )

    if use_prompt:
        context, context_mask = model.encode_prompt(prompt)
    else:
        if context is None or context_mask is None:
            raise ValueError("`context` and `context_mask` must be provided together.")
        if context.ndim == 2:
            context = context.unsqueeze(0)
        if context_mask.ndim == 1:
            context_mask = context_mask.unsqueeze(0)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                "`context/context_mask` must be [B,L,D]/[B,L], got "
                f"{tuple(context.shape)} and {tuple(context_mask.shape)}."
            )
        context = context.to(
            device=model.device,
            dtype=model.torch_dtype,
            non_blocking=True,
        )
        context_mask = context_mask.to(
            device=model.device,
            dtype=torch.bool,
            non_blocking=True,
        )

    if proprio is not None:
        if model.proprio_dim is None:
            raise ValueError("`proprio` was provided but proprio encoding is disabled.")
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape != (1, model.proprio_dim):
            raise ValueError(
                f"`proprio` must be [D] or [1,D] with D={model.proprio_dim}, "
                f"got {tuple(proprio.shape)}."
            )
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio.to(device=model.device, dtype=model.torch_dtype),
        )
    return context, context_mask


class BridgeWAMIDMAblation(AblationCheckpointMixin, BridgeWAM):
    """Two-stage IDM whose only Video-to-Action channel is Latent Bridge Queries."""

    def __init__(self, *args, ablation_config: dict[str, Any], **kwargs):
        super().__init__(*args, ablation_config=ablation_config, **kwargs)
        self.video_cond_noise_prob = float(
            ablation_config.get("video_cond_noise_prob", 0.5)
        )
        if not 0.0 <= self.video_cond_noise_prob <= 1.0:
            raise ValueError("`video_cond_noise_prob` must be in [0,1].")

    def _run_video_lbq_branch(
        self,
        *,
        latents_video: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        lbq_reads_full_video: bool,
        return_video_prediction: bool = True,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor, dict[str, Any]]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        kwargs = {
            "video_tokens": video_pre["tokens"],
            "video_freqs": video_pre["freqs"],
            "video_t_mod": video_pre["t_mod"],
            "video_context_payload": {
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            "video_attention_mask": video_attention_mask,
            "video_tokens_per_frame": int(video_pre["meta"]["tokens_per_frame"]),
        }
        if lbq_reads_full_video:
            video_tokens, lbq_tokens = self.bridge.forward_video_with_full_video_lbqs(
                **kwargs
            )
        else:
            video_tokens, lbq_tokens = self.bridge.forward_bridge_video(**kwargs)
        prediction = (
            self.video_expert.post_dit(video_tokens, video_pre)
            if return_video_prediction
            else None
        )
        return prediction, lbq_tokens, video_pre

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents_noisy = self.train_video_scheduler.add_noise(
            input_latents, noise_video, timestep_video
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, noise_video, timestep_video
        )
        if inputs["first_frame_latents"] is not None:
            latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

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

        cond_noise_mask = (
            torch.rand((batch_size,), device=self.device) < self.video_cond_noise_prob
        )
        timestep_video_cond = torch.zeros_like(
            timestep_video,
            dtype=input_latents.dtype,
            device=self.device,
        )
        latents_cond = input_latents
        if bool(cond_noise_mask.any()):
            sampled_cond_t = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=input_latents.dtype,
            )
            timestep_video_cond = torch.where(
                cond_noise_mask, sampled_cond_t, timestep_video_cond
            )
            cond_noise = torch.randn_like(input_latents)
            noised_cond = self.train_video_scheduler.add_noise(
                input_latents, cond_noise, sampled_cond_t
            )
            latents_cond = torch.where(
                cond_noise_mask.view(batch_size, 1, 1, 1, 1),
                noised_cond,
                input_latents,
            )
        if inputs["first_frame_latents"] is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        pred_video, noisy_lbq, _ = self._run_video_lbq_branch(
            latents_video=latents_noisy,
            timestep_video=timestep_video,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            lbq_reads_full_video=False,
        )
        _, cond_lbq, _ = self._run_video_lbq_branch(
            latents_video=latents_cond,
            timestep_video=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
            lbq_reads_full_video=True,
            return_video_prediction=False,
        )

        action_pre, self_lbq_context, lbq_context = self._prepare_action_dit_inputs(
            action_tokens=noisy_action,
            timestep=timestep_action,
            text_state_context=context,
            text_state_mask=context_mask,
            lbq_hidden=cond_lbq,
        )
        if self_lbq_context is not None:
            raise RuntimeError("BridgeWAM IDM requires `injection_mode=lbq_only`.")
        action_tokens = self.bridge.forward_bridge_action(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            self_lbq_context=None,
        )
        pred_action = self.action_expert.post_dit(action_tokens, action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device,
            dtype=loss_video_per_sample.dtype,
        )
        loss_video = (loss_video_per_sample * video_weight).mean()
        loss_action = self._compute_weighted_action_loss(
            pred_action=pred_action,
            target_action=target_action,
            action_is_pad=inputs["action_is_pad"],
            timestep_action=timestep_action,
        )
        loss_total = (
            self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        )
        return loss_total, {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
            "lbq_token_rms": float(
                noisy_lbq.detach().float().square().mean().sqrt().item()
            ),
            "lbq_context_rms": float(
                lbq_context.detach().float().square().mean().sqrt().item()
            ),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
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
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            test_action_with_infer_action=False,
            profile=profile,
        )

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
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
        test_action_with_infer_action: bool = False,
        profile: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._validate_inference_options(negative_prompt, text_cfg_scale, tiled)
        del action, test_action_with_infer_action
        self.eval()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[:2] != (1, 3):
            raise ValueError(
                "`input_image` must have shape [1,3,H,W] or [3,H,W], got "
                f"{tuple(input_image.shape)}."
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(
            height, width, num_video_frames
        )
        if (checked_h, checked_w, checked_t) != (
            height,
            width,
            num_video_frames,
        ):
            raise ValueError(
                "IDM inference requires spatial multiples of 16 and "
                "`num_video_frames % 4 == 1`."
            )

        context, context_mask = _prepare_inference_context(
            self,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
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
        first_frame_latents = self._encode_input_image_latents_tensor(
            input_image=input_image.to(device=self.device, dtype=self.torch_dtype),
            tiled=tiled,
        )
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(
            getattr(self.video_expert, "fuse_vae_embedding_in_latents", False)
        )

        video_ts, video_deltas = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(video_ts, video_deltas):
            timestep_video = step_t.unsqueeze(0).to(
                dtype=latents_video.dtype, device=self.device
            )
            pred_video = _profile_stage(
                profile,
                "video_generation_step",
                self.device,
                lambda: self._run_video_lbq_branch(
                    latents_video=latents_video,
                    timestep_video=timestep_video,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                    lbq_reads_full_video=False,
                )[0],
                append_ms=True,
            )
            latents_video = self.infer_video_scheduler.step(
                pred_video, step_delta, latents_video
            )
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        def _full_video_lbq_prefill():
            timestep_video = torch.zeros(
                (1,), dtype=latents_video.dtype, device=self.device
            )
            _, lbq_tokens, video_pre = self._run_video_lbq_branch(
                latents_video=latents_video,
                timestep_video=timestep_video,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                lbq_reads_full_video=True,
                return_video_prediction=False,
            )
            return lbq_tokens, video_pre

        lbq_tokens, video_pre = _profile_stage(
            profile,
            "full_video_lbq_prefill_once",
            self.device,
            _full_video_lbq_prefill,
        )
        action_conditioning = self.action_expert.prepare_conditioning(
            text_state_context=context,
            text_state_mask=context_mask,
            lbq_hidden=lbq_tokens,
        )
        if profile is not None:
            profile["video_seq_len"] = int(video_pre["tokens"].shape[1])
            profile["latent_bridge_queries"] = (
                self.bridge.latent_bridge_queries_config()
            )
            profile["action_cross_context_tokens"] = int(
                action_conditioning["cross_context"].shape[1]
            )

        action_ts, action_deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t, step_delta in zip(action_ts, action_deltas):
            timestep_action = step_t.unsqueeze(0).to(
                dtype=latents_action.dtype, device=self.device
            )
            pred_action = _profile_stage(
                profile,
                "action_dit_lbq_step",
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
            latents_action = self.infer_action_scheduler.step(
                pred_action, step_delta, latents_action
            )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }


class BridgeWAMJointAblation(AblationCheckpointMixin, BridgeWAM):
    """Synchronous 30-layer Video/LBQ/Action Joint upper bound."""

    @torch.no_grad()
    def infer_joint(self, *args, test_action_with_infer_action=False, **kwargs):
        """Joint ablation denoises all three streams; no action-only comparison."""
        return super().infer_joint(*args, test_action_with_infer_action=False, **kwargs)

    def _run_joint_training_branch(
        self,
        *,
        latents_video: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action: Optional[torch.Tensor],
        fuse_vae_embedding_in_latents: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        conditioning = self.action_expert.prepare_conditioning(
            text_state_context=context,
            text_state_mask=context_mask,
            lbq_hidden=None,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=conditioning["cross_context"],
            context_mask=conditioning["cross_mask"],
        )
        video_attention_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_tokens, action_tokens, lbq_tokens = self.bridge.forward_joint_with_lbqs(
            video_tokens=video_pre["tokens"],
            action_tokens=action_pre["tokens"],
            video_freqs=video_pre["freqs"],
            action_freqs=action_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            action_t_mod=action_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
        )
        return (
            self.video_expert.post_dit(video_tokens, video_pre),
            self.action_expert.post_dit(action_tokens, action_pre),
            {"lbq_tokens": lbq_tokens, "lbq_context": lbq_tokens},
        )

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int,
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
        del profile
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_video_frames,
            action_horizon=action_horizon,
            action=None,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            test_action_with_infer_action=False,
        )
