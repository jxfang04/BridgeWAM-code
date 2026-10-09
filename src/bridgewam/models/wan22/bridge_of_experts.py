from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from bridgewam.models.wan22.lbq import LatentBridgeQueries

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import multihead_attention, modulate, rope_apply
from bridgewam.utils.logging_config import get_logger

logger = get_logger(__name__)


class BridgeOfExperts(nn.Module):
    """Video/LBQ/Action backbone; Action never reads raw Video K/V.

    The historical routing buffer is serialized only for checkpoint identity.
    Joint ablations bootstrap their three-stream queries explicitly.
    """

    LATENT_BRIDGE_QUERIES_GENERATION_COUPLINGS = {
        "none",
        "future_video_reads_lbq",
    }
    LATENT_BRIDGE_QUERIES_INJECTION_MODES = {
        "lbq_only",
        "text_state_lbq",
        "text_state",
    }

    def __init__(
        self,
        mixtures: Dict[str, nn.Module],
        checkpoint_attention: bool = True,
        latent_bridge_queries: Optional[dict[str, Any]] = None,
        *,
        _joint_attention: bool = False,
    ):
        super().__init__()
        cfg = latent_bridge_queries or {}
        if not _joint_attention and (
            not cfg.get("enabled", False) or cfg.get("preserve_direct_video_kv", True)
        ):
            raise ValueError(
                "BridgeOfExperts requires enabled LBQs and preserve_direct_video_kv=false."
            )
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError(
                "`mixtures` must include both 'video' and 'action' experts."
            )

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.checkpoint_attention = checkpoint_attention
        if checkpoint_attention:
            logger.info(
                "Using gradient checkpointing for mixture attention. This will save memory but use more computation."
            )

        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        self.video_num_layers = len(video_expert.blocks)
        self.action_num_layers = len(action_expert.blocks)
        # Existing Video/LBQ code treats num_layers as Video depth.
        self.num_layers = self.video_num_layers
        self.action_architecture = str(getattr(action_expert, "architecture", "full"))
        lbq_requested = bool(self._cfg_get(latent_bridge_queries, "enabled", False))
        preserve_direct_video_kv_requested = bool(
            self._cfg_get(latent_bridge_queries, "preserve_direct_video_kv", True)
        )
        if self.action_architecture == "alternating_cross_self" and (
            not lbq_requested or preserve_direct_video_kv_requested
        ):
            raise ValueError(
                "alternating_cross_self requires latent_bridge_queries.enabled=true "
                "and latent_bridge_queries.preserve_direct_video_kv=false."
            )

        first_expert = video_expert
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            allow_separate_action_depth = (
                name == "action"
                and self.action_architecture == "alternating_cross_self"
                and lbq_requested
                and not preserve_direct_video_kv_requested
            )
            if (
                len(expert.blocks) != self.num_layers
                and not allow_separate_action_depth
            ):
                raise ValueError(
                    "Video and Action experts may use different depths only with "
                    "Latent Bridge Queries alternating_cross_self and preserve_direct_video_kv=false; "
                    f"got Video={self.num_layers}, Action={len(expert.blocks)}."
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        # Keep the registered buffer and registration order of the verified
        # checkpoint format. It does not enable a direct Video-to-Action route.
        self.register_buffer(
            "action_video_kv_layer_mask",
            torch.ones(self.action_num_layers, dtype=torch.bool),
            persistent=True,
        )

        self.latent_bridge_queries_enabled = bool(
            self._cfg_get(latent_bridge_queries, "enabled", False)
        )
        self.latent_bridge_queries_freeze_video_expert = bool(
            self._cfg_get(latent_bridge_queries, "freeze_video_expert", True)
        )
        self.latent_bridge_queries_train_action_cross_attention = bool(
            self._cfg_get(latent_bridge_queries, "train_action_cross_attention", True)
        )
        self.latent_bridge_queries_action_cross_attention_lr_scale = float(
            self._cfg_get(latent_bridge_queries, "action_cross_attention_lr_scale", 0.1)
        )
        self.latent_bridge_queries_strict_training_scope = bool(
            self._cfg_get(latent_bridge_queries, "strict_training_scope", False)
        )
        self.latent_bridge_queries_preserve_direct_video_kv = bool(
            self._cfg_get(latent_bridge_queries, "preserve_direct_video_kv", True)
        )
        self.latent_bridge_queries_injection_mode = str(
            self._cfg_get(latent_bridge_queries, "injection_mode", "lbq_only")
        )
        self.latent_bridge_queries_generation_coupling = str(
            self._cfg_get(latent_bridge_queries, "generation_coupling", "none")
        )
        if self.latent_bridge_queries_action_cross_attention_lr_scale <= 0.0:
            raise ValueError(
                "`latent_bridge_queries.action_cross_attention_lr_scale` must be positive."
            )
        if self.latent_bridge_queries_enabled:
            if (
                self.latent_bridge_queries_generation_coupling
                not in self.LATENT_BRIDGE_QUERIES_GENERATION_COUPLINGS
            ):
                raise ValueError(
                    "`latent_bridge_queries.generation_coupling` must be one of "
                    f"{sorted(self.LATENT_BRIDGE_QUERIES_GENERATION_COUPLINGS)}, got "
                    f"{self.latent_bridge_queries_generation_coupling!r}."
                )
            if (
                self.latent_bridge_queries_injection_mode
                not in self.LATENT_BRIDGE_QUERIES_INJECTION_MODES
            ):
                raise ValueError(
                    "`latent_bridge_queries.injection_mode` must be one of "
                    f"{sorted(self.LATENT_BRIDGE_QUERIES_INJECTION_MODES)}, got "
                    f"{self.latent_bridge_queries_injection_mode!r}."
                )
            if (
                not self.latent_bridge_queries_freeze_video_expert
                and self.latent_bridge_queries_strict_training_scope
            ):
                raise ValueError(
                    "Full-parameter Latent Bridge Queries training requires "
                    "`strict_training_scope=false`."
                )
            video_expert = self.mixtures["video"]
            action_expert = self.mixtures["action"]
            if getattr(action_expert, "lbq_embedding", None) is None:
                raise ValueError(
                    "Enabled Video Latent Bridge Queries requires ActionDiT `lbq_embedding`."
                )
            if int(action_expert.lbq_dim) != int(video_expert.hidden_dim):
                raise ValueError(
                    "ActionDiT `lbq_dim` must match Video DiT hidden dim, got "
                    f"{action_expert.lbq_dim} and {video_expert.hidden_dim}."
                )
            if (
                action_expert.conditioning_mode
                != self.latent_bridge_queries_injection_mode
            ):
                raise ValueError(
                    "ActionDiT conditioning mode and Video Latent Bridge Queries injection mode "
                    f"must match, got {action_expert.conditioning_mode!r} and "
                    f"{self.latent_bridge_queries_injection_mode!r}."
                )
            self.latent_bridge_queries = LatentBridgeQueries(
                video_hidden_dim=int(video_expert.hidden_dim),
                num_video_layers=self.video_num_layers,
                num_lbqs=int(self._cfg_get(latent_bridge_queries, "num_lbqs", 32)),
                start_layer=int(self._cfg_get(latent_bridge_queries, "start_layer", 0)),
                readout_layer=int(
                    self._cfg_get(latent_bridge_queries, "readout_layer", -1)
                ),
                lbq_attention=str(
                    self._cfg_get(
                        latent_bridge_queries, "lbq_attention", "bidirectional"
                    )
                ),
                lbq_rope_mode=str(
                    self._cfg_get(latent_bridge_queries, "lbq_rope_mode", "identity")
                ),
                eps=float(self._cfg_get(latent_bridge_queries, "eps", 1.0e-6)),
            )
            action_reference = next(action_expert.parameters())
            self.latent_bridge_queries.to(
                device=action_reference.device,
                dtype=action_reference.dtype,
            )
        else:
            self.latent_bridge_queries = None

        self._log_initialization()

    def _log_initialization(self) -> None:
        logger.info(
            "Initialized %s with experts=%s video_layers=%d action_layers=%d "
            "action_architecture=%s",
            type(self).__name__,
            self.expert_order,
            self.video_num_layers,
            self.action_num_layers,
            self.action_architecture,
        )
        logger.info(
            "Action visual conditioning: LBQ bridge; no direct Video K/V cache."
        )
        if self.latent_bridge_queries_enabled:
            logger.info(
                "BridgeWAM Latent Bridge Queries enabled: queries=%d start_layer=%d "
                "readout_layer=%d (requested=%d) generation_coupling=%s "
                "injection_mode=%s direct_video_kv=%s "
                "training_scope=%s action_cross_attn_lr_scale=%.4f",
                self.latent_bridge_queries.num_lbqs,
                self.latent_bridge_queries.start_layer,
                self.latent_bridge_queries.readout_layer,
                self.latent_bridge_queries.requested_readout_layer,
                self.latent_bridge_queries_generation_coupling,
                self.latent_bridge_queries_injection_mode,
                self.latent_bridge_queries_preserve_direct_video_kv,
                (
                    "video_frozen"
                    if self.latent_bridge_queries_freeze_video_expert
                    else "full_expert_finetune"
                ),
                self.latent_bridge_queries_action_cross_attention_lr_scale,
            )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B"
            )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            base_mod + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    @classmethod
    def normalize_latent_bridge_queries_checkpoint_config(
        cls,
        config: dict[str, Any],
        *,
        num_layers: int,
    ) -> dict[str, Any]:
        normalized = dict(config)
        injection_mode = str(normalized.get("injection_mode", "lbq_only"))
        if injection_mode == "replace_action_context":
            injection_mode = "lbq_only"
        normalized["injection_mode"] = injection_mode

        legacy_adapter = normalized.pop("conditioning_adapter", None)
        if legacy_adapter not in (None, "action_hidden_embedding"):
            raise ValueError(
                "Connector-based Latent Bridge Queries checkpoints cannot be loaded into "
                "BridgeWAM. Expected legacy `conditioning_adapter` to be "
                f"`action_hidden_embedding`, got {legacy_adapter!r}."
            )
        normalized.setdefault("generation_coupling", "none")
        legacy_readout_layer = normalized.pop(
            "connector_input_layer",
            int(num_layers) - 1,
        )
        if "readout_layer" not in normalized:
            normalized["readout_layer"] = legacy_readout_layer
        readout_layer = int(normalized["readout_layer"])
        if readout_layer < 0:
            readout_layer += int(num_layers)
        normalized["readout_layer"] = readout_layer
        for key in (
            "connector_dim",
            "connector_depth",
            "connector_num_heads",
            "connector_mlp_ratio",
        ):
            normalized.pop(key, None)
        return normalized

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return multihead_attention(
                q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask
            )

        if self.checkpoint_attention and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    @staticmethod
    def _cfg_get(cfg: Optional[dict], key: str, default=None):
        if cfg is None:
            return default
        if hasattr(cfg, "get"):
            return cfg.get(key, default)
        return default

    @property
    def requires_direct_video_kv_cache(self) -> bool:
        """Serialized legacy metadata never enables a direct Video K/V path."""
        return False

    def configure_lbq_trainable_parameters(self) -> None:
        if not self.latent_bridge_queries_enabled:
            return
        if not self.latent_bridge_queries_freeze_video_expert:
            self.train()
            self.requires_grad_(True)
            return
        self.requires_grad_(False)
        self.mixtures["video"].eval()
        self.mixtures["action"].train()
        self.latent_bridge_queries.train()
        self.latent_bridge_queries.requires_grad_(True)
        action_expert = self.mixtures["action"]
        action_expert.lbq_embedding.train()
        action_expert.lbq_embedding.requires_grad_(True)
        if action_expert.uses_text_state_context:
            action_expert.text_embedding.train()
            action_expert.text_embedding.requires_grad_(True)
        for module in self._action_conditioning_attention_modules():
            module.train()
            module.requires_grad_(True)

    def _action_conditioning_attention_modules(self) -> list[nn.Module]:
        action_expert = self.mixtures["action"]
        modules: list[nn.Module] = []
        for block in action_expert.blocks:
            if self.latent_bridge_queries_train_action_cross_attention and hasattr(
                block, "cross_attn"
            ):
                modules.append(block.cross_attn)
            if self.latent_bridge_queries_injection_mode == "text_state" and hasattr(
                block, "self_attn"
            ):
                modules.append(block.self_attn)
        return modules

    def lbq_parameter_groups(
        self,
        base_learning_rate: float,
    ) -> Optional[list[dict[str, Any]]]:
        if not self.latent_bridge_queries_enabled:
            return None
        if not self.latent_bridge_queries_freeze_video_expert:
            return None
        lbq_params = [
            parameter
            for parameter in self.latent_bridge_queries.parameters()
            if parameter.requires_grad
        ]
        action_expert = self.mixtures["action"]
        lbq_params.extend(
            parameter
            for parameter in action_expert.lbq_embedding.parameters()
            if parameter.requires_grad
        )
        if action_expert.uses_text_state_context:
            lbq_params.extend(
                parameter
                for parameter in action_expert.text_embedding.parameters()
                if parameter.requires_grad
            )
        action_attention_params = [
            parameter
            for module in self._action_conditioning_attention_modules()
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        groups = [
            {
                "name": "bridge_conditioning",
                "params": lbq_params,
                "lr": float(base_learning_rate),
            },
        ]
        if action_attention_params:
            groups.append(
                {
                    "name": "action_conditioning_attention",
                    "params": action_attention_params,
                    "lr": float(base_learning_rate)
                    * self.latent_bridge_queries_action_cross_attention_lr_scale,
                }
            )
        return groups

    def latent_bridge_queries_config(self) -> Optional[dict[str, Any]]:
        if not self.latent_bridge_queries_enabled:
            return None
        config = self.latent_bridge_queries.config_dict()
        config.update(
            {
                "injection_mode": self.latent_bridge_queries_injection_mode,
                "preserve_direct_video_kv": self.latent_bridge_queries_preserve_direct_video_kv,
                "freeze_video_expert": self.latent_bridge_queries_freeze_video_expert,
                "train_action_cross_attention": (
                    self.latent_bridge_queries_train_action_cross_attention
                ),
                "action_cross_attention_lr_scale": (
                    self.latent_bridge_queries_action_cross_attention_lr_scale
                ),
                "strict_training_scope": self.latent_bridge_queries_strict_training_scope,
                "generation_coupling": self.latent_bridge_queries_generation_coupling,
            }
        )
        action_expert = self.mixtures["action"]
        config.update(
            {
                "lbq_embedding_input_dim": int(action_expert.lbq_dim),
                "lbq_embedding_output_dim": int(action_expert.hidden_dim),
            }
        )
        return config

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(
            getattr(expert, "use_gradient_checkpointing", False)
        )
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """

        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    @staticmethod
    def _lbq_context_payload(
        video_context_payload: Optional[dict],
        num_lbqs: int,
    ) -> Optional[dict]:
        if video_context_payload is None:
            return None
        context = video_context_payload.get("context")
        if context is None:
            return video_context_payload
        context_mask = video_context_payload.get("mask")
        if context_mask is None:
            return {"context": context, "mask": None}
        if context_mask.ndim == 2:
            lbq_mask = context_mask.unsqueeze(1).expand(-1, num_lbqs, -1)
        elif context_mask.ndim == 3:
            lbq_mask = context_mask[:, :1, :].expand(-1, num_lbqs, -1)
        elif context_mask.ndim == 4:
            lbq_mask = context_mask[:, :, :1, :].expand(
                -1,
                -1,
                num_lbqs,
                -1,
            )
        else:
            raise ValueError(
                "Video context mask must be 2D, 3D, or 4D, got "
                f"{tuple(context_mask.shape)}."
            )
        return {"context": context, "mask": lbq_mask}

    def _forward_bridge_video(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_tokens_per_frame: int,
        stop_after_readout: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """Run the full Video path and return Latent Bridge Queries states from the configured layer."""
        if not self.latent_bridge_queries_enabled or self.latent_bridge_queries is None:
            raise RuntimeError(
                "Video Latent Bridge Queries forward requires "
                "`latent_bridge_queries.enabled=true`."
            )
        if video_attention_mask.ndim != 2:
            raise ValueError(
                "`video_attention_mask` must be 2D [Sv,Sv], got "
                f"{tuple(video_attention_mask.shape)}."
            )
        if video_attention_mask.shape != (
            video_tokens.shape[1],
            video_tokens.shape[1],
        ):
            raise ValueError(
                "Video attention mask shape must match Video sequence length, got "
                f"{tuple(video_attention_mask.shape)} and {video_tokens.shape[1]}."
            )
        first_frame_tokens = min(
            int(video_tokens_per_frame),
            int(video_tokens.shape[1]),
        )
        if first_frame_tokens <= 0:
            raise ValueError(
                "Latent Bridge Queries requires at least one first-frame Video token."
            )

        expert = self.mixtures["video"]
        x_video = video_tokens
        lbq_tokens = None
        lbq_attention_mask = torch.ones(
            (
                self.latent_bridge_queries.num_lbqs,
                first_frame_tokens + self.latent_bridge_queries.num_lbqs,
            ),
            dtype=torch.bool,
            device=video_tokens.device,
        )
        lbq_freqs = torch.ones(
            (
                self.latent_bridge_queries.num_lbqs,
                1,
                video_freqs.shape[-1],
            ),
            dtype=video_freqs.dtype,
            device=video_freqs.device,
        )
        if video_t_mod.ndim == 4:
            lbq_t_mod = video_t_mod[:, :1].expand(
                -1,
                self.latent_bridge_queries.num_lbqs,
                -1,
                -1,
            )
        elif video_t_mod.ndim == 3:
            lbq_t_mod = video_t_mod
        else:
            raise ValueError(
                f"Video time modulation must be 3D or 4D, got {video_t_mod.ndim}D."
            )
        lbq_context_payload = self._lbq_context_payload(
            video_context_payload,
            self.latent_bridge_queries.num_lbqs,
        )

        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_video,
                k_video,
                v_video,
                residual_video,
                gate_msa_video,
                shift_mlp_video,
                scale_mlp_video,
                gate_mlp_video,
                checkpoint_video,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            lbq_active = (
                self.latent_bridge_queries.start_layer
                <= layer_idx
                <= self.latent_bridge_queries.readout_layer
            )
            couple_future_video = (
                lbq_active
                and self.latent_bridge_queries_generation_coupling
                == "future_video_reads_lbq"
                and first_frame_tokens < x_video.shape[1]
            )
            # Future rows are computed below with LBQ K/V. Do not first compute
            # the Video-only rows that would immediately be discarded. Keep all
            # Video keys and the original mask for the retained query rows.
            video_query_count = (
                first_frame_tokens if couple_future_video else x_video.shape[1]
            )
            mixed_video = self._mixed_attention(
                q_cat=q_video[:, :video_query_count],
                k_cat=k_video,
                v_cat=v_video,
                attention_mask=video_attention_mask[:video_query_count],
            )

            lbq_state = None
            if lbq_active:
                if lbq_tokens is None:
                    lbq_tokens = self.latent_bridge_queries.initial_lbq_tokens(x_video)
                (
                    q_lbq,
                    k_lbq,
                    v_lbq,
                    residual_lbq,
                    gate_msa_lbq,
                    shift_mlp_lbq,
                    scale_mlp_lbq,
                    gate_mlp_lbq,
                    checkpoint_lbq,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=lbq_tokens,
                    freqs=lbq_freqs,
                    t_mod=lbq_t_mod,
                )
                mixed_lbq = self._mixed_attention(
                    q_cat=q_lbq,
                    k_cat=torch.cat(
                        [k_video[:, :first_frame_tokens], k_lbq],
                        dim=1,
                    ),
                    v_cat=torch.cat(
                        [v_video[:, :first_frame_tokens], v_lbq],
                        dim=1,
                    ),
                    attention_mask=lbq_attention_mask,
                )
                if couple_future_video:
                    future_video_mask = torch.cat(
                        [
                            video_attention_mask[first_frame_tokens:],
                            torch.ones(
                                (
                                    x_video.shape[1] - first_frame_tokens,
                                    self.latent_bridge_queries.num_lbqs,
                                ),
                                dtype=torch.bool,
                                device=video_attention_mask.device,
                            ),
                        ],
                        dim=1,
                    )
                    mixed_future_video = self._mixed_attention(
                        q_cat=q_video[:, first_frame_tokens:],
                        k_cat=torch.cat([k_video, k_lbq], dim=1),
                        v_cat=torch.cat([v_video, v_lbq], dim=1),
                        attention_mask=future_video_mask,
                    )
                    mixed_video = torch.cat(
                        [
                            mixed_video[:, :first_frame_tokens],
                            mixed_future_video,
                        ],
                        dim=1,
                    )
                lbq_state = (
                    residual_lbq,
                    gate_msa_lbq,
                    shift_mlp_lbq,
                    scale_mlp_lbq,
                    gate_mlp_lbq,
                    checkpoint_lbq,
                    mixed_lbq,
                )

            x_video = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_video,
                gate_msa=gate_msa_video,
                shift_mlp=shift_mlp_video,
                scale_mlp=scale_mlp_video,
                gate_mlp=gate_mlp_video,
                use_gradient_checkpointing=checkpoint_video,
                mixed_slice=mixed_video,
                context_payload=video_context_payload,
            )

            if lbq_state is not None:
                (
                    residual_lbq,
                    gate_msa_lbq,
                    shift_mlp_lbq,
                    scale_mlp_lbq,
                    gate_mlp_lbq,
                    checkpoint_lbq,
                    mixed_lbq,
                ) = lbq_state
                lbq_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=residual_lbq,
                    gate_msa=gate_msa_lbq,
                    shift_mlp=shift_mlp_lbq,
                    scale_mlp=scale_mlp_lbq,
                    gate_mlp=gate_mlp_lbq,
                    use_gradient_checkpointing=checkpoint_lbq,
                    mixed_slice=mixed_lbq,
                    context_payload=lbq_context_payload,
                )
            if (
                stop_after_readout
                and layer_idx >= self.latent_bridge_queries.readout_layer
            ):
                break

        if lbq_tokens is None:
            raise RuntimeError("Latent Bridge Queries tokens were never initialized.")
        return x_video, lbq_tokens

    def forward_bridge_video(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_tokens_per_frame: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return final Video tokens and selected-layer LBQ tokens for training."""
        return self._forward_bridge_video(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
            video_tokens_per_frame=video_tokens_per_frame,
        )

    def prefill_bridge(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_tokens_per_frame: int,
    ) -> torch.Tensor:
        """Compute fixed LBQ conditioning once, stopping at the configured readout."""
        _, lbq_tokens = self._forward_bridge_video(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
            video_tokens_per_frame=video_tokens_per_frame,
            stop_after_readout=True,
        )
        return lbq_tokens

    def forward_bridge_action(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: Optional[dict],
        self_lbq_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Denoise Action from projected LBQs; no raw Video tokens or cache."""
        action_seq_len = int(action_tokens.shape[1])
        action_self_mask = torch.ones(
            (action_seq_len, action_seq_len),
            dtype=torch.bool,
            device=action_tokens.device,
        )

        expert = self.mixtures["action"]
        x = action_tokens
        if self.action_architecture == "alternating_cross_self":
            if self.latent_bridge_queries_preserve_direct_video_kv:
                raise ValueError(
                    "alternating_cross_self requires "
                    "latent_bridge_queries.preserve_direct_video_kv=false."
                )
            context = (
                None
                if action_context_payload is None
                else action_context_payload.get("context")
            )
            context_mask = (
                None
                if action_context_payload is None
                else action_context_payload.get("mask")
            )
            for block in expert.blocks:
                x = gradient_checkpoint_forward(
                    block,
                    bool(expert.use_gradient_checkpointing),
                    x,
                    context,
                    action_t_mod,
                    action_freqs,
                    context_mask=context_mask,
                    self_attn_mask=action_self_mask,
                    lbq_context=self_lbq_context,
                )
            return x

        for layer_idx in range(self.action_num_layers):
            block = expert.blocks[layer_idx]
            # Action query/key/value are still step-dependent and must be recomputed each step.
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            k_parts = []
            v_parts = []
            base_mask = action_self_mask
            k_parts.append(k_action)
            v_parts.append(v_action)
            if self_lbq_context is not None:
                k_lbq = block.self_attn.norm_k(block.self_attn.k(self_lbq_context))
                v_lbq = block.self_attn.v(self_lbq_context)
                k_parts.append(k_lbq)
                v_parts.append(v_lbq)
                lbq_mask = torch.ones(
                    (base_mask.shape[0], self_lbq_context.shape[1]),
                    dtype=torch.bool,
                    device=base_mask.device,
                )
                base_mask = torch.cat([base_mask, lbq_mask], dim=-1)

            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=torch.cat(k_parts, dim=1),
                v_cat=torch.cat(v_parts, dim=1),
                attention_mask=base_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
            )
        return x
