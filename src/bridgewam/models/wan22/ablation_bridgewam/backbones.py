from __future__ import annotations

from typing import Any, Optional

import torch

from bridgewam.models.wan22.lbq import LatentBridgeQueries

from ..bridge_of_experts import BridgeOfExperts, logger


class AblationBridge(BridgeOfExperts):
    """BridgeWAM backbone with an experiment identity in checkpoints."""

    def __init__(self, *args, ablation_type: str, **kwargs):
        self.ablation_type = str(ablation_type)
        super().__init__(*args, **kwargs)

    def latent_bridge_queries_config(self) -> Optional[dict[str, Any]]:
        config = super().latent_bridge_queries_config()
        if config is not None:
            config["ablation_type"] = self.ablation_type
            ablation_config = getattr(self, "ablation_config", None)
            if ablation_config is not None:
                config["ablation_config"] = dict(ablation_config)
        return config


class FrozenVideoBridge(AblationBridge):
    """Freeze Video/proprio while training every Action parameter and Latent Bridge Queries."""

    def configure_lbq_trainable_parameters(self) -> None:
        if not self.latent_bridge_queries_enabled or self.latent_bridge_queries is None:
            raise RuntimeError(
                "Frozen-Video ablation requires Video Latent Bridge Queries."
            )

        self.requires_grad_(False)
        self.mixtures["video"].eval()
        self.mixtures["action"].train()
        self.mixtures["action"].requires_grad_(True)
        self.latent_bridge_queries.train()
        self.latent_bridge_queries.requires_grad_(True)

    def lbq_parameter_groups(
        self,
        base_learning_rate: float,
    ) -> list[dict[str, Any]]:
        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("Frozen-Video ablation has no trainable parameters.")
        return [
            {
                "name": "action_and_lbq",
                "params": parameters,
                "lr": float(base_learning_rate),
            }
        ]


class FullVideoBridge(AblationBridge):
    """Latent Bridge Queries variant that can read all frames in an IDM condition branch."""

    def forward_video_with_full_video_lbqs(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_tokens_per_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._forward_video_with_scoped_lbqs(
            video_tokens=video_tokens,
            video_freqs=video_freqs,
            video_t_mod=video_t_mod,
            video_context_payload=video_context_payload,
            video_attention_mask=video_attention_mask,
            video_tokens_per_frame=video_tokens_per_frame,
            lbq_source_token_count=int(video_tokens.shape[1]),
        )

    def _forward_video_with_scoped_lbqs(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
        video_tokens_per_frame: int,
        lbq_source_token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.latent_bridge_queries_enabled or self.latent_bridge_queries is None:
            raise RuntimeError(
                "Full-video LBQ forward requires Video Latent Bridge Queries."
            )
        if video_attention_mask.ndim != 2 or video_attention_mask.shape != (
            video_tokens.shape[1],
            video_tokens.shape[1],
        ):
            raise ValueError(
                "Video attention mask must be square and match Video tokens, got "
                f"{tuple(video_attention_mask.shape)} for {video_tokens.shape[1]} tokens."
            )

        first_frame_tokens = min(
            int(video_tokens_per_frame), int(video_tokens.shape[1])
        )
        lbq_source_token_count = int(lbq_source_token_count)
        if not first_frame_tokens <= lbq_source_token_count <= video_tokens.shape[1]:
            raise ValueError(
                "LBQ source token count must include the first frame and stay "
                f"within the Video sequence, got {lbq_source_token_count}."
            )

        expert = self.mixtures["video"]
        x_video = video_tokens
        lbq_tokens = None
        lbq_attention_mask = torch.ones(
            (
                self.latent_bridge_queries.num_lbqs,
                lbq_source_token_count + self.latent_bridge_queries.num_lbqs,
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
                -1, self.latent_bridge_queries.num_lbqs, -1, -1
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
                        [k_video[:, :lbq_source_token_count], k_lbq], dim=1
                    ),
                    v_cat=torch.cat(
                        [v_video[:, :lbq_source_token_count], v_lbq], dim=1
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

        if lbq_tokens is None:
            raise RuntimeError("Latent Bridge Queries tokens were never initialized.")
        return x_video, lbq_tokens


class JointAttentionBridge(BridgeOfExperts):
    """Thirty-layer synchronous Video/LBQ/Action attention for the Joint ablation."""

    def __init__(
        self,
        *,
        mixtures,
        latent_bridge_queries: dict[str, Any],
        checkpoint_attention: bool,
        ablation_type: str,
    ):
        self.ablation_type = str(ablation_type)
        super().__init__(
            mixtures=mixtures,
            checkpoint_attention=checkpoint_attention,
            _joint_attention=True,
            latent_bridge_queries=None,
        )
        if not bool(self._cfg_get(latent_bridge_queries, "enabled", False)):
            raise ValueError(
                "Joint ablation requires `latent_bridge_queries.enabled=true`."
            )
        if self.action_num_layers != self.video_num_layers:
            raise ValueError(
                "Joint ablation requires equal Video and Action depth, got "
                f"{self.video_num_layers} and {self.action_num_layers}."
            )
        if self.action_architecture != "full":
            raise ValueError("Joint ablation requires `action architecture=full`.")

        video_expert = self.mixtures["video"]
        self.latent_bridge_queries_enabled = True
        self.latent_bridge_queries_freeze_video_expert = False
        self.latent_bridge_queries_train_action_cross_attention = True
        self.latent_bridge_queries_action_cross_attention_lr_scale = 1.0
        self.latent_bridge_queries_strict_training_scope = False
        self.latent_bridge_queries_preserve_direct_video_kv = False
        self.latent_bridge_queries_injection_mode = "text_state"
        self.latent_bridge_queries_generation_coupling = "none"
        self.latent_bridge_queries = LatentBridgeQueries(
            video_hidden_dim=int(video_expert.hidden_dim),
            num_video_layers=self.video_num_layers,
            num_lbqs=int(self._cfg_get(latent_bridge_queries, "num_lbqs", 32)),
            start_layer=int(self._cfg_get(latent_bridge_queries, "start_layer", 0)),
            readout_layer=int(
                self._cfg_get(latent_bridge_queries, "readout_layer", -1)
            ),
            lbq_attention=str(
                self._cfg_get(latent_bridge_queries, "lbq_attention", "bidirectional")
            ),
            lbq_rope_mode=str(
                self._cfg_get(latent_bridge_queries, "lbq_rope_mode", "identity")
            ),
            eps=float(self._cfg_get(latent_bridge_queries, "eps", 1.0e-6)),
        )
        if self.latent_bridge_queries.start_layer != 0 or (
            self.latent_bridge_queries.readout_layer != self.video_num_layers - 1
        ):
            raise ValueError(
                "Joint ablation requires Latent Bridge Queries participation in every layer "
                "(`start_layer=0`, `readout_layer=-1`)."
            )
        reference = next(self.mixtures["action"].parameters())
        self.latent_bridge_queries.to(device=reference.device, dtype=reference.dtype)
        self._log_initialization()

    def _log_initialization(self) -> None:
        # The base constructor runs before this subclass installs its LBQs.
        if self.latent_bridge_queries is None:
            return
        logger.info(
            "Initialized %s: synchronous Video/LBQ/Action attention; "
            "direct cross-stream K/V enabled; video_layers=%d action_layers=%d "
            "queries=%d training_scope=full_expert_finetune",
            type(self).__name__,
            self.video_num_layers,
            self.action_num_layers,
            self.latent_bridge_queries.num_lbqs,
        )

    def latent_bridge_queries_config(self) -> dict[str, Any]:
        config = self.latent_bridge_queries.config_dict()
        config.update(
            {
                "ablation_type": self.ablation_type,
                "injection_mode": "text_state",
                "preserve_direct_video_kv": False,
                "freeze_video_expert": False,
                "train_action_cross_attention": True,
                "action_cross_attention_lr_scale": 1.0,
                "strict_training_scope": False,
                "generation_coupling": "none",
                "joint_attention": True,
            }
        )
        ablation_config = getattr(self, "ablation_config", None)
        if ablation_config is not None:
            config["ablation_config"] = dict(ablation_config)
        return config

    @staticmethod
    def build_joint_attention_mask(
        *,
        video_attention_mask: torch.Tensor,
        num_lbq_tokens: int,
        action_seq_len: int,
    ) -> torch.Tensor:
        if video_attention_mask.ndim != 2 or (
            video_attention_mask.shape[0] != video_attention_mask.shape[1]
        ):
            raise ValueError("Joint Video attention mask must be square and 2D.")
        video_seq_len = int(video_attention_mask.shape[0])
        total = video_seq_len + int(num_lbq_tokens) + int(action_seq_len)
        mask = torch.ones(
            (total, total),
            dtype=torch.bool,
            device=video_attention_mask.device,
        )
        # Only Video-to-Video retains first-frame causality. Every cross-stream
        # edge and LBQ/Action self edge is bidirectional.
        mask[:video_seq_len, :video_seq_len] = video_attention_mask
        return mask

    def forward_joint_with_lbqs(
        self,
        *,
        video_tokens: torch.Tensor,
        action_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        action_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_context_payload: Optional[dict],
        action_context_payload: Optional[dict],
        video_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        video_expert = self.mixtures["video"]
        action_expert = self.mixtures["action"]
        x_video = video_tokens
        x_action = action_tokens
        x_lbq = self.latent_bridge_queries.initial_lbq_tokens(video_tokens)

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
                -1, self.latent_bridge_queries.num_lbqs, -1, -1
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
        attention_mask = self.build_joint_attention_mask(
            video_attention_mask=video_attention_mask,
            num_lbq_tokens=self.latent_bridge_queries.num_lbqs,
            action_seq_len=x_action.shape[1],
        )

        for layer_idx in range(self.num_layers):
            video_io = self._build_expert_attention_io(
                expert=video_expert,
                block=video_expert.blocks[layer_idx],
                x=x_video,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            lbq_io = self._build_expert_attention_io(
                expert=video_expert,
                block=video_expert.blocks[layer_idx],
                x=x_lbq,
                freqs=lbq_freqs,
                t_mod=lbq_t_mod,
            )
            action_io = self._build_expert_attention_io(
                expert=action_expert,
                block=action_expert.blocks[layer_idx],
                x=x_action,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            q_video, k_video, v_video = video_io[:3]
            q_lbq, k_lbq, v_lbq = lbq_io[:3]
            q_action, k_action, v_action = action_io[:3]
            mixed = self._mixed_attention(
                q_cat=torch.cat([q_video, q_lbq, q_action], dim=1),
                k_cat=torch.cat([k_video, k_lbq, k_action], dim=1),
                v_cat=torch.cat([v_video, v_lbq, v_action], dim=1),
                attention_mask=attention_mask,
            )

            video_end = x_video.shape[1]
            lbq_end = video_end + x_lbq.shape[1]
            x_video = self._post_joint_stream(
                block=video_expert.blocks[layer_idx],
                io=video_io,
                mixed_slice=mixed[:, :video_end],
                context_payload=video_context_payload,
            )
            x_lbq = self._post_joint_stream(
                block=video_expert.blocks[layer_idx],
                io=lbq_io,
                mixed_slice=mixed[:, video_end:lbq_end],
                context_payload=lbq_context_payload,
            )
            x_action = self._post_joint_stream(
                block=action_expert.blocks[layer_idx],
                io=action_io,
                mixed_slice=mixed[:, lbq_end:],
                context_payload=action_context_payload,
            )

        return x_video, x_action, x_lbq

    def _post_joint_stream(
        self,
        *,
        block,
        io,
        mixed_slice: torch.Tensor,
        context_payload: Optional[dict],
    ) -> torch.Tensor:
        (
            _q,
            _k,
            _v,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        ) = io
        return self._apply_post_with_optional_checkpoint(
            block=block,
            residual_x=residual_x,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mixed_slice=mixed_slice,
            context_payload=context_payload,
        )
