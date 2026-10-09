import os
import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from bridgewam.utils.logging_config import get_logger

from .helpers.gradient import gradient_checkpoint_forward
from .wan_video_dit import (
    CrossAttention,
    DiTBlock,
    SelfAttention,
    multihead_attention,
    modulate,
    rope_apply,
    sinusoidal_embedding_1d,
    precompute_freqs_cis,
)

logger = get_logger(__name__)


class _AlternatingActionBlockBase(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int, eps: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_dim = ffn_dim
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.modulation = nn.Parameter(
            torch.randn(1, 6, hidden_dim) / hidden_dim**0.5
        )

    def _modulation(self, t_mod: torch.Tensor):
        has_seq = t_mod.ndim == 4
        chunk_dim = 2 if has_seq else 1
        values = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            values = tuple(value.squeeze(2) for value in values)
        return values

    @staticmethod
    def _residual(x: torch.Tensor, gate: torch.Tensor, value: torch.Tensor):
        return x + gate * value

    def _apply_ffn(
        self,
        x: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        ffn_input = modulate(self.norm2(x), shift_mlp, scale_mlp)
        return self._residual(x, gate_mlp, self.ffn(ffn_input))


class ActionCrossAttentionBlock(_AlternatingActionBlockBase):
    block_type = "cross"

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float,
    ):
        super().__init__(hidden_dim=hidden_dim, ffn_dim=ffn_dim, eps=eps)
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.cross_attn = CrossAttention(
            hidden_dim, attn_head_dim, num_heads, eps
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        lbq_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del freqs, self_attn_mask, lbq_context
        if context is None:
            raise ValueError("Action cross-attention block requires context.")
        if context_mask is not None and context_mask.ndim == 3:
            context_mask = context_mask.unsqueeze(1)
        (
            shift_attn,
            scale_attn,
            _gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._modulation(t_mod)
        attn_input = modulate(self.norm1(x), shift_attn, scale_attn)
        x = x + self.cross_attn(attn_input, context, ctx_mask=context_mask)
        return self._apply_ffn(x, shift_mlp, scale_mlp, gate_mlp)


class ActionMixSelfAttention(SelfAttention):
    """Action self-attention with optional LBQ tokens exposed as K/V only."""

    @staticmethod
    def _append_lbq_mask(
        self_attn_mask: Optional[torch.Tensor],
        num_lbq_tokens: int,
    ) -> Optional[torch.Tensor]:
        if self_attn_mask is None:
            return None
        lbq_mask = torch.ones(
            (*self_attn_mask.shape[:-1], int(num_lbq_tokens)),
            dtype=torch.bool,
            device=self_attn_mask.device,
        )
        return torch.cat([self_attn_mask, lbq_mask], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        lbq_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_action = self.norm_q(self.q(x))
        k_action = self.norm_k(self.k(x))
        v_action = self.v(x)
        q_action = rope_apply(q_action, freqs, self.num_heads)
        k_action = rope_apply(k_action, freqs, self.num_heads)

        if lbq_context is None:
            k = k_action
            v = v_action
            attention_mask = self_attn_mask
        else:
            if lbq_context.ndim != 3 or lbq_context.shape[0] != x.shape[0]:
                raise ValueError(
                    "`lbq_context` must be [B,M,D] with the same batch size as "
                    f"Action tokens, got {tuple(lbq_context.shape)}."
                )
            if lbq_context.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"LBQ context hidden dim must be {self.hidden_dim}, "
                    f"got {lbq_context.shape[-1]}."
                )
            # LBQ tokens have no Action horizon position, so their keys use
            # identity RoPE while Action keys keep the normal horizon RoPE.
            k_lbq = self.norm_k(self.k(lbq_context))
            v_lbq = self.v(lbq_context)
            k = torch.cat([k_action, k_lbq], dim=1)
            v = torch.cat([v_action, v_lbq], dim=1)
            attention_mask = self._append_lbq_mask(
                self_attn_mask,
                lbq_context.shape[1],
            )

        readout = multihead_attention(
            q=q_action,
            k=k,
            v=v,
            num_heads=self.num_heads,
            ctx_mask=attention_mask,
        )
        return self.o(readout)


class ActionSelfAttentionBlock(_AlternatingActionBlockBase):
    block_type = "self"

    def __init__(
        self,
        hidden_dim: int,
        attn_head_dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float,
    ):
        super().__init__(hidden_dim=hidden_dim, ffn_dim=ffn_dim, eps=eps)
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.self_attn = ActionMixSelfAttention(
            hidden_dim, attn_head_dim, num_heads, eps
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        lbq_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del context, context_mask
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self._modulation(t_mod)
        attn_input = modulate(self.norm1(x), shift_attn, scale_attn)
        x = self._residual(
            x,
            gate_attn,
            self.self_attn(
                attn_input,
                freqs,
                self_attn_mask=self_attn_mask,
                lbq_context=lbq_context,
            ),
        )
        return self._apply_ffn(x, shift_mlp, scale_mlp, gate_mlp)


class ActionHead(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=False)
        self.proj = nn.Linear(hidden_dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, hidden_dim) / hidden_dim**0.5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = (self.modulation.to(dtype=t.dtype, device=t.device) + t.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.squeeze(1)
        scale = scale.squeeze(1)
        return self.proj(self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class ActionDiT(nn.Module):
    SUPPORTED_ARCHITECTURES = {"full", "alternating_cross_self"}
    SUPPORTED_CONDITIONING_MODES = {
        "lbq_only",
        "text_state_lbq",
        "text_state",
    }
    ACTION_BACKBONE_SKIP_PREFIXES = (
        "action_encoder.",
        "head.",
        "lbq_embedding.",
        "position_embedding.",
    )
    ACTION_BACKBONE_META_KEYS = (
        "hidden_dim",
        "ffn_dim",
        "num_layers",
        "num_heads",
        "attn_head_dim",
        "text_dim",
        "freq_dim",
        "eps",
    )

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        ffn_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        lbq_dim: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
        architecture: str = "full",
        conditioning_mode: str = "text_state",
        add_pos_embed: bool = False,
        max_action_horizon: int = 1024,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.ffn_dim = ffn_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.lbq_dim = None if lbq_dim is None else int(lbq_dim)
        self.architecture = str(architecture)
        self.conditioning_mode = str(conditioning_mode)
        self.add_pos_embed = bool(add_pos_embed)
        self.max_action_horizon = int(max_action_horizon)
        self.pretrained_source_layers: Optional[list[int]] = None

        if self.architecture not in self.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                "`architecture` must be one of "
                f"{sorted(self.SUPPORTED_ARCHITECTURES)}, got {architecture!r}."
            )
        if self.conditioning_mode not in self.SUPPORTED_CONDITIONING_MODES:
            raise ValueError(
                "`conditioning_mode` must be one of "
                f"{sorted(self.SUPPORTED_CONDITIONING_MODES)}, "
                f"got {conditioning_mode!r}."
            )
        if self.conditioning_mode in {"lbq_only", "text_state_lbq"} and self.lbq_dim is None:
            raise ValueError(
                f"`conditioning_mode={self.conditioning_mode}` requires `lbq_dim`."
            )
        if int(num_layers) <= 0:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}.")
        if self.architecture == "alternating_cross_self" and int(num_layers) % 2:
            raise ValueError(
                "`alternating_cross_self` requires an even `num_layers`, "
                f"got {num_layers}."
            )
        if self.max_action_horizon <= 0:
            raise ValueError(
                "`max_action_horizon` must be positive, "
                f"got {max_action_horizon}."
            )
        if num_heads <= 0:
            raise ValueError(f"`num_heads` must be > 0, got {num_heads}")
        if attn_head_dim <= 0:
            raise ValueError(f"`attn_head_dim` must be > 0, got {attn_head_dim}")
        if attn_head_dim % 2 != 0:
            raise ValueError(f"`attn_head_dim` must be even for RoPE, got {attn_head_dim}")

        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.position_embedding = (
            nn.Embedding(self.max_action_horizon, hidden_dim)
            if self.add_pos_embed
            else None
        )
        if self.position_embedding is not None:
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        # Preserve the historical initialization order across conditioning
        # modes. In lbq_only this module remains present but is not executed.
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.lbq_embedding = (
            nn.Sequential(
                nn.Linear(self.lbq_dim, hidden_dim),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if self.lbq_dim is not None
            else None
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6))
        if self.architecture == "full":
            blocks = [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    attn_head_dim=attn_head_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        else:
            blocks = [
                (
                    ActionCrossAttentionBlock(
                        hidden_dim=hidden_dim,
                        attn_head_dim=attn_head_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        eps=eps,
                    )
                    if layer_idx % 2 == 0
                    else ActionSelfAttentionBlock(
                        hidden_dim=hidden_dim,
                        attn_head_dim=attn_head_dim,
                        num_heads=num_heads,
                        ffn_dim=ffn_dim,
                        eps=eps,
                    )
                )
                for layer_idx in range(num_layers)
            ]
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(hidden_dim, action_dim)
        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)

        self.use_gradient_checkpointing = use_gradient_checkpointing

    @property
    def block_types(self) -> list[str]:
        if self.architecture == "full":
            return ["full"] * len(self.blocks)
        return [block.block_type for block in self.blocks]

    def architecture_config(self) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "num_layers": len(self.blocks),
            "block_types": self.block_types,
            "conditioning_mode": self.conditioning_mode,
            "uses_lbq_self_attention": bool(
                self.conditioning_mode == "text_state" and self.lbq_embedding is not None
            ),
            "add_pos_embed": self.add_pos_embed,
            "max_action_horizon": self.max_action_horizon,
        }

    @property
    def uses_text_state_context(self) -> bool:
        return self.conditioning_mode in {"text_state", "text_state_lbq"}

    @property
    def uses_lbq_context(self) -> bool:
        return self.lbq_embedding is not None

    def embed_lbq_context(self, lbq_hidden: torch.Tensor) -> torch.Tensor:
        if self.lbq_embedding is None or self.lbq_dim is None:
            raise RuntimeError(
                "ActionDiT LBQ embedding is disabled; construct the model with "
                "`lbq_dim` to use Latent Bridge Queries action conditioning."
            )
        if lbq_hidden.ndim != 3:
            raise ValueError(
                f"`lbq_hidden` must be [B,M,D], got {tuple(lbq_hidden.shape)}."
            )
        if lbq_hidden.shape[-1] != self.lbq_dim:
            raise ValueError(
                f"LBQ hidden dim must be {self.lbq_dim}, "
                f"got {lbq_hidden.shape[-1]}."
            )
        return self.lbq_embedding(lbq_hidden)

    def prepare_conditioning(
        self,
        *,
        text_state_context: Optional[torch.Tensor],
        text_state_mask: Optional[torch.Tensor],
        lbq_hidden: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Project fixed conditioning once before Action denoising blocks."""

        text_context = None
        if self.uses_text_state_context:
            if text_state_context is None or text_state_mask is None:
                raise ValueError(
                    f"`conditioning_mode={self.conditioning_mode}` requires "
                    "Text+State context and mask."
                )
            if text_state_context.ndim != 3:
                raise ValueError(
                    "`text_state_context` must be [B,L,D], got "
                    f"{tuple(text_state_context.shape)}."
                )
            if text_state_context.shape[-1] != self.text_dim:
                raise ValueError(
                    f"Text+State context dim must be {self.text_dim}, "
                    f"got {text_state_context.shape[-1]}."
                )
            if text_state_mask.ndim != 2 or text_state_mask.shape != text_state_context.shape[:2]:
                raise ValueError(
                    "`text_state_mask` must match Text+State [B,L], got "
                    f"{tuple(text_state_mask.shape)}."
                )
            text_context = self.text_embedding(text_state_context)

        lbq_context = None
        if self.uses_lbq_context:
            if lbq_hidden is None:
                raise ValueError(
                    f"`conditioning_mode={self.conditioning_mode}` requires LBQ hidden tokens."
                )
            lbq_context = self.embed_lbq_context(lbq_hidden)

        if self.conditioning_mode == "lbq_only":
            cross_context = lbq_context
            cross_mask = torch.ones(
                lbq_context.shape[:2],
                dtype=torch.bool,
                device=lbq_context.device,
            )
            self_lbq_context = None
        elif self.conditioning_mode == "text_state_lbq":
            lbq_mask = torch.ones(
                lbq_context.shape[:2],
                dtype=torch.bool,
                device=lbq_context.device,
            )
            cross_context = torch.cat([text_context, lbq_context], dim=1)
            cross_mask = torch.cat([text_state_mask, lbq_mask], dim=1)
            self_lbq_context = None
        else:
            cross_context = text_context
            cross_mask = text_state_mask
            # Baseline ActionDiT has no lbq_embedding and therefore retains
            # ordinary Action-only self-attention in text_state mode.
            self_lbq_context = lbq_context

        return {
            "cross_context": cross_context,
            "cross_mask": cross_mask,
            "lbq_context": lbq_context,
            "self_lbq_context": self_lbq_context,
        }

    @classmethod
    def backbone_key_set(cls, keys) -> set[str]:
        return {
            key
            for key in keys
            if not any(key.startswith(prefix) for prefix in cls.ACTION_BACKBONE_SKIP_PREFIXES)
        }

    @classmethod
    def from_pretrained(
        cls,
        action_dit_config: dict[str, Any],
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ) -> "ActionDiT":
        if action_dit_config is None:
            raise ValueError("`action_dit_config` is required for ActionDiT.from_pretrained().")
        if skip_dit_load_from_pretrain:
            logger.info(
                "Skipping ActionDiT pretrained load (`skip_dit_load_from_pretrain=True`); "
                "initializing action expert randomly and expecting checkpoint override."
            )
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        if not action_dit_pretrained_path:
            logger.info("No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.")
            return cls(**action_dit_config).to(device=device, dtype=torch_dtype)
        from pathlib import Path
        p = Path(action_dit_pretrained_path)
        if not p.is_absolute():
            p = Path(__file__).resolve().parents[4] / p
        action_dit_pretrained_path = str(p)
        if not os.path.isfile(action_dit_pretrained_path):
            raise FileNotFoundError(
                f"`action_dit_pretrained_path` does not exist: {action_dit_pretrained_path}"
            )

        action_cfg = dict(action_dit_config)
        action_expert = cls(**action_cfg).to(device=device, dtype=torch_dtype)
        action_state = action_expert.state_dict()
        expected_backbone_keys = cls.backbone_key_set(action_state.keys())

        payload = torch.load(action_dit_pretrained_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                f"Invalid action backbone payload type from {action_dit_pretrained_path}: {type(payload)}"
            )
        
        policy = payload.get("policy", {})
        if policy:
            logger.info(f"ActionDiT backbone payload policy: {policy}")

        meta = payload.get("meta")
        expected_meta = {
            "hidden_dim": int(action_cfg["hidden_dim"]),
            "ffn_dim": int(action_cfg["ffn_dim"]),
            "num_layers": int(action_cfg["num_layers"]),
            "num_heads": int(action_cfg["num_heads"]),
            "attn_head_dim": int(action_cfg["attn_head_dim"]),
            "text_dim": int(action_cfg["text_dim"]),
            "freq_dim": int(action_cfg["freq_dim"]),
            "eps": float(action_cfg["eps"]),
        }
        architecture = str(action_cfg.get("architecture", "full"))
        for key in cls.ACTION_BACKBONE_META_KEYS:
            if key not in meta:
                raise ValueError(f"`meta.{key}` missing in {action_dit_pretrained_path}")
            expected_value = expected_meta[key]
            got_value = meta[key]
            if key == "eps":
                if abs(float(got_value) - float(expected_value)) > 1e-12:
                    raise ValueError(
                        f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                        f"expected {expected_value}, got {got_value}"
                    )
            elif key == "num_layers" and architecture == "alternating_cross_self":
                if int(got_value) <= 0:
                    raise ValueError(
                        f"`meta.num_layers` must be positive, got {got_value}."
                    )
            elif int(got_value) != int(expected_value):
                raise ValueError(
                    f"`meta.{key}` mismatch in {action_dit_pretrained_path}: "
                    f"expected {expected_value}, got {got_value}"
                )

        backbone_state_dict = payload.get("backbone_state_dict")
        if not isinstance(backbone_state_dict, dict):
            raise ValueError(
                f"`backbone_state_dict` must be a dict in {action_dit_pretrained_path}, "
                f"got {type(backbone_state_dict)}"
            )

        merged_state = dict(action_state)
        if architecture == "alternating_cross_self":
            source_num_layers = int(meta["num_layers"])
            target_num_layers = int(action_cfg["num_layers"])
            if target_num_layers == 1:
                source_layers = [0]
            else:
                source_layers = [
                    round(idx * (source_num_layers - 1) / (target_num_layers - 1))
                    for idx in range(target_num_layers)
                ]
            source_keys = {}
            for key in expected_backbone_keys:
                if key.startswith("blocks."):
                    _, target_idx, suffix = key.split(".", 2)
                    source_key = f"blocks.{source_layers[int(target_idx)]}.{suffix}"
                else:
                    source_key = key
                source_keys[key] = source_key
            missing_keys = sorted(
                source_key
                for source_key in source_keys.values()
                if source_key not in backbone_state_dict
            )
            if missing_keys:
                raise ValueError(
                    "Alternating ActionDiT pretrained mapping is incomplete. "
                    f"Missing source keys: {missing_keys[:10]}"
                    f"{'...' if len(missing_keys) > 10 else ''}."
                )
            logger.info(
                "Mapping alternating ActionDiT layers from pretrained source: %s.",
                source_layers,
            )
            action_expert.pretrained_source_layers = source_layers
        else:
            provided_keys = set(backbone_state_dict.keys())
            missing_keys = sorted(expected_backbone_keys - provided_keys)
            unexpected_keys = sorted(provided_keys - expected_backbone_keys)
            if missing_keys or unexpected_keys:
                raise ValueError(
                    "Action backbone key mismatch in preprocessed payload. "
                    f"missing={missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}, "
                    f"unexpected={unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
                )
            source_keys = {key: key for key in expected_backbone_keys}

        for key, source_key in source_keys.items():
            value = backbone_state_dict[source_key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"`backbone_state_dict[{source_key}]` must be torch.Tensor in {action_dit_pretrained_path}, "
                    f"got {type(value)}"
                )
            target = merged_state[key]
            if tuple(value.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for `{key}` in {action_dit_pretrained_path}: "
                    f"expected {tuple(target.shape)}, got {tuple(value.shape)}"
                )
            merged_state[key] = value.to(device=target.device, dtype=target.dtype)

        action_expert.load_state_dict(merged_state, strict=True)
        logger.info(
            "Loaded ActionDiT backbone from %s (keys=%d; random_kept_prefixes=%s).",
            action_dit_pretrained_path,
            len(expected_backbone_keys),
            list(cls.ACTION_BACKBONE_SKIP_PREFIXES),
        )
        return action_expert.to(device=device, dtype=torch_dtype)

    def pre_dit(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if action_tokens.ndim != 3:
            raise ValueError(
                f"`action_tokens` must be 3D [B, T, action_dim], got shape {tuple(action_tokens.shape)}"
            )
        if action_tokens.shape[2] != self.action_dim:
            raise ValueError(
                f"`action_tokens` last dim must be {self.action_dim}, got {action_tokens.shape[2]}"
            )
        if timestep.ndim != 1:
            raise ValueError(f"`timestep` must be 1D [B] or [1], got shape {tuple(timestep.shape)}")
        if context is None or context.ndim != 3:
            raise ValueError(
                "Projected Action cross-attention `context` must be [B,L,D]."
            )
        if context.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Projected Action context dim must be {self.hidden_dim}, "
                f"got {context.shape[-1]}."
            )

        batch_size = action_tokens.shape[0]
        if context.shape[0] != batch_size:
            raise ValueError(
                f"Batch mismatch between action tokens and text context: {batch_size} vs {context.shape[0]}"
            )
        if timestep.shape[0] not in (1, batch_size):
            raise ValueError(
                f"`timestep` length must be 1 or batch_size({batch_size}), got {timestep.shape[0]}"
            )
        if timestep.shape[0] == 1 and batch_size > 1:
            if self.training:
                raise ValueError("During training, action timestep length must match batch_size.")
            timestep = timestep.expand(batch_size)

        if context_mask is None:
            context_mask = torch.ones(
                (batch_size, context.shape[1]), dtype=torch.bool, device=context.device
            )
        else:
            if context_mask.ndim != 2:
                raise ValueError(f"`context_mask` must be 2D [B, L], got shape {tuple(context_mask.shape)}")
            if context_mask.shape[0] != batch_size or context_mask.shape[1] != context.shape[1]:
                raise ValueError(
                    f"`context_mask` shape must match `context` shape [B, L], got {tuple(context_mask.shape)} vs {tuple(context.shape)}"
                )

        seq_len = action_tokens.shape[1]
        if seq_len > self.freqs.shape[0]:
            raise ValueError(
                f"Action token length {seq_len} exceeds RoPE cache {self.freqs.shape[0]}."
            )

        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        if self.position_embedding is not None:
            if seq_len > self.max_action_horizon:
                raise ValueError(
                    f"Action token length {seq_len} exceeds position embedding "
                    f"capacity {self.max_action_horizon}."
                )
            position_ids = torch.arange(seq_len, device=tokens.device)
            tokens = tokens + self.position_embedding(position_ids).unsqueeze(0).to(
                dtype=tokens.dtype
            )
        context_attn_mask = context_mask.unsqueeze(1).expand(-1, seq_len, -1)
        freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)

        return {
            "tokens": tokens,
            "freqs": freqs,
            "t": t,
            "t_mod": t_mod,
            "context": context,
            "context_mask": context_attn_mask,
            "meta": {
                "batch_size": batch_size,
                "seq_len": seq_len,
            },
        }

    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)

    def forward(
        self,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor] = None,
        lbq_hidden: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        conditioning = self.prepare_conditioning(
            text_state_context=context,
            text_state_mask=context_mask,
            lbq_hidden=lbq_hidden,
        )
        pre_state = self.pre_dit(
            action_tokens=action_tokens,
            timestep=timestep,
            context=conditioning["cross_context"],
            context_mask=conditioning["cross_mask"],
        )
        x = pre_state["tokens"]
        context = pre_state["context"]
        t_mod = pre_state["t_mod"]
        freqs = pre_state["freqs"]
        context_mask = pre_state["context_mask"]

        for block in self.blocks:
            block_kwargs = {"context_mask": context_mask}
            if self.architecture == "alternating_cross_self":
                block_kwargs["lbq_context"] = conditioning["self_lbq_context"]
            elif conditioning["self_lbq_context"] is not None:
                raise RuntimeError(
                    "Full ActionDiT mixed LBQ self-attention is executed by BridgeOfExperts; "
                    "use the BridgeWAM forward path instead of standalone ActionDiT."
                )
            x = gradient_checkpoint_forward(
                block,
                self.use_gradient_checkpointing,
                x,
                context,
                t_mod,
                freqs,
                **block_kwargs,
            )

        return self.post_dit(x, pre_state)
