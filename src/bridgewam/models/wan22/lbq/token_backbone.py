from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LatentBridgeQueries(nn.Module):
    """Learned bridge tokens propagated through selected Video DiT blocks."""

    SUPPORTED_ATTENTION_MODES = {"bidirectional"}
    SUPPORTED_ROPE_MODES = {"identity"}

    def __init__(
        self,
        *,
        video_hidden_dim: int,
        num_video_layers: int,
        num_lbqs: int = 32,
        start_layer: int = 0,
        readout_layer: int = -1,
        lbq_attention: str = "bidirectional",
        lbq_rope_mode: str = "identity",
        eps: float = 1.0e-6,
    ):
        super().__init__()
        if int(num_lbqs) <= 0:
            raise ValueError(f"`num_lbqs` must be positive, got {num_lbqs}.")
        if not 0 <= int(start_layer) < int(num_video_layers):
            raise ValueError(
                f"`start_layer` must be in [0, {num_video_layers}), got {start_layer}."
            )

        requested_readout_layer = int(readout_layer)
        resolved_readout_layer = (
            requested_readout_layer
            if requested_readout_layer >= 0
            else int(num_video_layers) + requested_readout_layer
        )
        if not 0 <= resolved_readout_layer < int(num_video_layers):
            raise ValueError(
                "`readout_layer` must resolve into "
                f"[0, {num_video_layers}), got {requested_readout_layer}."
            )
        if resolved_readout_layer < int(start_layer):
            raise ValueError(
                "`readout_layer` must not precede `start_layer`: "
                f"{resolved_readout_layer} < {start_layer}."
            )
        if str(lbq_attention) not in self.SUPPORTED_ATTENTION_MODES:
            raise ValueError(
                f"`lbq_attention` must be one of {sorted(self.SUPPORTED_ATTENTION_MODES)}, "
                f"got {lbq_attention!r}."
            )
        if str(lbq_rope_mode) not in self.SUPPORTED_ROPE_MODES:
            raise ValueError(
                f"`lbq_rope_mode` must be one of {sorted(self.SUPPORTED_ROPE_MODES)}, "
                f"got {lbq_rope_mode!r}."
            )

        self.video_hidden_dim = int(video_hidden_dim)
        self.num_video_layers = int(num_video_layers)
        self.num_lbqs = int(num_lbqs)
        self.start_layer = int(start_layer)
        self.requested_readout_layer = requested_readout_layer
        self.readout_layer = resolved_readout_layer
        self.lbq_attention = str(lbq_attention)
        self.lbq_rope_mode = str(lbq_rope_mode)
        self.eps = float(eps)

        self.lbq_embeddings = nn.Parameter(
            torch.randn(1, self.num_lbqs, self.video_hidden_dim)
            / self.video_hidden_dim**0.5
        )

    def initial_lbq_tokens(self, reference: torch.Tensor) -> torch.Tensor:
        if reference.ndim != 3:
            raise ValueError(
                f"`reference` must be [B,S,D], got {tuple(reference.shape)}."
            )
        if reference.shape[-1] != self.video_hidden_dim:
            raise ValueError(
                f"Video hidden dim must be {self.video_hidden_dim}, "
                f"got {reference.shape[-1]}."
            )
        return self.lbq_embeddings.to(
            device=reference.device,
            dtype=reference.dtype,
        ).expand(reference.shape[0], -1, -1)

    def config_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "num_lbqs": self.num_lbqs,
            "start_layer": self.start_layer,
            "readout_layer": self.readout_layer,
            "lbq_attention": self.lbq_attention,
            "lbq_rope_mode": self.lbq_rope_mode,
            "eps": self.eps,
        }
