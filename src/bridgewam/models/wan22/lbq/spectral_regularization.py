"""Training-only, per-example spectral diversity of LBQ hidden states."""

import math

import torch
import torch.nn.functional as F


def lbq_spectral_diversity(
    hidden: torch.Tensor, *, diagnostics: bool = False
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return log(N) - spectral entropy, without mixing examples or adding jitter.

    Rows are normalized before forming an N x N Gram matrix. Zero rows stay
    zero: an all-zero representation has rank zero, not artificial diversity.
    The eigensolver always uses FP32, including under mixed-precision training.
    """
    if hidden.ndim != 3 or any(size == 0 for size in hidden.shape):
        raise ValueError("LBQ spectral input must be nonempty [batch, tokens, hidden_dim].")
    if not hidden.is_floating_point():
        raise ValueError("LBQ spectral input must be floating point.")
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        values = hidden.float()
        if not torch.isfinite(values).all():
            raise ValueError("LBQ spectral input contains non-finite hidden states.")
        count = values.shape[1]
        normalized = F.normalize(values, dim=-1, eps=1.0e-6)
        gram = normalized @ normalized.transpose(-1, -2)
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        energy = eigenvalues.sum(dim=-1, keepdim=True)
        probabilities = eigenvalues / energy.clamp_min(1.0e-12)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(-1)
        loss = (math.log(count) - entropy).mean()
        if count == 1:
            loss = values.sum() * 0.0

        metrics = {}
        if diagnostics:
            with torch.no_grad():
                norms = values.norm(dim=-1)
                effective_rank = torch.where(energy.squeeze(-1) > 0, entropy.exp(), 0.0)
                off_diagonal = (
                    (gram.sum((-1, -2)) - gram.diagonal(dim1=-2, dim2=-1).sum(-1))
                    / (count * (count - 1))
                    if count > 1 else torch.zeros_like(entropy)
                )
                metrics = {
                    "effective_rank": effective_rank.mean().item(),
                    "top1_mass": probabilities[:, -1].mean().item(),
                    "top4_mass": probabilities[:, -min(4, count):].sum(-1).mean().item(),
                    "token_norm": norms.mean().item(),
                    "low_norm_fraction": (norms < 1.0e-6).float().mean().item(),
                    "mean_cosine": off_diagonal.mean().item(),
                }
    return loss, metrics
