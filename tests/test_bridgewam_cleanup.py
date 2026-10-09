"""Behavioral regressions for shared LBQ prefill and pruned attention."""
from unittest.mock import patch

import pytest
import torch

from bridgewam.models.wan22.action_dit import ActionDiT
from bridgewam.models.wan22.bridge_of_experts import BridgeOfExperts
from tests.test_alternating_action_dit import _action_config
from tests.test_latent_bridge_queries import _MiniExpert, _lbq_config
from tests.test_lbq_spectral_regularization import make_model


def test_public_action_inference_prefills_once_and_reuses_identical_conditions():
    model = make_model()
    model.video_expert.fuse_vae_embedding_in_latents = True
    first = torch.randn(1, 2, 1, 2, 2)
    kwargs = dict(prompt=None, input_image=torch.zeros(1, 3, 16, 16), action_horizon=4,
                  context=torch.randn(1, 3, 8), context_mask=torch.ones(1, 3, dtype=torch.bool),
                  seed=82, num_inference_steps=3)
    with patch.object(model, "_encode_input_image_latents_tensor", return_value=first), \
         patch.object(model, "_prefill_video_for_action", wraps=model._prefill_video_for_action) as prefill, \
         patch.object(model, "_predict_action_noise_with_cache", wraps=model._predict_action_noise_with_cache) as step:
        output = model.infer_action(**kwargs)
    assert prefill.call_count == 1 and step.call_count == 3
    assert output["action"].shape == (4, 2)
    conditions = [call.kwargs["action_cross_context"] for call in step.call_args_list]
    assert all(condition is conditions[0] for condition in conditions)
    assert all("video_kv_cache" not in call.kwargs for call in step.call_args_list)


@pytest.mark.parametrize("coupling", ["none", "future_video_reads_lbq"])
@pytest.mark.parametrize("frames", [1, 3])
@pytest.mark.parametrize("checkpoint", [False, True])
def test_attention_pruning_respects_active_layer_interval(coupling, frames, checkpoint):
    config = _lbq_config(readout_layer=1, generation_coupling=coupling)
    config["start_layer"] = 1
    action = ActionDiT(**_action_config(architecture="full", num_layers=3))
    bridge = BridgeOfExperts({"video": _MiniExpert(3), "action": action}, checkpoint, latent_bridge_queries=config)
    tokens = torch.randn(1, frames * 2, 8, requires_grad=True)
    kwargs = dict(video_tokens=tokens, video_freqs=torch.ones(frames*2, 1, 2, dtype=torch.complex128),
                  video_t_mod=torch.randn(1, frames*2, 6, 8), video_context_payload=None,
                  video_attention_mask=torch.ones(frames*2, frames*2, dtype=torch.bool),
                  video_tokens_per_frame=2)
    with patch.object(bridge, "_mixed_attention", wraps=bridge._mixed_attention) as attend:
        video, lbq = bridge.forward_bridge_video(**kwargs)
        calls = [(c.kwargs["q_cat"].shape[1], c.kwargs["k_cat"].shape[1]) for c in attend.call_args_list]
    count = frames * 2
    expected = [(count, count), (count, count), (3, 5), (count, count)]
    if coupling == "future_video_reads_lbq" and frames > 1:
        expected = [(count, count), (2, count), (3, 5), (count-2, count+3), (count, count)]
    assert calls == expected
    (video.square().mean() + lbq.square().mean()).backward()
    assert torch.isfinite(tokens.grad).all()
    assert bridge.latent_bridge_queries.lbq_embeddings.grad.norm() > 0


def test_boe_logs_effective_route(caplog):
    caplog.set_level("INFO")
    model = make_model()
    caplog.clear()
    BridgeOfExperts(model.bridge.mixtures, False, latent_bridge_queries=model.bridge.latent_bridge_queries_config())
    assert "Initialized BridgeOfExperts" in caplog.text
    assert "full_expert_finetune" in caplog.text
    assert "no direct Video K/V cache" in caplog.text
    assert "Initialized MoT" not in caplog.text
    assert "Action direct Video K/V layers" not in caplog.text
