import unittest

import torch
from torch import nn

from bridgewam.models.wan22.action_dit import ActionDiT
from bridgewam.models.wan22.bridge_of_experts import BridgeOfExperts
from bridgewam.models.wan22.wan_video_dit import DiTBlock
from bridgewam.models.wan22.lbq import LatentBridgeQueries


class _MiniExpert(nn.Module):
    def __init__(self, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = 8
        self.text_dim = 8
        self.action_dim = 2
        self.num_heads = 2
        self.attn_head_dim = 4
        self.use_gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=8,
                    attn_head_dim=4,
                    num_heads=2,
                    ffn_dim=16,
                )
                for _ in range(num_layers)
            ]
        )


def _action_config(
    conditioning_mode: str = "lbq_only",
    *,
    architecture: str = "full",
    num_layers: int = 2,
):
    return {
        "action_dim": 2,
        "hidden_dim": 8,
        "ffn_dim": 16,
        "text_dim": 12,
        "freq_dim": 8,
        "eps": 1.0e-6,
        "num_heads": 2,
        "attn_head_dim": 4,
        "num_layers": num_layers,
        "lbq_dim": 8,
        "architecture": architecture,
        "conditioning_mode": conditioning_mode,
        "add_pos_embed": architecture == "alternating_cross_self",
        "max_action_horizon": 16,
    }


def _lbq_config(
    *,
    preserve_direct_video_kv: bool = False,
    injection_mode: str = "lbq_only",
    readout_layer: int = -1,
    generation_coupling: str = "none",
    freeze_video_expert: bool = True,
    strict_training_scope: bool = True,
):
    return {
        "enabled": True,
        "injection_mode": injection_mode,
        "num_lbqs": 3,
        "start_layer": 0,
        "readout_layer": readout_layer,
        "generation_coupling": generation_coupling,
        "lbq_attention": "bidirectional",
        "lbq_rope_mode": "identity",
        "preserve_direct_video_kv": preserve_direct_video_kv,
        "freeze_video_expert": freeze_video_expert,
        "train_action_cross_attention": True,
        "action_cross_attention_lr_scale": 1.0,
        "strict_training_scope": strict_training_scope,
    }


class TestLatentBridgeQueries(unittest.TestCase):
    def test_module_contains_only_learned_query_tokens(self):
        module = LatentBridgeQueries(
            video_hidden_dim=8,
            num_video_layers=4,
            num_lbqs=3,
            start_layer=1,
            readout_layer=-2,
        )

        self.assertEqual(module.readout_layer, 2)
        self.assertEqual(list(dict(module.named_parameters())), ["lbq_embeddings"])
        self.assertEqual(list(dict(module.named_modules())), [""])
        self.assertEqual(module.config_dict()["readout_layer"], 2)
        output = module.initial_lbq_tokens(torch.zeros(2, 4, 8))
        self.assertEqual(output.shape, (2, 3, 8))

    def test_readout_layer_validation(self):
        with self.assertRaisesRegex(ValueError, "must not precede"):
            LatentBridgeQueries(
                video_hidden_dim=8,
                num_video_layers=4,
                start_layer=2,
                readout_layer=1,
            )
        with self.assertRaisesRegex(ValueError, "must resolve"):
            LatentBridgeQueries(
                video_hidden_dim=8,
                num_video_layers=4,
                readout_layer=-5,
            )

    def test_text_projector_is_initialized_in_every_conditioning_mode(self):
        lbq_only = ActionDiT(**_action_config("lbq_only"))
        text_state_lbq = ActionDiT(**_action_config("text_state_lbq"))
        text_state = ActionDiT(**_action_config("text_state"))

        self.assertIsNotNone(lbq_only.text_embedding)
        self.assertIsNotNone(lbq_only.lbq_embedding)
        self.assertIsNotNone(text_state_lbq.text_embedding)
        self.assertIsNotNone(text_state_lbq.lbq_embedding)
        self.assertIsNotNone(text_state.text_embedding)
        self.assertIsNotNone(text_state.lbq_embedding)
        self.assertTrue(
            any(key.startswith("text_embedding.") for key in lbq_only.state_dict())
        )

    def test_conditioning_modes_preserve_effective_random_initialization(self):
        state_by_mode = {}
        with torch.random.fork_rng():
            for mode in ("lbq_only", "text_state_lbq", "text_state"):
                torch.manual_seed(42)
                action = ActionDiT(**_action_config(mode))
                state_by_mode[mode] = {
                    key: value.detach().clone()
                    for key, value in action.state_dict().items()
                }

        reference = state_by_mode["lbq_only"]
        for mode in ("text_state_lbq", "text_state"):
            self.assertEqual(reference.keys(), state_by_mode[mode].keys())
            for key, value in reference.items():
                self.assertTrue(
                    torch.equal(value, state_by_mode[mode][key]),
                    msg=f"Initialization differs for {mode}:{key}",
                )

    def test_three_conditioning_modes_route_exact_context_sources(self):
        text = torch.randn(2, 5, 12)
        text_mask = torch.tensor(
            [[True, True, False, True, True], [True, False, True, True, True]]
        )
        lbq = torch.randn(2, 3, 8)

        for mode in ("lbq_only", "text_state_lbq", "text_state"):
            with self.subTest(mode=mode):
                action = ActionDiT(**_action_config(mode))
                text_calls = []
                hook = None
                if mode == "lbq_only":
                    hook = action.text_embedding.register_forward_hook(
                        lambda *args: text_calls.append(1)
                    )
                result = action.prepare_conditioning(
                    text_state_context=text,
                    text_state_mask=text_mask,
                    lbq_hidden=lbq,
                )
                if hook is not None:
                    hook.remove()
                if mode == "lbq_only":
                    self.assertEqual(text_calls, [])
                    self.assertEqual(result["cross_context"].shape[1], 3)
                    self.assertIsNone(result["self_lbq_context"])
                elif mode == "text_state_lbq":
                    self.assertEqual(result["cross_context"].shape[1], 8)
                    self.assertTrue(
                        torch.equal(result["cross_mask"][:, :5], text_mask)
                    )
                    self.assertIsNone(result["self_lbq_context"])
                else:
                    self.assertEqual(result["cross_context"].shape[1], 5)
                    self.assertTrue(torch.equal(result["cross_mask"], text_mask))
                    self.assertEqual(result["self_lbq_context"].shape, (2, 3, 8))

    def test_removed_modes_are_rejected(self):
        for mode in ("state_lbq", "text_lbq", "replace_action_context"):
            with self.subTest(mode=mode), self.assertRaisesRegex(
                ValueError, "conditioning_mode"
            ):
                ActionDiT(**_action_config(mode))

    def test_video_lbq_tokens_are_queries_and_keys_values(self):
        video = _MiniExpert(num_layers=3)
        action = ActionDiT(**_action_config("lbq_only", num_layers=3))
        bridge = BridgeOfExperts(
            mixtures={"video": video, "action": action},
            checkpoint_attention=False,
            latent_bridge_queries=_lbq_config(readout_layer=1),
        )
        attention_shapes = []

        def fake_attention(q_cat, k_cat, v_cat, attention_mask):
            attention_shapes.append(
                (q_cat.shape[1], k_cat.shape[1], v_cat.shape[1], attention_mask.shape)
            )
            return torch.zeros_like(q_cat)

        bridge._mixed_attention = fake_attention
        video_tokens = torch.randn(1, 4, 8, requires_grad=True)
        _, lbq_tokens = bridge.forward_bridge_video(
            video_tokens=video_tokens,
            video_freqs=torch.ones(4, 1, 2, dtype=torch.complex128),
            video_t_mod=torch.zeros(1, 4, 6, 8),
            video_context_payload=None,
            video_attention_mask=torch.ones(4, 4, dtype=torch.bool),
            video_tokens_per_frame=2,
        )

        self.assertEqual(lbq_tokens.shape, (1, 3, 8))
        # At active layers LBQ Q reads first-frame Video K/V plus LBQ K/V.
        self.assertEqual(
            [shape[:3] for shape in attention_shapes],
            [(4, 4, 4), (3, 5, 5), (4, 4, 4), (3, 5, 5), (4, 4, 4)],
        )
        lbq_tokens.square().mean().backward()
        self.assertIsNotNone(bridge.latent_bridge_queries.lbq_embeddings.grad)

    def test_prefill_without_direct_video_kv_stops_at_readout(self):
        video = _MiniExpert(num_layers=4)
        action = ActionDiT(
            **_action_config(
                "lbq_only",
                architecture="alternating_cross_self",
                num_layers=2,
            )
        )
        bridge = BridgeOfExperts(
            mixtures={"video": video, "action": action},
            checkpoint_attention=False,
            latent_bridge_queries=_lbq_config(
                preserve_direct_video_kv=False,
                readout_layer=1,
            ),
        )
        calls = []
        bridge._mixed_attention = lambda q_cat, k_cat, v_cat, attention_mask: (
            calls.append((q_cat.shape[1], k_cat.shape[1])) or torch.zeros_like(q_cat)
        )

        lbq_tokens = bridge.prefill_bridge(
            video_tokens=torch.randn(1, 4, 8),
            video_freqs=torch.ones(4, 1, 2, dtype=torch.complex128),
            video_t_mod=torch.zeros(1, 4, 6, 8),
            video_context_payload=None,
            video_attention_mask=torch.ones(4, 4, dtype=torch.bool),
            video_tokens_per_frame=2,
        )

        self.assertEqual(lbq_tokens.shape, (1, 3, 8))
        self.assertEqual(calls, [(4, 4), (3, 5), (4, 4), (3, 5)])

    def test_future_video_queries_read_lbq_kv_and_backpropagate(self):
        video = _MiniExpert(num_layers=1)
        action = ActionDiT(**_action_config("lbq_only", num_layers=1))
        bridge = BridgeOfExperts(
            mixtures={"video": video, "action": action},
            checkpoint_attention=False,
            latent_bridge_queries=_lbq_config(
                readout_layer=0,
                generation_coupling="future_video_reads_lbq",
            ),
        )
        calls = []

        def fake_attention(q_cat, k_cat, v_cat, attention_mask):
            calls.append((q_cat.shape[1], k_cat.shape[1]))
            if q_cat.shape[1] == 2 and k_cat.shape[1] == 7:
                return v_cat[:, -3:].mean(dim=1, keepdim=True).expand(-1, 2, -1)
            return torch.zeros_like(q_cat)

        bridge._mixed_attention = fake_attention
        video_output, _ = bridge.forward_bridge_video(
            video_tokens=torch.randn(1, 4, 8),
            video_freqs=torch.ones(4, 1, 2, dtype=torch.complex128),
            video_t_mod=torch.zeros(1, 4, 6, 8),
            video_context_payload=None,
            video_attention_mask=torch.ones(4, 4, dtype=torch.bool),
            video_tokens_per_frame=2,
        )

        self.assertEqual(calls, [(2, 4), (3, 5), (2, 7)])
        video_output[:, 2:].square().mean().backward()
        self.assertIsNotNone(bridge.latent_bridge_queries.lbq_embeddings.grad)


    def test_legacy_action_hidden_metadata_maps_without_connector(self):
        normalized = BridgeOfExperts.normalize_latent_bridge_queries_checkpoint_config(
            {
                "injection_mode": "lbq_only",
                "conditioning_adapter": "action_hidden_embedding",
                "connector_input_layer": -2,
                "connector_dim": 1024,
            },
            num_layers=30,
        )
        self.assertEqual(normalized["readout_layer"], 28)
        self.assertNotIn("conditioning_adapter", normalized)
        self.assertNotIn("connector_dim", normalized)

        with self.assertRaisesRegex(ValueError, "Connector-based"):
            BridgeOfExperts.normalize_latent_bridge_queries_checkpoint_config(
                {"conditioning_adapter": "connector_context"},
                num_layers=30,
            )


if __name__ == "__main__":
    unittest.main()
