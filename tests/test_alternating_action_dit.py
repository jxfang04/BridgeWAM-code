import tempfile
import unittest
from unittest.mock import patch

import torch

from bridgewam.models.wan22.action_dit import (
    ActionCrossAttentionBlock,
    ActionDiT,
    ActionMixSelfAttention,
    ActionSelfAttentionBlock,
)
from bridgewam.models.wan22.bridge_of_experts import BridgeOfExperts

from tests.test_latent_bridge_queries import _MiniExpert, _lbq_config


def _action_config(**overrides):
    config = {
        "action_dim": 2,
        "hidden_dim": 8,
        "ffn_dim": 16,
        "text_dim": 8,
        "freq_dim": 8,
        "eps": 1.0e-6,
        "num_heads": 2,
        "attn_head_dim": 4,
        "num_layers": 2,
        "lbq_dim": 8,
        "architecture": "alternating_cross_self",
        "conditioning_mode": "lbq_only",
        "add_pos_embed": True,
        "max_action_horizon": 16,
    }
    config.update(overrides)
    return config


class TestAlternatingActionDiT(unittest.TestCase):
    def test_requires_positive_even_depth(self):
        with self.assertRaisesRegex(ValueError, "even"):
            ActionDiT(**_action_config(num_layers=3))
        with self.assertRaisesRegex(ValueError, "positive"):
            ActionDiT(**_action_config(num_layers=0))

    def test_two_layers_are_cross_then_self_without_unused_attention(self):
        action = ActionDiT(**_action_config())

        self.assertEqual(action.block_types, ["cross", "self"])
        self.assertIsInstance(action.blocks[0], ActionCrossAttentionBlock)
        self.assertTrue(hasattr(action.blocks[0], "cross_attn"))
        self.assertFalse(hasattr(action.blocks[0], "self_attn"))
        self.assertIsInstance(action.blocks[1], ActionSelfAttentionBlock)
        self.assertTrue(hasattr(action.blocks[1], "self_attn"))
        self.assertFalse(hasattr(action.blocks[1], "cross_attn"))

    def test_position_embedding_distinguishes_action_horizon_slots(self):
        action = ActionDiT(**_action_config())
        pre = action.pre_dit(
            action_tokens=torch.zeros(1, 4, 2),
            timestep=torch.zeros(1),
            context=torch.zeros(1, 3, 8),
            context_mask=torch.ones(1, 3, dtype=torch.bool),
        )

        self.assertFalse(torch.equal(pre["tokens"][:, 0], pre["tokens"][:, 1]))

    def test_bridge_allows_30_style_video_depth_and_two_action_layers(self):
        action = ActionDiT(**_action_config())
        bridge = BridgeOfExperts(
            mixtures={
                "video": _MiniExpert(num_layers=4),
                "action": action,
            },
            checkpoint_attention=False,
            latent_bridge_queries=_lbq_config(
                preserve_direct_video_kv=False,
                freeze_video_expert=False,
                strict_training_scope=False,
            ),
        )
        cross_calls = []
        self_calls = []
        action.blocks[0].cross_attn.register_forward_hook(
            lambda *args: cross_calls.append(1)
        )
        action.blocks[1].self_attn.register_forward_hook(
            lambda *args: self_calls.append(1)
        )
        action_seq_len = 4
        output = bridge.forward_bridge_action(
            action_tokens=torch.randn(1, action_seq_len, 8),
            action_freqs=action.freqs[:action_seq_len].view(
                action_seq_len, 1, -1
            ),
            action_t_mod=torch.zeros(1, 6, 8),
            action_context_payload={
                "context": torch.randn(1, 3, 8),
                "mask": torch.ones(1, action_seq_len, 3, dtype=torch.bool),
            },
        )

        self.assertEqual(output.shape, (1, action_seq_len, 8))
        self.assertEqual(len(cross_calls), 1)
        self.assertEqual(len(self_calls), 1)
        self.assertEqual(bridge.video_num_layers, 4)
        self.assertEqual(bridge.action_num_layers, 2)

    def test_text_state_self_block_reads_action_plus_lbq_kv(self):
        attention = ActionMixSelfAttention(8, 4, 2, 1.0e-6)
        captured = {}

        def fake_attention(q, k, v, num_heads, ctx_mask=None):
            captured.update(
                q_len=q.shape[1],
                k_len=k.shape[1],
                v_len=v.shape[1],
                mask_shape=None if ctx_mask is None else tuple(ctx_mask.shape),
                num_heads=num_heads,
            )
            return torch.zeros_like(q)

        with patch(
            "bridgewam.models.wan22.action_dit.multihead_attention",
            side_effect=fake_attention,
        ):
            output = attention(
                torch.randn(1, 4, 8),
                freqs=torch.ones(4, 1, 2, dtype=torch.complex128),
                self_attn_mask=torch.ones(4, 4, dtype=torch.bool),
                lbq_context=torch.randn(1, 3, 8),
            )

        self.assertEqual(output.shape, (1, 4, 8))
        self.assertEqual(captured["q_len"], 4)
        self.assertEqual(captured["k_len"], 7)
        self.assertEqual(captured["v_len"], 7)
        self.assertEqual(captured["mask_shape"], (4, 7))

    def test_pretrained_mapping_uses_first_and_last_source_layers(self):
        source_config = _action_config(
            architecture="full",
            add_pos_embed=False,
            num_layers=4,
        )
        source = ActionDiT(**source_config)
        with torch.no_grad():
            source.blocks[0].cross_attn.q.weight.fill_(1.25)
            source.blocks[3].self_attn.q.weight.fill_(2.5)
        source_state = source.state_dict()
        backbone_state = {
            key: value
            for key, value in source_state.items()
            if key in ActionDiT.backbone_key_set(source_state.keys())
        }
        payload = {
            "meta": {
                "hidden_dim": 8,
                "ffn_dim": 16,
                "num_layers": 4,
                "num_heads": 2,
                "attn_head_dim": 4,
                "text_dim": 8,
                "freq_dim": 8,
                "eps": 1.0e-6,
            },
            "backbone_state_dict": backbone_state,
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save(payload, checkpoint.name)
            target = ActionDiT.from_pretrained(
                action_dit_config=_action_config(),
                action_dit_pretrained_path=checkpoint.name,
                device="cpu",
                torch_dtype=torch.float32,
            )

        self.assertEqual(target.pretrained_source_layers, [0, 3])
        self.assertTrue(
            torch.equal(
                target.blocks[0].cross_attn.q.weight,
                source.blocks[0].cross_attn.q.weight,
            )
        )
        self.assertTrue(
            torch.equal(
                target.blocks[1].self_attn.q.weight,
                source.blocks[3].self_attn.q.weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
