import tempfile
import unittest

import torch
from torch import nn

from bridgewam.models.wan22.action_dit import ActionDiT
from bridgewam.models.wan22.ablation_bridgewam.factory import _validate_topology
from bridgewam.models.wan22.ablation_bridgewam.models import (
    AblationCheckpointMixin,
    FrozenVideoBridgeWAM,
)
from bridgewam.models.wan22.ablation_bridgewam.backbones import (
    FrozenVideoBridge,
    FullVideoBridge,
    JointAttentionBridge,
)

from tests.test_latent_bridge_queries import (
    _MiniExpert,
    _action_config,
    _lbq_config,
)


def _alternating_action(mode: str = "lbq_only") -> ActionDiT:
    return ActionDiT(
        **_action_config(
            mode,
            architecture="alternating_cross_self",
            num_layers=2,
        )
    )


class TestFrozenVideoAblation(unittest.TestCase):
    def test_training_scope_is_full_action_plus_lbq(self):
        video = _MiniExpert(num_layers=2)
        action = _alternating_action()
        bridge = FrozenVideoBridge(
            mixtures={"video": video, "action": action},
            checkpoint_attention=False,
            latent_bridge_queries=_lbq_config(
                preserve_direct_video_kv=False,
                generation_coupling="none",
                freeze_video_expert=True,
                strict_training_scope=True,
            ),
            ablation_type="frozen_video_action_reader",
        )

        bridge.configure_lbq_trainable_parameters()

        self.assertTrue(all(not p.requires_grad for p in video.parameters()))
        self.assertTrue(all(p.requires_grad for p in action.parameters()))
        self.assertTrue(bridge.latent_bridge_queries.lbq_embeddings.requires_grad)
        grouped = {
            id(parameter)
            for group in bridge.lbq_parameter_groups(1.0e-4)
            for parameter in group["params"]
        }
        actual = {id(p) for p in bridge.parameters() if p.requires_grad}
        self.assertEqual(grouped, actual)

    def test_query_has_action_gradient_but_no_video_loss_gradient(self):
        video = _MiniExpert(num_layers=2)
        action = _alternating_action()
        bridge = FrozenVideoBridge(
            mixtures={"video": video, "action": action},
            checkpoint_attention=False,
            latent_bridge_queries=_lbq_config(
                preserve_direct_video_kv=False,
                generation_coupling="none",
                freeze_video_expert=True,
                strict_training_scope=True,
            ),
            ablation_type="frozen_video_action_reader",
        )
        bridge.configure_lbq_trainable_parameters()
        video_tokens, lbq_tokens = bridge.forward_bridge_video(
            video_tokens=torch.randn(1, 4, 8, requires_grad=True),
            video_freqs=torch.ones(4, 1, 2, dtype=torch.complex128),
            video_t_mod=torch.zeros(1, 4, 6, 8),
            video_context_payload=None,
            video_attention_mask=torch.ones(4, 4, dtype=torch.bool),
            video_tokens_per_frame=2,
        )
        lbq_embeddings = bridge.latent_bridge_queries.lbq_embeddings
        video_grad = torch.autograd.grad(
            video_tokens.square().mean(),
            lbq_embeddings,
            retain_graph=True,
            allow_unused=True,
        )[0]
        self.assertIsNone(video_grad)

        conditioning = action.prepare_conditioning(
            text_state_context=None,
            text_state_mask=None,
            lbq_hidden=lbq_tokens,
        )
        action_pre = action.pre_dit(
            action_tokens=torch.randn(1, 4, 2),
            timestep=torch.zeros(1),
            context=conditioning["cross_context"],
            context_mask=conditioning["cross_mask"],
        )
        action_hidden = bridge.forward_bridge_action(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
        )
        action.post_dit(action_hidden, action_pre).square().mean().backward()

        self.assertIsNotNone(lbq_embeddings.grad)
        self.assertGreater(float(lbq_embeddings.grad.abs().sum()), 0.0)
        self.assertTrue(all(parameter.grad is None for parameter in video.parameters()))
        self.assertTrue(
            any(parameter.grad is not None for parameter in action.parameters())
        )

    def test_release_loader_only_overwrites_video_and_proprio(self):
        holder = type("Holder", (), {})()
        holder.video_expert = nn.Linear(3, 4)
        holder.proprio_encoder = nn.Linear(2, 5)
        expected_video = {
            key: torch.full_like(value, 2.0)
            for key, value in holder.video_expert.state_dict().items()
        }
        expected_proprio = {
            key: torch.full_like(value, 3.0)
            for key, value in holder.proprio_encoder.state_dict().items()
        }
        payload = {
            "mot": {
                f"mixtures.video.{key}": value
                for key, value in expected_video.items()
            },
            "proprio_encoder": expected_proprio,
        }
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            torch.save(payload, checkpoint.name)
            FrozenVideoBridgeWAM.load_release_video_and_proprio(
                holder, checkpoint.name
            )

        for key, value in holder.video_expert.state_dict().items():
            self.assertTrue(torch.equal(value, expected_video[key]))
        for key, value in holder.proprio_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, expected_proprio[key]))


class TestIDMLBQScope(unittest.TestCase):
    def test_full_video_lbqs_read_all_frames_and_future_reads_lbq(self):
        video = _MiniExpert(num_layers=1)
        action = ActionDiT(**_action_config("lbq_only", num_layers=1))
        bridge = FullVideoBridge(
            mixtures={"video": video, "action": action},
            checkpoint_attention=False,
            latent_bridge_queries=_lbq_config(
                preserve_direct_video_kv=False,
                readout_layer=0,
                generation_coupling="future_video_reads_lbq",
                freeze_video_expert=False,
                strict_training_scope=False,
            ),
            ablation_type="idm",
        )
        calls = []
        bridge._mixed_attention = lambda q_cat, k_cat, v_cat, attention_mask: (
            calls.append((q_cat.shape[1], k_cat.shape[1]))
            or torch.zeros_like(q_cat)
        )
        bridge.forward_video_with_full_video_lbqs(
            video_tokens=torch.randn(1, 4, 8),
            video_freqs=torch.ones(4, 1, 2, dtype=torch.complex128),
            video_t_mod=torch.zeros(1, 4, 6, 8),
            video_context_payload=None,
            video_attention_mask=torch.ones(4, 4, dtype=torch.bool),
            video_tokens_per_frame=2,
        )

        self.assertEqual(calls, [(2, 4), (3, 7), (2, 7)])


class TestJointLBQAblation(unittest.TestCase):
    def test_joint_mask_preserves_only_video_internal_causality(self):
        video_mask = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        mask = JointAttentionBridge.build_joint_attention_mask(
            video_attention_mask=video_mask,
            num_lbq_tokens=3,
            action_seq_len=2,
        )

        self.assertEqual(mask.shape, (9, 9))
        self.assertTrue(torch.equal(mask[:4, :4], video_mask))
        self.assertTrue(mask[:4, 4:].all())
        self.assertTrue(mask[4:, :].all())

    def test_joint_forward_updates_all_three_streams_each_layer(self):
        video = _MiniExpert(num_layers=2)
        action_config = _action_config(
            "text_state", architecture="full", num_layers=2
        )
        action_config["lbq_dim"] = None
        action = ActionDiT(**action_config)
        bridge = JointAttentionBridge(
            mixtures={"video": video, "action": action},
            latent_bridge_queries=_lbq_config(
                preserve_direct_video_kv=False,
                injection_mode="text_state",
                readout_layer=-1,
                freeze_video_expert=False,
                strict_training_scope=False,
            ),
            checkpoint_attention=False,
            ablation_type="joint",
        )
        calls = []
        bridge._mixed_attention = lambda q_cat, k_cat, v_cat, attention_mask: (
            calls.append((q_cat.shape[1], tuple(attention_mask.shape)))
            or torch.zeros_like(q_cat)
        )
        video_out, action_out, lbq_out = bridge.forward_joint_with_lbqs(
            video_tokens=torch.randn(1, 4, 8),
            action_tokens=torch.randn(1, 2, 8),
            video_freqs=torch.ones(4, 1, 2, dtype=torch.complex128),
            action_freqs=torch.ones(2, 1, 2, dtype=torch.complex128),
            video_t_mod=torch.zeros(1, 4, 6, 8),
            action_t_mod=torch.zeros(1, 6, 8),
            video_context_payload=None,
            action_context_payload=None,
            video_attention_mask=torch.ones(4, 4, dtype=torch.bool),
        )

        self.assertEqual(video_out.shape, (1, 4, 8))
        self.assertEqual(action_out.shape, (1, 2, 8))
        self.assertEqual(lbq_out.shape, (1, 3, 8))
        self.assertEqual(calls, [(9, (9, 9)), (9, (9, 9))])

    def test_joint_requires_equal_full_depth(self):
        with self.assertRaisesRegex(
            ValueError, "different depths|equal Video and Action depth"
        ):
            JointAttentionBridge(
                mixtures={
                    "video": _MiniExpert(num_layers=2),
                    "action": ActionDiT(
                        **_action_config(
                            "text_state", architecture="full", num_layers=1
                        )
                    ),
                },
                latent_bridge_queries=_lbq_config(
                    preserve_direct_video_kv=False,
                    injection_mode="text_state",
                    readout_layer=-1,
                    freeze_video_expert=False,
                    strict_training_scope=False,
                ),
                checkpoint_attention=False,
                ablation_type="joint",
            )


class TestAblationTopologyValidation(unittest.TestCase):
    def _validate(self, kind, mode, coupling, architecture, action_layers, freeze):
        _validate_topology(
            ablation_type=kind,
            video_config={"num_layers": 30},
            action_config={
                "architecture": architecture,
                "num_layers": action_layers,
            },
            lbq_config={
                "enabled": True,
                "num_lbqs": 32,
                "start_layer": 0,
                "readout_layer": -1,
                "lbq_attention": "bidirectional",
                "lbq_rope_mode": "identity",
                "preserve_direct_video_kv": False,
                "injection_mode": mode,
                "generation_coupling": coupling,
                "freeze_video_expert": freeze,
            },
        )

    def test_all_four_locked_topologies(self):
        self._validate(
            "frozen_video_action_reader",
            "lbq_only",
            "none",
            "alternating_cross_self",
            2,
            True,
        )
        self._validate(
            "lbq_kv_mot",
            "text_state",
            "future_video_reads_lbq",
            "alternating_cross_self",
            2,
            False,
        )
        self._validate(
            "idm",
            "lbq_only",
            "future_video_reads_lbq",
            "alternating_cross_self",
            2,
            False,
        )
        self._validate("joint", "text_state", "none", "full", 30, False)

    def test_wrong_route_fails_before_model_construction(self):
        with self.assertRaisesRegex(ValueError, "Invalid topology"):
            self._validate(
                "lbq_kv_mot",
                "lbq_only",
                "future_video_reads_lbq",
                "alternating_cross_self",
                2,
                False,
            )


class _CheckpointBase:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.bridge = _CheckpointBridge()
        self.action_expert = _CheckpointAction()
        self.proprio_encoder = None
        self.torch_dtype = torch.float32

    def load_checkpoint(self, path, optimizer=None):
        del optimizer
        return torch.load(path, map_location="cpu")


class _CheckpointBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    @staticmethod
    def latent_bridge_queries_config():
        return {"enabled": True}


class _CheckpointAction:
    pretrained_source_layers = None

    @staticmethod
    def architecture_config():
        return {"architecture": "test"}


class _CheckpointTagged(AblationCheckpointMixin, _CheckpointBase):
    pass


class TestAblationCheckpointIdentity(unittest.TestCase):
    def test_checkpoint_identity_is_required_and_strict(self):
        identity = {"type": "idm", "video_cond_noise_prob": 0.5}
        source = _CheckpointTagged(ablation_config=identity)
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            source.save_checkpoint(checkpoint.name, step=12)
            self.assertEqual(source.load_checkpoint(checkpoint.name)["step"], 12)

            mismatched = _CheckpointTagged(
                ablation_config={"type": "idm", "video_cond_noise_prob": 0.25}
            )
            with self.assertRaisesRegex(ValueError, "ablation checkpoint/config"):
                mismatched.load_checkpoint(checkpoint.name)


if __name__ == "__main__":
    unittest.main()
