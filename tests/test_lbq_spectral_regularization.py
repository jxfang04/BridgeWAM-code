import math
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from bridgewam.models.wan22.factory import create_bridgewam

from bridgewam.models.wan22.action_dit import ActionDiT
from bridgewam.models.wan22.bridgewam import BridgeWAM
from bridgewam.models.wan22.bridge_of_experts import BridgeOfExperts
from bridgewam.models.wan22.wan_video_dit import WanVideoDiT
from bridgewam.models.wan22.lbq.spectral_regularization import lbq_spectral_diversity
from tests.test_alternating_action_dit import _action_config
from tests.test_latent_bridge_queries import _lbq_config


ROOT = Path(__file__).resolve().parents[1]
TASK = "libero_uncond_2cam224_lbqs_spectral_2layer_fullfinetune_1e-4"
BASE_TASK = "libero_uncond_2cam224_lbqs_only_2layer_alternating_cross_self_fullfinetune_1e-4"


def make_model(weight=0.0, diagnostics=False):
    video = WanVideoDiT(
        hidden_dim=8, in_dim=2, ffn_dim=16, out_dim=2, text_dim=8,
        freq_dim=8, eps=1e-6, patch_size=(1, 1, 1), num_heads=2,
        attn_head_dim=4, num_layers=2, has_image_input=False,
        seperated_timestep=True,
        video_attention_mask_mode="first_frame_causal",
    )
    action = ActionDiT(**_action_config())
    lbq_config = _lbq_config(
        preserve_direct_video_kv=False, generation_coupling="future_video_reads_lbq",
        freeze_video_expert=False, strict_training_scope=False,
    )
    bridge = BridgeOfExperts(
        mixtures={"video": video, "action": action},
        checkpoint_attention=False, latent_bridge_queries=lbq_config,
    )
    return BridgeWAM(
        video, action, bridge, nn.Identity(), text_dim=8, proprio_dim=2,
        loss_lambda_lbq_spectral=weight, lbq_spectral_diagnostics=diagnostics,
    )


def inputs(model):
    latents = torch.randn(2, 2, 2, 2, 2)
    return {
        "input_latents": latents,
        "first_frame_latents": latents[:, :, :1],
        "context": model.proprio_encoder(torch.randn(2, 3, 2)),
        "context_mask": torch.ones(2, 3, dtype=torch.bool),
        "action": torch.randn(2, 4, 2),
        "action_is_pad": None, "image_is_pad": None,
        "fuse_vae_embedding_in_latents": True,
    }


class TestSpectralDiversity(unittest.TestCase):
    def test_orthogonal_and_collapsed(self):
        loss, stats = lbq_spectral_diversity(torch.eye(8)[None], diagnostics=True)
        self.assertAlmostEqual(loss.item(), 0, places=6)
        self.assertAlmostEqual(stats["effective_rank"], 8, places=5)
        loss, stats = lbq_spectral_diversity(torch.ones(1, 8, 16), diagnostics=True)
        self.assertAlmostEqual(loss.item(), math.log(8), places=5)
        self.assertAlmostEqual(stats["effective_rank"], 1, places=5)
        self.assertAlmostEqual(stats["top1_mass"], 1, places=5)

    def test_token_scale_and_permutation_invariance(self):
        hidden = torch.randn(2, 5, 12)
        original, _ = lbq_spectral_diversity(hidden)
        scaled, _ = lbq_spectral_diversity(hidden * torch.arange(1, 6)[None, :, None])
        permuted, _ = lbq_spectral_diversity(hidden[:, [3, 0, 2, 4, 1]])
        torch.testing.assert_close(original, scaled, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(original, permuted, atol=1e-6, rtol=1e-5)

    def test_batch_is_not_flattened(self):
        hidden = torch.stack((torch.eye(4), torch.ones(4, 4)))
        batched, _ = lbq_spectral_diversity(hidden)
        individual = torch.stack([lbq_spectral_diversity(x[None])[0] for x in hidden]).mean()
        torch.testing.assert_close(batched, individual)
        self.assertAlmostEqual(batched.item(), math.log(4) / 2, places=5)

    def test_zero_singleton_and_rank_deficient_have_finite_gradients(self):
        for hidden in (torch.zeros(2, 4, 8), torch.randn(2, 1, 8),
                       torch.ones(2, 4, 8), torch.eye(4)[None]):
            with self.subTest(shape=hidden.shape):
                hidden.requires_grad_()
                loss, stats = lbq_spectral_diversity(hidden, diagnostics=True)
                loss.backward()
                self.assertTrue(torch.isfinite(hidden.grad).all())
                self.assertTrue(math.isfinite(loss.item()))
                if hidden.detach().count_nonzero() == 0:
                    self.assertEqual(stats["effective_rank"], 0)
                    self.assertEqual(stats["low_norm_fraction"], 1)
                if hidden.shape[1] == 1:
                    self.assertEqual(loss.item(), 0)

    def test_fp32_eigensolver_under_autocast(self):
        hidden = torch.randn(2, 5, 16, dtype=torch.bfloat16, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss, _ = lbq_spectral_diversity(hidden)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(hidden.grad.float().norm().item(), 0)

    def test_invalid_inputs_fail(self):
        for hidden in (torch.zeros(2, 3), torch.zeros(0, 3, 8),
                       torch.ones(1, 3, 8, dtype=torch.long),
                       torch.full((1, 3, 8), float("nan"))):
            with self.assertRaises(ValueError):
                lbq_spectral_diversity(hidden)

    def test_function_does_not_consume_rng(self):
        hidden = torch.randn(2, 3, 8)
        state = torch.get_rng_state().clone()
        lbq_spectral_diversity(hidden, diagnostics=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


class TestTrainingIntegration(unittest.TestCase):
    def test_initialization_state_and_parameter_keys_unchanged(self):
        torch.manual_seed(42)
        baseline = make_model()
        state = torch.get_rng_state().clone()
        torch.manual_seed(42)
        regularized = make_model(1e-4, True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(baseline.state_dict().keys(), regularized.state_dict().keys())
        for key, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, regularized.state_dict()[key]), key)
        self.assertEqual(
            [n for n, p in baseline.named_parameters() if p.requires_grad],
            [n for n, p in regularized.named_parameters() if p.requires_grad],
        )
        regularized.load_state_dict(baseline.state_dict(), strict=True)

    def test_zero_weight_diagnostics_preserves_loss_gradients_and_rng(self):
        model = make_model()
        snapshots = []
        for diagnostics in (False, True):
            model.zero_grad(set_to_none=True)
            model.lbq_spectral_diagnostics = diagnostics
            torch.manual_seed(123)
            batch = inputs(model)
            with patch.object(model, "build_inputs", return_value=batch):
                loss, metrics = model.training_loss({})
            loss.backward()
            snapshots.append((loss.detach(), {
                n: None if p.grad is None else p.grad.clone()
                for n, p in model.named_parameters()
            }, torch.get_rng_state().clone()))
            self.assertEqual("lbq_readout_effective_rank" in metrics, diagnostics)
        self.assertTrue(torch.equal(snapshots[0][0], snapshots[1][0]))
        self.assertTrue(torch.equal(snapshots[0][2], snapshots[1][2]))
        for name, grad in snapshots[0][1].items():
            other = snapshots[1][1][name]
            self.assertTrue(other is None if grad is None else torch.equal(grad, other), name)

    def test_default_training_never_calls_spectral_helper(self):
        model = make_model()
        with patch.object(model, "build_inputs", return_value=inputs(model)), patch(
            "bridgewam.models.wan22.bridgewam.lbq_spectral_diversity"
        ) as spectral:
            model.training_loss({})
        spectral.assert_not_called()

    def test_regularizer_gradient_routes_upstream_not_to_action(self):
        model = make_model(1e-4, True)
        # Isolate the extra loss, keeping the actual Video/LBQ/Action forward.
        model.loss_lambda_action = model.loss_lambda_video = 0.0
        batch = inputs(model)
        with patch.object(model, "build_inputs", return_value=batch):
            loss, metrics = model.training_loss({})
        self.assertAlmostEqual(loss.item(), metrics["loss_lbq_spectral"], places=8)
        loss.backward()
        self.assertGreater(model.bridge.latent_bridge_queries.lbq_embeddings.grad.norm().item(), 0)
        self.assertGreater(model.video_expert.blocks[0].self_attn.v.weight.grad.norm().item(), 0)
        self.assertGreater(model.proprio_encoder.weight.grad.norm().item(), 0)
        for parameter in model.action_expert.parameters():
            self.assertTrue(parameter.grad is None or parameter.grad.count_nonzero() == 0)
        for source in ("embedding", "readout", "context"):
            self.assertIn(f"lbq_{source}_effective_rank", metrics)

    def test_loss_is_original_plus_weighted_penalty(self):
        model = make_model()
        losses = []
        for weight in (0.0, 0.01):
            model.loss_lambda_lbq_spectral = weight
            torch.manual_seed(123)
            with patch.object(model, "build_inputs", return_value=inputs(model)):
                loss, metrics = model.training_loss({})
            losses.append(loss.detach())
        torch.testing.assert_close(losses[1], losses[0] + metrics["loss_lbq_spectral"])

    def test_invalid_weights_fail(self):
        for weight in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "nonnegative"):
                make_model(weight)

    def test_inference_unchanged_and_never_calls_regularizer(self):
        model = make_model()
        latent = torch.randn(1, 2, 1, 2, 2)
        context = torch.randn(1, 3, 8)
        outputs = []
        with patch.object(model, "_encode_input_image_latents_tensor", return_value=latent), patch(
            "bridgewam.models.wan22.bridgewam.lbq_spectral_diversity"
        ) as spectral:
            for weight, diagnostics in ((0.0, False), (1e-4, True)):
                model.loss_lambda_lbq_spectral = weight
                model.lbq_spectral_diagnostics = diagnostics
                outputs.append(model.infer_action(
                    prompt=None, input_image=torch.zeros(1, 3, 16, 16),
                    action_horizon=4, context=context,
                    context_mask=torch.ones(1, 3, dtype=torch.bool),
                    proprio=torch.ones(1, 2), num_inference_steps=10, seed=42,
                )["action"])
        spectral.assert_not_called()
        self.assertTrue(torch.equal(*outputs))

    def test_checkpoint_schema_is_unchanged(self):
        model = make_model()
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "base.pt"
            experiment = Path(directory) / "spectral.pt"
            model.save_checkpoint(baseline)
            model.loss_lambda_lbq_spectral = 1e-4
            model.lbq_spectral_diagnostics = True
            model.save_checkpoint(experiment)
            first = torch.load(baseline, weights_only=False)
            second = torch.load(experiment, weights_only=False)
            self.assertEqual(first.keys(), second.keys())
            for key, value in first.items():
                if isinstance(value, dict):
                    self.assertEqual(value.keys(), second[key].keys())
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, torch.Tensor):
                            self.assertTrue(torch.equal(subvalue, second[key][subkey]))
                        else:
                            self.assertEqual(subvalue, second[key][subkey])
                else:
                    self.assertEqual(value, second[key])
            model.load_checkpoint(baseline)


class TestSpectralConfiguration(unittest.TestCase):
    def test_task_only_changes_loss_configuration(self):
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
            base = compose(config_name="train", overrides=[f"task={BASE_TASK}"])
            experiment = compose(config_name="train", overrides=[f"task={TASK}"])
            control = compose(config_name="train", overrides=[
                f"task={TASK}", "model.loss.lambda_lbq_spectral=0.0"
            ])
        self.assertEqual(experiment.model.loss.lambda_lbq_spectral, 1e-4)
        self.assertTrue(experiment.model.loss.lbq_spectral_diagnostics)
        self.assertEqual(control.model.loss.lambda_lbq_spectral, 0)
        base_dict = OmegaConf.to_container(base, resolve=False)
        exp_dict = OmegaConf.to_container(experiment, resolve=False)
        base_dict["model"].pop("loss")
        exp_dict["model"].pop("loss")
        self.assertEqual(base_dict, exp_dict)

    def test_model_factory_forwards_spectral_configuration(self):
        model = make_model()
        components = SimpleNamespace(dit=model.video_expert, vae=model.vae,
            text_encoder=None, tokenizer=None, dit_path=None, vae_path=None,
            text_encoder_path=None, tokenizer_path=None)
        with patch("bridgewam.models.wan22.bridgewam.load_wan22_ti2v_5b_components", return_value=components), patch.object(
            ActionDiT, "from_pretrained", return_value=model.action_expert
        ):
            actual = create_bridgewam(
                model_id="test", tokenizer_model_id="test", device="cpu", model_dtype=torch.float32,
                video_dit_config={"text_dim": 8, "hidden_dim": 8}, action_dit_config=_action_config(),
                latent_bridge_queries=_lbq_config(preserve_direct_video_kv=False,
                    freeze_video_expert=False, strict_training_scope=False),
                action_scheduler={"train_shift": 5, "infer_shift": 5, "num_train_timesteps": 1000},
                loss=OmegaConf.create({"lambda_lbq_spectral": 1e-4, "lbq_spectral_diagnostics": True}),
            )
        self.assertEqual(actual.loss_lambda_lbq_spectral, 1e-4)
        self.assertTrue(actual.lbq_spectral_diagnostics)


if __name__ == "__main__":
    unittest.main()
