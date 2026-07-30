import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from network.model import AINet
    from pretrain import H2HPretrainDataset


def make_config(partner_joints=3):
    return SimpleNamespace(
        motion=SimpleNamespace(
            dim1=3 * 3,
            dim2=partner_joints * 3,
            harper_input_length_dct=8,
            harper_target_length_train=3,
        ),
        motion_mlp=SimpleNamespace(
            embed_dim=16,
            hidden_dim=16,
            dropout=0.0,
            intra_layers=1,
            inter_layers=1,
            attn_heads=4,
        ),
        intent=SimpleNamespace(
            latent_dim=4,
            freeze_human_backbone_stage2=True,
        ),
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this test runtime")
class IntentModelTest(unittest.TestCase):
    def test_conditional_kl_is_zero_for_identical_gaussians(self):
        zeros = torch.zeros(3, 4)
        kl = AINet._kl_normal(zeros, zeros, zeros, zeros)
        torch.testing.assert_close(kl, torch.zeros(3))

    def test_stage1_posterior_forward_and_backward(self):
        torch.manual_seed(1)
        model = AINet(make_config())
        model.set_stage(1)
        model.train()

        dct_a = torch.randn(2, 8, 9)
        dct_b = torch.randn(2, 8, 9)
        history_a = torch.randn(2, 8, 9)
        history_b = torch.randn(2, 8, 9)
        future_a = torch.randn(2, 3, 9)
        future_b = torch.randn(2, 3, 9)
        valid = torch.tensor([1.0, 0.0])

        pred_a, *_ = model(
            dct_a,
            dct_b,
            history_motion1=history_a,
            history_motion2=history_b,
            future_motion1=future_a,
            future_motion2=future_b,
            interaction_valid=valid,
        )
        self.assertEqual(tuple(pred_a.shape), (2, 8, 9))
        self.assertEqual(tuple(model.last_pred_partner.shape), (2, 8, 9))
        self.assertEqual(tuple(model.last_prior_mu.shape), (2, 4))
        self.assertEqual(tuple(model.last_posterior_mu.shape), (2, 4))
        self.assertEqual(tuple(model.last_kl_per_sample.shape), (2,))

        loss = (
            pred_a.square().mean()
            + model.last_pred_partner.square().mean()
            + model.last_kl_per_sample[0]
            + model.last_intent_token_loss_per_sample[0]
        )
        loss.backward()
        self.assertIsNotNone(model.intent_prior[0].weight.grad)

    def test_inference_uses_history_prior_without_future(self):
        model = AINet(make_config())
        model.eval()
        dct_a = torch.randn(2, 8, 9)
        dct_b = torch.randn(2, 8, 9)
        history_a = torch.randn(2, 8, 9)
        history_b = torch.randn(2, 8, 9)

        with torch.no_grad():
            pred_a, *_ = model(
                dct_a,
                dct_b,
                history_motion1=history_a,
                history_motion2=history_b,
            )
        self.assertEqual(tuple(pred_a.shape), (2, 8, 9))
        self.assertIsNone(model.last_posterior_mu)
        self.assertTrue(torch.count_nonzero(model.last_kl_per_sample) == 0)

    def test_h2h_pair_trains_cross_attention_token_and_intent_modules(self):
        torch.manual_seed(2)
        model = AINet(make_config())
        model.train()
        pred_a, *_ = model(
            torch.randn(2, 8, 9),
            torch.randn(2, 8, 9),
            history_motion1=torch.randn(2, 8, 9),
            history_motion2=torch.randn(2, 8, 9),
            future_motion1=torch.randn(2, 3, 9),
            future_motion2=torch.randn(2, 3, 9),
            interaction_valid=torch.ones(2),
        )
        loss = (
            pred_a.square().mean()
            + model.last_pred_partner.square().mean()
            + model.last_kl_per_sample.mean()
            + model.last_intent_token_loss_per_sample.mean()
        )
        loss.backward()

        cross_grads = [parameter.grad for parameter in model.cross_blocks.parameters()]
        token_cross_grads = [
            parameter.grad for parameter in model.interaction_encoder.cross_attention.parameters()
        ]
        prior_grads = [parameter.grad for parameter in model.intent_prior.parameters()]
        posterior_grads = [parameter.grad for parameter in model.intent_posterior.parameters()]
        decoder_grads = [parameter.grad for parameter in model.future_token_decoder.parameters()]

        def has_nonzero_gradient(gradients):
            return any(
                gradient is not None and torch.count_nonzero(gradient).item() > 0
                for gradient in gradients
            )

        self.assertTrue(has_nonzero_gradient(cross_grads))
        self.assertTrue(has_nonzero_gradient(token_cross_grads))
        self.assertTrue(has_nonzero_gradient(prior_grads))
        self.assertTrue(has_nonzero_gradient(posterior_grads))
        self.assertTrue(has_nonzero_gradient(decoder_grads))

    def test_synthetic_npz_uses_source_intent_eligibility_not_provenance_mask(self):
        motion = np.zeros((12, 21, 3), dtype=np.float32)
        motion[:, :, 1] = np.linspace(0.0, 0.2, 12, dtype=np.float32)[:, None]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.savez_compressed(
                path / "synthetic_pair.npz",
                person_a=motion,
                person_b=motion + np.float32(1.0),
                joint_layout="optitrack21",
                unit="m",
                unit_scale_to_m=np.float32(1.0),
                synthetic=np.bool_(True),
                interaction_valid=np.float32(0.0),
            )
            cfg = {
                "datasets": {
                    "normalization": {
                        "target_layout": "optitrack21",
                        "unit": "m",
                        "root_center": True,
                        "validate_scale": False,
                    },
                    "sources": [
                        {
                            "name": "synthetic-test",
                            "path": str(path),
                            "enabled": True,
                            "intent_supervision": True,
                        }
                    ],
                }
            }
            dataset = H2HPretrainDataset(cfg, obs_len=8, pred_len=3)
            _, _, intent_mask = dataset[0]

        self.assertEqual(intent_mask.item(), 1.0)
        self.assertTrue(dataset.intent_flags[0])
        self.assertFalse(dataset.recorded_pair_flags[0])
        self.assertTrue(dataset.synthetic_flags[0])
        self.assertEqual(dataset.sample_sources[0], "synthetic-test")
        self.assertEqual(dataset.source_window_counts["synthetic-test"], 2)

    def test_stage1_checkpoint_loads_into_larger_robot_skeleton(self):
        stage1 = AINet(make_config(partner_joints=3))
        stage2 = AINet(make_config(partner_joints=4))
        report = stage2.load_compatible_state_dict(stage1.state_dict())

        self.assertIn("intra_r.0.aspatial", report["skipped_shape"])
        self.assertIn("ifb.joint_embed", report["skipped_shape"])
        critical_missing = [
            key
            for key in report["missing"]
            if key.startswith(("interaction_encoder.", "intent_prior.", "intent_posterior."))
        ]
        self.assertEqual(critical_missing, [])
        token_cross_key = "interaction_encoder.cross_attention.spatial.in_proj_weight"
        torch.testing.assert_close(
            stage2.state_dict()[token_cross_key],
            stage1.state_dict()[token_cross_key],
        )

        stage2.set_stage(2)
        stage2.eval()
        with torch.no_grad():
            prediction, *_ = stage2(
                torch.randn(2, 8, 9),
                torch.randn(2, 8, 12),
                history_motion1=torch.randn(2, 8, 9),
                history_motion2=torch.randn(2, 8, 12),
            )
        self.assertEqual(tuple(prediction.shape), (2, 8, 9))
        self.assertEqual(tuple(stage2.last_interaction_token.shape), (2, 16))

    def test_stage2_freezes_only_the_human_motion_backbone(self):
        model = AINet(make_config(partner_joints=4))
        model.set_stage(2)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder_h.parameters()))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.intra_h.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.intra_r.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.cross_blocks.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.intent_prior.parameters()))


if __name__ == "__main__":
    unittest.main()
