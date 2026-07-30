import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from network.model import AINet


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

    def test_synthetic_pair_mask_blocks_interaction_gradients(self):
        model = AINet(make_config())
        model.train()
        pred_a, *_ = model(
            torch.randn(2, 8, 9),
            torch.randn(2, 8, 9),
            history_motion1=torch.randn(2, 8, 9),
            history_motion2=torch.randn(2, 8, 9),
            interaction_valid=torch.zeros(2),
        )
        pred_a.square().mean().backward()

        cross_grads = [parameter.grad for parameter in model.cross_blocks.parameters()]
        prior_grads = [parameter.grad for parameter in model.intent_prior.parameters()]
        self.assertTrue(
            all(gradient is None or torch.count_nonzero(gradient) == 0 for gradient in cross_grads)
        )
        self.assertTrue(
            all(gradient is None or torch.count_nonzero(gradient) == 0 for gradient in prior_grads)
        )

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
