import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from config.config_utils import build_h2h_model_config, load_yaml
from dataset.skeleton_utils import canonicalize_motion, scene_center_pair
from network.model import AINet


def get_dct_matrix(n: int):
    dct_m = np.eye(n)
    for k in np.arange(n):
        for i in np.arange(n):
            w = np.sqrt(2 / n)
            if k == 0:
                w = np.sqrt(1 / n)
            dct_m[k, i] = w * np.cos(np.pi * (i + 0.5) * k / n)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def _npz_scalar(z, key: str, default=None):
    if key not in z.files:
        return default
    value = np.asarray(z[key])
    if value.size != 1:
        return default
    item = value.reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return item


class H2HPretrainDataset(Dataset):
    def __init__(self, cfg: dict, obs_len: int, pred_len: int, target_joints: int = 21):
        if target_joints != 21:
            raise ValueError("H2H pretraining now uses the fixed HARPER/OptiTrack 21-joint layout")
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.win = obs_len + pred_len
        self.target_joints = target_joints
        self.files = []
        self.index = []
        self.intent_flags = []
        self.recorded_pair_flags = []
        self.synthetic_flags = []
        self.sample_sources = []
        self.source_window_counts = {}
        normalisation = cfg.get("datasets", {}).get("normalization", {})
        target_layout = str(normalisation.get("target_layout", "optitrack21")).lower()
        target_unit = str(normalisation.get("unit", "m")).lower()
        if target_layout not in {"optitrack21", "harper21", "canonical21"}:
            raise ValueError(f"Unsupported pretraining target layout: {target_layout}")
        if target_unit not in {"m", "meter", "metre"}:
            raise ValueError(f"Pretraining target coordinates must use metres, got {target_unit}")
        self.root_center = bool(normalisation.get("root_center", True))
        self.validate_scale = bool(normalisation.get("validate_scale", True))
        self.min_extent_m = float(normalisation.get("min_extent_m", 0.25))
        self.max_extent_m = float(normalisation.get("max_extent_m", 3.5))
        skipped = 0

        sources = cfg.get("datasets", {}).get("sources", [])
        for src in sources:
            if not src.get("enabled", True):
                continue
            source_name = str(src.get("name", src["path"]))
            p = Path(src["path"])
            if not p.exists():
                warnings.warn(
                    f"Enabled H2H source does not exist and will be skipped: "
                    f"{source_name} ({p})",
                    stacklevel=2,
                )
                continue
            self.source_window_counts.setdefault(source_name, 0)
            for fp in sorted(p.glob("*.npz")):
                try:
                    with np.load(fp, allow_pickle=True) as z:
                        if "person_a" not in z.files or "person_b" not in z.files:
                            # Skip index-only files (e.g. current MuPots files).
                            skipped += 1
                            continue
                        layout = str(_npz_scalar(z, "joint_layout", src.get("joint_layout", "auto")))
                        unit = str(_npz_scalar(z, "unit", "")).lower()
                        default_scale = 1.0 if unit in {"m", "meter", "metre"} else src.get("unit_scale", 1.0)
                        unit_scale = float(_npz_scalar(z, "unit_scale_to_m", default_scale))
                        synthetic = bool(
                            _npz_scalar(
                                z,
                                "synthetic",
                                src.get("converted_to_multi_person", False),
                            )
                        )
                        file_intent_enabled = bool(
                            _npz_scalar(z, "intent_training_eligible", True)
                        )
                        source_intent_enabled = bool(
                            src.get("intent_supervision", file_intent_enabled)
                        )
                        file_interaction_valid = bool(
                            _npz_scalar(
                                z,
                                "interaction_valid",
                                not synthetic,
                            )
                        )
                        # Training eligibility and data provenance are separate.
                        # Random AMASS/H36M pairs are valid for the trajectory
                        # latent and cross-attention token when the source is
                        # enabled, while recorded synchronous pairs retain a
                        # higher sampling weight below.
                        intent_valid = source_intent_enabled and file_intent_enabled
                        recorded_pair = bool(
                            _npz_scalar(
                                z,
                                "recorded_synchronous",
                                file_interaction_valid and not synthetic,
                            )
                        )

                        # Validate semantics, finiteness and metric scale before
                        # adding any windows, so bad files never fail mid-epoch.
                        a = canonicalize_motion(
                            z["person_a"],
                            layout,
                            unit_scale,
                            validate_scale=self.validate_scale,
                            min_extent_m=self.min_extent_m,
                            max_extent_m=self.max_extent_m,
                        )
                        b = canonicalize_motion(
                            z["person_b"],
                            layout,
                            unit_scale,
                            validate_scale=self.validate_scale,
                            min_extent_m=self.min_extent_m,
                            max_extent_m=self.max_extent_m,
                        )
                        t = min(a.shape[0], b.shape[0])
                        if t < self.win:
                            skipped += 1
                            continue
                except (OSError, ValueError, KeyError):
                    skipped += 1
                    continue

                file_id = len(self.files)
                self.files.append((fp, layout, unit_scale, intent_valid, recorded_pair, synthetic))
                window_count = t - self.win + 1
                self.source_window_counts[source_name] += window_count
                for st in range(0, window_count):
                    self.index.append((file_id, st))
                    self.intent_flags.append(intent_valid)
                    self.recorded_pair_flags.append(recorded_pair)
                    self.synthetic_flags.append(synthetic)
                    self.sample_sources.append(source_name)

        # Compatibility alias for callers that used the old field.  Its
        # semantics are now "eligible for interaction/intent training".
        self.interaction_flags = self.intent_flags

        if skipped:
            warnings.warn(
                f"Skipped {skipped} H2H files with missing trajectories, incompatible joints, "
                "invalid coordinates/scale, or insufficient frames.",
                stacklevel=2,
            )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_id, st = self.index[idx]
        fp, layout, unit_scale, intent_valid, _, _ = self.files[file_id]
        with np.load(fp, allow_pickle=True) as z:
            a = canonicalize_motion(
                z["person_a"],
                layout,
                unit_scale,
                validate_scale=False,
            )
            b = canonicalize_motion(
                z["person_b"],
                layout,
                unit_scale,
                validate_scale=False,
            )

        a = a[st : st + self.win]  # [win, J, 3]
        b = b[st : st + self.win]
        if self.root_center:
            a, b = scene_center_pair(a, b)
        data = np.concatenate([a.reshape(self.win, -1), b.reshape(self.win, -1)], axis=-1)  # [win, 2*J*3]

        motion_input = data[: self.obs_len]
        motion_target = data[self.obs_len :]
        # This mask controls cross-attention, interaction-token, KL and latent
        # conditioning.  Provenance is tracked separately for sampling.
        intent_mask = torch.tensor(float(intent_valid), dtype=torch.float32)
        return torch.from_numpy(motion_input), torch.from_numpy(motion_target), intent_mask


def gen_velocity(m):
    return m[:, 1:] - m[:, :-1]


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=values.dtype).reshape(-1)
    return torch.sum(values.reshape(-1) * mask) / torch.clamp(mask.sum(), min=1.0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config/h2h_pretrain_cfg.yml")
    parser.add_argument("--work-dir", type=str, default="./ckpt_h2h_pretrain")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full training by epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.cfg)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 888))
    torch.manual_seed(seed)
    np.random.seed(seed)
    Path(args.work_dir).mkdir(parents=True, exist_ok=True)

    obs_len = int(cfg["sequence"]["obs_len"])
    pred_len = int(cfg["sequence"]["pred_len"])
    coord_dim = int(cfg["sequence"].get("coord_dim", 3))
    target_joints = int(cfg["sequence"].get("target_joints", 21))
    relative_output = bool(cfg["sequence"].get("relative_output", True))
    if coord_dim != 3:
        raise ValueError(f"Only xyz coordinates are supported, got coord_dim={coord_dim}")
    domains = {
        "input": str(cfg["sequence"].get("input_domain", "dct")).lower(),
        "output": str(cfg["sequence"].get("output_domain", "dct")).lower(),
        "supervision": str(cfg["sequence"].get("supervision_domain", "time")).lower(),
        "reconstruction": str(cfg["sequence"].get("reconstruction_domain", "dct")).lower(),
    }
    expected_domains = {"input": "dct", "output": "dct", "supervision": "time", "reconstruction": "dct"}
    if domains != expected_domains:
        raise ValueError(f"Unsupported domain configuration {domains}; expected {expected_domains}")
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    num_workers = int(cfg["train"]["num_workers"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    lambda_pre = float(cfg["train"]["lambda_pre"])
    lambda_rec = float(cfg["train"]["lambda_rec"])
    intent_cfg = cfg.get("intent", {})
    lambda_partner = float(intent_cfg.get("partner_prediction_weight", 1.0))
    lambda_kl = float(intent_cfg.get("kl_weight", 1.0e-3))
    lambda_intent_token = float(intent_cfg.get("token_prediction_weight", 0.1))
    kl_warmup_steps = max(1, int(intent_cfg.get("kl_warmup_steps", 10000)))

    dataset = H2HPretrainDataset(cfg, obs_len=obs_len, pred_len=pred_len, target_joints=target_joints)
    if len(dataset) == 0:
        raise RuntimeError("No valid training windows found. Check data_aug files and cfg paths.")
    intent_pair_count = int(sum(dataset.intent_flags))
    recorded_pair_count = int(sum(dataset.recorded_pair_flags))
    synthetic_pair_count = int(sum(dataset.synthetic_flags))
    if intent_pair_count == 0:
        raise RuntimeError("No H2H windows are enabled for interaction-token/intent training.")
    real_pair_sampling_weight = float(intent_cfg.get("real_pair_sampling_weight", 4.0))
    if real_pair_sampling_weight <= 0:
        raise ValueError("intent.real_pair_sampling_weight must be positive")
    balance_sources = bool(intent_cfg.get("balance_sources", True))
    sample_weight_values = []
    source_sampling_mass = {name: 0.0 for name in dataset.source_window_counts}
    for is_recorded, source_name in zip(
        dataset.recorded_pair_flags,
        dataset.sample_sources,
    ):
        weight = (real_pair_sampling_weight if is_recorded else 1.0) / (
            dataset.source_window_counts[source_name] if balance_sources else 1.0
        )
        sample_weight_values.append(weight)
        source_sampling_mass[source_name] += weight
    sample_weights = torch.tensor(sample_weight_values, dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(dataset), replacement=True)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    print(
        f"H2H windows: {len(dataset)} total, {intent_pair_count} intent-enabled "
        f"({intent_pair_count / len(dataset):.1%}), {recorded_pair_count} recorded synchronous, "
        f"{synthetic_pair_count} synthetic; source balancing={balance_sources}, "
        f"recorded-pair sampling weight={real_pair_sampling_weight:g}"
    )
    total_sampling_mass = float(sample_weights.sum())
    print("H2H windows and expected sampling share by source:")
    for name, count in dataset.source_window_counts.items():
        share = source_sampling_mass[name] / total_sampling_mass
        print(f"  {name}: {count} windows, {share:.1%} sampled")

    config = build_h2h_model_config(cfg)

    model = AINet(config).cuda()
    model.set_stage(int(cfg.get("stage", 1)))
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    dct_m, idct_m = get_dct_matrix(obs_len)
    dct_m = torch.tensor(dct_m).float().cuda().unsqueeze(0)
    idct_m = torch.tensor(idct_m).float().cuda().unsqueeze(0)

    step = 0
    for ep in range(epochs):
        pbar = tqdm(loader, desc=f"pretrain epoch {ep+1}/{epochs}")
        for motion_input, motion_target, intent_valid in pbar:
            motion_input = motion_input.cuda()  # [B,T,126]
            motion_target = motion_target.cuda()  # [B,P,126]
            intent_valid = intent_valid.cuda()
            b, t, _ = motion_input.shape
            in_features = target_joints * coord_dim

            src1 = motion_input[:, :, :in_features]
            src2 = motion_input[:, :, in_features:]
            tgt1 = motion_target[:, :, :in_features]
            tgt2 = motion_target[:, :, in_features:]
            src1_dct = torch.matmul(dct_m[:, :, :obs_len], src1)
            src2_dct = torch.matmul(dct_m[:, :, :obs_len], src2)

            pred1_dct, _, _, _, _ = model(
                src1_dct,
                src2_dct,
                history_motion1=src1,
                history_motion2=src2,
                future_motion1=tgt1,
                future_motion2=tgt2,
                interaction_valid=intent_valid,
            )
            pred2_dct = model.last_pred_partner
            # The model predicts DCT coefficients. Convert them back to time
            # before comparing with the time-domain future target.
            pred1_time = torch.matmul(idct_m, pred1_dct)[:, :pred_len]
            pred2_time = torch.matmul(idct_m, pred2_dct)[:, :pred_len]
            if relative_output:
                pred1_time = pred1_time + src1[:, -1:, :]
                pred2_time = pred2_time + src2[:, -1:, :]

            pred1_xyz = pred1_time.reshape(b, pred_len, target_joints, coord_dim)
            pred2_xyz = pred2_time.reshape(b, pred_len, target_joints, coord_dim)
            tgt1_xyz = tgt1.reshape(b, pred_len, target_joints, coord_dim)
            tgt2_xyz = tgt2.reshape(b, pred_len, target_joints, coord_dim)
            loss_person1 = torch.mean(torch.norm(pred1_xyz - tgt1_xyz, dim=-1))
            loss_person2 = torch.mean(torch.norm(pred2_xyz - tgt2_xyz, dim=-1))
            loss_pre = loss_person1 + lambda_partner * loss_person2

            rec1 = model.last_recon_h
            rec2 = model.last_recon_r
            # Reconstruction heads also operate in the DCT domain.
            src1_xyz = src1_dct.reshape(b, t, target_joints, coord_dim).reshape(-1, coord_dim)
            src2_xyz = src2_dct.reshape(b, t, target_joints, coord_dim).reshape(-1, coord_dim)
            rec1_xyz = rec1.reshape(-1, 3)
            rec2_xyz = rec2.reshape(-1, 3)
            loss_rec = torch.mean(torch.norm(rec1_xyz - src1_xyz, dim=-1)) + torch.mean(
                torch.norm(rec2_xyz - src2_xyz, dim=-1)
            )

            kl_warmup = min(1.0, float(step + 1) / float(kl_warmup_steps))
            loss_kl = masked_mean(model.last_kl_per_sample, intent_valid)
            loss_intent_token = masked_mean(
                model.last_intent_token_loss_per_sample,
                intent_valid,
            )
            total = (
                lambda_pre * loss_pre
                + lambda_rec * loss_rec
                + lambda_kl * kl_warmup * loss_kl
                + lambda_intent_token * loss_intent_token
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            step += 1
            pbar.set_postfix(
                {
                    "loss": f"{total.item():.4f}",
                    "p1": f"{loss_person1.item():.4f}",
                    "p2": f"{loss_person2.item():.4f}",
                    "rec": f"{loss_rec.item():.4f}",
                    "kl": f"{loss_kl.item():.4f}",
                    "intent": f"{loss_intent_token.item():.4f}",
                    "intent_valid": f"{intent_valid.mean().item():.2f}",
                }
            )

            if args.max_steps > 0 and step >= args.max_steps:
                break

        ckpt = Path(args.work_dir) / f"pretrain_stage1_epoch{ep+1}.pth"
        torch.save(model.state_dict(), ckpt)
        if args.max_steps > 0 and step >= args.max_steps:
            break

    final_ckpt = Path(args.work_dir) / "pretrain_stage1_final.pth"
    torch.save(model.state_dict(), final_ckpt)
    print(f"saved: {final_ckpt}")
    print(f"total steps: {step}")


if __name__ == "__main__":
    main()
