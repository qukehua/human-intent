import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.io import loadmat

try:
    from dataset.skeleton_utils import (
        canonicalize_motion,
        joint_names_array,
        mirror_canonical_x,
    )
except ModuleNotFoundError:
    from skeleton_utils import canonicalize_motion, joint_names_array, mirror_canonical_x


def rotation_z(deg: float) -> np.ndarray:
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def augment_second_person(person_a: np.ndarray, rotate_deg: float = 35.0, mirror_x: bool = True) -> np.ndarray:
    # person_a: canonical [T, 21, 3] in metres.
    person_b = mirror_canonical_x(person_a) if mirror_x else person_a.copy()
    r = rotation_z(rotate_deg)
    person_b = person_b @ r.T
    person_b[..., 0] += 0.6  # lateral offset to avoid overlap
    return person_b


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_canonical_pair(
    out_fp: Path,
    person_a: np.ndarray,
    person_b: np.ndarray,
    source: Path,
    *,
    synthetic: bool,
) -> None:
    np.savez_compressed(
        out_fp,
        person_a=person_a.astype(np.float32),
        person_b=person_b.astype(np.float32),
        source=str(source),
        synthetic=synthetic,
        joint_layout="optitrack21",
        joint_names=joint_names_array(),
        unit="m",
        unit_scale_to_m=np.float32(1.0),
    )


def process_3dpw(root: Path, out_dir: Path, max_files: int) -> dict:
    files = sorted((root / "sequenceFiles").rglob("*.pkl"))
    info = {"dataset": "3DPW", "is_multi_person": True, "files_seen": len(files), "files_written": 0}
    ensure_dir(out_dir)
    for idx, fp in enumerate(files[:max_files]):
        with open(fp, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        joints = data.get("jointPositions", None)
        if joints is None or len(joints) < 2:
            continue
        # each person: [T, 72], reshape to [T, 24, 3]
        try:
            p1 = canonicalize_motion(
                np.asarray(joints[0], dtype=np.float32).reshape(-1, 24, 3),
                "smpl24",
                unit_scale=1.0,
            )
            p2 = canonicalize_motion(
                np.asarray(joints[1], dtype=np.float32).reshape(-1, 24, 3),
                "smpl24",
                unit_scale=1.0,
            )
        except ValueError:
            continue
        out_fp = out_dir / f"{fp.stem}.npz"
        save_canonical_pair(out_fp, p1, p2, fp, synthetic=False)
        info["files_written"] += 1
    return info


def process_amass(root: Path, out_dir: Path, max_files: int) -> dict:
    files = sorted(root.rglob("*_poses.npz"))
    info = {
        "dataset": "amass",
        "is_multi_person": False,
        "files_seen": len(files),
        "files_written": 0,
        "files_skipped_no_joint_positions": 0,
    }
    ensure_dir(out_dir)
    for fp in files[:max_files]:
        with np.load(fp, allow_pickle=True) as z:
            joint_key = next(
                (key for key in ("joints", "joint_positions", "jointPositions") if key in z.files),
                None,
            )
            if joint_key is None:
                # Raw AMASS contains SMPL pose parameters, not 3-D joints.
                # A body-model forward pass is required before this preprocessor.
                info["files_skipped_no_joint_positions"] += 1
                continue
            raw_joints = np.asarray(z[joint_key], dtype=np.float32)
        if raw_joints.ndim == 2 and raw_joints.shape[1] % 3 == 0:
            raw_joints = raw_joints.reshape(raw_joints.shape[0], -1, 3)
        if raw_joints.ndim != 3 or raw_joints.shape[-1] != 3:
            info["files_skipped_no_joint_positions"] += 1
            continue
        layout = "optitrack21" if raw_joints.shape[1] == 21 else "smpl24"
        try:
            p1 = canonicalize_motion(raw_joints, layout, unit_scale=1.0)
        except ValueError:
            info["files_skipped_no_joint_positions"] += 1
            continue
        p2 = augment_second_person(p1, rotate_deg=25.0, mirror_x=True)
        out_fp = out_dir / f"{fp.stem}.npz"
        save_canonical_pair(out_fp, p1, p2, fp, synthetic=True)
        info["files_written"] += 1
    return info


def process_h36m(root: Path, out_dir: Path, max_files: int) -> dict:
    files = sorted((root / "dataset").rglob("*.txt"))
    info = {"dataset": "h3.6m", "is_multi_person": False, "files_seen": len(files), "files_written": 0}
    ensure_dir(out_dir)
    for fp in files[:max_files]:
        arr = np.loadtxt(fp, delimiter=",", dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] % 3 != 0:
            continue
        j = arr.shape[1] // 3
        if j != 32:
            # Common 99-D H36M text exports contain exponential-map angles,
            # not 33 XYZ joints. Never reinterpret those values as positions.
            continue
        try:
            p1 = canonicalize_motion(
                arr.reshape(arr.shape[0], j, 3),
                "h36m32",
                unit_scale=0.001,
            )
        except ValueError:
            continue
        p2 = augment_second_person(p1, rotate_deg=40.0, mirror_x=True)
        out_fp = out_dir / f"{fp.parent.name}_{fp.stem}.npz"
        save_canonical_pair(out_fp, p1, p2, fp, synthetic=True)
        info["files_written"] += 1
    return info


def process_mupots(root: Path, out_dir: Path, max_files: int) -> dict:
    annot_files = sorted(root.rglob("annot.mat"))
    info = {"dataset": "MuPots-3d", "is_multi_person": True, "files_seen": len(annot_files), "files_written": 0}
    ensure_dir(out_dir)
    # Keep multi-person source as index for pretrain loader.
    for fp in annot_files[:max_files]:
        mat = loadmat(fp)
        key = "annotations" if "annotations" in mat else None
        if key is None:
            continue
        out_fp = out_dir / f"{fp.parent.name}_index.npz"
        np.savez_compressed(out_fp, source=str(fp), key=key)
        info["files_written"] += 1
    return info


def main():
    parser = argparse.ArgumentParser(description="Check H2H datasets and build two-person augmented data.")
    parser.add_argument("--datasets-root", type=str, default="/data/user/qkh/datasets")
    parser.add_argument("--max-files-per-dataset", type=int, default=200)
    args = parser.parse_args()

    root = Path(args.datasets_root)
    mappings = {
        "3DPW": root / "3DPW",
        "amass": root / "amass",
        "h3.6m": root / "h3.6m",
        "MuPots-3d": root / "MuPots-3d",
    }

    reports = []
    reports.append(process_3dpw(mappings["3DPW"], mappings["3DPW"] / "data_aug", args.max_files_per_dataset))
    reports.append(process_amass(mappings["amass"], mappings["amass"] / "data_aug", args.max_files_per_dataset))
    reports.append(process_h36m(mappings["h3.6m"], mappings["h3.6m"] / "data_aug", args.max_files_per_dataset))
    reports.append(process_mupots(mappings["MuPots-3d"], mappings["MuPots-3d"] / "data_aug", args.max_files_per_dataset))

    summary = {
        "datasets_root": str(root),
        "results": reports,
    }
    out = root / "h2h_pretrain_data_aug_report.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved report -> {out}")


if __name__ == "__main__":
    main()
