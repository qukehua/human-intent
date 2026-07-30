from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from dataset.cmu_interaction_converter import lowpass_downsample
    from dataset.h36m_fk import h36m_expmap_to_xyz
    from dataset.skeleton_utils import (
        canonicalize_motion,
        joint_names_array,
    )
except ModuleNotFoundError:
    from cmu_interaction_converter import lowpass_downsample
    from h36m_fk import h36m_expmap_to_xyz
    from skeleton_utils import canonicalize_motion, joint_names_array


@dataclass(frozen=True)
class MotionRecord:
    source: Path
    motion: np.ndarray


def rotation_y(deg: float) -> np.ndarray:
    """Y-up yaw rotation shared by HARPER, CMU and the canonical skeleton."""

    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _file_limit(files: list[Path], max_files: int) -> list[Path]:
    return files if max_files <= 0 else files[:max_files]


def _scalar_text(value) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _safe_stem(path: Path) -> str:
    return "_".join(path.stem.replace(" ", "_").split())


def save_canonical_pair(
    out_fp: Path,
    person_a: np.ndarray,
    person_b: np.ndarray,
    source_a: Path,
    source_b: Path,
    *,
    synthetic: bool,
    dataset: str,
    pairing_strategy: str,
    source_fps: float,
    target_fps: float,
    split: str = "train",
) -> None:
    """Save the only pair format accepted by Stage-1 pretraining."""

    if person_a.ndim != 3 or person_a.shape[1:] != (21, 3):
        raise ValueError(f"person_a must be [T, 21, 3], got {person_a.shape}")
    if person_b.ndim != 3 or person_b.shape[1:] != (21, 3):
        raise ValueError(f"person_b must be [T, 21, 3], got {person_b.shape}")
    if min(person_a.shape[0], person_b.shape[0]) < 1:
        raise ValueError("Cannot save an empty person pair")
    # Keep the legacy interaction_valid field as recorded-pair provenance so
    # existing data and checkpoints remain readable.  Stage-1 training uses
    # intent_training_eligible (or the source config) to decide whether the
    # pair contributes to cross-attention/token/KL objectives.
    interaction_valid = np.float32(0.0 if synthetic else 1.0)
    np.savez_compressed(
        out_fp,
        person_a=person_a.astype(np.float32),
        person_b=person_b.astype(np.float32),
        source_a=str(source_a),
        source_b=str(source_b),
        dataset=dataset,
        split=split,
        synthetic=np.bool_(synthetic),
        interaction_valid=interaction_valid,
        recorded_synchronous=np.bool_(not synthetic),
        intent_training_eligible=np.bool_(True),
        pairing_strategy=pairing_strategy,
        joint_layout="optitrack21",
        joint_names=joint_names_array(),
        unit="m",
        unit_scale_to_m=np.float32(1.0),
        source_fps=np.float32(source_fps),
        target_fps=np.float32(target_fps),
    )


def _derangement(count: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random permutation without pairing a clip with itself."""

    if count < 2:
        raise ValueError("At least two distinct motions are required for cross-sequence pairing")
    base = np.arange(count)
    for _ in range(128):
        shuffled = rng.permutation(count)
        if np.all(shuffled != base):
            return shuffled
    shift = int(rng.integers(1, count))
    return np.roll(base, shift)


def place_in_shared_scene(
    motion_a: np.ndarray,
    motion_b: np.ndarray,
    rng: np.random.Generator,
    *,
    min_distance_m: float,
    max_distance_m: float,
    max_pair_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Crop two clips, root-normalise them independently, then place B in A's scene."""

    if min_distance_m <= 0 or max_distance_m < min_distance_m:
        raise ValueError("Pair distance must satisfy 0 < min_distance <= max_distance")
    length = min(motion_a.shape[0], motion_b.shape[0])
    if max_pair_frames > 0:
        length = min(length, max_pair_frames)
    if length < 1:
        raise ValueError("Cannot pair empty motions")
    start_a = int(rng.integers(0, motion_a.shape[0] - length + 1))
    start_b = int(rng.integers(0, motion_b.shape[0] - length + 1))
    person_a = np.asarray(motion_a[start_a : start_a + length], dtype=np.float32).copy()
    person_b = np.asarray(motion_b[start_b : start_b + length], dtype=np.float32).copy()

    # Keep each actor's own trajectory and articulation. Only their initial
    # roots are aligned before assigning a relative scene transform.
    person_a -= person_a[0, 0].copy()
    person_b -= person_b[0, 0].copy()
    yaw_deg = float(rng.uniform(-180.0, 180.0))
    person_b = person_b @ rotation_y(yaw_deg).T
    distance_m = float(rng.uniform(min_distance_m, max_distance_m))
    bearing = float(rng.uniform(-np.pi, np.pi))
    translation = np.asarray(
        [distance_m * np.cos(bearing), 0.0, distance_m * np.sin(bearing)],
        dtype=np.float32,
    )
    person_b += translation
    metadata = {
        "start_a": start_a,
        "start_b": start_b,
        "yaw_b_deg": yaw_deg,
        "distance_m": distance_m,
    }
    return person_a, person_b, metadata


def pair_motion_records(
    records: list[MotionRecord],
    out_dir: Path,
    *,
    dataset: str,
    seed: int,
    source_fps: float,
    target_fps: float,
    min_distance_m: float,
    max_distance_m: float,
    max_pair_frames: int,
) -> dict:
    ensure_dir(out_dir)
    info = {
        "dataset": dataset,
        "is_multi_person": False,
        "augmentation": "random_cross_sequence_shared_scene",
        "synthetic": True,
        "human_latent_supervision": True,
        "usable_single_motion_files": len(records),
        "files_written": 0,
        "frames_written": 0,
    }
    if len(records) < 2:
        info["status"] = "skipped"
        info["reason"] = "fewer than two usable single-person motion clips"
        return info

    rng = np.random.default_rng(seed)
    partner_indices = _derangement(len(records), rng)
    for index_a, index_b in enumerate(partner_indices):
        record_a = records[index_a]
        record_b = records[int(index_b)]
        person_a, person_b, placement = place_in_shared_scene(
            record_a.motion,
            record_b.motion,
            rng,
            min_distance_m=min_distance_m,
            max_distance_m=max_distance_m,
            max_pair_frames=max_pair_frames,
        )
        filename = f"{index_a:05d}_{_safe_stem(record_a.source)}__{_safe_stem(record_b.source)}.npz"
        save_canonical_pair(
            out_dir / filename,
            person_a,
            person_b,
            record_a.source,
            record_b.source,
            synthetic=True,
            dataset=dataset,
            pairing_strategy="random_cross_sequence_shared_scene",
            source_fps=source_fps,
            target_fps=target_fps,
        )
        info["files_written"] += 1
        info["frames_written"] += int(person_a.shape[0])
        if info["files_written"] == 1:
            info["example_placement"] = placement
        if info["files_written"] % 250 == 0 or info["files_written"] == len(records):
            print(
                f"{dataset} pair generation: {info['files_written']}/"
                f"{len(records)} files",
                flush=True,
            )
    info["status"] = "ok"
    return info


def process_3dpw(root: Path, out_dir: Path, max_files: int) -> dict:
    """Convert real two-person train/validation clips; never augment them."""

    files: list[Path] = []
    for split in ("train", "validation"):
        files.extend(sorted((root / "sequenceFiles" / split).glob("*.pkl")))
    files = _file_limit(files, max_files)
    info = {
        "dataset": "3DPW",
        "is_multi_person": True,
        "augmentation": "none",
        "synthetic": False,
        "human_latent_supervision": True,
        "splits": ["train", "validation"],
        "test_split_excluded": True,
        "files_seen": len(files),
        "files_written": 0,
        "frames_written": 0,
    }
    ensure_dir(out_dir)
    for fp in files:
        with open(fp, "rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        joints = data.get("jointPositions")
        if joints is None or len(joints) < 2:
            continue
        try:
            person_a = canonicalize_motion(
                np.asarray(joints[0], dtype=np.float32).reshape(-1, 24, 3),
                "smpl24",
                unit_scale=1.0,
            )
            person_b = canonicalize_motion(
                np.asarray(joints[1], dtype=np.float32).reshape(-1, 24, 3),
                "smpl24",
                unit_scale=1.0,
            )
        except ValueError:
            continue
        frame_count = min(person_a.shape[0], person_b.shape[0])
        split = fp.parent.name
        save_canonical_pair(
            out_dir / f"{split}_{_safe_stem(fp)}.npz",
            person_a[:frame_count],
            person_b[:frame_count],
            fp,
            fp,
            synthetic=False,
            dataset="3DPW",
            pairing_strategy="recorded_synchronous_pair",
            source_fps=30.0,
            target_fps=30.0,
            split=split,
        )
        info["files_written"] += 1
        info["frames_written"] += frame_count
    info["status"] = "ok"
    return info


def load_h36m_records(
    root: Path,
    *,
    max_files: int,
    target_fps: float,
) -> tuple[list[MotionRecord], dict]:
    files = _file_limit(sorted((root / "dataset").rglob("*.txt")), max_files)
    records: list[MotionRecord] = []
    skipped = 0
    for fp in files:
        try:
            expmap = np.loadtxt(fp, delimiter=",", dtype=np.float32)
            xyz_mm = h36m_expmap_to_xyz(expmap, y_up=True)
            canonical = canonicalize_motion(xyz_mm, "h36m32", unit_scale=0.001)
            canonical = lowpass_downsample(canonical, source_fps=50.0, target_fps=target_fps)
        except (OSError, ValueError):
            skipped += 1
            continue
        records.append(MotionRecord(fp, canonical))
    return records, {
        "files_seen": len(files),
        "files_converted_to_xyz": len(records),
        "files_skipped": skipped,
    }


def process_h36m(
    root: Path,
    out_dir: Path,
    *,
    max_files: int,
    seed: int,
    target_fps: float,
    min_distance_m: float,
    max_distance_m: float,
    max_pair_frames: int,
) -> dict:
    records, conversion = load_h36m_records(
        root,
        max_files=max_files,
        target_fps=target_fps,
    )
    info = pair_motion_records(
        records,
        out_dir,
        dataset="h3.6m",
        seed=seed,
        source_fps=50.0,
        target_fps=target_fps,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        max_pair_frames=max_pair_frames,
    )
    info.update(conversion)
    info["input_representation"] = "99D exponential map"
    info["conversion"] = (
        "root-trajectory integration + standard Human3.6M 32-joint forward kinematics"
    )
    return info


class SmplhJointExtractor:
    """Recover AMASS SMPL-H kinematic joints from the official model NPZs.

    The Extended SMPL+H archive distributed for AMASS uses
    ``smplh/{gender}/model.npz``.  It is not the flat ``SMPLH_FEMALE.pkl``
    layout expected by the ``smplx`` package.  Joint recovery only needs the
    shaped rest joints and the kinematic chain, so doing it directly in NumPy
    also avoids generating all 6,890 mesh vertices.
    """

    def __init__(self, model_dir: Path, *, device: str, batch_size: int):
        if not model_dir.exists():
            raise RuntimeError(f"SMPL-H model directory does not exist: {model_dir}")
        self.model_dir = model_dir
        # Kept in the public signature for CLI compatibility. Joint-only
        # extraction is deliberately CPU/NumPy and does not allocate a mesh.
        self.device = "cpu"
        self.requested_device = device
        self.batch_size = max(1, batch_size)
        self.models = {}

    def _model_path(self, gender: str) -> Path:
        upper = gender.upper()
        candidates = (
            self.model_dir / "smplh" / gender / "model.npz",
            self.model_dir / gender / "model.npz",
            self.model_dir / "smplh" / f"SMPLH_{upper}.npz",
            self.model_dir / f"SMPLH_{upper}.npz",
        )
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        checked = "\n  ".join(str(path) for path in candidates)
        raise RuntimeError(
            f"Could not find the licensed SMPL-H {gender} model. Checked:\n  {checked}"
        )

    def _model(self, gender: str):
        gender = gender.lower()
        if gender not in {"male", "female", "neutral"}:
            gender = "neutral"
        if gender not in self.models:
            model_path = self._model_path(gender)
            with np.load(model_path, allow_pickle=True) as data:
                required = {
                    "J_regressor",
                    "kintree_table",
                    "shapedirs",
                    "v_template",
                }
                missing = required.difference(data.files)
                if missing:
                    raise RuntimeError(
                        f"{model_path} is missing SMPL-H arrays: {sorted(missing)}"
                    )
                model = {
                    name: np.asarray(data[name]).copy()
                    for name in required
                }
            parents = np.asarray(model["kintree_table"][0], dtype=np.int64)
            parents[0] = -1
            if model["shapedirs"].shape[-1] < 16:
                raise RuntimeError(
                    f"{model_path} has only {model['shapedirs'].shape[-1]} betas; "
                    "AMASS requires the Extended SMPL+H 16-beta model"
                )
            if model["J_regressor"].shape[0] != parents.size:
                raise RuntimeError(
                    f"Inconsistent joint count in {model_path}: "
                    f"regressor={model['J_regressor'].shape[0]}, tree={parents.size}"
                )
            model["parents"] = parents
            self.models[gender] = model
        return self.models[gender]

    @staticmethod
    def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
        """Vectorized Rodrigues formula for arrays ending in three values."""

        vectors = np.asarray(axis_angle, dtype=np.float64)
        theta_sq = np.sum(vectors * vectors, axis=-1)
        theta = np.sqrt(theta_sq)
        small = theta_sq < 1.0e-12
        scale_sin = np.empty_like(theta)
        scale_cos = np.empty_like(theta)
        scale_sin[~small] = np.sin(theta[~small]) / theta[~small]
        scale_cos[~small] = (
            1.0 - np.cos(theta[~small])
        ) / theta_sq[~small]
        # Taylor expansions retain precision around the zero pose.
        scale_sin[small] = 1.0 - theta_sq[small] / 6.0
        scale_cos[small] = 0.5 - theta_sq[small] / 24.0

        skew = np.zeros(vectors.shape[:-1] + (3, 3), dtype=np.float64)
        x, y, z = np.moveaxis(vectors, -1, 0)
        skew[..., 0, 1] = -z
        skew[..., 0, 2] = y
        skew[..., 1, 0] = z
        skew[..., 1, 2] = -x
        skew[..., 2, 0] = -y
        skew[..., 2, 1] = x
        identity = np.broadcast_to(np.eye(3), skew.shape)
        return (
            identity
            + scale_sin[..., None, None] * skew
            + scale_cos[..., None, None] * np.matmul(skew, skew)
        )

    def __call__(
        self,
        poses: np.ndarray,
        trans: np.ndarray,
        betas: np.ndarray,
        gender: str,
    ) -> np.ndarray:
        model = self._model(gender)
        parents = model["parents"]
        joint_count = parents.size
        poses = np.asarray(poses, dtype=np.float64)
        trans = np.asarray(trans, dtype=np.float64)
        betas = np.asarray(betas, dtype=np.float64).reshape(-1)
        if poses.ndim != 2 or poses.shape[1] < joint_count * 3:
            raise ValueError(
                f"SMPL-H poses must be [T, {joint_count * 3}], got {poses.shape}"
            )
        if trans.shape != (poses.shape[0], 3):
            raise ValueError(
                f"SMPL-H translation must be [T, 3], got {trans.shape}"
            )
        if betas.size < 16:
            raise ValueError(f"AMASS requires 16 betas, got {betas.size}")

        shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)[..., :16]
        v_shaped = np.asarray(model["v_template"], dtype=np.float64) + np.einsum(
            "vci,i->vc",
            shapedirs,
            betas[:16],
            optimize=True,
        )
        rest_joints = np.asarray(
            model["J_regressor"], dtype=np.float64
        ) @ v_shaped
        rest_offsets = rest_joints.copy()
        rest_offsets[1:] -= rest_joints[parents[1:]]

        outputs = []
        for start in range(0, poses.shape[0], self.batch_size):
            stop = min(start + self.batch_size, poses.shape[0])
            local_rotations = self._axis_angle_to_matrix(
                poses[start:stop, : joint_count * 3].reshape(
                    -1, joint_count, 3
                )
            )
            global_rotations = np.empty_like(local_rotations)
            posed_joints = np.empty(
                (stop - start, joint_count, 3), dtype=np.float64
            )
            global_rotations[:, 0] = local_rotations[:, 0]
            posed_joints[:, 0] = rest_joints[0]
            for joint in range(1, joint_count):
                parent = parents[joint]
                posed_joints[:, joint] = (
                    posed_joints[:, parent]
                    + np.einsum(
                        "bij,j->bi",
                        global_rotations[:, parent],
                        rest_offsets[joint],
                        optimize=True,
                    )
                )
                global_rotations[:, joint] = np.matmul(
                    global_rotations[:, parent],
                    local_rotations[:, joint],
                )
            posed_joints += trans[start:stop, None, :]
            outputs.append(posed_joints[:, :22])
        return np.concatenate(outputs, axis=0).astype(np.float32)


def load_amass_records(
    root: Path,
    *,
    model_dir: Path,
    max_files: int,
    target_fps: float,
    device: str,
    batch_size: int,
) -> tuple[list[MotionRecord], dict]:
    files = _file_limit(sorted(root.rglob("*_poses.npz")), max_files)
    info = {
        "files_seen": len(files),
        "files_converted_to_xyz": 0,
        "files_skipped": 0,
    }
    extractor = SmplhJointExtractor(model_dir, device=device, batch_size=batch_size)
    records: list[MotionRecord] = []
    for file_index, fp in enumerate(files, start=1):
        try:
            with np.load(fp, allow_pickle=True) as data:
                poses = np.asarray(data["poses"], dtype=np.float32)
                trans = np.asarray(data["trans"], dtype=np.float32)
                betas = np.asarray(data["betas"], dtype=np.float32)
                gender = _scalar_text(data["gender"])
                source_fps = float(np.asarray(data["mocap_framerate"]).reshape(-1)[0])
            if poses.ndim != 2 or poses.shape[1] < 156 or source_fps < target_fps:
                raise ValueError("unsupported AMASS pose shape or source frame rate")
            joints = extractor(poses[:, :156], trans, betas, gender)
            canonical = canonicalize_motion(joints, "smpl24", unit_scale=1.0)
            canonical = lowpass_downsample(
                canonical,
                source_fps=source_fps,
                target_fps=target_fps,
            )
        except (OSError, KeyError, ValueError, RuntimeError):
            info["files_skipped"] += 1
            continue
        records.append(MotionRecord(fp, canonical))
        if file_index % 250 == 0 or file_index == len(files):
            print(
                f"AMASS joint recovery: {file_index}/{len(files)} files "
                f"({len(records)} usable, {info['files_skipped']} skipped)",
                flush=True,
            )
    info["files_converted_to_xyz"] = len(records)
    return records, info


def process_amass(
    root: Path,
    out_dir: Path,
    *,
    model_dir: Path | None,
    max_files: int,
    seed: int,
    target_fps: float,
    min_distance_m: float,
    max_distance_m: float,
    max_pair_frames: int,
    device: str,
    batch_size: int,
) -> dict:
    base = {
        "dataset": "amass",
        "is_multi_person": False,
        "augmentation": "random_cross_sequence_shared_scene",
        "synthetic": True,
        "human_latent_supervision": True,
        "input_representation": "SMPL-H pose parameters",
    }
    if model_dir is None:
        return {
            **base,
            "status": "blocked",
            "reason": (
                "No licensed SMPL-H body-model directory was supplied. Raw AMASS poses "
                "are rotations, not 3-D joints; use --smplh-model-dir after downloading SMPL-H."
            ),
            "files_seen": len(list(root.rglob("*_poses.npz"))),
            "files_written": 0,
        }
    try:
        records, conversion = load_amass_records(
            root,
            model_dir=model_dir,
            max_files=max_files,
            target_fps=target_fps,
            device=device,
            batch_size=batch_size,
        )
    except RuntimeError as exc:
        return {
            **base,
            "status": "blocked",
            "reason": str(exc),
            "files_written": 0,
        }
    info = pair_motion_records(
        records,
        out_dir,
        dataset="amass",
        seed=seed,
        # Every MotionRecord has already been resampled to the common rate;
        # original AMASS clips may each have a different capture rate.
        source_fps=target_fps,
        target_fps=target_fps,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        max_pair_frames=max_pair_frames,
    )
    info.update(conversion)
    return info


def mupots_policy(root: Path) -> dict:
    annotation_count = len(list(root.rglob("annot.mat")))
    return {
        "dataset": "MuPoTS-3D",
        "is_multi_person": True,
        "files_seen": annotation_count,
        "status": "disabled",
        "human_latent_supervision": False,
        "reason": (
            "MuPoTS-3D is a multi-person 3-D pose TEST benchmark, not a dedicated "
            "intent-labelled interaction training set. Training on it would leak the test benchmark."
        ),
        "license_note": "local license permits non-commercial use only",
    }


def _first_existing(paths: Iterable[Path]) -> Path:
    candidates = list(paths)
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build Stage-1 H2H data. H36M/AMASS are random cross-clip synthetic "
            "pairs; every generated pair can supervise cross-attention and the "
            "trajectory intent latent, while recorded pairs retain provenance."
        )
    )
    parser.add_argument("--datasets-root", type=Path, default=Path(r"D:\datasets"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("h36m", "amass", "3dpw", "mupots"),
        default=("h36m", "amass", "3dpw", "mupots"),
    )
    parser.add_argument("--max-files-per-dataset", type=int, default=0, help="0 uses all files")
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-pair-distance-m", type=float, default=0.8)
    parser.add_argument("--max-pair-distance-m", type=float, default=2.0)
    parser.add_argument("--max-pair-frames", type=int, default=0, help="0 keeps the full common duration")
    parser.add_argument("--smplh-model-dir", type=Path, default=None)
    parser.add_argument("--smpl-device", type=str, default="cuda")
    parser.add_argument("--smpl-batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.datasets_root
    roots = {
        "3dpw": _first_existing((root / "3DPW" / "3DPW", root / "3DPW")),
        "amass": _first_existing((root / "amass" / "amass", root / "amass")),
        "h36m": _first_existing((root / "h3.6m" / "h3.6m", root / "h3.6m")),
        "mupots": _first_existing(
            (
                root / "MuPots-3d" / "MuPots-3d",
                root / "MuPots-3d",
            )
        ),
    }

    reports = []
    selected = set(args.datasets)
    if "3dpw" in selected:
        reports.append(
            process_3dpw(
                roots["3dpw"],
                roots["3dpw"] / "data_aug",
                args.max_files_per_dataset,
            )
        )
    if "h36m" in selected:
        reports.append(
            process_h36m(
                roots["h36m"],
                roots["h36m"] / "data_aug",
                max_files=args.max_files_per_dataset,
                seed=args.seed,
                target_fps=args.target_fps,
                min_distance_m=args.min_pair_distance_m,
                max_distance_m=args.max_pair_distance_m,
                max_pair_frames=args.max_pair_frames,
            )
        )
    if "amass" in selected:
        reports.append(
            process_amass(
                roots["amass"],
                roots["amass"] / "data_aug",
                model_dir=args.smplh_model_dir,
                max_files=args.max_files_per_dataset,
                seed=args.seed + 1,
                target_fps=args.target_fps,
                min_distance_m=args.min_pair_distance_m,
                max_distance_m=args.max_pair_distance_m,
                max_pair_frames=args.max_pair_frames,
                device=args.smpl_device,
                batch_size=args.smpl_batch_size,
            )
        )
    if "mupots" in selected:
        reports.append(mupots_policy(roots["mupots"]))

    summary = {
        "datasets_root": str(root),
        "seed": args.seed,
        "target_fps": args.target_fps,
        "policy": {
            "synthetic_cross_sequence_pairs": (
                "motion + cross-attention interaction token + trajectory intent latent; "
                "interaction_valid=0 is retained only as legacy provenance"
            ),
            "recorded_synchronous_pairs": (
                "same objectives with higher Stage-1 sampling weight; interaction_valid=1"
            ),
        },
        "results": reports,
    }
    out = root / "h2h_pretrain_data_aug_report.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved report -> {out}")


if __name__ == "__main__":
    main()
