"""Build temporal two-person Stage-1 data from MPI-INF-3DHP annotations.

Official MuCo-3DHP composites independently sampled single-person frames for
multi-person pose training.  A motion-intent model instead needs continuous
trajectories, so this converter reads the underlying MPI-INF-3DHP sequences,
splits them into clips, pairs clips from different subjects, and places both
actors in one metric scene.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from dataset.cmu_interaction_converter import lowpass_downsample
    from dataset.data_prepocess import place_in_shared_scene
    from dataset.skeleton_utils import canonicalize_motion, joint_names_array
except ModuleNotFoundError:
    from cmu_interaction_converter import lowpass_downsample
    from data_prepocess import place_in_shared_scene
    from skeleton_utils import canonicalize_motion, joint_names_array


@dataclass(frozen=True)
class MucoClip:
    source: Path
    subject: int
    sequence: int
    source_fps: float
    start_frame_30hz: int
    motion: np.ndarray


def _loadmat(path: Path, variable_names: tuple[str, ...]) -> dict:
    try:
        from scipy.io import loadmat
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MuCo conversion requires scipy (already listed in requirements.txt)"
        ) from exc
    return loadmat(path, variable_names=list(variable_names))


def subject_sequence_from_path(path: Path) -> tuple[int, int]:
    try:
        subject = int(path.parent.parent.name.removeprefix("S"))
        sequence = int(path.parent.name.removeprefix("Seq"))
    except ValueError as exc:
        raise ValueError(f"Expected .../Sx/Seqy/annot.mat, got {path}") from exc
    return subject, sequence


def official_source_fps(subject: int, sequence: int) -> float:
    """Return the per-sequence FPS from mpii_get_sequence_info.m."""

    if not 1 <= subject <= 8 or sequence not in {1, 2}:
        raise ValueError(f"Unsupported MPI-INF-3DHP sequence S{subject}/Seq{sequence}")
    return 50.0 if (subject, sequence) in {
        (1, 2),
        (3, 1),
        (3, 2),
        (5, 1),
        (5, 2),
    } else 25.0


def resample_motion(
    motion: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    if target_fps <= source_fps:
        return lowpass_downsample(
            motion,
            source_fps=source_fps,
            target_fps=target_fps,
        )
    # 25 Hz sequences need a modest interpolation to the common 30 Hz grid.
    values = np.asarray(motion, dtype=np.float64)
    target_count = int(
        np.floor((values.shape[0] - 1) * target_fps / source_fps)
    ) + 1
    source_time = np.arange(values.shape[0], dtype=np.float64) / source_fps
    target_time = np.arange(target_count, dtype=np.float64) / target_fps
    flat = values.reshape(values.shape[0], -1)
    output = np.empty((target_count, flat.shape[1]), dtype=np.float64)
    for column in range(flat.shape[1]):
        output[:, column] = np.interp(target_time, source_time, flat[:, column])
    return output.reshape((-1, *values.shape[1:])).astype(np.float32)


def load_continuous_motion(
    annotation_path: Path,
    *,
    camera: int = 0,
    annotation_stream: str = "univ_annot3",
    source_fps: float = 50.0,
    target_fps: float = 30.0,
) -> np.ndarray:
    if annotation_stream not in {"annot3", "univ_annot3"}:
        raise ValueError("annotation_stream must be annot3 or univ_annot3")
    data = _loadmat(
        annotation_path,
        ("cameras", "frames", annotation_stream),
    )
    cameras = np.asarray(data["cameras"]).reshape(-1)
    matches = np.flatnonzero(cameras == camera)
    if matches.size != 1:
        raise ValueError(
            f"{annotation_path} contains {matches.size} entries for camera {camera}"
        )
    annotations = np.asarray(data[annotation_stream]).reshape(-1)[int(matches[0])]
    frames = np.asarray(data["frames"]).reshape(-1)
    frame_count = min(len(frames), annotations.shape[0])
    if annotations.ndim != 2 or annotations.shape[1] != 28 * 3:
        raise ValueError(
            f"Expected {annotation_stream} [T,84], got {annotations.shape}"
        )

    points = annotations[:frame_count].reshape(frame_count, 28, 3).astype(np.float32)
    # MPI camera coordinates use +Y downward. HARPER and the other Stage-1
    # sources use +Y upward.
    points[..., 1] *= -1.0
    canonical = canonicalize_motion(
        points,
        "mpiinf3dhp28",
        unit_scale=0.001,
        validate_scale=True,
    )
    return resample_motion(
        canonical,
        source_fps=source_fps,
        target_fps=target_fps,
    )


def split_motion_into_clips(
    motion: np.ndarray,
    *,
    clip_frames: int,
    stride_frames: int,
    min_frames: int,
) -> list[tuple[int, np.ndarray]]:
    if min_frames <= 0 or clip_frames < min_frames or stride_frames <= 0:
        raise ValueError("Require clip_frames >= min_frames > 0 and stride_frames > 0")
    clips = []
    for start in range(0, motion.shape[0], stride_frames):
        clip = motion[start : start + clip_frames]
        if clip.shape[0] >= min_frames:
            clips.append((start, clip.copy()))
    return clips


def different_subject_partner_indices(
    clips: list[MucoClip],
    rng: np.random.Generator,
) -> np.ndarray:
    if len(clips) < 2:
        raise ValueError("At least two clips are required")
    subjects = np.asarray([clip.subject for clip in clips])
    unique_subjects = np.unique(subjects)
    groups = {
        subject: rng.permutation(np.flatnonzero(subjects == subject))
        for subject in unique_subjects
    }
    maximum_group = max(len(indices) for indices in groups.values())
    if maximum_group > len(clips) // 2:
        raise ValueError("No one-to-one different-subject pairing is possible")

    # Arrange each subject contiguously in random subject order, then rotate by
    # the largest group size. No group is wide enough to overlap itself after
    # this rotation, and every clip is used exactly once as a partner.
    subject_order = rng.permutation(unique_subjects)
    ordered = np.concatenate([groups[subject] for subject in subject_order])
    paired_order = np.roll(ordered, -maximum_group)
    partners = np.empty(len(clips), dtype=np.int64)
    partners[ordered] = paired_order
    if np.any(subjects[partners] == subjects):
        raise AssertionError("Different-subject pairing construction failed")
    return partners


def _clip_name(clip: MucoClip) -> str:
    stop = clip.start_frame_30hz + clip.motion.shape[0] - 1
    return (
        f"S{clip.subject}_Seq{clip.sequence}_"
        f"f{clip.start_frame_30hz:06d}-{stop:06d}"
    )


def save_pair(
    path: Path,
    actor_a: MucoClip,
    actor_b: MucoClip,
    person_a: np.ndarray,
    person_b: np.ndarray,
    *,
    camera: int,
    annotation_stream: str,
    target_fps: float,
    placement: dict,
) -> None:
    np.savez_compressed(
        path,
        person_a=np.asarray(person_a, dtype=np.float32),
        person_b=np.asarray(person_b, dtype=np.float32),
        source_a=str(actor_a.source),
        source_b=str(actor_b.source),
        subject_a=np.int16(actor_a.subject),
        subject_b=np.int16(actor_b.subject),
        sequence_a=np.int8(actor_a.sequence),
        sequence_b=np.int8(actor_b.sequence),
        source_fps_a=np.float32(actor_a.source_fps),
        source_fps_b=np.float32(actor_b.source_fps),
        clip_start_a_30hz=np.int32(actor_a.start_frame_30hz),
        clip_start_b_30hz=np.int32(actor_b.start_frame_30hz),
        dataset="MuCo-3DHP-temporal",
        split="train",
        synthetic=np.bool_(True),
        interaction_valid=np.float32(0.0),
        recorded_synchronous=np.bool_(False),
        intent_training_eligible=np.bool_(True),
        pairing_strategy="different_subject_cross_sequence_shared_scene",
        joint_layout="optitrack21",
        joint_names=joint_names_array(),
        unit="m",
        unit_scale_to_m=np.float32(1.0),
        target_fps=np.float32(target_fps),
        camera=np.int8(camera),
        annotation_stream=annotation_stream,
        yaw_b_deg=np.float32(placement["yaw_b_deg"]),
        pair_distance_m=np.float32(placement["distance_m"]),
    )


def convert_dataset(
    input_root: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    camera: int = 0,
    annotation_stream: str = "univ_annot3",
    source_fps: float = 0.0,
    target_fps: float = 30.0,
    clip_frames: int = 300,
    stride_frames: int = 300,
    min_frames: int = 70,
    min_pair_distance_m: float = 0.8,
    max_pair_distance_m: float = 2.0,
    overwrite: bool = False,
    max_sequences: int = 0,
) -> dict:
    annotation_files = sorted(input_root.glob("S*/Seq*/annot.mat"))
    if max_sequences > 0:
        annotation_files = annotation_files[:max_sequences]
    if not annotation_files:
        raise FileNotFoundError(f"No S*/Seq*/annot.mat files found under {input_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("*.npz"))
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains {len(existing)} NPZ files; "
            "use --overwrite only when intentionally replacing them"
        )
    if existing and overwrite:
        for path in existing:
            path.unlink()

    clips: list[MucoClip] = []
    sequence_report = []
    for index, annotation_path in enumerate(annotation_files, start=1):
        subject, sequence = subject_sequence_from_path(annotation_path)
        sequence_source_fps = (
            float(source_fps)
            if source_fps > 0
            else official_source_fps(subject, sequence)
        )
        motion = load_continuous_motion(
            annotation_path,
            camera=camera,
            annotation_stream=annotation_stream,
            source_fps=sequence_source_fps,
            target_fps=target_fps,
        )
        sequence_clips = split_motion_into_clips(
            motion,
            clip_frames=clip_frames,
            stride_frames=stride_frames,
            min_frames=min_frames,
        )
        clips.extend(
            MucoClip(
                annotation_path,
                subject,
                sequence,
                sequence_source_fps,
                start,
                clip,
            )
            for start, clip in sequence_clips
        )
        sequence_report.append(
            {
                "subject": subject,
                "sequence": sequence,
                "source_fps": sequence_source_fps,
                "frames_30hz": int(motion.shape[0]),
                "clips": len(sequence_clips),
            }
        )
        print(
            f"MuCo trajectory extraction: {index}/{len(annotation_files)} "
            f"S{subject}/Seq{sequence} ({sequence_source_fps:g} Hz), "
            f"{motion.shape[0]} frames, "
            f"{len(sequence_clips)} clips",
            flush=True,
        )

    rng = np.random.default_rng(seed)
    partners = different_subject_partner_indices(clips, rng)
    files_written = 0
    frames_written = 0
    for index_a, index_b in enumerate(partners):
        actor_a = clips[index_a]
        actor_b = clips[int(index_b)]
        person_a, person_b, placement = place_in_shared_scene(
            actor_a.motion,
            actor_b.motion,
            rng,
            min_distance_m=min_pair_distance_m,
            max_distance_m=max_pair_distance_m,
        )
        filename = f"{index_a:05d}_{_clip_name(actor_a)}__{_clip_name(actor_b)}.npz"
        save_pair(
            output_dir / filename,
            actor_a,
            actor_b,
            person_a,
            person_b,
            camera=camera,
            annotation_stream=annotation_stream,
            target_fps=target_fps,
            placement=placement,
        )
        files_written += 1
        frames_written += int(person_a.shape[0])
        if files_written % 100 == 0 or files_written == len(clips):
            print(
                f"MuCo pair generation: {files_written}/{len(clips)} files",
                flush=True,
            )

    report = {
        "dataset": "MuCo-3DHP-temporal",
        "status": "ok",
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "annotation_stream": annotation_stream,
        "camera": camera,
        "source_fps": (
            source_fps if source_fps > 0 else "official per-sequence 25/50 Hz"
        ),
        "target_fps": target_fps,
        "clip_frames": clip_frames,
        "stride_frames": stride_frames,
        "minimum_training_frames": min_frames,
        "pairing": "one-to-one random permutation with different subjects",
        "synthetic": True,
        "recorded_synchronous": False,
        "intent_training_eligible": True,
        "sequences": sequence_report,
        "sequence_count": len(annotation_files),
        "files_written": files_written,
        "frames_written": frames_written,
    }
    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved report -> {report_path}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert continuous MPI-INF-3DHP annotations into synthetic "
            "two-person temporal MuCo Stage-1 data"
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            r"D:\datasets\MuPots-3d\MuPots-3d\MuCo-3DHP\MPI-INF-3DHP"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            r"D:\datasets\MuPots-3d\MuPots-3d\MuCo-3DHP\data_aug"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--annotation-stream",
        choices=("annot3", "univ_annot3"),
        default="univ_annot3",
    )
    parser.add_argument(
        "--source-fps",
        type=float,
        default=0.0,
        help="Override all source FPS values; 0 uses official per-sequence 25/50 Hz",
    )
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--clip-frames", type=int, default=300)
    parser.add_argument("--stride-frames", type=int, default=300)
    parser.add_argument("--min-frames", type=int, default=70)
    parser.add_argument("--min-pair-distance-m", type=float, default=0.8)
    parser.add_argument("--max-pair-distance-m", type=float, default=2.0)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = convert_dataset(
        args.input_root,
        args.output_dir,
        seed=args.seed,
        camera=args.camera,
        annotation_stream=args.annotation_stream,
        source_fps=args.source_fps,
        target_fps=args.target_fps,
        clip_frames=args.clip_frames,
        stride_frames=args.stride_frames,
        min_frames=args.min_frames,
        min_pair_distance_m=args.min_pair_distance_m,
        max_pair_distance_m=args.max_pair_distance_m,
        overwrite=args.overwrite,
        max_sequences=args.max_sequences,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
