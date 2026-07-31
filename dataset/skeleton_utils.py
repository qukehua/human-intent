from __future__ import annotations

from typing import Iterable

import numpy as np


# HARPER uses the OptiTrack Baseline skeleton. Keep this order as the single
# representation shared by all human-motion pretraining sources.
CANONICAL_21_JOINT_NAMES = (
    "Hips",
    "Spine",
    "Spine1",
    "Neck",
    "Head",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
)

CANONICAL_ROOT_INDEX = 0

# SMPL-24:
# pelvis, l/r hip, spine1, l/r knee, spine2, l/r ankle, spine3,
# l/r foot, neck, l/r collar, head, l/r shoulder, l/r elbow,
# l/r wrist, l/r hand.
SMPL24_TO_OPTITRACK21 = np.asarray(
    [
        0,
        3,
        9,
        12,
        15,
        13,
        16,
        18,
        20,
        14,
        17,
        19,
        21,
        1,
        4,
        7,
        10,
        2,
        5,
        8,
        11,
    ],
    dtype=np.int64,
)

# MPI-INF-3DHP 28-joint order:
# spine3, spine4, spine2, spine, pelvis, neck, head, head_top,
# left clavicle/shoulder/elbow/wrist/hand, right equivalents,
# left hip/knee/ankle/foot/toe, right equivalents.
MPI_INF_3DHP_28_JOINT_NAMES = (
    "spine3",
    "spine4",
    "spine2",
    "spine",
    "pelvis",
    "neck",
    "head",
    "head_top",
    "left_clavicle",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "right_clavicle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "left_toe",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "right_toe",
)

LEFT_RIGHT_PAIRS = (
    (5, 9),
    (6, 10),
    (7, 11),
    (8, 12),
    (13, 17),
    (14, 18),
    (15, 19),
    (16, 20),
)


def _normalise_layout_name(layout: str) -> str:
    name = str(layout).strip().lower().replace("_", "").replace("-", "").replace(".", "")
    aliases = {
        "canonical21": "optitrack21",
        "harper21": "optitrack21",
        "optitrack21": "optitrack21",
        "smpl": "smpl24",
        "smpl24": "smpl24",
        "3dpw": "smpl24",
        "amass": "smpl24",
        "h36m": "h36m32",
        "h36m32": "h36m32",
        "human36m": "h36m32",
        "mpiinf3dhp": "mpiinf3dhp28",
        "mpiinf3dhp28": "mpiinf3dhp28",
        "muco3dhp": "mpiinf3dhp28",
        "auto": "auto",
    }
    if name not in aliases:
        raise ValueError(f"Unsupported joint layout: {layout!r}")
    return aliases[name]


def _as_motion(motion: np.ndarray) -> np.ndarray:
    array = np.asarray(motion, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] % 3 == 0:
        array = array.reshape(array.shape[0], -1, 3)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Motion must have shape [T, J, 3], got {array.shape}")
    if array.shape[0] < 1:
        raise ValueError("Motion contains no frames")
    if not np.isfinite(array).all():
        raise ValueError("Motion contains NaN or infinite coordinates")
    return array


def layout_is_compatible(joint_count: int, layout: str) -> bool:
    name = _normalise_layout_name(layout)
    if name == "auto":
        return joint_count in {21, 22, 24, 32}
    if name == "optitrack21":
        return joint_count == 21
    if name == "smpl24":
        return joint_count in {22, 24}
    if name == "h36m32":
        return joint_count == 32
    if name == "mpiinf3dhp28":
        return joint_count == 28
    return False


def _h36m32_to_optitrack21(motion: np.ndarray) -> np.ndarray:
    # Human3.6M 32-joint position order. Static end sites are intentionally
    # ignored. OptiTrack has explicit clavicles, which H36M does not, so use
    # thorax/shoulder midpoints for those two joints.
    output = np.empty((motion.shape[0], 21, 3), dtype=np.float32)
    output[:, 0] = motion[:, 0]  # Hips
    output[:, 1] = motion[:, 12]  # Spine
    output[:, 2] = motion[:, 13]  # Spine1 / thorax
    output[:, 3] = motion[:, 14]  # Neck
    output[:, 4] = motion[:, 15]  # Head
    output[:, 5] = 0.5 * (motion[:, 13] + motion[:, 17])  # LeftShoulder
    output[:, 6] = motion[:, 17]  # LeftArm
    output[:, 7] = motion[:, 18]  # LeftForeArm
    output[:, 8] = motion[:, 19]  # LeftHand/wrist
    output[:, 9] = 0.5 * (motion[:, 13] + motion[:, 25])  # RightShoulder
    output[:, 10] = motion[:, 25]  # RightArm
    output[:, 11] = motion[:, 26]  # RightForeArm
    output[:, 12] = motion[:, 27]  # RightHand/wrist
    output[:, 13] = motion[:, 6]  # LeftUpLeg
    output[:, 14] = motion[:, 7]  # LeftLeg
    output[:, 15] = motion[:, 8]  # LeftFoot
    output[:, 16] = motion[:, 9]  # LeftToeBase
    output[:, 17] = motion[:, 1]  # RightUpLeg
    output[:, 18] = motion[:, 2]  # RightLeg
    output[:, 19] = motion[:, 3]  # RightFoot
    output[:, 20] = motion[:, 4]  # RightToeBase
    return output


def mpi_inf_3dhp_28_to_optitrack21(motion: np.ndarray) -> np.ndarray:
    """Map the official MPI-INF-3DHP 28-joint order to HARPER-21.

    As in the H36M and CMU converters, canonical shoulder tokens are placed
    between the thorax and anatomical shoulder, while arm/forearm/hand tokens
    use shoulder/elbow/wrist positions.
    """

    source = _as_motion(motion)
    if source.shape[1] != len(MPI_INF_3DHP_28_JOINT_NAMES):
        raise ValueError(f"Expected MPI-INF-3DHP [T, 28, 3], got {source.shape}")
    output = np.empty((source.shape[0], 21, 3), dtype=np.float32)
    output[:, 0] = source[:, 4]  # pelvis
    output[:, 1] = source[:, 3]  # spine
    output[:, 2] = source[:, 1]  # upper thorax / spine4
    output[:, 3] = source[:, 5]  # neck
    output[:, 4] = source[:, 6]  # head
    output[:, 5] = 0.5 * (source[:, 1] + source[:, 9])
    output[:, 6] = source[:, 9]  # left shoulder
    output[:, 7] = source[:, 10]  # left elbow
    output[:, 8] = source[:, 11]  # left wrist
    output[:, 9] = 0.5 * (source[:, 1] + source[:, 14])
    output[:, 10] = source[:, 14]  # right shoulder
    output[:, 11] = source[:, 15]  # right elbow
    output[:, 12] = source[:, 16]  # right wrist
    output[:, 13] = source[:, 18]  # left hip
    output[:, 14] = source[:, 19]  # left knee
    output[:, 15] = source[:, 20]  # left ankle
    output[:, 16] = source[:, 22]  # left toe
    output[:, 17] = source[:, 23]  # right hip
    output[:, 18] = source[:, 24]  # right knee
    output[:, 19] = source[:, 25]  # right ankle
    output[:, 20] = source[:, 27]  # right toe
    return output


def _median_body_extent(motion: np.ndarray) -> float:
    frame_min = motion.min(axis=1)
    frame_max = motion.max(axis=1)
    extent = np.linalg.norm(frame_max - frame_min, axis=-1)
    return float(np.median(extent))


def canonicalize_motion(
    motion: np.ndarray,
    layout: str,
    unit_scale: float = 1.0,
    *,
    validate_scale: bool = True,
    min_extent_m: float = 0.25,
    max_extent_m: float = 3.5,
) -> np.ndarray:
    """Map a motion sequence to HARPER/OptiTrack-21 coordinates in metres."""

    source = _as_motion(motion)
    name = _normalise_layout_name(layout)
    if name == "auto":
        if source.shape[1] == 21:
            name = "optitrack21"
        elif source.shape[1] in {22, 24}:
            name = "smpl24"
        elif source.shape[1] == 32:
            name = "h36m32"
        elif source.shape[1] == 28:
            name = "mpiinf3dhp28"
        else:
            raise ValueError(f"Cannot infer layout from {source.shape[1]} joints")

    if not layout_is_compatible(source.shape[1], name):
        raise ValueError(f"Layout {name} is incompatible with {source.shape[1]} joints")

    if name == "optitrack21":
        canonical = source.copy()
    elif name == "smpl24":
        canonical = source[:, SMPL24_TO_OPTITRACK21].copy()
    elif name == "h36m32":
        canonical = _h36m32_to_optitrack21(source)
    elif name == "mpiinf3dhp28":
        canonical = mpi_inf_3dhp_28_to_optitrack21(source)
    else:
        raise AssertionError(f"Unhandled layout: {name}")

    scale = float(unit_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"unit_scale must be positive and finite, got {unit_scale}")
    canonical *= scale

    if validate_scale:
        extent = _median_body_extent(canonical)
        if not min_extent_m <= extent <= max_extent_m:
            raise ValueError(
                f"Implausible body extent {extent:.3f} m after scaling; "
                f"expected [{min_extent_m:.3f}, {max_extent_m:.3f}] m"
            )
    return canonical


def scene_center_pair(
    person_a: np.ndarray,
    person_b: np.ndarray,
    *,
    root_index: int = CANONICAL_ROOT_INDEX,
) -> tuple[np.ndarray, np.ndarray]:
    """Use person A's first-frame root as a shared origin for both people."""

    if person_a.shape[-2:] != (21, 3) or person_b.shape[-2:] != (21, 3):
        raise ValueError("scene_center_pair expects two canonical [T, 21, 3] motions")
    origin = np.asarray(person_a[0, root_index], dtype=np.float32).copy()
    return person_a - origin, person_b - origin


def mirror_canonical_x(motion: np.ndarray, swap_left_right: bool = True) -> np.ndarray:
    """Mirror a canonical skeleton while preserving anatomical joint labels."""

    mirrored = np.asarray(motion, dtype=np.float32).copy()
    if mirrored.ndim != 3 or mirrored.shape[1:] != (21, 3):
        raise ValueError(f"Expected [T, 21, 3], got {mirrored.shape}")
    mirrored[..., 0] *= -1.0
    if swap_left_right:
        for left, right in LEFT_RIGHT_PAIRS:
            left_value = mirrored[:, left].copy()
            mirrored[:, left] = mirrored[:, right]
            mirrored[:, right] = left_value
    return mirrored


def joint_names_array(names: Iterable[str] = CANONICAL_21_JOINT_NAMES) -> np.ndarray:
    return np.asarray(tuple(names), dtype="<U32")
