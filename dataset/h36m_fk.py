from __future__ import annotations

import numpy as np


# Standard 32-joint Human3.6M kinematic tree and offsets used by the
# Martinez et al. motion-prediction preprocessing. Offsets are millimetres.
H36M_PARENTS = np.asarray(
    [
        -1,
        0,
        1,
        2,
        3,
        4,
        0,
        6,
        7,
        8,
        9,
        0,
        11,
        12,
        13,
        14,
        12,
        16,
        17,
        18,
        19,
        20,
        19,
        22,
        12,
        24,
        25,
        26,
        27,
        28,
        27,
        30,
    ],
    dtype=np.int64,
)

H36M_OFFSETS_MM = np.asarray(
    [
        0.000000,
        0.000000,
        0.000000,
        -132.948591,
        0.000000,
        0.000000,
        0.000000,
        -442.894612,
        0.000000,
        0.000000,
        -454.206447,
        0.000000,
        0.000000,
        0.000000,
        162.767078,
        0.000000,
        0.000000,
        74.999437,
        132.948826,
        0.000000,
        0.000000,
        0.000000,
        -442.894413,
        0.000000,
        0.000000,
        -454.206590,
        0.000000,
        0.000000,
        0.000000,
        162.767426,
        0.000000,
        0.000000,
        74.999948,
        0.000000,
        0.100000,
        0.000000,
        0.000000,
        233.383263,
        0.000000,
        0.000000,
        257.077681,
        0.000000,
        0.000000,
        121.134938,
        0.000000,
        0.000000,
        115.002227,
        0.000000,
        0.000000,
        257.077681,
        0.000000,
        0.000000,
        151.034226,
        0.000000,
        0.000000,
        278.882773,
        0.000000,
        0.000000,
        251.733451,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        99.999627,
        0.000000,
        100.000188,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        257.077681,
        0.000000,
        0.000000,
        151.031437,
        0.000000,
        0.000000,
        278.892924,
        0.000000,
        0.000000,
        251.728680,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
        99.999888,
        0.000000,
        137.499922,
        0.000000,
        0.000000,
        0.000000,
        0.000000,
    ],
    dtype=np.float64,
).reshape(32, 3)

def expmap_to_rotmat(expmap: np.ndarray) -> np.ndarray:
    """Vectorised Rodrigues conversion for arrays ending in three values."""

    values = np.asarray(expmap, dtype=np.float64)
    if values.shape[-1] != 3:
        raise ValueError(f"Expected exponential maps ending in 3, got {values.shape}")
    theta = np.linalg.norm(values, axis=-1, keepdims=True)
    axis = values / np.maximum(theta, np.finfo(np.float64).eps)
    x, y, z = np.moveaxis(axis, -1, 0)
    zeros = np.zeros_like(x)
    skew = np.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        axis=-1,
    ).reshape((*values.shape[:-1], 3, 3))
    identity = np.broadcast_to(np.eye(3, dtype=np.float64), skew.shape)
    sin_theta = np.sin(theta)[..., None]
    one_minus_cos = (1.0 - np.cos(theta))[..., None]
    return identity + sin_theta * skew + one_minus_cos * np.matmul(skew, skew)


def recover_root_transform(expmap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integrate SRNN-style root translation and rotation increments."""

    angles = np.asarray(expmap, dtype=np.float64)
    if angles.ndim == 1:
        angles = angles[None]
    if angles.ndim != 2 or angles.shape[1] != 99:
        raise ValueError(f"Expected Human3.6M exponential maps [T, 99], got {angles.shape}")

    translations = np.empty((angles.shape[0], 3), dtype=np.float64)
    rotations = np.empty((angles.shape[0], 3, 3), dtype=np.float64)
    previous_translation = np.zeros(3, dtype=np.float64)
    previous_rotation = np.eye(3, dtype=np.float64)
    root_differences = expmap_to_rotmat(angles[:, 3:6])
    for frame in range(angles.shape[0]):
        translations[frame] = (
            previous_translation + previous_rotation.T @ angles[frame, :3]
        )
        rotations[frame] = root_differences[frame] @ previous_rotation
        previous_translation = translations[frame]
        previous_rotation = rotations[frame]
    return translations, rotations


def h36m_expmap_to_xyz(
    expmap: np.ndarray,
    *,
    y_up: bool = True,
    recover_trajectory: bool = True,
) -> np.ndarray:
    """Convert standard Human3.6M 99-D exponential maps to [T, 32, 3].

    The returned coordinates are millimetres. ``y_up=True`` keeps the raw
    Human3.6M skeleton's vertical axis aligned with the project's Y-up
    HARPER/OptiTrack convention. Set it to false only to reproduce the
    historical motion-prediction visualisation order ``[x, z, y]``.
    ``recover_trajectory`` integrates the root translation/orientation
    increments stored by the SRNN motion-prediction preprocessing.
    """

    angles = np.asarray(expmap, dtype=np.float64)
    if angles.ndim == 1:
        angles = angles[None]
    if angles.ndim != 2 or angles.shape[1] != 99:
        raise ValueError(f"Expected Human3.6M exponential maps [T, 99], got {angles.shape}")
    if not np.isfinite(angles).all():
        raise ValueError("Human3.6M exponential maps contain NaN or infinity")

    frame_count = angles.shape[0]
    local_rotations = expmap_to_rotmat(angles[:, 3:].reshape(frame_count, 32, 3))
    if recover_trajectory:
        root_positions, root_rotations = recover_root_transform(angles)
        local_rotations[:, 0] = root_rotations
    else:
        root_positions = angles[:, :3]

    xyz = np.empty((frame_count, 32, 3), dtype=np.float64)
    global_rotations = np.empty((frame_count, 32, 3, 3), dtype=np.float64)
    for joint, parent in enumerate(H36M_PARENTS):
        if parent < 0:
            global_rotations[:, joint] = local_rotations[:, joint]
            xyz[:, joint] = H36M_OFFSETS_MM[joint] + root_positions
        else:
            xyz[:, joint] = (
                np.einsum(
                    "ti,tij->tj",
                    np.broadcast_to(H36M_OFFSETS_MM[joint], (frame_count, 3)),
                    global_rotations[:, parent],
                )
                + xyz[:, parent]
            )
            global_rotations[:, joint] = np.matmul(
                local_rotations[:, joint],
                global_rotations[:, parent],
            )

    if not y_up:
        xyz = xyz[..., [0, 2, 1]]
    return xyz.astype(np.float32)
