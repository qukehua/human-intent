from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from dataset.skeleton_utils import CANONICAL_21_JOINT_NAMES, joint_names_array
except ModuleNotFoundError:
    from skeleton_utils import CANONICAL_21_JOINT_NAMES, joint_names_array


CMU_SOURCE_FPS = 120


@dataclass(frozen=True)
class Bone:
    name: str
    direction: np.ndarray
    length: float
    axis: np.ndarray
    axis_order: str
    dof: tuple[str, ...]
    parent: str


@dataclass(frozen=True)
class Skeleton:
    bones: dict[str, Bone]
    children: dict[str, tuple[str, ...]]
    traversal: tuple[str, ...]
    root_order: tuple[str, ...]
    root_axis_order: str
    root_position: np.ndarray
    root_orientation: np.ndarray
    length_scale_to_m: float


@dataclass(frozen=True)
class Motion:
    frame_ids: np.ndarray
    channels: tuple[dict[str, np.ndarray], ...]
    angle_unit: str


def _clean_lines(path: Path) -> list[str]:
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _section(lines: list[str], name: str) -> tuple[int, int]:
    start = next((i for i, line in enumerate(lines) if line.lower() == name.lower()), None)
    if start is None:
        raise ValueError(f"Missing {name} section")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(":")), len(lines))
    return start + 1, end


def _parse_float_vector(values: Iterable[str], expected: int, field: str) -> np.ndarray:
    parsed = np.asarray([float(value) for value in values], dtype=np.float64)
    if parsed.shape != (expected,):
        raise ValueError(f"{field} must contain {expected} values, got {parsed.shape}")
    if not np.isfinite(parsed).all():
        raise ValueError(f"{field} contains NaN or infinity")
    return parsed


def _build_traversal(children: dict[str, tuple[str, ...]], bone_names: set[str]) -> tuple[str, ...]:
    traversal: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise ValueError(f"Cycle in ASF hierarchy at {name}")
        if name in visited:
            return
        visiting.add(name)
        traversal.append(name)
        for child in children.get(name, ()):
            visit(child)
        visiting.remove(name)
        visited.add(name)

    visit("root")
    expected = {"root", *bone_names}
    if visited != expected:
        missing = sorted(expected - visited)
        extra = sorted(visited - expected)
        raise ValueError(f"Disconnected ASF hierarchy: missing={missing}, extra={extra}")
    return tuple(traversal)


def parse_asf(path: Path) -> Skeleton:
    """Parse the CMU ASF subset used by the interaction recordings."""

    path = Path(path)
    lines = _clean_lines(path)

    units_start, units_end = _section(lines, ":units")
    length_unit = None
    for line in lines[units_start:units_end]:
        tokens = line.split()
        if tokens[0].lower() == "length":
            length_unit = float(tokens[1])
    if length_unit is None or not np.isfinite(length_unit) or length_unit <= 0:
        raise ValueError(f"{path}: invalid or missing :units length")
    # CMU documents that ASF values were multiplied by :units length before
    # being stored and are otherwise inches.
    length_scale_to_m = 0.0254 / length_unit

    root_start, root_end = _section(lines, ":root")
    root_fields: dict[str, list[str]] = {}
    for line in lines[root_start:root_end]:
        tokens = line.split()
        root_fields[tokens[0].lower()] = tokens[1:]
    root_order = tuple(token.lower() for token in root_fields.get("order", ()))
    if set(root_order) != {"tx", "ty", "tz", "rx", "ry", "rz"}:
        raise ValueError(f"{path}: unsupported root order {root_order}")
    root_axis_order = "".join(root_fields.get("axis", ("XYZ",))).upper()
    if root_axis_order != "XYZ":
        raise ValueError(f"{path}: only CMU XYZ root axes are supported, got {root_axis_order}")
    root_position = _parse_float_vector(root_fields.get("position", ()), 3, "root position")
    root_orientation = _parse_float_vector(root_fields.get("orientation", ()), 3, "root orientation")

    bonedata_start, bonedata_end = _section(lines, ":bonedata")
    raw_bones: dict[str, dict[str, object]] = {}
    i = bonedata_start
    while i < bonedata_end:
        if lines[i].lower() != "begin":
            i += 1
            continue
        i += 1
        block: dict[str, object] = {}
        while i < bonedata_end and lines[i].lower() != "end":
            tokens = lines[i].split()
            key = tokens[0].lower()
            if key == "name":
                block["name"] = tokens[1].lower()
            elif key == "direction":
                block["direction"] = _parse_float_vector(tokens[1:], 3, "bone direction")
            elif key == "length":
                block["length"] = float(tokens[1])
            elif key == "axis":
                block["axis"] = _parse_float_vector(tokens[1:4], 3, "bone axis")
                block["axis_order"] = tokens[4].upper()
            elif key == "dof":
                block["dof"] = tuple(token.lower() for token in tokens[1:])
            i += 1
        required = {"name", "direction", "length", "axis", "axis_order"}
        missing = required - block.keys()
        if missing:
            raise ValueError(f"{path}: incomplete ASF bone block, missing {sorted(missing)}")
        name = str(block["name"])
        if str(block["axis_order"]) != "XYZ":
            raise ValueError(f"{path}: bone {name} uses unsupported axis order {block['axis_order']}")
        dof = tuple(block.get("dof", ()))
        unsupported = set(dof) - {"rx", "ry", "rz"}
        if unsupported:
            raise ValueError(f"{path}: bone {name} has unsupported DOFs {sorted(unsupported)}")
        raw_bones[name] = block
        i += 1

    hierarchy_start, hierarchy_end = _section(lines, ":hierarchy")
    parents: dict[str, str] = {}
    child_lists: dict[str, list[str]] = {"root": []}
    for line in lines[hierarchy_start:hierarchy_end]:
        if line.lower() in {"begin", "end"}:
            continue
        tokens = [token.lower() for token in line.split()]
        parent, children = tokens[0], tokens[1:]
        child_lists.setdefault(parent, [])
        for child in children:
            if child in parents:
                raise ValueError(f"{path}: bone {child} has multiple parents")
            parents[child] = parent
            child_lists[parent].append(child)
            child_lists.setdefault(child, [])

    bone_names = set(raw_bones)
    if set(parents) != bone_names:
        raise ValueError(
            f"{path}: hierarchy/bone mismatch, no parent for {sorted(bone_names - set(parents))}"
        )
    unknown = ({name for values in child_lists.values() for name in values} | set(child_lists)) - {
        "root",
        *bone_names,
    }
    if unknown:
        raise ValueError(f"{path}: hierarchy references unknown bones {sorted(unknown)}")

    children = {name: tuple(values) for name, values in child_lists.items()}
    traversal = _build_traversal(children, bone_names)
    bones = {
        name: Bone(
            name=name,
            direction=np.asarray(block["direction"], dtype=np.float64),
            length=float(block["length"]) * length_scale_to_m,
            axis=np.asarray(block["axis"], dtype=np.float64),
            axis_order=str(block["axis_order"]),
            dof=tuple(block.get("dof", ())),
            parent=parents[name],
        )
        for name, block in raw_bones.items()
    }
    return Skeleton(
        bones=bones,
        children=children,
        traversal=traversal,
        root_order=root_order,
        root_axis_order=root_axis_order,
        root_position=root_position * length_scale_to_m,
        root_orientation=root_orientation,
        length_scale_to_m=length_scale_to_m,
    )


def parse_amc(path: Path, skeleton: Skeleton) -> Motion:
    """Parse a fully-specified AMC file and validate it against its ASF."""

    path = Path(path)
    angle_unit = "deg"
    current_id: int | None = None
    current: dict[str, np.ndarray] | None = None
    frame_ids: list[int] = []
    frames: list[dict[str, np.ndarray]] = []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(":"):
            if line.upper() == ":RADIANS":
                angle_unit = "rad"
            elif line.upper() == ":DEGREES":
                angle_unit = "deg"
            continue
        if re.fullmatch(r"\d+", line):
            if current is not None and current_id is not None:
                frames.append(current)
                frame_ids.append(current_id)
            current_id = int(line)
            current = {}
            continue
        if current is None:
            raise ValueError(f"{path}: motion channel before first frame: {line}")
        tokens = line.split()
        name = tokens[0].lower()
        if name == "root":
            expected = len(skeleton.root_order)
        elif name in skeleton.bones:
            expected = len(skeleton.bones[name].dof)
        else:
            raise ValueError(f"{path}: AMC references unknown bone {name}")
        values = _parse_float_vector(tokens[1:], expected, f"{path.name}:{name}")
        current[name] = values

    if current is not None and current_id is not None:
        frames.append(current)
        frame_ids.append(current_id)
    if not frames:
        raise ValueError(f"{path}: no AMC frames")

    ids = np.asarray(frame_ids, dtype=np.int64)
    if not np.array_equal(ids, np.arange(ids[0], ids[0] + len(ids))):
        raise ValueError(f"{path}: AMC frame numbers are not contiguous")
    for index, frame in enumerate(frames):
        if "root" not in frame:
            raise ValueError(f"{path}: frame {ids[index]} has no root channel")
        missing = [name for name, bone in skeleton.bones.items() if bone.dof and name not in frame]
        if missing:
            raise ValueError(f"{path}: frame {ids[index]} misses channels {missing}")
    return Motion(frame_ids=ids, channels=tuple(frames), angle_unit=angle_unit)


def _axis_rotation(axis: str, angles_rad: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles_rad, dtype=np.float64)
    c = np.cos(angles)
    s = np.sin(angles)
    output = np.zeros((angles.shape[0], 3, 3), dtype=np.float64)
    if axis == "X":
        output[:, 0, 0] = 1.0
        output[:, 1, 1] = c
        output[:, 1, 2] = -s
        output[:, 2, 1] = s
        output[:, 2, 2] = c
    elif axis == "Y":
        output[:, 0, 0] = c
        output[:, 0, 2] = s
        output[:, 1, 1] = 1.0
        output[:, 2, 0] = -s
        output[:, 2, 2] = c
    elif axis == "Z":
        output[:, 0, 0] = c
        output[:, 0, 1] = -s
        output[:, 1, 0] = s
        output[:, 1, 1] = c
        output[:, 2, 2] = 1.0
    else:
        raise ValueError(f"Unknown rotation axis {axis}")
    return output


def _batch_euler(angles: np.ndarray, order: str = "XYZ", *, degrees: bool = True) -> np.ndarray:
    values = np.asarray(angles, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(order):
        raise ValueError(f"Euler angles must be [T,{len(order)}], got {values.shape}")
    if degrees:
        values = np.deg2rad(values)
    result = np.broadcast_to(np.eye(3, dtype=np.float64), (values.shape[0], 3, 3)).copy()
    for index, axis in enumerate(order):
        result = np.matmul(_axis_rotation(axis, values[:, index]), result)
    return result


def _static_euler(angles: np.ndarray, order: str = "XYZ") -> np.ndarray:
    return _batch_euler(np.asarray(angles, dtype=np.float64)[None], order=order)[0]


def _motion_xyz(motion: Motion, name: str, order: tuple[str, ...]) -> np.ndarray:
    output = np.zeros((len(motion.channels), 3), dtype=np.float64)
    channel_index = {"rx": 0, "ry": 1, "rz": 2}
    for frame_index, frame in enumerate(motion.channels):
        values = frame.get(name)
        if values is None:
            continue
        for value_index, dof in enumerate(order):
            if dof in channel_index:
                output[frame_index, channel_index[dof]] = values[value_index]
    return output


def _root_translation(motion: Motion, skeleton: Skeleton) -> np.ndarray:
    output = np.zeros((len(motion.channels), 3), dtype=np.float64)
    channel_index = {"tx": 0, "ty": 1, "tz": 2}
    for frame_index, frame in enumerate(motion.channels):
        values = frame["root"]
        for value_index, dof in enumerate(skeleton.root_order):
            if dof in channel_index:
                output[frame_index, channel_index[dof]] = values[value_index]
    return output * skeleton.length_scale_to_m + skeleton.root_position


def forward_kinematics(skeleton: Skeleton, motion: Motion) -> dict[str, np.ndarray]:
    """Return root and bone endpoints in the shared CMU world frame."""

    frame_count = len(motion.channels)
    root_angles = _motion_xyz(motion, "root", skeleton.root_order)
    root_motion = _batch_euler(
        root_angles,
        order="XYZ",
        degrees=motion.angle_unit == "deg",
    )
    root_axis = _static_euler(skeleton.root_orientation, skeleton.root_axis_order)
    root_local = np.matmul(np.matmul(root_axis[None], root_motion), root_axis.T[None])

    rotations: dict[str, np.ndarray] = {"root": root_local}
    endpoints: dict[str, np.ndarray] = {"root": _root_translation(motion, skeleton)}

    for name in skeleton.traversal[1:]:
        bone = skeleton.bones[name]
        angles = _motion_xyz(motion, name, bone.dof)
        motion_rotation = _batch_euler(
            angles,
            order="XYZ",
            degrees=motion.angle_unit == "deg",
        )
        axis = _static_euler(bone.axis, bone.axis_order)
        local_rotation = np.matmul(np.matmul(axis[None], motion_rotation), axis.T[None])
        global_rotation = np.matmul(rotations[bone.parent], local_rotation)
        offset = bone.direction * bone.length
        world_offset = np.einsum("tij,j->ti", global_rotation, offset)
        endpoints[name] = endpoints[bone.parent] + world_offset
        rotations[name] = global_rotation

    for name, values in endpoints.items():
        if values.shape != (frame_count, 3) or not np.isfinite(values).all():
            raise ValueError(f"Invalid FK result for {name}: {values.shape}")
    return endpoints


def cmu_endpoints_to_canonical21(endpoints: dict[str, np.ndarray]) -> np.ndarray:
    """Create the Stage-1 21-joint interface without discarding pair geometry."""

    required = {
        "root",
        "lowerback",
        "thorax",
        "upperneck",
        "head",
        "lclavicle",
        "lhumerus",
        "lradius",
        "rclavicle",
        "rhumerus",
        "rradius",
        "lhipjoint",
        "lfemur",
        "ltibia",
        "lfoot",
        "rhipjoint",
        "rfemur",
        "rtibia",
        "rfoot",
    }
    missing = required - endpoints.keys()
    if missing:
        raise ValueError(f"CMU skeleton is missing canonical joints {sorted(missing)}")

    thorax = endpoints["thorax"]
    joints = (
        endpoints["root"],
        endpoints["lowerback"],
        thorax,
        endpoints["upperneck"],
        endpoints["head"],
        0.5 * (thorax + endpoints["lclavicle"]),
        endpoints["lclavicle"],
        endpoints["lhumerus"],
        endpoints["lradius"],
        0.5 * (thorax + endpoints["rclavicle"]),
        endpoints["rclavicle"],
        endpoints["rhumerus"],
        endpoints["rradius"],
        endpoints["lhipjoint"],
        endpoints["lfemur"],
        endpoints["ltibia"],
        endpoints["lfoot"],
        endpoints["rhipjoint"],
        endpoints["rfemur"],
        endpoints["rtibia"],
        endpoints["rfoot"],
    )
    output = np.stack(joints, axis=1).astype(np.float32)
    if output.shape[1:] != (len(CANONICAL_21_JOINT_NAMES), 3):
        raise AssertionError(f"Unexpected canonical CMU shape {output.shape}")
    return output


def lowpass_downsample(
    motion: np.ndarray,
    source_fps: int = CMU_SOURCE_FPS,
    target_fps: int = 30,
) -> np.ndarray:
    """Windowed-sinc antialiasing followed by integer-rate decimation."""

    values = np.asarray(motion, dtype=np.float64)
    if source_fps == target_fps:
        return values.astype(np.float32)
    if source_fps <= 0 or target_fps <= 0 or source_fps % target_fps != 0:
        raise ValueError(f"Only integer downsampling is supported, got {source_fps}->{target_fps}")
    factor = source_fps // target_fps
    half_width = 8 * factor
    sample = np.arange(-half_width, half_width + 1, dtype=np.float64)
    cutoff = 0.45 / factor
    kernel = 2.0 * cutoff * np.sinc(2.0 * cutoff * sample)
    kernel *= np.hamming(kernel.size)
    kernel /= kernel.sum()

    flat = values.reshape(values.shape[0], -1)
    padded = np.pad(flat, ((half_width, half_width), (0, 0)), mode="edge")
    filtered = np.empty_like(flat)
    for column in range(flat.shape[1]):
        filtered[:, column] = np.convolve(padded[:, column], kernel, mode="valid")
    return filtered[::factor].reshape((-1, *values.shape[1:])).astype(np.float32)


def _median_extent(motion: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(motion.max(axis=1) - motion.min(axis=1), axis=-1)))


def _resolve_manifest_path(root: Path, value: str) -> Path:
    return root.joinpath(*value.replace("\\", "/").split("/"))


def convert_manifest(
    input_root: Path,
    output_dir: Path,
    *,
    target_fps: int = 30,
    overwrite: bool = False,
) -> dict[str, object]:
    input_root = Path(input_root).resolve()
    output_dir = Path(output_dir).resolve()
    manifest_path = input_root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"CMU interaction manifest not found: {manifest_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    skeleton_cache: dict[Path, Skeleton] = {}
    reports: list[dict[str, object]] = []

    for row in rows:
        pair = row["pair"]
        trial = int(row["trial"])
        asf_a = _resolve_manifest_path(input_root, row["asf_a"])
        amc_a = _resolve_manifest_path(input_root, row["amc_a"])
        asf_b = _resolve_manifest_path(input_root, row["asf_b"])
        amc_b = _resolve_manifest_path(input_root, row["amc_b"])
        output_path = output_dir / f"{pair}_trial_{trial:02d}.npz"

        if output_path.exists() and not overwrite:
            reports.append(
                {
                    "output": str(output_path),
                    "pair": pair,
                    "trial": trial,
                    "status": "skipped_existing",
                }
            )
            continue

        if asf_a not in skeleton_cache:
            skeleton_cache[asf_a] = parse_asf(asf_a)
        if asf_b not in skeleton_cache:
            skeleton_cache[asf_b] = parse_asf(asf_b)
        skeleton_a = skeleton_cache[asf_a]
        skeleton_b = skeleton_cache[asf_b]
        motion_a = parse_amc(amc_a, skeleton_a)
        motion_b = parse_amc(amc_b, skeleton_b)
        if not np.array_equal(motion_a.frame_ids, motion_b.frame_ids):
            raise ValueError(f"{pair} trial {trial}: A/B AMC frame numbers differ")

        person_a_120 = cmu_endpoints_to_canonical21(forward_kinematics(skeleton_a, motion_a))
        person_b_120 = cmu_endpoints_to_canonical21(forward_kinematics(skeleton_b, motion_b))
        person_a = lowpass_downsample(person_a_120, CMU_SOURCE_FPS, target_fps)
        person_b = lowpass_downsample(person_b_120, CMU_SOURCE_FPS, target_fps)
        if person_a.shape != person_b.shape:
            raise ValueError(f"{pair} trial {trial}: converted A/B shapes differ")
        if person_a.shape[0] < 2 or not (np.isfinite(person_a).all() and np.isfinite(person_b).all()):
            raise ValueError(f"{pair} trial {trial}: invalid converted coordinates")

        extent_a = _median_extent(person_a)
        extent_b = _median_extent(person_b)
        if not (0.8 <= extent_a <= 2.5 and 0.8 <= extent_b <= 2.5):
            raise ValueError(
                f"{pair} trial {trial}: implausible body extent A={extent_a:.3f}, B={extent_b:.3f} m"
            )
        root_distance = np.linalg.norm(person_b[:, 0] - person_a[:, 0], axis=-1)
        if float(np.max(root_distance)) > 12.0:
            raise ValueError(
                f"{pair} trial {trial}: implausible A/B root distance {np.max(root_distance):.3f} m"
            )

        tmp_path = output_path.with_suffix(".npz.tmp")
        try:
            with tmp_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    person_a=person_a,
                    person_b=person_b,
                    source_a=str(amc_a),
                    source_b=str(amc_b),
                    source_manifest=str(manifest_path),
                    action=row["action"],
                    pair=pair,
                    trial=np.int32(trial),
                    synthetic=np.bool_(False),
                    interaction_valid=np.float32(1.0),
                    joint_layout="optitrack21",
                    joint_names=joint_names_array(),
                    unit="m",
                    unit_scale_to_m=np.float32(1.0),
                    source_fps=np.int32(CMU_SOURCE_FPS),
                    fps=np.int32(target_fps),
                    source_frame_count=np.int32(len(motion_a.frame_ids)),
                    root_centered=np.bool_(False),
                    coordinate_system="cmu_world_y_up",
                )
            tmp_path.replace(output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        reports.append(
            {
                "output": str(output_path),
                "pair": pair,
                "trial": trial,
                "action": row["action"],
                "status": "written",
                "source_frames": int(len(motion_a.frame_ids)),
                "output_frames": int(person_a.shape[0]),
                "extent_a_m": extent_a,
                "extent_b_m": extent_b,
                "median_root_distance_m": float(np.median(root_distance)),
                "max_root_distance_m": float(np.max(root_distance)),
            }
        )

    written = [report for report in reports if report["status"] == "written"]
    summary: dict[str, object] = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "source_fps": CMU_SOURCE_FPS,
        "target_fps": target_fps,
        "manifest_rows": len(rows),
        "files_written": len(written),
        "files_skipped": len(reports) - len(written),
        "total_output_frames": int(sum(int(report["output_frames"]) for report in written)),
        "sequences": reports,
    }
    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paired CMU ASF/AMC interaction motions to Stage-1 NPZ files."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(r"D:\datasets\cmu_mocap\cmu_mocap\human_interaction"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_root / "data_aug"
    summary = convert_manifest(
        args.input_root,
        output_dir,
        target_fps=args.target_fps,
        overwrite=args.overwrite,
    )
    printable = {key: value for key, value in summary.items() if key != "sequences"}
    print(json.dumps(printable, indent=2))
    print(f"conversion report -> {Path(summary['output_dir']) / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
