import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.skeleton_utils import (  # noqa: E402
    SMPL24_TO_OPTITRACK21,
    canonicalize_motion,
    mirror_canonical_x,
    scene_center_pair,
)


class SkeletonUtilsTest(unittest.TestCase):
    def test_smpl24_mapping_has_stable_joint_semantics(self):
        motion = np.zeros((2, 24, 3), dtype=np.float32)
        motion[:, :, 0] = np.arange(24, dtype=np.float32)
        canonical = canonicalize_motion(motion, "smpl24", validate_scale=False)
        np.testing.assert_array_equal(canonical[0, :, 0], SMPL24_TO_OPTITRACK21)

    def test_h36m_mapping_and_millimetre_conversion(self):
        motion = np.zeros((2, 32, 3), dtype=np.float32)
        motion[:, :, 0] = np.arange(32, dtype=np.float32) * 1000.0
        canonical = canonicalize_motion(
            motion,
            "h36m32",
            unit_scale=0.001,
            validate_scale=False,
        )
        self.assertAlmostEqual(float(canonical[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(canonical[0, 5, 0]), 15.0, places=5)
        self.assertAlmostEqual(float(canonical[0, 6, 0]), 17.0, places=5)
        self.assertAlmostEqual(float(canonical[0, 17, 0]), 1.0, places=5)

    def test_scene_center_preserves_inter_person_displacement(self):
        person_a = np.ones((2, 21, 3), dtype=np.float32)
        person_b = person_a + np.asarray([2.0, -1.0, 0.5], dtype=np.float32)
        before = person_b - person_a
        centered_a, centered_b = scene_center_pair(person_a, person_b)
        np.testing.assert_allclose(centered_a[0, 0], 0.0)
        np.testing.assert_allclose(centered_b - centered_a, before)

    def test_mirror_swaps_anatomical_left_and_right(self):
        motion = np.zeros((1, 21, 3), dtype=np.float32)
        motion[0, 6, 0] = -0.4
        motion[0, 10, 0] = 0.7
        mirrored = mirror_canonical_x(motion)
        self.assertAlmostEqual(float(mirrored[0, 6, 0]), -0.7)
        self.assertAlmostEqual(float(mirrored[0, 10, 0]), 0.4)

    def test_unsupported_joint_count_is_rejected(self):
        with self.assertRaises(ValueError):
            canonicalize_motion(
                np.zeros((2, 17, 3), dtype=np.float32),
                "auto",
                validate_scale=False,
            )
        with self.assertRaises(ValueError):
            canonicalize_motion(
                np.zeros((2, 33, 3), dtype=np.float32),
                "h36m32",
                validate_scale=False,
            )

    def test_scale_validation_catches_millimetres_without_conversion(self):
        motion_mm = np.zeros((2, 21, 3), dtype=np.float32)
        motion_mm[:, :, 2] = np.linspace(0.0, 1800.0, 21, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Implausible body extent"):
            canonicalize_motion(motion_mm, "optitrack21", unit_scale=1.0)
        motion_m = canonicalize_motion(motion_mm, "optitrack21", unit_scale=0.001)
        self.assertAlmostEqual(float(motion_m[0, -1, 2]), 1.8, places=5)


if __name__ == "__main__":
    unittest.main()
