import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_prepocess import (  # noqa: E402
    MotionRecord,
    pair_motion_records,
    place_in_shared_scene,
)
from dataset.h36m_fk import (  # noqa: E402
    H36M_OFFSETS_MM,
    H36M_PARENTS,
    expmap_to_rotmat,
    h36m_expmap_to_xyz,
    recover_root_transform,
)


class H36mPreprocessingTest(unittest.TestCase):
    def test_zero_pose_follows_offsets_and_tree(self):
        xyz = h36m_expmap_to_xyz(np.zeros((1, 99), dtype=np.float32))
        expected = np.empty((32, 3), dtype=np.float32)
        for joint, parent in enumerate(H36M_PARENTS):
            expected[joint] = H36M_OFFSETS_MM[joint]
            if parent >= 0:
                expected[joint] += expected[parent]
        np.testing.assert_allclose(xyz[0], expected, atol=1.0e-5)

    def test_expmap_rodrigues_rotation(self):
        rotation = expmap_to_rotmat(np.asarray([[0.0, np.pi / 2.0, 0.0]]))[0]
        result = np.asarray([1.0, 0.0, 0.0]) @ rotation
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0], atol=1.0e-6)

    def test_root_increments_are_integrated(self):
        angles = np.zeros((3, 99), dtype=np.float32)
        angles[:, 0] = 1.0
        translations, rotations = recover_root_transform(angles)
        np.testing.assert_allclose(translations[:, 0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            rotations,
            np.broadcast_to(np.eye(3), rotations.shape),
            atol=1.0e-7,
        )

    def test_scene_pair_uses_distinct_motion_and_y_up_translation(self):
        motion_a = np.zeros((10, 21, 3), dtype=np.float32)
        motion_b = np.ones((12, 21, 3), dtype=np.float32)
        motion_a[:, :, 0] += np.arange(10)[:, None]
        motion_b[:, :, 2] += np.arange(12)[:, None]
        person_a, person_b, metadata = place_in_shared_scene(
            motion_a,
            motion_b,
            np.random.default_rng(7),
            min_distance_m=1.0,
            max_distance_m=1.0,
        )
        self.assertEqual(person_a.shape, (10, 21, 3))
        self.assertEqual(person_b.shape, (10, 21, 3))
        self.assertAlmostEqual(
            float(np.linalg.norm(person_b[0, 0, [0, 2]] - person_a[0, 0, [0, 2]])),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(float(person_b[0, 0, 1]), 0.0, places=6)
        self.assertIn("yaw_b_deg", metadata)

    def test_saved_synthetic_pairs_are_intent_training_eligible(self):
        a = MotionRecord(Path("a.txt"), np.zeros((5, 21, 3), dtype=np.float32))
        b = MotionRecord(Path("b.txt"), np.ones((5, 21, 3), dtype=np.float32))
        with tempfile.TemporaryDirectory() as directory:
            report = pair_motion_records(
                [a, b],
                Path(directory),
                dataset="test",
                seed=42,
                source_fps=50,
                target_fps=30,
                min_distance_m=0.8,
                max_distance_m=1.2,
                max_pair_frames=0,
            )
            files = sorted(Path(directory).glob("*.npz"))
            self.assertEqual(len(files), 2)
            with np.load(files[0]) as data:
                self.assertTrue(bool(data["synthetic"]))
                # Legacy field records that this is not a synchronous capture.
                self.assertEqual(float(data["interaction_valid"]), 0.0)
                self.assertFalse(bool(data["recorded_synchronous"]))
                self.assertTrue(bool(data["intent_training_eligible"]))
                self.assertNotEqual(str(data["source_a"]), str(data["source_b"]))
        self.assertTrue(report["human_latent_supervision"])


if __name__ == "__main__":
    unittest.main()
