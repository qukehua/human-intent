import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.data_prepocess import SmplhJointExtractor  # noqa: E402


class SmplhJointExtractorTest(unittest.TestCase):
    def _model_root(self, directory: str) -> Path:
        root = Path(directory) / "body_models"
        model_dir = root / "smplh" / "female"
        model_dir.mkdir(parents=True)
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        regressor = np.zeros((3, 4), dtype=np.float64)
        regressor[np.arange(3), np.arange(3)] = 1.0
        parents = np.asarray(
            [
                [np.iinfo(np.uint32).max, 0, 1],
                [0, 1, 2],
            ],
            dtype=np.uint32,
        )
        np.savez(
            model_dir / "model.npz",
            J_regressor=regressor,
            kintree_table=parents,
            shapedirs=np.zeros((4, 3, 16), dtype=np.float64),
            v_template=vertices,
        )
        return root

    def test_zero_pose_recovers_shaped_joints_and_translation(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor = SmplhJointExtractor(
                self._model_root(directory),
                device="cuda",
                batch_size=2,
            )
            result = extractor(
                np.zeros((2, 9), dtype=np.float32),
                np.asarray([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]]),
                np.zeros(16),
                "female",
            )
            expected = np.asarray(
                [
                    [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 3.0]],
                    [[-1.0, 0.0, 0.5], [0.0, 0.0, 0.5], [1.0, 0.0, 0.5]],
                ],
                dtype=np.float32,
            )
            np.testing.assert_allclose(result, expected, atol=1.0e-6)
            self.assertEqual(extractor.device, "cpu")

    def test_root_rotation_propagates_through_kinematic_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            extractor = SmplhJointExtractor(
                self._model_root(directory),
                device="cpu",
                batch_size=1,
            )
            pose = np.zeros((1, 9), dtype=np.float64)
            pose[0, 2] = np.pi / 2.0
            result = extractor(
                pose,
                np.zeros((1, 3)),
                np.zeros(16),
                "female",
            )
            np.testing.assert_allclose(
                result[0],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                atol=1.0e-6,
            )


if __name__ == "__main__":
    unittest.main()
