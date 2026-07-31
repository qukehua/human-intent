import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.muco3dhp_converter import (  # noqa: E402
    MucoClip,
    different_subject_partner_indices,
    official_source_fps,
    resample_motion,
    split_motion_into_clips,
)
from dataset.skeleton_utils import mpi_inf_3dhp_28_to_optitrack21  # noqa: E402


class Muco3dhpConverterTest(unittest.TestCase):
    def test_official_28_joint_mapping_preserves_semantics(self):
        source = np.zeros((1, 28, 3), dtype=np.float32)
        source[0, :, 0] = np.arange(28)
        output = mpi_inf_3dhp_28_to_optitrack21(source)

        self.assertEqual(output.shape, (1, 21, 3))
        self.assertEqual(float(output[0, 0, 0]), 4.0)  # pelvis
        self.assertEqual(float(output[0, 6, 0]), 9.0)  # left shoulder
        self.assertEqual(float(output[0, 8, 0]), 11.0)  # left wrist
        self.assertEqual(float(output[0, 13, 0]), 18.0)  # left hip
        self.assertEqual(float(output[0, 16, 0]), 22.0)  # left toe
        self.assertEqual(float(output[0, 20, 0]), 27.0)  # right toe
        self.assertEqual(float(output[0, 5, 0]), 5.0)  # thorax/shoulder midpoint

    def test_clip_splitting_keeps_only_trainable_tail(self):
        motion = np.zeros((675, 21, 3), dtype=np.float32)
        clips = split_motion_into_clips(
            motion,
            clip_frames=300,
            stride_frames=300,
            min_frames=70,
        )
        self.assertEqual([start for start, _ in clips], [0, 300, 600])
        self.assertEqual([clip.shape[0] for _, clip in clips], [300, 300, 75])

    def test_pairing_is_one_to_one_and_uses_different_subjects(self):
        clips = [
            MucoClip(
                Path(f"S{subject}/Seq1/annot.mat"),
                subject,
                1,
                25.0,
                i,
                np.zeros((70, 21, 3)),
            )
            for i, subject in enumerate((1, 1, 2, 2, 3, 3, 4, 4))
        ]
        partners = different_subject_partner_indices(
            clips,
            np.random.default_rng(42),
        )
        self.assertEqual(sorted(partners.tolist()), list(range(len(clips))))
        for index, partner in enumerate(partners):
            self.assertNotEqual(clips[index].subject, clips[int(partner)].subject)

    def test_official_fps_and_25_to_30_resampling_preserve_duration(self):
        self.assertEqual(official_source_fps(1, 1), 25.0)
        self.assertEqual(official_source_fps(1, 2), 50.0)
        self.assertEqual(official_source_fps(3, 1), 50.0)
        motion = np.zeros((26, 21, 3), dtype=np.float32)
        motion[:, :, 0] = np.arange(26, dtype=np.float32)[:, None]
        resampled = resample_motion(motion, source_fps=25.0, target_fps=30.0)
        self.assertEqual(resampled.shape[0], 31)
        self.assertAlmostEqual(float(resampled[-1, 0, 0]), 25.0, places=5)


if __name__ == "__main__":
    unittest.main()
