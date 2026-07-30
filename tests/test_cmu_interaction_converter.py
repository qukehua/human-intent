import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cmu_interaction_converter import (  # noqa: E402
    cmu_endpoints_to_canonical21,
    forward_kinematics,
    lowpass_downsample,
    parse_amc,
    parse_asf,
)


MINIMAL_ASF = """
:version 1.10
:name test
:units
  mass 1.0
  length 1.0
  angle deg
:root
  order TX TY TZ RX RY RZ
  axis XYZ
  position 0 0 0
  orientation 0 0 0
:bonedata
  begin
    id 1
    name arm
    direction 1 0 0
    length 1
    axis 0 0 0 XYZ
    dof rz
    limits (-180 180)
  end
:hierarchy
  begin
    root arm
  end
"""

MINIMAL_AMC = """
#!OML:ASF test.asf
:FULLY-SPECIFIED
:DEGREES
1
root 1 2 3 0 0 0
arm 90
2
root 2 2 3 0 0 90
arm 0
"""


class CmuInteractionConverterTest(unittest.TestCase):
    def test_asf_amc_forward_kinematics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asf_path = root / "test.asf"
            amc_path = root / "test.amc"
            asf_path.write_text(MINIMAL_ASF)
            amc_path.write_text(MINIMAL_AMC)

            skeleton = parse_asf(asf_path)
            motion = parse_amc(amc_path, skeleton)
            endpoints = forward_kinematics(skeleton, motion)

        scale = 0.0254
        np.testing.assert_allclose(endpoints["root"][0], np.asarray([1, 2, 3]) * scale)
        np.testing.assert_allclose(
            endpoints["arm"][0],
            np.asarray([1, 3, 3]) * scale,
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            endpoints["arm"][1],
            np.asarray([2, 3, 3]) * scale,
            atol=1.0e-7,
        )

    def test_canonical_mapping_preserves_expected_semantics(self):
        names = (
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
        )
        endpoints = {
            name: np.full((2, 3), index, dtype=np.float32)
            for index, name in enumerate(names)
        }
        canonical = cmu_endpoints_to_canonical21(endpoints)
        self.assertEqual(canonical.shape, (2, 21, 3))
        np.testing.assert_allclose(canonical[:, 0], endpoints["root"])
        np.testing.assert_allclose(canonical[:, 6], endpoints["lclavicle"])
        np.testing.assert_allclose(canonical[:, 13], endpoints["lhipjoint"])
        np.testing.assert_allclose(
            canonical[:, 5],
            0.5 * (endpoints["thorax"] + endpoints["lclavicle"]),
        )

    def test_lowpass_downsample_preserves_constant_motion(self):
        motion = np.full((121, 21, 3), 2.5, dtype=np.float32)
        output = lowpass_downsample(motion, source_fps=120, target_fps=30)
        self.assertEqual(output.shape, (31, 21, 3))
        np.testing.assert_allclose(output, 2.5, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
