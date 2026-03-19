import unittest
import numpy as np

from scripts.rotation_gradient import verify_globally_disjoint


class TestVerifyGloballyDisjoint(unittest.TestCase):

    def test_disjoint_identity_rotation(self):
        # Points uniformly spaced by 1.0 on all axes
        encodings = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0]
        ])
        R = np.eye(2)  # Identity matrix (no rotation)
        delta = 0.5

        is_disjoint, min_gap, min_gaps_array = verify_globally_disjoint(encodings, R, delta)

        self.assertTrue(is_disjoint, "Should be disjoint as gaps (1.0) > delta (0.5)")
        self.assertAlmostEqual(min_gap, 1.0)
        np.testing.assert_almost_equal(min_gaps_array, np.array([1.0, 1.0]))

    def test_not_disjoint_identity_rotation(self):
        # Points spaced by 0.1 on axis 0, and 1.0 on axis 1
        encodings = np.array([
            [0.0, 0.0],
            [0.1, 1.0]
        ])
        R = np.eye(2)
        delta = 0.5

        is_disjoint, min_gap, min_gaps_array = verify_globally_disjoint(encodings, R, delta)

        self.assertFalse(is_disjoint, "Should fail because gap on axis 0 (0.1) < delta (0.5)")
        self.assertAlmostEqual(min_gap, 0.1)
        np.testing.assert_almost_equal(min_gaps_array, np.array([0.1, 1.0]))

    def test_rotation_effect(self):
        # Points at (0, 0) and (1, 1). Gap is 1.0 on both axes initially.
        encodings = np.array([
            [0.0, 0.0],
            [1.0, 1.0]
        ])

        # Rotate by 45 degrees. The vector [1, 1] becomes [sqrt(2), 0]
        theta = np.pi / 4
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        delta = 0.5

        is_disjoint, min_gap, min_gaps_array = verify_globally_disjoint(encodings, R, delta)

        # Now the gap on the Y-axis (axis 1) should be 0.0
        self.assertFalse(is_disjoint, "Rotation should collapse the Y-axis projection")
        self.assertAlmostEqual(min_gap, 0.0, places=7)


if __name__ == '__main__':
    unittest.main()
