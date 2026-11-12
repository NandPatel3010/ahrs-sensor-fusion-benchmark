import unittest
from math import cos, pi, sin

import numpy as np

from ahrs.filters import Complementary, GyroOnly, Madgwick, Mahony
from ahrs.metrics import angular_error, slope_per_minute, tilt_error


class FilterTests(unittest.TestCase):
    def test_gyro_integrates_positive_body_yaw(self):
        attitude = GyroOnly((1, 0, 0, 0))
        for _ in range(1000):
            estimate = attitude.update((0, 0, pi / 2), (0, 0, 9.81), 0.001)
        reference = np.array([[cos(pi / 4), 0, 0, sin(pi / 4)]])
        self.assertLess(angular_error(np.array([estimate]), reference)[0], 0.001)

    def test_fusion_corrects_a_tilt_error_at_rest(self):
        tilted = (cos(pi / 12), sin(pi / 12), 0, 0)
        reference = np.array([[1, 0, 0, 0]])
        for implementation in (Complementary, Mahony, Madgwick):
            with self.subTest(filter=implementation.__name__):
                attitude = implementation(tilted)
                for _ in range(3000):
                    estimate = attitude.update((0, 0, 0), (0, 0, 9.81), 0.005)
                self.assertLess(tilt_error(np.array([estimate]), reference)[0], 5)

    def test_quaternion_sign_and_yaw_only_error(self):
        reference = np.array([[1, 0, 0, 0]])
        yaw = np.array([[0.5, 0, 0, sin(pi / 3)]])
        np.testing.assert_allclose(angular_error(-reference, reference), [0])
        np.testing.assert_allclose(angular_error(yaw, reference), [120])
        np.testing.assert_allclose(tilt_error(yaw, reference), [0])

    def test_static_slope_has_units_of_degrees_per_minute(self):
        time = np.arange(30)
        self.assertAlmostEqual(slope_per_minute(time, time / 10), 6)

    def test_mahony_rejects_tilt_bias_at_rest_but_heading_drifts(self):
        reference = np.array([[1, 0, 0, 0]])
        rate_bias = pi / 360  # 0.5 deg/s
        estimates = []
        for gyro in ((rate_bias, 0, 0), (0, 0, rate_bias)):
            attitude = Mahony(reference[0])
            for _ in range(6000):
                pose = attitude.update(gyro, (0, 0, 9.81), 0.01)
            estimates.append(pose)

        tilt_bias, yaw_bias = np.array(estimates)
        self.assertLess(angular_error(tilt_bias, reference)[0], 0.05)
        self.assertAlmostEqual(angular_error(yaw_bias, reference)[0], 30, places=4)


if __name__ == "__main__":
    unittest.main()
