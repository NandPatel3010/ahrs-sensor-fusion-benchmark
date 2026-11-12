"""Orientation errors and drift metrics for six-axis IMU evaluation."""

import numpy as np

from .quaternion import rotation_matrices


def angular_error(estimate, reference):
    dot = np.abs(np.sum(estimate * reference, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))


def tilt_error(estimate, reference):
    est_up = rotation_matrices(estimate)[:, 2, :]
    ref_up = rotation_matrices(reference)[:, 2, :]
    cross = np.linalg.norm(np.cross(est_up, ref_up), axis=1)
    dot = np.sum(est_up * ref_up, axis=1)
    return np.degrees(np.arctan2(cross, dot))


def initial_rest(trial):
    """Return up to the first 25 s of labeled rest with valid optical samples."""
    first_motion = np.flatnonzero(trial.movement)[0]
    rest = trial.valid_reference & (trial.time < trial.time[0] + 25)
    rest[first_motion:] = False
    return rest


def slope_per_minute(time, error):
    if len(time) < 2 or time[-1] - time[0] < 5:
        return None
    centered = time - time.mean()
    return float(60 * np.dot(centered, error - error.mean()) / np.dot(centered, centered))


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))
