"""Hamilton quaternions (w, x, y, z), rotating body vectors into world ENU."""

from math import hypot

import numpy as np


def product(q, p):
    w, x, y, z = q
    a, b, c, d = p
    return (
        w * a - x * b - y * c - z * d,
        w * b + x * a + y * d - z * c,
        w * c - x * d + y * a + z * b,
        w * d + x * c - y * b + z * a,
    )


def unit(values):
    length = hypot(*values)
    if length == 0:
        raise ValueError("Cannot normalize a zero vector")
    return tuple(value / length for value in values)


def gravity_body(q):
    """World +Z expressed in body coordinates, matching resting specific force."""
    w, x, y, z = q
    return (
        2 * (x * z - w * y),
        2 * (w * x + y * z),
        1 - 2 * (x * x + y * y),
    )


def cross(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx


def rotation_matrices(q):
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((
        np.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)), axis=-1),
        np.stack((2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)), axis=-1),
        np.stack((2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)), axis=-1),
    ), axis=-2)


def rpy_deg(q):
    """ZYX Euler angles for plotting. Keep yaw and roll in [-180, 180]."""
    r = rotation_matrices(q)
    return np.degrees(np.column_stack((
        np.arctan2(r[:, 2, 1], r[:, 2, 2]),
        np.arcsin(np.clip(-r[:, 2, 0], -1, 1)),
        np.arctan2(r[:, 1, 0], r[:, 0, 0]),
    )))
