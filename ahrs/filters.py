"""Recursive filter updates. Each accepts gyroscope rad/s, accel m/s² and dt s."""

from math import atan2, cos, expm1, hypot, sin

from .quaternion import cross, gravity_body, product, unit


def _gyro_step(q, gyro, dt, correction=(0.0, 0.0, 0.0, 0.0)):
    derivative = product(q, (0.0, *gyro))
    return unit(tuple(
        component + dt * (0.5 * rate + feedback)
        for component, rate, feedback in zip(q, derivative, correction)
    ))


class GyroOnly:
    """Integrate angular rate without a gravity reference."""

    def __init__(self, q0):
        self.q = unit(q0)

    def update(self, gyro, accel, dt):
        self.q = _gyro_step(self.q, gyro, dt)
        return self.q


class Complementary(GyroOnly):
    """Apply a fraction of the shortest rotation from measured to predicted up."""

    def __init__(self, q0, tau=1.0):
        super().__init__(q0)
        self.tau = tau

    def update(self, gyro, accel, dt):
        q = _gyro_step(self.q, gyro, dt)
        if hypot(*accel) > 1e-12:
            measured = unit(accel)
            predicted = gravity_body(q)
            axis = cross(measured, predicted)
            length = hypot(*axis)
            dot = sum(a * b for a, b in zip(measured, predicted))
            if length < 1e-12 and dot < 0:
                # Opposite vectors admit any perpendicular axis.
                basis = (1, 0, 0) if abs(measured[0]) < 0.9 else (0, 1, 0)
                axis = cross(measured, basis)
            if hypot(*axis) > 1e-12:
                weight = -expm1(-dt / self.tau)
                half = 0.5 * weight * atan2(length, dot)
                correction = (cos(half), *(sin(half) * v for v in unit(axis)))
                q = unit(product(q, correction))
        self.q = q
        return q


class Mahony(GyroOnly):
    """Apply proportional feedback and integrate observable gyro bias."""

    def __init__(self, q0, kp=1.0, ki=0.10):
        super().__init__(q0)
        self.kp, self.ki = kp, ki
        self.bias = (0.0, 0.0, 0.0)

    def update(self, gyro, accel, dt):
        error = (0.0, 0.0, 0.0)
        if hypot(*accel) > 1e-12:
            error = cross(unit(accel), gravity_body(self.q))
            self.bias = tuple(b - self.ki * e * dt for b, e in zip(self.bias, error))
        corrected = tuple(g - b + self.kp * e for g, b, e in zip(gyro, self.bias, error))
        self.q = _gyro_step(self.q, corrected, dt)
        return self.q


class Madgwick(GyroOnly):
    """Use the normalized gradient of the gravity residual to correct q-dot."""

    def __init__(self, q0, beta=0.035):
        super().__init__(q0)
        self.beta = beta

    def update(self, gyro, accel, dt):
        correction = (0.0, 0.0, 0.0, 0.0)
        if hypot(*accel) > 1e-12:
            w, x, y, z = self.q
            fx, fy, fz = (a - b for a, b in zip(gravity_body(self.q), unit(accel)))
            # J(q).T @ f(q) for Madgwick's three-component gravity objective.
            gradient = (
                -2 * y * fx + 2 * x * fy,
                2 * z * fx + 2 * w * fy - 4 * x * fz,
                -2 * w * fx + 2 * z * fy - 4 * y * fz,
                2 * x * fx + 2 * y * fy,
            )
            if hypot(*gradient) > 1e-12:
                correction = tuple(-self.beta * value for value in unit(gradient))
        self.q = _gyro_step(self.q, gyro, dt, correction)
        return self.q
