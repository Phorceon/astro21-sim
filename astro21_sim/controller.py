from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_continuous_are

from .constants import ControllerConfig, ReferenceConfig
from .orbits import mean_motion, ss_shape_parameter


@dataclass(frozen=True)
class LQRData:
    a_matrix: np.ndarray
    b_matrix: np.ndarray
    gain: np.ndarray
    riccati: np.ndarray
    orbit_rate_rad_s: float
    ss_shape_parameter: float


@dataclass(frozen=True)
class ControlSnapshot:
    command_cb_chaser: float
    target_cb: float
    theta1_basis_value: float
    theta2_basis_value: float
    uy: float


def compute_lqr_data(config: ReferenceConfig) -> LQRData:
    orbit_rate = mean_motion(config.earth.mu, config.chaser_orbit.semi_major_axis_m)
    c_value = ss_shape_parameter(config.earth, config.chaser_orbit)
    a_coeff = 2.0 * orbit_rate * c_value
    b_coeff = (5.0 * c_value**2 - 2.0) * orbit_rate**2
    a_matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [b_coeff, 0.0, 0.0, a_coeff],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -a_coeff, 0.0, 0.0],
        ],
        dtype=float,
    )
    b_matrix = np.array([[0.0], [0.0], [0.0], [config.input_channel_sign]], dtype=float)
    p = solve_continuous_are(a_matrix, b_matrix, config.controller.q_matrix, np.array([[config.controller.r_scalar]]))
    gain = np.linalg.solve(np.array([[config.controller.r_scalar]]), b_matrix.T @ p).reshape(-1)
    return LQRData(a_matrix, b_matrix, gain, p, orbit_rate, c_value)


def fourier_basis(t_seconds: float, orbit_rate_rad_s: float) -> np.ndarray:
    return np.array([1.0, np.sin(orbit_rate_rad_s * t_seconds), np.cos(orbit_rate_rad_s * t_seconds)], dtype=float)


def theta_basis_value(theta: np.ndarray, basis: np.ndarray) -> float:
    return float(theta @ basis)


def projection(theta: np.ndarray, theta_dot: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    projected = theta_dot.copy()
    for idx in range(theta.size):
        if theta[idx] >= upper[idx] and theta_dot[idx] > 0.0:
            projected[idx] = 0.0
        if theta[idx] <= lower[idx] and theta_dot[idx] < 0.0:
            projected[idx] = 0.0
    return projected


def saturate_ballistic_coefficient(config: ReferenceConfig, commanded_cb: float) -> float:
    return float(
        np.clip(
            commanded_cb,
            config.spacecraft.chaser_ballistic_coefficient_min,
            config.spacecraft.chaser_ballistic_coefficient_max,
        )
    )


def control_law(
    config: ReferenceConfig,
    lqr: LQRData,
    t_seconds: float,
    x_state: np.ndarray,
    theta1_hat: np.ndarray,
    theta2_hat: np.ndarray,
    target_cb: float,
) -> ControlSnapshot:
    basis = fourier_basis(t_seconds, lqr.orbit_rate_rad_s)
    theta1_basis = max(theta_basis_value(theta1_hat, basis), config.controller.min_theta1_basis_value)
    theta2_basis = theta_basis_value(theta2_hat, basis)
    commanded_cb = (theta2_basis - float(lqr.gain @ x_state)) / theta1_basis
    commanded_cb = saturate_ballistic_coefficient(config, commanded_cb)
    uy = theta1_basis * commanded_cb - theta2_basis
    return ControlSnapshot(commanded_cb, target_cb, theta1_basis, theta2_basis, uy)


def adaptive_law(
    controller: ControllerConfig,
    lqr: LQRData,
    t_seconds: float,
    x_state: np.ndarray,
    theta1_hat: np.ndarray,
    commanded_cb: float,
) -> tuple[np.ndarray, np.ndarray]:
    basis = fourier_basis(t_seconds, lqr.orbit_rate_rad_s)
    y1 = np.array([commanded_cb, commanded_cb * basis[1], commanded_cb * basis[2]], dtype=float)
    y2 = np.array([-1.0, -basis[1], -basis[2]], dtype=float)
    feedback_scalar = (lqr.b_matrix.T @ lqr.riccati.T @ x_state).item()
    theta1_dot = controller.gamma1 @ (2.0 * y1 * feedback_scalar)
    theta1_dot = projection(theta1_hat, theta1_dot, controller.theta1_lower, controller.theta1_upper)
    theta2_dot = controller.gamma2 @ (2.0 * y2 * feedback_scalar)
    return theta1_dot, theta2_dot
