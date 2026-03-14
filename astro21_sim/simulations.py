from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .atmosphere import AtmosphereModel, SinusoidalDensityModel
from .constants import OrbitElements, ReferenceConfig, SimulationCase, TargetPerturbation
from .controller import LQRData, adaptive_law, compute_lqr_data, control_law, fourier_basis
from .orbits import (
    initial_in_plane_state,
    mean_motion,
    oe_to_eci,
    orbital_elements_with_perturbation,
    relative_speed_to_atmosphere_squared,
    relative_state_lvlh,
)


@dataclass(frozen=True)
class SimulationResult:
    time_s: np.ndarray
    x_history: np.ndarray
    cb_chaser_history: np.ndarray
    cb_target_history: np.ndarray
    theta1_hat_history: np.ndarray
    theta2_hat_history: np.ndarray
    orbit_path_xy_m: np.ndarray
    atmosphere_source: str
    lqr: LQRData
    initial_state: np.ndarray


def _target_elements(config: ReferenceConfig, case: SimulationCase) -> OrbitElements:
    return orbital_elements_with_perturbation(config.chaser_orbit, case.target_perturbation)


def _make_time_grid(case: SimulationCase) -> np.ndarray:
    return np.arange(0.0, case.duration_hours * 3600.0 + case.sample_step_s, case.sample_step_s)


def simulate_ss(config: ReferenceConfig) -> SimulationResult:
    lqr = compute_lqr_data(config)
    target_orbit = _target_elements(config, config.ss_case)
    chaser_r0, chaser_v0 = oe_to_eci(config.earth, config.chaser_orbit)
    target_r0, target_v0 = oe_to_eci(config.earth, target_orbit)
    x0 = initial_in_plane_state(config.earth, config.chaser_orbit, config.ss_case.target_perturbation)
    theta10 = config.controller.theta1_hat0.copy()
    theta20 = config.controller.theta2_hat0.copy()
    vr_c_sq = relative_speed_to_atmosphere_squared(config.earth, chaser_r0, chaser_v0)
    vr_t_sq = relative_speed_to_atmosphere_squared(config.earth, target_r0, target_v0)
    density_model = SinusoidalDensityModel(config.density_fit)
    t_eval = _make_time_grid(config.ss_case)

    def rhs(t_seconds: float, state: np.ndarray) -> np.ndarray:
        x_state = state[:4]
        theta1_hat = state[4:7]
        theta2_hat = state[7:10]
        snapshot = control_law(
            config,
            lqr,
            t_seconds,
            x_state,
            theta1_hat,
            theta2_hat,
            config.spacecraft.target_ballistic_coefficient,
        )
        rho_c = density_model.density(t_seconds, lqr.orbit_rate_rad_s)
        rho_t = density_model.density(t_seconds, lqr.orbit_rate_rad_s)
        uy = rho_c * vr_c_sq * snapshot.command_cb_chaser - rho_t * vr_t_sq * snapshot.target_cb
        x_dot = lqr.a_matrix @ x_state + lqr.b_matrix.reshape(-1) * uy
        theta1_dot, theta2_dot = adaptive_law(
            config.controller, lqr, t_seconds, x_state, theta1_hat, snapshot.command_cb_chaser
        )
        return np.concatenate([x_dot, theta1_dot, theta2_dot])

    y0 = np.concatenate([x0, theta10, theta20])
    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        t_eval=t_eval,
        rtol=config.solver_rtol,
        atol=config.solver_atol,
        method="DOP853",
    )
    x_history = sol.y[:4].T
    theta1_hat_history = sol.y[4:7].T
    theta2_hat_history = sol.y[7:10].T
    cb_chaser_history = np.zeros_like(sol.t)
    cb_target_history = np.full_like(sol.t, config.spacecraft.target_ballistic_coefficient)
    orbit_path_xy = np.column_stack([x_history[:, 0], x_history[:, 2]])
    for idx, t_seconds in enumerate(sol.t):
        snapshot = control_law(
            config,
            lqr,
            t_seconds,
            x_history[idx],
            theta1_hat_history[idx],
            theta2_hat_history[idx],
            config.spacecraft.target_ballistic_coefficient,
        )
        cb_chaser_history[idx] = snapshot.command_cb_chaser
    return SimulationResult(
        time_s=sol.t,
        x_history=x_history,
        cb_chaser_history=cb_chaser_history,
        cb_target_history=cb_target_history,
        theta1_hat_history=theta1_hat_history,
        theta2_hat_history=theta2_hat_history,
        orbit_path_xy_m=orbit_path_xy,
        atmosphere_source="sinusoidal-fit",
        lqr=lqr,
        initial_state=x0,
    )


def _gravity_and_j2_acceleration(config: ReferenceConfig, r_eci: np.ndarray) -> np.ndarray:
    x, y, z = r_eci
    r_norm = np.linalg.norm(r_eci)
    mu = config.earth.mu
    re = config.earth.radius_m
    j2 = config.earth.j2
    factor = 1.5 * j2 * mu * re**2 / r_norm**5
    z_ratio_sq = (z / r_norm) ** 2
    a_gravity = -mu * r_eci / r_norm**3
    a_j2 = factor * np.array(
        [
            x * (5.0 * z_ratio_sq - 1.0),
            y * (5.0 * z_ratio_sq - 1.0),
            z * (5.0 * z_ratio_sq - 3.0),
        ]
    )
    return a_gravity + a_j2


def _drag_acceleration(config: ReferenceConfig, density_kg_m3: float, ballistic_coefficient: float, r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    omega = np.array([0.0, 0.0, config.earth.omega_rad_s])
    v_rel = v_eci - np.cross(omega, r_eci)
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm <= 0.0:
        return np.zeros(3)
    return -density_kg_m3 * ballistic_coefficient * v_rel_norm**2 * (v_rel / v_rel_norm)


def _target_ballistic_coefficient(config: ReferenceConfig, t_seconds: float) -> float:
    omega_tumble = 2.0 * np.pi * config.full_target_tumble_rpm / 60.0
    nominal = config.spacecraft.target_ballistic_coefficient
    return nominal * (1.0 + config.full_target_tumble_fraction * np.sin(omega_tumble * t_seconds))


def simulate_full(config: ReferenceConfig, require_exact_atmosphere: bool = False) -> SimulationResult:
    lqr = compute_lqr_data(config)
    atmosphere_model = AtmosphereModel(config, require_exact=require_exact_atmosphere)
    target_orbit = _target_elements(config, config.full_case)
    chaser_r0, chaser_v0 = oe_to_eci(config.earth, config.chaser_orbit)
    target_r0, target_v0 = oe_to_eci(config.earth, target_orbit)
    rho0, rho_dot0 = relative_state_lvlh(chaser_r0, chaser_v0, target_r0, target_v0)
    initial_x = np.array([rho0[0], rho_dot0[0], rho0[1], rho_dot0[1]], dtype=float)
    t_eval = _make_time_grid(config.full_case)

    def rhs(t_seconds: float, state: np.ndarray) -> np.ndarray:
        chaser_r = state[0:3]
        chaser_v = state[3:6]
        target_r = state[6:9]
        target_v = state[9:12]
        theta1_hat = state[12:15]
        theta2_hat = state[15:18]
        rho_lvlh, rho_dot_lvlh = relative_state_lvlh(chaser_r, chaser_v, target_r, target_v)
        x_state = np.array([rho_lvlh[0], rho_dot_lvlh[0], rho_lvlh[1], rho_dot_lvlh[1]], dtype=float)
        target_cb = _target_ballistic_coefficient(config, t_seconds)
        snapshot = control_law(config, lqr, t_seconds, x_state, theta1_hat, theta2_hat, target_cb)
        theta1_dot, theta2_dot = adaptive_law(
            config.controller, lqr, t_seconds, x_state, theta1_hat, snapshot.command_cb_chaser
        )
        chaser_density = atmosphere_model.density(config.earth, t_seconds, chaser_r)
        target_density = atmosphere_model.density(config.earth, t_seconds, target_r)
        chaser_acc = _gravity_and_j2_acceleration(config, chaser_r) + _drag_acceleration(
            config, chaser_density.density_kg_m3, snapshot.command_cb_chaser, chaser_r, chaser_v
        )
        target_acc = _gravity_and_j2_acceleration(config, target_r) + _drag_acceleration(
            config, target_density.density_kg_m3, target_cb, target_r, target_v
        )
        return np.concatenate([chaser_v, chaser_acc, target_v, target_acc, theta1_dot, theta2_dot])

    y0 = np.concatenate(
        [
            chaser_r0,
            chaser_v0,
            target_r0,
            target_v0,
            config.controller.theta1_hat0.copy(),
            config.controller.theta2_hat0.copy(),
        ]
    )
    sol = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        y0,
        t_eval=t_eval,
        rtol=config.solver_rtol,
        atol=config.solver_atol,
        method="DOP853",
    )
    x_history = np.zeros((sol.t.size, 4))
    theta1_hat_history = sol.y[12:15].T
    theta2_hat_history = sol.y[15:18].T
    cb_chaser_history = np.zeros_like(sol.t)
    cb_target_history = np.zeros_like(sol.t)
    orbit_path_xy = np.zeros((sol.t.size, 2))
    atmosphere_source = "unknown"
    for idx, t_seconds in enumerate(sol.t):
        chaser_r = sol.y[0:3, idx]
        chaser_v = sol.y[3:6, idx]
        target_r = sol.y[6:9, idx]
        target_v = sol.y[9:12, idx]
        rho_lvlh, rho_dot_lvlh = relative_state_lvlh(chaser_r, chaser_v, target_r, target_v)
        x_state = np.array([rho_lvlh[0], rho_dot_lvlh[0], rho_lvlh[1], rho_dot_lvlh[1]], dtype=float)
        x_history[idx] = x_state
        orbit_path_xy[idx] = [rho_lvlh[0], rho_lvlh[1]]
        target_cb = _target_ballistic_coefficient(config, t_seconds)
        snapshot = control_law(
            config, lqr, t_seconds, x_state, theta1_hat_history[idx], theta2_hat_history[idx], target_cb
        )
        cb_chaser_history[idx] = snapshot.command_cb_chaser
        cb_target_history[idx] = target_cb
        atmosphere_source = atmosphere_model.density(config.earth, t_seconds, chaser_r).source
    return SimulationResult(
        time_s=sol.t,
        x_history=x_history,
        cb_chaser_history=cb_chaser_history,
        cb_target_history=cb_target_history,
        theta1_hat_history=theta1_hat_history,
        theta2_hat_history=theta2_hat_history,
        orbit_path_xy_m=orbit_path_xy,
        atmosphere_source=atmosphere_source,
        lqr=lqr,
        initial_state=initial_x,
    )

