from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from .constants import EarthConstants, OrbitElements, TargetPerturbation


def mean_motion(mu: float, semi_major_axis_m: float) -> float:
    return np.sqrt(mu / semi_major_axis_m**3)


def orbital_elements_with_perturbation(
    base: OrbitElements, perturbation: TargetPerturbation
) -> OrbitElements:
    return OrbitElements(
        semi_major_axis_m=base.semi_major_axis_m + perturbation.delta_a_m,
        eccentricity=perturbation.eccentricity,
        inclination_rad=base.inclination_rad,
        raan_rad=base.raan_rad,
        arg_perigee_rad=base.arg_perigee_rad,
        true_anomaly_rad=base.true_anomaly_rad + perturbation.delta_true_anomaly_rad,
    )


def oe_to_eci(earth: EarthConstants, elements: OrbitElements) -> tuple[np.ndarray, np.ndarray]:
    a = elements.semi_major_axis_m
    e = elements.eccentricity
    nu = elements.true_anomaly_rad
    p = a * (1.0 - e**2)
    r_pf = np.array(
        [
            p * np.cos(nu) / (1.0 + e * np.cos(nu)),
            p * np.sin(nu) / (1.0 + e * np.cos(nu)),
            0.0,
        ]
    )
    v_pf = np.array(
        [
            -np.sqrt(earth.mu / p) * np.sin(nu),
            np.sqrt(earth.mu / p) * (e + np.cos(nu)),
            0.0,
        ]
    )
    c_raan = np.cos(elements.raan_rad)
    s_raan = np.sin(elements.raan_rad)
    c_arg = np.cos(elements.arg_perigee_rad)
    s_arg = np.sin(elements.arg_perigee_rad)
    c_inc = np.cos(elements.inclination_rad)
    s_inc = np.sin(elements.inclination_rad)
    rotation = np.array(
        [
            [
                c_raan * c_arg - s_raan * s_arg * c_inc,
                -c_raan * s_arg - s_raan * c_arg * c_inc,
                s_raan * s_inc,
            ],
            [
                s_raan * c_arg + c_raan * s_arg * c_inc,
                -s_raan * s_arg + c_raan * c_arg * c_inc,
                -c_raan * s_inc,
            ],
            [
                s_arg * s_inc,
                c_arg * s_inc,
                c_inc,
            ],
        ]
    )
    return rotation @ r_pf, rotation @ v_pf


def lvlh_frame(r_eci: np.ndarray, v_eci: np.ndarray) -> np.ndarray:
    x_hat = r_eci / np.linalg.norm(r_eci)
    z_hat = np.cross(r_eci, v_eci)
    z_hat = z_hat / np.linalg.norm(z_hat)
    y_hat = np.cross(z_hat, x_hat)
    return np.vstack([x_hat, y_hat, z_hat])


def relative_state_lvlh(
    chaser_r_eci: np.ndarray,
    chaser_v_eci: np.ndarray,
    target_r_eci: np.ndarray,
    target_v_eci: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = lvlh_frame(chaser_r_eci, chaser_v_eci)
    rho_eci = target_r_eci - chaser_r_eci
    rho_dot_eci = target_v_eci - chaser_v_eci
    omega_lvlh_eci = np.cross(chaser_r_eci, chaser_v_eci) / np.linalg.norm(chaser_r_eci) ** 2
    rho_lvlh = frame @ rho_eci
    rho_dot_lvlh = frame @ (rho_dot_eci - np.cross(omega_lvlh_eci, rho_eci))
    return rho_lvlh, rho_dot_lvlh


def initial_in_plane_state(
    earth: EarthConstants,
    chaser: OrbitElements,
    perturbation: TargetPerturbation,
) -> np.ndarray:
    target = orbital_elements_with_perturbation(chaser, perturbation)
    chaser_r, chaser_v = oe_to_eci(earth, chaser)
    target_r, target_v = oe_to_eci(earth, target)
    rho_lvlh, rho_dot_lvlh = relative_state_lvlh(chaser_r, chaser_v, target_r, target_v)
    return np.array([rho_lvlh[0], rho_dot_lvlh[0], rho_lvlh[1], rho_dot_lvlh[1]], dtype=float)


def relative_speed_to_atmosphere_squared(
    earth: EarthConstants, r_eci: np.ndarray, v_eci: np.ndarray
) -> float:
    omega = np.array([0.0, 0.0, earth.omega_rad_s])
    v_rel = v_eci - np.cross(omega, r_eci)
    return float(v_rel @ v_rel)


def ss_shape_parameter(earth: EarthConstants, orbit: OrbitElements) -> float:
    return float(
        np.sqrt(
            1.0
            + 3.0
            * earth.j2
            * earth.radius_m**2
            / (8.0 * orbit.semi_major_axis_m**2)
            * (1.0 + 3.0 * np.cos(2.0 * orbit.inclination_rad))
        )
    )


def eci_to_geocentric(
    earth: EarthConstants,
    epoch_utc: datetime,
    t_seconds: float,
    r_eci: np.ndarray,
) -> tuple[float, float, float]:
    dt = timedelta(seconds=float(t_seconds))
    _ = epoch_utc + dt
    theta = earth.omega_rad_s * float(t_seconds)
    c_theta = np.cos(theta)
    s_theta = np.sin(theta)
    rotation = np.array(
        [[c_theta, s_theta, 0.0], [-s_theta, c_theta, 0.0], [0.0, 0.0, 1.0]]
    )
    r_ecef = rotation @ r_eci
    radius = np.linalg.norm(r_ecef)
    altitude_m = radius - earth.radius_m
    latitude_rad = np.arcsin(r_ecef[2] / radius)
    longitude_rad = np.arctan2(r_ecef[1], r_ecef[0])
    return float(latitude_rad), float(longitude_rad), float(altitude_m)

