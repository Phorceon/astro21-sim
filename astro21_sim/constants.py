from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np


@dataclass(frozen=True)
class EarthConstants:
    mu: float = 3.986004418e14
    radius_m: float = 6_378_137.0
    j2: float = 1.08262668e-3
    omega_rad_s: float = 7.2921159e-5


@dataclass(frozen=True)
class OrbitElements:
    semi_major_axis_m: float
    eccentricity: float
    inclination_rad: float
    raan_rad: float
    arg_perigee_rad: float
    true_anomaly_rad: float


@dataclass(frozen=True)
class TargetPerturbation:
    delta_a_m: float
    eccentricity: float
    delta_true_anomaly_rad: float


@dataclass(frozen=True)
class SpacecraftConfig:
    drag_coefficient: float
    chaser_mass_kg: float
    target_mass_kg: float
    chaser_area_min_m2: float
    chaser_area_max_m2: float
    target_area_m2: float

    @property
    def target_ballistic_coefficient(self) -> float:
        return self.target_area_m2 * self.drag_coefficient / (2.0 * self.target_mass_kg)

    @property
    def chaser_ballistic_coefficient_min(self) -> float:
        return self.chaser_area_min_m2 * self.drag_coefficient / (2.0 * self.chaser_mass_kg)

    @property
    def chaser_ballistic_coefficient_max(self) -> float:
        return self.chaser_area_max_m2 * self.drag_coefficient / (2.0 * self.chaser_mass_kg)


@dataclass(frozen=True)
class ControllerConfig:
    q_matrix: np.ndarray
    r_scalar: float
    gamma1: np.ndarray
    gamma2: np.ndarray
    theta1_hat0: np.ndarray
    theta2_hat0: np.ndarray
    theta1_lower: np.ndarray
    theta1_upper: np.ndarray
    theta2_lower: np.ndarray
    theta2_upper: np.ndarray
    min_theta1_basis_value: float


@dataclass(frozen=True)
class DensityFitConfig:
    d1: float
    d2: float
    d3: float


@dataclass(frozen=True)
class SimulationCase:
    name: str
    target_perturbation: TargetPerturbation
    duration_hours: float
    sample_step_s: float


@dataclass(frozen=True)
class AtmosphereConfig:
    epoch_utc: datetime
    f107: float
    f107a: float
    ap: float
    fallback_scale_height_m: float


@dataclass(frozen=True)
class ReferenceConfig:
    earth: EarthConstants
    chaser_orbit: OrbitElements
    spacecraft: SpacecraftConfig
    controller: ControllerConfig
    density_fit: DensityFitConfig
    ss_case: SimulationCase
    full_case: SimulationCase
    atmosphere: AtmosphereConfig
    input_channel_sign: float
    full_target_tumble_rpm: float
    full_target_tumble_fraction: float
    solver_rtol: float
    solver_atol: float


def build_reference_config() -> ReferenceConfig:
    earth = EarthConstants()
    chaser_orbit = OrbitElements(
        semi_major_axis_m=6.7131e6,
        eccentricity=0.0,
        inclination_rad=np.deg2rad(51.94),
        raan_rad=np.deg2rad(206.36),
        arg_perigee_rad=np.deg2rad(101.07),
        true_anomaly_rad=np.deg2rad(108.08),
    )
    spacecraft = SpacecraftConfig(
        drag_coefficient=2.2,
        chaser_mass_kg=3.0,
        target_mass_kg=1.5,
        chaser_area_min_m2=0.01,
        chaser_area_max_m2=0.5,
        target_area_m2=0.2,
    )
    controller = ControllerConfig(
        q_matrix=np.diag([180.0, 1.0, 1.8, 1.0]),
        r_scalar=1.8e16,
        gamma1=1.0e-21 * np.eye(3),
        gamma2=1.5e-21 * np.eye(3),
        theta1_hat0=np.array([3.8e-4, 0.0, 0.0], dtype=float),
        theta2_hat0=np.array([-3.5e-5, 0.0, 0.0], dtype=float),
        theta1_lower=np.array([1.0e-6, -5.0e-5, -5.0e-5], dtype=float),
        theta1_upper=np.array([1.0e-3, 5.0e-5, 5.0e-5], dtype=float),
        theta2_lower=np.array([-1.0e-3, -1.0e-3, -1.0e-3], dtype=float),
        theta2_upper=np.array([1.0e-3, 1.0e-3, 1.0e-3], dtype=float),
        min_theta1_basis_value=1.0e-6,
    )
    density_fit = DensityFitConfig(
        d1=3.3319e-12,
        d2=-7.1895e-13,
        d3=1.3008e-13,
    )
    ss_case = SimulationCase(
        name="ss_reference",
        target_perturbation=TargetPerturbation(
            delta_a_m=325.0,
            eccentricity=5.0e-5,
            delta_true_anomaly_rad=np.deg2rad(0.05),
        ),
        duration_hours=60.0,
        sample_step_s=60.0,
    )
    full_case = SimulationCase(
        name="full_reference",
        target_perturbation=TargetPerturbation(
            delta_a_m=-62.5,
            eccentricity=5.0e-5,
            delta_true_anomaly_rad=np.deg2rad(0.095),
        ),
        duration_hours=62.0,
        sample_step_s=13.0,
    )
    atmosphere = AtmosphereConfig(
        epoch_utc=datetime(2020, 3, 4, 0, 0, tzinfo=timezone.utc),
        f107=150.0,
        f107a=150.0,
        ap=4.0,
        fallback_scale_height_m=50_000.0,
    )
    return ReferenceConfig(
        earth=earth,
        chaser_orbit=chaser_orbit,
        spacecraft=spacecraft,
        controller=controller,
        density_fit=density_fit,
        ss_case=ss_case,
        full_case=full_case,
        atmosphere=atmosphere,
        input_channel_sign=1.0,
        full_target_tumble_rpm=5.0,
        full_target_tumble_fraction=0.10,
        solver_rtol=1.0e-9,
        solver_atol=1.0e-12,
    )
