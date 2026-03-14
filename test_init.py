import numpy as np

from astro21_sim.constants import build_reference_config
from astro21_sim.orbits import initial_in_plane_state


def describe_case(name: str, delta_a_m: float, delta_nu_rad: float, x0: np.ndarray) -> None:
    delta_nu_deg = np.rad2deg(delta_nu_rad)
    within_delta_a = abs(delta_a_m) <= 500.0
    within_delta_nu = abs(delta_nu_deg) <= 0.2
    print(f"{name} Case:")
    print(f"  delta_a={delta_a_m:.2f} m (within paper bound: {within_delta_a})")
    print(f"  delta_nu={delta_nu_deg:.6f} deg (within paper bound: {within_delta_nu})")
    print(f"  Init state: x={x0[0]:.2f}, xdot={x0[1]:.4f}, y={x0[2]:.2f}, ydot={x0[3]:.4f}")


config = build_reference_config()

ss = config.ss_case.target_perturbation
ss_x0 = initial_in_plane_state(config.earth, config.chaser_orbit, ss)
describe_case("SS", ss.delta_a_m, ss.delta_true_anomaly_rad, ss_x0)

full = config.full_case.target_perturbation
full_x0 = initial_in_plane_state(config.earth, config.chaser_orbit, full)
describe_case("Full", full.delta_a_m, full.delta_true_anomaly_rad, full_x0)
