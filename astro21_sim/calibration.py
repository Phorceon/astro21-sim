from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import differential_evolution

from .constants import ControllerConfig, ReferenceConfig, SimulationCase, TargetPerturbation
from .regression import FULL_PAPER_TARGETS, SS_PAPER_TARGETS, compare_result_to_targets, report_score
from .simulations import simulate_full, simulate_ss


def calibrate_ss_reference(
    base_config: ReferenceConfig,
    maxiter: int = 10,
    popsize: int = 8,
    seed: int = 7,
    allow_sign_flip: bool = False,
) -> tuple[ReferenceConfig, float]:
    candidates = []
    signs = [1.0, -1.0] if allow_sign_flip else [base_config.input_channel_sign]
    for sign in signs:
        seed_config = replace(base_config, input_channel_sign=sign)
        result = differential_evolution(
            lambda vec: _score_report(
                compare_result_to_targets(
                    "ss",
                    simulate_ss(_with_calibration_params(seed_config, vec, mode="ss")),
                    SS_PAPER_TARGETS,
                )
            ),
            bounds=[
                (-500.0, 500.0),
                (-0.2, 0.2),
                (2.5e-4, 5.0e-4),
                (-1.0e-4, 1.0e-4),
            ],
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            polish=False,
            updating="deferred",
            workers=1,
        )
        candidates.append((_with_calibration_params(seed_config, result.x, mode="ss"), float(result.fun)))
    return min(candidates, key=lambda item: item[1])


def calibrate_full_reference(
    base_config: ReferenceConfig,
    maxiter: int = 6,
    popsize: int = 6,
    seed: int = 7,
    require_exact_atmosphere: bool = True,
    allow_sign_flip: bool = False,
) -> tuple[ReferenceConfig, float]:
    candidates = []
    signs = [1.0, -1.0] if allow_sign_flip else [base_config.input_channel_sign]
    for sign in signs:
        seed_config = replace(base_config, input_channel_sign=sign)
        result = differential_evolution(
            lambda vec: _score_report(
                compare_result_to_targets(
                    "full",
                    simulate_full(_with_calibration_params(seed_config, vec, mode="full"), require_exact_atmosphere=require_exact_atmosphere),
                    FULL_PAPER_TARGETS,
                )
            ),
            bounds=[
                (-500.0, 500.0),
                (-0.2, 0.2),
                (2.5e-4, 5.0e-4),
                (-1.0e-4, 1.0e-4),
            ],
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            polish=False,
            updating="deferred",
            workers=1,
        )
        candidates.append((_with_calibration_params(seed_config, result.x, mode="full"), float(result.fun)))
    return min(candidates, key=lambda item: item[1])


def _with_calibration_params(base_config: ReferenceConfig, vec: np.ndarray, mode: str) -> ReferenceConfig:
    delta_a_m, delta_nu_deg, theta1_0, theta2_0 = [float(item) for item in vec]
    controller = replace(
        base_config.controller,
        theta1_hat0=np.array([theta1_0, 0.0, 0.0], dtype=float),
        theta2_hat0=np.array([theta2_0, 0.0, 0.0], dtype=float),
    )
    case = SimulationCase(
        name=f"{mode}_calibrated",
        target_perturbation=TargetPerturbation(
            delta_a_m=delta_a_m,
            eccentricity=5.0e-5,
            delta_true_anomaly_rad=np.deg2rad(delta_nu_deg),
        ),
        duration_hours=base_config.ss_case.duration_hours if mode == "ss" else base_config.full_case.duration_hours,
        sample_step_s=base_config.ss_case.sample_step_s if mode == "ss" else base_config.full_case.sample_step_s,
    )
    if mode == "ss":
        return replace(base_config, controller=controller, ss_case=case)
    return replace(base_config, controller=controller, full_case=case)


def _score_report(report) -> float:
    return report_score(report)
