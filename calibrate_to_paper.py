from __future__ import annotations

import argparse

from astro21_sim.calibration import calibrate_full_reference, calibrate_ss_reference
from astro21_sim.constants import build_reference_config
from astro21_sim.regression import FULL_PAPER_TARGETS, SS_PAPER_TARGETS, compare_result_to_targets, format_report, report_score
from astro21_sim.simulations import simulate_full, simulate_ss


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate hidden reference parameters against paper-derived targets.")
    parser.add_argument("--mode", choices=("ss", "full"), required=True)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--popsize", type=int, default=None)
    args = parser.parse_args()

    config = build_reference_config()
    if args.mode == "ss":
        calibrated, score = calibrate_ss_reference(
            config,
            maxiter=args.maxiter or 10,
            popsize=args.popsize or 8,
        )
        result = simulate_ss(calibrated)
        report = compare_result_to_targets("ss", result, SS_PAPER_TARGETS)
        print(f"score={score} report_score={report_score(report)}")
        print(
            "best_params: "
            f"delta_a={calibrated.ss_case.target_perturbation.delta_a_m:.6f}, "
            f"delta_nu_deg={calibrated.ss_case.target_perturbation.delta_true_anomaly_rad * 180.0 / 3.141592653589793:.6f}, "
            f"input_sign={calibrated.input_channel_sign:.0f}, "
            f"theta1_hat0_0={calibrated.controller.theta1_hat0[0]:.6e}, "
            f"theta2_hat0_0={calibrated.controller.theta2_hat0[0]:.6e}"
        )
        print(format_report(report))
        return

    calibrated, score = calibrate_full_reference(
        config,
        maxiter=args.maxiter or 6,
        popsize=args.popsize or 6,
        require_exact_atmosphere=True,
    )
    result = simulate_full(calibrated, require_exact_atmosphere=True)
    report = compare_result_to_targets("full", result, FULL_PAPER_TARGETS)
    print(f"score={score} report_score={report_score(report)}")
    print(
        "best_params: "
        f"delta_a={calibrated.full_case.target_perturbation.delta_a_m:.6f}, "
        f"delta_nu_deg={calibrated.full_case.target_perturbation.delta_true_anomaly_rad * 180.0 / 3.141592653589793:.6f}, "
        f"input_sign={calibrated.input_channel_sign:.0f}, "
        f"theta1_hat0_0={calibrated.controller.theta1_hat0[0]:.6e}, "
        f"theta2_hat0_0={calibrated.controller.theta2_hat0[0]:.6e}"
    )
    print(format_report(report))


if __name__ == "__main__":
    main()
