"""
Parametric study: controller tuning optimization for fastest convergence
under chaotic atmospheric conditions.

Sweeps controller parameters (Q weights, R, Γ₁, Γ₂) across multiple chaos
scenarios and produces heatmaps of convergence time.

Usage:
    python3 parametric_study.py                    # Run all sweeps
    python3 parametric_study.py --grid 8           # Use 8×8 grid (faster)
    python3 parametric_study.py --scenarios nominal high_density
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(WORKSPACE_ROOT / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from astro21_sim.constants import (
    ControllerConfig,
    DensityFitConfig,
    ReferenceConfig,
    SimulationCase,
    TargetPerturbation,
    build_reference_config,
)
from astro21_sim.simulations import simulate_ss

# ---------------------------------------------------------------------------
# Convergence metric
# ---------------------------------------------------------------------------

def convergence_time_hours(
    result, threshold_m: float = 100.0, sustained: bool = True,
) -> float | None:
    """Time (hours) when ||[Δx, Δy]|| drops below threshold and stays below."""
    t_h = result.time_s / 3600.0
    norms = np.linalg.norm(result.x_history[:, [0, 2]], axis=1)
    below = norms < threshold_m
    if sustained:
        for idx in range(below.size):
            if below[idx] and np.all(below[idx:]):
                return float(t_h[idx])
        return None
    else:
        first = np.argmax(below)
        return float(t_h[first]) if below[first] else None


# ---------------------------------------------------------------------------
# Chaos scenarios
# ---------------------------------------------------------------------------

def _make_scenario(
    base: ReferenceConfig,
    name: str,
    density_scale: float = 1.0,
    oscillation_scale: float = 1.0,
    delta_a_scale: float = 1.0,
    delta_nu_scale: float = 1.0,
    duration_hours: float = 80.0,
) -> tuple[str, ReferenceConfig]:
    """Build a chaos scenario by scaling baseline parameters."""
    d = base.density_fit
    density_fit = DensityFitConfig(
        d1=d.d1 * density_scale,
        d2=d.d2 * oscillation_scale,
        d3=d.d3 * oscillation_scale,
    )
    p = base.ss_case.target_perturbation
    case = SimulationCase(
        name=f"ss_{name}",
        target_perturbation=TargetPerturbation(
            delta_a_m=p.delta_a_m * delta_a_scale,
            eccentricity=p.eccentricity,
            delta_true_anomaly_rad=p.delta_true_anomaly_rad * delta_nu_scale,
        ),
        duration_hours=duration_hours,
        sample_step_s=base.ss_case.sample_step_s,
    )
    return name, replace(base, density_fit=density_fit, ss_case=case)


SCENARIO_BUILDERS: dict[str, dict] = {
    "nominal":        dict(density_scale=1.0, oscillation_scale=1.0),
    "high_density":   dict(density_scale=3.0, oscillation_scale=1.0),
    "low_density":    dict(density_scale=0.3, oscillation_scale=1.0),
    "density_storm":  dict(density_scale=1.5, oscillation_scale=3.0),
    "large_sep":      dict(delta_a_scale=2.0, delta_nu_scale=2.0),
    "extreme_chaos":  dict(density_scale=3.0, oscillation_scale=3.0, delta_a_scale=1.5),
}


def build_scenarios(
    base: ReferenceConfig, selected: list[str] | None = None,
) -> list[tuple[str, ReferenceConfig]]:
    keys = selected or list(SCENARIO_BUILDERS.keys())
    return [_make_scenario(base, k, **SCENARIO_BUILDERS[k]) for k in keys]


# ---------------------------------------------------------------------------
# Parameter sweep engine
# ---------------------------------------------------------------------------

def _apply_controller_params(
    config: ReferenceConfig,
    q11: float | None = None,
    q33: float | None = None,
    r_scalar: float | None = None,
    gamma1_diag: float | None = None,
    gamma2_diag: float | None = None,
) -> ReferenceConfig:
    """Return a new config with the specified controller param overrides."""
    ctrl = config.controller
    diag = np.diag(ctrl.q_matrix).copy()
    if q11 is not None:
        diag[0] = q11
    if q33 is not None:
        diag[2] = q33
    new_ctrl = replace(
        ctrl,
        q_matrix=np.diag(diag),
        r_scalar=r_scalar if r_scalar is not None else ctrl.r_scalar,
        gamma1=np.eye(3) * (gamma1_diag if gamma1_diag is not None else ctrl.gamma1[0, 0]),
        gamma2=np.eye(3) * (gamma2_diag if gamma2_diag is not None else ctrl.gamma2[0, 0]),
    )
    return replace(config, controller=new_ctrl)


def sweep_2d(
    config: ReferenceConfig,
    param_x: str,
    values_x: np.ndarray,
    param_y: str,
    values_y: np.ndarray,
    threshold_m: float = 100.0,
) -> np.ndarray:
    """Run 2D parameter sweep, return convergence_time matrix [ny, nx]."""
    result_matrix = np.full((len(values_y), len(values_x)), np.nan)
    for iy, vy in enumerate(values_y):
        for ix, vx in enumerate(values_x):
            overrides = {param_x: vx, param_y: vy}
            cfg = _apply_controller_params(config, **overrides)
            try:
                sim = simulate_ss(cfg)
                t_conv = convergence_time_hours(sim, threshold_m)
                result_matrix[iy, ix] = t_conv if t_conv is not None else np.nan
            except Exception:
                result_matrix[iy, ix] = np.nan
    return result_matrix


# ---------------------------------------------------------------------------
# Heatmap plotting
# ---------------------------------------------------------------------------

PARAM_LABELS = {
    "q11": r"$Q_{11}$ (radial weight)",
    "q33": r"$Q_{33}$ (along-track weight)",
    "r_scalar": r"$R$ (control cost)",
    "gamma1_diag": r"$\Gamma_1$ (adapt. rate Θ₁)",
    "gamma2_diag": r"$\Gamma_2$ (adapt. rate Θ₂)",
}


def plot_heatmap(
    values_x: np.ndarray,
    values_y: np.ndarray,
    data: np.ndarray,
    param_x: str,
    param_y: str,
    scenario_name: str,
    output_path: Path,
    threshold_m: float = 100.0,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 7))

    # Mask NaN (did not converge)
    masked = np.ma.masked_invalid(data)
    valid = masked.compressed()

    if valid.size == 0:
        ax.text(0.5, 0.5, "No convergence\nfor any parameter\ncombination",
                transform=ax.transAxes, ha="center", va="center", fontsize=16, color="red")
    else:
        vmin, vmax = float(np.min(valid)), float(np.max(valid))
        if vmin <= 0:
            vmin = 0.1
        im = ax.pcolormesh(
            np.arange(len(values_x) + 1),
            np.arange(len(values_y) + 1),
            masked,
            cmap="RdYlGn_r",
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )
        cbar = fig.colorbar(im, ax=ax, label="Convergence time [h]")

        # Mark best (minimum convergence time)
        best_iy, best_ix = np.unravel_index(np.nanargmin(data), data.shape)
        ax.plot(best_ix + 0.5, best_iy + 0.5, "k*", markersize=18, markeredgewidth=1.5,
                label=f"Best: {data[best_iy, best_ix]:.1f}h")
        ax.legend(loc="upper right", fontsize=10)

        # Mark non-convergent cells
        for iy in range(data.shape[0]):
            for ix in range(data.shape[1]):
                if np.isnan(data[iy, ix]):
                    ax.text(ix + 0.5, iy + 0.5, "✗", ha="center", va="center",
                            fontsize=10, color="gray", alpha=0.7)

    # Axis labels with actual values
    x_use_log = _is_log_spaced(values_x)
    y_use_log = _is_log_spaced(values_y)
    ax.set_xticks(np.arange(len(values_x)) + 0.5)
    ax.set_xticklabels([_fmt(v) for v in values_x], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(values_y)) + 0.5)
    ax.set_yticklabels([_fmt(v) for v in values_y], fontsize=8)

    ax.set_xlabel(PARAM_LABELS.get(param_x, param_x), fontsize=12)
    ax.set_ylabel(PARAM_LABELS.get(param_y, param_y), fontsize=12)
    ax.set_title(f"Convergence time — {scenario_name}\n(threshold = {threshold_m:.0f} m)", fontsize=13)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _is_log_spaced(values: np.ndarray) -> bool:
    if len(values) < 3 or np.any(values <= 0):
        return False
    ratios = values[1:] / values[:-1]
    return float(np.std(ratios) / np.mean(ratios)) < 0.1


def _fmt(value: float) -> str:
    if abs(value) >= 1e4 or (0 < abs(value) < 0.01):
        return f"{value:.1e}"
    return f"{value:.3g}"


# ---------------------------------------------------------------------------
# Robustness heatmap (worst-case across scenarios)
# ---------------------------------------------------------------------------

def plot_robustness_heatmap(
    values_x: np.ndarray,
    values_y: np.ndarray,
    all_data: dict[str, np.ndarray],
    param_x: str,
    param_y: str,
    output_path: Path,
    threshold_m: float = 100.0,
) -> Path:
    """Max convergence time across all scenarios — lower = more robust."""
    stacked = np.stack(list(all_data.values()), axis=0)
    worst_case = np.nanmax(stacked, axis=0)
    # If any scenario failed, mark as NaN
    any_nan = np.any(np.isnan(stacked), axis=0)
    worst_case[any_nan] = np.nan
    return plot_heatmap(
        values_x, values_y, worst_case,
        param_x, param_y, "ROBUST (worst-case across all)",
        output_path, threshold_m,
    )


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

SWEEP_DEFINITIONS = [
    {
        "name": "Q11_vs_R",
        "param_x": "q11",
        "param_y": "r_scalar",
        "values_x": np.array([20, 50, 100, 150, 180, 220, 300, 400, 600, 1000]),
        "values_y": np.array([1e14, 5e14, 1e15, 5e15, 1e16, 1.8e16, 5e16, 1e17, 5e17, 1e18]),
    },
    {
        "name": "Gamma1_vs_Gamma2",
        "param_x": "gamma1_diag",
        "param_y": "gamma2_diag",
        "values_x": np.array([1e-23, 5e-23, 1e-22, 5e-22, 1e-21, 5e-21, 1e-20, 5e-20, 1e-19]),
        "values_y": np.array([1e-23, 5e-23, 1e-22, 5e-22, 1.5e-21, 5e-21, 1e-20, 5e-20, 1e-19]),
    },
    {
        "name": "Q11_vs_Q33",
        "param_x": "q11",
        "param_y": "q33",
        "values_x": np.array([10, 30, 60, 100, 150, 180, 250, 400, 700, 1200]),
        "values_y": np.array([0.1, 0.3, 0.6, 1.0, 1.8, 3.0, 5.0, 10.0, 20.0, 50.0]),
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parametric study for adaptive diff-drag controller.")
    parser.add_argument("--output-dir", default="outputs/parametric_study", help="Output directory.")
    parser.add_argument("--grid", type=int, default=None, help="Override grid size (NxN).")
    parser.add_argument("--threshold", type=float, default=100.0, help="Convergence threshold in meters.")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Scenario names to run (default: all).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    base_config = build_reference_config()
    scenarios = build_scenarios(base_config, args.scenarios)

    total_sims = 0
    for sweep in SWEEP_DEFINITIONS:
        nx = args.grid or len(sweep["values_x"])
        ny = args.grid or len(sweep["values_y"])
        total_sims += nx * ny * len(scenarios)

    print(f"Running {total_sims} simulations across {len(scenarios)} scenarios "
          f"and {len(SWEEP_DEFINITIONS)} sweep pairs...")
    print(f"Convergence threshold: {args.threshold:.0f} m")
    print(f"Estimated time: ~{total_sims * 0.35 / 60:.1f} minutes\n")

    t0 = time.time()
    run_count = 0
    summary_lines = ["# Parametric Study Results\n"]

    for sweep in SWEEP_DEFINITIONS:
        sweep_name = sweep["name"]
        px, py = sweep["param_x"], sweep["param_y"]
        vx = sweep["values_x"] if args.grid is None else np.geomspace(
            sweep["values_x"][0], sweep["values_x"][-1], args.grid
        ) if sweep["values_x"][0] > 0 else np.linspace(
            sweep["values_x"][0], sweep["values_x"][-1], args.grid
        )
        vy = sweep["values_y"] if args.grid is None else np.geomspace(
            sweep["values_y"][0], sweep["values_y"][-1], args.grid
        ) if sweep["values_y"][0] > 0 else np.linspace(
            sweep["values_y"][0], sweep["values_y"][-1], args.grid
        )

        print(f"━━━ Sweep: {sweep_name} ({len(vx)}×{len(vy)}) ━━━")
        all_data: dict[str, np.ndarray] = {}

        for scenario_name, scenario_config in scenarios:
            print(f"  Scenario: {scenario_name}...", end=" ", flush=True)
            data = sweep_2d(scenario_config, px, vx, py, vy, args.threshold)
            all_data[scenario_name] = data
            run_count += len(vx) * len(vy)

            path = plot_heatmap(
                vx, vy, data, px, py, scenario_name,
                output_dir / f"{sweep_name}_{scenario_name}.png",
                args.threshold,
            )

            valid = data[~np.isnan(data)]
            if valid.size > 0:
                best_iy, best_ix = np.unravel_index(np.nanargmin(data), data.shape)
                print(f"best={data[best_iy, best_ix]:.1f}h "
                      f"({px}={_fmt(vx[best_ix])}, {py}={_fmt(vy[best_iy])}), "
                      f"converged={valid.size}/{data.size}")
                summary_lines.append(
                    f"**{sweep_name} / {scenario_name}**: "
                    f"Best = {data[best_iy, best_ix]:.1f}h at "
                    f"{px}={_fmt(vx[best_ix])}, {py}={_fmt(vy[best_iy])}\n"
                )
            else:
                print("NO convergence for any params!")
                summary_lines.append(f"**{sweep_name} / {scenario_name}**: No convergence\n")

        # Robustness (worst-case) heatmap
        robust_path = plot_robustness_heatmap(
            vx, vy, all_data, px, py,
            output_dir / f"{sweep_name}_ROBUST.png",
            args.threshold,
        )
        # Find robust optimum
        stacked = np.stack(list(all_data.values()), axis=0)
        worst = np.nanmax(stacked, axis=0)
        worst[np.any(np.isnan(stacked), axis=0)] = np.nan
        valid_worst = worst[~np.isnan(worst)]
        if valid_worst.size > 0:
            best_iy, best_ix = np.unravel_index(np.nanargmin(worst), worst.shape)
            print(f"  ★ ROBUST optimum: {worst[best_iy, best_ix]:.1f}h "
                  f"({px}={_fmt(vx[best_ix])}, {py}={_fmt(vy[best_iy])})")
            summary_lines.append(
                f"\n**★ {sweep_name} ROBUST optimum**: "
                f"{worst[best_iy, best_ix]:.1f}h at "
                f"{px}={_fmt(vx[best_ix])}, {py}={_fmt(vy[best_iy])}\n"
            )
        print()

    elapsed = time.time() - t0
    print(f"Done! {run_count} simulations in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Results saved to {output_dir}/")

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
