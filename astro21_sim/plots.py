from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .simulations import SimulationResult


plt.rcParams.update(
    {
        "figure.figsize": (11, 8),
        "axes.grid": True,
        "grid.alpha": 0.35,
        "font.size": 11,
    }
)


def _time_hours(result: SimulationResult) -> np.ndarray:
    return result.time_s / 3600.0


def _save(fig: plt.Figure, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def make_relative_states_figure(result: SimulationResult, title: str) -> plt.Figure:
    t_hr = _time_hours(result)
    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1.0])
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_xy = fig.add_subplot(gs[:, 1])

    ax_pos.plot(t_hr, result.x_history[:, 0], label=r"$\Delta x$")
    ax_pos.plot(t_hr, result.x_history[:, 2], label=r"$\Delta y$")
    ax_pos.set_ylabel(r"$\Delta x,\Delta y\ [m]$")
    ax_pos.set_xlabel("Time [h]")
    ax_pos.legend(loc="upper right")

    ax_vel.plot(t_hr, result.x_history[:, 1], label=r"$\Delta \dot{x}$")
    ax_vel.plot(t_hr, result.x_history[:, 3], label=r"$\Delta \dot{y}$")
    ax_vel.set_ylabel(r"$\Delta \dot{x},\Delta \dot{y}\ [m/s]$")
    ax_vel.set_xlabel("Time [h]")
    ax_vel.legend(loc="upper right")

    ax_xy.plot(result.orbit_path_xy_m[:, 0], result.orbit_path_xy_m[:, 1], label="Target")
    ax_xy.scatter(
        [result.orbit_path_xy_m[0, 0]],
        [result.orbit_path_xy_m[0, 1]],
        facecolors="none",
        edgecolors="tab:red",
        s=80,
        label="Initial Position",
    )
    ax_xy.scatter(
        [result.orbit_path_xy_m[-1, 0]],
        [result.orbit_path_xy_m[-1, 1]],
        marker="x",
        color="tab:red",
        s=70,
        label="Final Position",
    )
    ax_xy.set_xlabel(r"$\Delta x\ [m]$")
    ax_xy.set_ylabel(r"$\Delta y\ [m]$")
    ax_xy.legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_relative_states(result: SimulationResult, output_dir: Path, filename: str, title: str) -> Path:
    fig = make_relative_states_figure(result, title)
    return _save(fig, output_dir, filename)


def make_control_figure(result: SimulationResult, title: str) -> plt.Figure:
    t_hr = _time_hours(result)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(t_hr, result.cb_target_history, label=r"$C_{b,t}$")
    ax.plot(t_hr, result.cb_chaser_history, label=r"$C_{b,c}$")
    ax.set_xlabel("Time [h]")
    ax.set_ylabel(r"$C_b\ [m^2/kg]$")
    ax.legend(loc="upper right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_control(result: SimulationResult, output_dir: Path, filename: str, title: str) -> Path:
    fig = make_control_figure(result, title)
    return _save(fig, output_dir, filename)


def make_estimates_figure(result: SimulationResult, title: str) -> plt.Figure:
    t_hr = _time_hours(result)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for idx in range(3):
        axes[0].plot(t_hr, result.theta1_hat_history[:, idx], label=rf"$\hat{{\Theta}}_1({idx + 1})$")
        # The paper's estimate plots use the target-drag term sign convention, which is opposite
        # the stored adaptive state used in the controller implementation.
        axes[1].plot(t_hr, -result.theta2_hat_history[:, idx], label=rf"$\hat{{\Theta}}_2({idx + 1})$")
    axes[0].set_ylabel(r"$\hat{\Theta}_1 [(kg)/(m s^2)]$")
    axes[1].set_ylabel(r"$\hat{\Theta}_2 [(kg)/(m s^2)]$")
    axes[1].set_xlabel("Time [h]")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_estimates(result: SimulationResult, output_dir: Path, filename: str, title: str) -> Path:
    fig = make_estimates_figure(result, title)
    return _save(fig, output_dir, filename)
