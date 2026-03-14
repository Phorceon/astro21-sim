from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import ReferenceConfig
from .simulations import SimulationResult, simulate_full, simulate_ss


@dataclass(frozen=True)
class SeriesAnchor:
    time_h: float
    target: float
    tolerance: float


@dataclass(frozen=True)
class SeriesTarget:
    metric: str
    series_key: str
    anchors: tuple[SeriesAnchor, ...]


@dataclass(frozen=True)
class PathAnchor:
    time_h: float
    x_target_m: float
    y_target_m: float
    tolerance_m: float


@dataclass(frozen=True)
class StatisticTarget:
    metric: str
    series_key: str
    reducer: str
    start_h: float | None
    end_h: float | None
    target: float | None
    tolerance: float | None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class PaperFigureTargets:
    series_targets: tuple[SeriesTarget, ...]
    path_anchors: tuple[PathAnchor, ...]
    statistic_targets: tuple[StatisticTarget, ...]


@dataclass(frozen=True)
class ResultSummary:
    initial_x_m: float
    initial_y_m: float
    initial_xdot_m_s: float
    initial_ydot_m_s: float
    dy_peak_m: float
    dy_peak_time_h: float
    cb_initial: float
    cb_min: float
    cb_max: float
    cb_final: float
    theta1_1_final: float
    theta2_1_final: float
    path_turns: int
    completion_time_h: float | None
    final_norm_m: float


@dataclass(frozen=True)
class RegressionCheck:
    metric: str
    actual: float | int | None
    target: float | int | None
    delta: float | None
    tolerance: float | None
    passed: bool
    note: str

    @property
    def normalized_error(self) -> float:
        if self.delta is None or self.tolerance in (None, 0.0):
            return 0.0
        return abs(float(self.delta)) / float(self.tolerance)


@dataclass(frozen=True)
class RegressionReport:
    name: str
    summary: ResultSummary
    checks: tuple[RegressionCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


SS_PAPER_TARGETS = PaperFigureTargets(
    series_targets=(
        SeriesTarget(
            metric="dx",
            series_key="dx",
            anchors=(
                SeriesAnchor(0.0, 430.0, 100.0),
                SeriesAnchor(5.0, -40.0, 220.0),
                SeriesAnchor(10.0, 60.0, 180.0),
                SeriesAnchor(20.0, 20.0, 120.0),
                SeriesAnchor(40.0, 0.0, 60.0),
                SeriesAnchor(60.0, 0.0, 30.0),
            ),
        ),
        SeriesTarget(
            metric="dy",
            series_key="dy",
            anchors=(
                SeriesAnchor(0.0, 5860.0, 280.0),
                SeriesAnchor(5.0, 4700.0, 900.0),
                SeriesAnchor(10.0, 6900.0, 600.0),
                SeriesAnchor(20.0, 0.0, 3500.0),
                SeriesAnchor(40.0, 80.0, 160.0),
                SeriesAnchor(60.0, 0.0, 45.0),
            ),
        ),
        SeriesTarget(
            metric="dxdot",
            series_key="dxdot",
            anchors=(
                SeriesAnchor(0.0, 0.366, 0.08),
                SeriesAnchor(5.0, 0.00, 0.20),
                SeriesAnchor(10.0, 0.00, 0.20),
                SeriesAnchor(20.0, 0.00, 0.18),
                SeriesAnchor(40.0, 0.00, 0.12),
                SeriesAnchor(60.0, 0.00, 0.03),
            ),
        ),
        SeriesTarget(
            metric="dydot",
            series_key="dydot",
            anchors=(
                SeriesAnchor(0.0, -0.80, 0.25),
                SeriesAnchor(5.0, -0.20, 0.22),
                SeriesAnchor(10.0, 0.00, 0.45),
                SeriesAnchor(20.0, 0.00, 0.20),
                SeriesAnchor(40.0, 0.00, 0.08),
                SeriesAnchor(60.0, 0.00, 0.06),
            ),
        ),
        SeriesTarget(
            metric="cb_chaser",
            series_key="cb_chaser",
            anchors=(
                SeriesAnchor(0.0, 0.004, 0.010),
                SeriesAnchor(5.0, 0.180, 0.020),
                SeriesAnchor(10.0, 0.175, 0.025),
                SeriesAnchor(20.0, 0.150, 0.040),
                SeriesAnchor(40.0, 0.147, 0.015),
                SeriesAnchor(60.0, 0.147, 0.006),
            ),
        ),
        SeriesTarget(
            metric="theta1_1",
            series_key="theta1_1",
            anchors=(
                SeriesAnchor(0.0, 3.80e-4, 2.0e-5),
                SeriesAnchor(10.0, 3.70e-4, 2.5e-5),
                SeriesAnchor(20.0, 3.70e-4, 2.5e-5),
                SeriesAnchor(40.0, 3.85e-4, 3.5e-5),
                SeriesAnchor(60.0, 3.85e-4, 3.5e-5),
            ),
        ),
        SeriesTarget(
            metric="theta2_1_display",
            series_key="theta2_1_display",
            anchors=(
                SeriesAnchor(0.0, 3.5e-5, 2.0e-5),
                SeriesAnchor(10.0, -3.5e-5, 5.0e-5),
                SeriesAnchor(20.0, -4.5e-5, 3.0e-5),
                SeriesAnchor(40.0, -5.4e-5, 1.5e-5),
                SeriesAnchor(60.0, -5.6e-5, 1.5e-5),
            ),
        ),
    ),
    path_anchors=(
        PathAnchor(0.0, 430.0, 5860.0, 180.0),
        PathAnchor(5.0, -120.0, 5200.0, 600.0),
        PathAnchor(10.0, 0.0, 6200.0, 550.0),
        PathAnchor(20.0, 0.0, 0.0, 6000.0),
        PathAnchor(30.0, 0.0, 0.0, 2000.0),
        PathAnchor(40.0, 0.0, 250.0, 180.0),
        PathAnchor(60.0, 0.0, 0.0, 60.0),
    ),
    statistic_targets=(
        StatisticTarget("dy_peak", "dy", "max", None, None, 7000.0, 550.0),
        StatisticTarget("dy_peak_time_h", "dy", "argmax_time", None, None, 8.0, 3.0),
        StatisticTarget("cb_chaser_min_early", "cb_chaser", "min", 0.0, 8.0, None, None, upper=0.015),
        StatisticTarget("cb_chaser_max_early", "cb_chaser", "max", 0.0, 15.0, None, None, lower=0.175),
        StatisticTarget("theta1_2_near_zero", "theta1_2", "max_abs", None, None, None, None, upper=1.5e-5),
        StatisticTarget("theta1_3_near_zero", "theta1_3", "max_abs", None, None, None, None, upper=1.5e-5),
        StatisticTarget("theta2_2_near_zero", "theta2_2_display", "max_abs", None, None, None, None, upper=4.0e-5),
        StatisticTarget("theta2_3_near_zero", "theta2_3_display", "max_abs", None, None, None, None, upper=1.5e-5),
    ),
)

FULL_PAPER_TARGETS = PaperFigureTargets(
    series_targets=(
        SeriesTarget(
            metric="dx",
            series_key="dx",
            anchors=(
                SeriesAnchor(0.0, 30.0, 90.0),
                SeriesAnchor(5.0, 20.0, 150.0),
                SeriesAnchor(10.0, 40.0, 160.0),
                SeriesAnchor(20.0, 20.0, 120.0),
                SeriesAnchor(40.0, 0.0, 70.0),
                SeriesAnchor(62.0, 0.0, 30.0),
            ),
        ),
        SeriesTarget(
            metric="dy",
            series_key="dy",
            anchors=(
                SeriesAnchor(0.0, 11100.0, 350.0),
                SeriesAnchor(5.0, 3500.0, 2500.0),
                SeriesAnchor(10.0, 700.0, 1600.0),
                SeriesAnchor(20.0, 80.0, 300.0),
                SeriesAnchor(40.0, 0.0, 120.0),
                SeriesAnchor(62.0, 0.0, 45.0),
            ),
        ),
        SeriesTarget(
            metric="dxdot",
            series_key="dxdot",
            anchors=(
                SeriesAnchor(0.0, 0.22, 0.20),
                SeriesAnchor(5.0, 0.00, 0.18),
                SeriesAnchor(10.0, 0.00, 0.15),
                SeriesAnchor(20.0, 0.00, 0.10),
                SeriesAnchor(40.0, 0.00, 0.05),
                SeriesAnchor(62.0, 0.00, 0.02),
            ),
        ),
        SeriesTarget(
            metric="dydot",
            series_key="dydot",
            anchors=(
                SeriesAnchor(0.0, -0.18, 0.16),
                SeriesAnchor(5.0, -0.50, 0.30),
                SeriesAnchor(10.0, -0.10, 0.25),
                SeriesAnchor(20.0, 0.00, 0.10),
                SeriesAnchor(40.0, 0.00, 0.05),
                SeriesAnchor(62.0, 0.00, 0.02),
            ),
        ),
        SeriesTarget(
            metric="cb_chaser",
            series_key="cb_chaser",
            anchors=(
                SeriesAnchor(0.0, 0.183, 0.010),
                SeriesAnchor(5.0, 0.120, 0.040),
                SeriesAnchor(10.0, 0.135, 0.030),
                SeriesAnchor(20.0, 0.147, 0.015),
                SeriesAnchor(40.0, 0.147, 0.010),
                SeriesAnchor(62.0, 0.147, 0.006),
            ),
        ),
        SeriesTarget(
            metric="theta1_1",
            series_key="theta1_1",
            anchors=(
                SeriesAnchor(0.0, 3.80e-4, 2.0e-5),
                SeriesAnchor(10.0, 3.75e-4, 3.5e-5),
                SeriesAnchor(20.0, 3.85e-4, 3.5e-5),
                SeriesAnchor(40.0, 3.85e-4, 3.5e-5),
                SeriesAnchor(62.0, 3.85e-4, 3.5e-5),
            ),
        ),
        SeriesTarget(
            metric="theta2_1_display",
            series_key="theta2_1_display",
            anchors=(
                SeriesAnchor(0.0, 3.5e-5, 2.0e-5),
                SeriesAnchor(10.0, -3.0e-5, 3.0e-5),
                SeriesAnchor(20.0, -4.7e-5, 2.0e-5),
                SeriesAnchor(40.0, -5.5e-5, 1.5e-5),
                SeriesAnchor(62.0, -5.7e-5, 1.5e-5),
            ),
        ),
    ),
    path_anchors=(
        PathAnchor(0.0, 30.0, 11100.0, 220.0),
        PathAnchor(5.0, 0.0, 0.0, 6000.0),
        PathAnchor(10.0, 0.0, 0.0, 5500.0),
        PathAnchor(20.0, 0.0, 0.0, 800.0),
        PathAnchor(30.0, 0.0, 150.0, 180.0),
        PathAnchor(40.0, 0.0, 30.0, 90.0),
        PathAnchor(62.0, 0.0, 0.0, 60.0),
    ),
    statistic_targets=(
        StatisticTarget("dy_peak", "dy", "max", None, None, 11100.0, 500.0),
        StatisticTarget("dy_peak_time_h", "dy", "argmax_time", None, None, 0.0, 2.0),
        StatisticTarget("cb_chaser_min_early", "cb_chaser", "min", 0.0, 15.0, None, None, upper=0.105),
        StatisticTarget("cb_chaser_max_early", "cb_chaser", "max", 0.0, 5.0, None, None, lower=0.175),
        StatisticTarget("theta1_2_near_zero", "theta1_2", "max_abs", None, None, None, None, upper=1.5e-5),
        StatisticTarget("theta1_3_near_zero", "theta1_3", "max_abs", None, None, None, None, upper=1.5e-5),
        StatisticTarget("theta2_2_near_zero", "theta2_2_display", "max_abs", None, None, None, None, upper=1.5e-5),
        StatisticTarget("theta2_3_near_zero", "theta2_3_display", "max_abs", None, None, None, None, upper=1.5e-5),
        StatisticTarget("completion_time_h", "dy", "completion_time", None, None, 42.6, 8.0),
    ),
)


def summarize_result(result: SimulationResult) -> ResultSummary:
    t_hr = result.time_s / 3600.0
    dy = result.x_history[:, 2]
    peak_idx = int(np.argmax(dy))
    path_turns = _count_turning_points(result.orbit_path_xy_m[:, 0])
    completion_time_h = _completion_time_hours(result.x_history, t_hr)
    return ResultSummary(
        initial_x_m=float(result.x_history[0, 0]),
        initial_y_m=float(result.x_history[0, 2]),
        initial_xdot_m_s=float(result.x_history[0, 1]),
        initial_ydot_m_s=float(result.x_history[0, 3]),
        dy_peak_m=float(dy[peak_idx]),
        dy_peak_time_h=float(t_hr[peak_idx]),
        cb_initial=float(result.cb_chaser_history[0]),
        cb_min=float(np.min(result.cb_chaser_history)),
        cb_max=float(np.max(result.cb_chaser_history)),
        cb_final=float(result.cb_chaser_history[-1]),
        theta1_1_final=float(result.theta1_hat_history[-1, 0]),
        theta2_1_final=float((-result.theta2_hat_history[:, 0])[-1]),
        path_turns=path_turns,
        completion_time_h=completion_time_h,
        final_norm_m=float(np.linalg.norm(result.x_history[-1, [0, 2]])),
    )


def compare_result_to_targets(name: str, result: SimulationResult, targets: PaperFigureTargets) -> RegressionReport:
    summary = summarize_result(result)
    checks: list[RegressionCheck] = []
    for target in targets.series_targets:
        series = _series_from_result(result, target.series_key)
        for anchor in target.anchors:
            actual = _interpolate(result.time_s / 3600.0, series, anchor.time_h)
            checks.append(_check(f"{target.metric}@{anchor.time_h:.1f}h", actual, anchor.target, anchor.tolerance))
    for anchor in targets.path_anchors:
        x_actual = _interpolate(result.time_s / 3600.0, result.orbit_path_xy_m[:, 0], anchor.time_h)
        y_actual = _interpolate(result.time_s / 3600.0, result.orbit_path_xy_m[:, 1], anchor.time_h)
        radial_error = float(np.hypot(x_actual - anchor.x_target_m, y_actual - anchor.y_target_m))
        checks.append(
            _check(
                f"path@{anchor.time_h:.1f}h",
                radial_error,
                0.0,
                anchor.tolerance_m,
            )
        )
    for target in targets.statistic_targets:
        checks.append(_statistic_check(result, target))
    return RegressionReport(name=name, summary=summary, checks=tuple(checks))


def run_paper_regression(config: ReferenceConfig, require_exact_atmosphere: bool = True) -> tuple[RegressionReport, RegressionReport]:
    ss_result = simulate_ss(config)
    full_result = simulate_full(config, require_exact_atmosphere=require_exact_atmosphere)
    ss_report = compare_result_to_targets("ss", ss_result, SS_PAPER_TARGETS)
    full_report = compare_result_to_targets("full", full_result, FULL_PAPER_TARGETS)
    return ss_report, full_report


def report_score(report: RegressionReport) -> float:
    score = 0.0
    for check in report.checks:
        normalized = check.normalized_error
        score += normalized**2
        if not check.passed:
            score += 4.0 + normalized
    if report.summary.completion_time_h is None:
        score += 100.0
    return score


def format_report(report: RegressionReport) -> str:
    lines = [f"[{report.name}] passed={report.passed}"]
    summary = report.summary
    lines.append(
        "summary: "
        f"init=({summary.initial_x_m:.3f}, {summary.initial_y_m:.3f}, "
        f"{summary.initial_xdot_m_s:.6f}, {summary.initial_ydot_m_s:.6f}) "
        f"dy_peak={summary.dy_peak_m:.3f}@{summary.dy_peak_time_h:.3f}h "
        f"cb=[{summary.cb_min:.6f}, {summary.cb_max:.6f}] final_cb={summary.cb_final:.6f} "
        f"theta1f={summary.theta1_1_final:.6e} theta2f={summary.theta2_1_final:.6e} "
        f"turns={summary.path_turns} completion={summary.completion_time_h} "
        f"score={report_score(report):.4f}"
    )
    for check in report.checks:
        status = "PASS" if check.passed else "FAIL"
        delta = "None" if check.delta is None else f"{check.delta:.6g}"
        tol = "None" if check.tolerance is None else f"{check.tolerance:.6g}"
        lines.append(
            f"  {status} {check.metric}: actual={check.actual} target={check.target} delta={delta} tol={tol} note={check.note}"
        )
    return "\n".join(lines)


def save_report_text(path: Path, reports: tuple[RegressionReport, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n".join(format_report(report) for report in reports)
    path.write_text(text, encoding="utf-8")


def _series_from_result(result: SimulationResult, series_key: str) -> np.ndarray:
    if series_key == "dx":
        return result.x_history[:, 0]
    if series_key == "dxdot":
        return result.x_history[:, 1]
    if series_key == "dy":
        return result.x_history[:, 2]
    if series_key == "dydot":
        return result.x_history[:, 3]
    if series_key == "cb_chaser":
        return result.cb_chaser_history
    if series_key == "theta1_1":
        return result.theta1_hat_history[:, 0]
    if series_key == "theta1_2":
        return result.theta1_hat_history[:, 1]
    if series_key == "theta1_3":
        return result.theta1_hat_history[:, 2]
    if series_key == "theta2_1_display":
        return -result.theta2_hat_history[:, 0]
    if series_key == "theta2_2_display":
        return -result.theta2_hat_history[:, 1]
    if series_key == "theta2_3_display":
        return -result.theta2_hat_history[:, 2]
    raise KeyError(f"Unsupported series key: {series_key}")


def _interpolate(t_h: np.ndarray, series: np.ndarray, time_h: float) -> float:
    return float(np.interp(time_h, t_h, series))


def _statistic_check(result: SimulationResult, target: StatisticTarget) -> RegressionCheck:
    t_h = result.time_s / 3600.0
    series = _series_from_result(result, target.series_key)
    if target.reducer == "completion_time":
        actual = _completion_time_hours(result.x_history, t_h)
        if actual is None or target.target is None or target.tolerance is None:
            return RegressionCheck(target.metric, actual, target.target, None, target.tolerance, False, "completion not reached")
        return _check(target.metric, float(actual), target.target, target.tolerance)

    window = np.ones_like(t_h, dtype=bool)
    if target.start_h is not None:
        window &= t_h >= target.start_h
    if target.end_h is not None:
        window &= t_h <= target.end_h
    subset = series[window]
    subset_t = t_h[window]
    if subset.size == 0:
        return RegressionCheck(target.metric, None, target.target, None, target.tolerance, False, "empty window")

    if target.reducer == "max":
        actual = float(np.max(subset))
    elif target.reducer == "min":
        actual = float(np.min(subset))
    elif target.reducer == "max_abs":
        actual = float(np.max(np.abs(subset)))
    elif target.reducer == "argmax_time":
        actual = float(subset_t[int(np.argmax(subset))])
    else:
        raise ValueError(f"Unsupported reducer: {target.reducer}")

    if target.target is not None and target.tolerance is not None:
        return _check(target.metric, actual, target.target, target.tolerance)
    if target.lower is not None and actual < target.lower:
        return RegressionCheck(target.metric, actual, target.lower, actual - target.lower, target.lower, False, f"actual >= {target.lower}")
    if target.upper is not None and actual > target.upper:
        return RegressionCheck(target.metric, actual, target.upper, actual - target.upper, target.upper, False, f"actual <= {target.upper}")
    bound = target.lower if target.lower is not None else target.upper
    note = f"actual >= {target.lower}" if target.lower is not None else f"actual <= {target.upper}"
    return RegressionCheck(target.metric, actual, bound, 0.0 if bound is not None else None, bound, True, note)


def _count_turning_points(signal: np.ndarray) -> int:
    if signal.size < 3:
        return 0
    first_derivative = np.diff(signal)
    threshold = max(1.0, 0.01 * np.max(np.abs(first_derivative)))
    signs = np.sign(np.where(np.abs(first_derivative) < threshold, 0.0, first_derivative))
    filtered = signs[signs != 0.0]
    if filtered.size < 2:
        return 0
    return int(np.sum(filtered[1:] * filtered[:-1] < 0))


def _completion_time_hours(x_history: np.ndarray, t_hours: np.ndarray) -> float | None:
    norms = np.linalg.norm(x_history[:, [0, 2]], axis=1)
    below = norms < 10.0
    for idx in range(below.size):
        if below[idx] and np.all(below[idx:]):
            return float(t_hours[idx])
    return None


def _check(metric: str, actual: float, target: float, tolerance: float) -> RegressionCheck:
    delta = float(actual - target)
    return RegressionCheck(
        metric=metric,
        actual=actual,
        target=target,
        delta=delta,
        tolerance=tolerance,
        passed=abs(delta) <= tolerance,
        note=f"|delta| <= {tolerance}",
    )
