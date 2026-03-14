from __future__ import annotations

import argparse
import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
MPL_DIR = WORKSPACE_ROOT / ".mplconfig"
XDG_CACHE_DIR = WORKSPACE_ROOT / ".cache"
MPL_DIR.mkdir(exist_ok=True)
XDG_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

from astro21_sim import build_reference_config, simulate_full, simulate_ss
from astro21_sim.plots import plot_control, plot_estimates, plot_relative_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the astro21 simulation suite.")
    parser.add_argument("--output-dir", default="outputs", help="Directory where PNG plots are written.")
    parser.add_argument(
        "--require-exact-atmosphere",
        action="store_true",
        help="Fail if no NRLMSISE-00 Python backend is installed.",
    )
    parser.add_argument("--skip-full", action="store_true", help="Only run the SS dynamics suite.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    config = build_reference_config()

    ss_result = simulate_ss(config)
    plot_relative_states(ss_result, output_dir, "fig3_ss_relative_states.png", "Fig. 3 Recreation")
    plot_control(ss_result, output_dir, "fig4_ss_control.png", "Fig. 4 Recreation")
    plot_estimates(ss_result, output_dir, "fig5_ss_estimates.png", "Fig. 5 Recreation")

    print("Generated SS suite.")
    print(f"  atmosphere: {ss_result.atmosphere_source}")
    print(f"  final state: {ss_result.x_history[-1]}")

    if args.skip_full:
        return

    full_result = simulate_full(config, require_exact_atmosphere=args.require_exact_atmosphere)
    plot_relative_states(full_result, output_dir, "fig6_full_relative_states.png", "Fig. 6 Recreation")
    plot_control(full_result, output_dir, "fig7_full_control.png", "Fig. 7 Recreation")
    plot_estimates(full_result, output_dir, "fig8_full_estimates.png", "Fig. 8 Recreation")
    print("Generated full suite.")
    print(f"  atmosphere: {full_result.atmosphere_source}")
    print(f"  final state: {full_result.x_history[-1]}")


if __name__ == "__main__":
    main()
