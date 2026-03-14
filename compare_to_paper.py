from __future__ import annotations

import argparse
from pathlib import Path

from astro21_sim.constants import build_reference_config
from astro21_sim.regression import format_report, run_paper_regression, save_report_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current astro21 outputs against paper-derived regression targets.")
    parser.add_argument(
        "--output",
        default="outputs/paper_regression_report.txt",
        help="Path where the textual regression report is written.",
    )
    parser.add_argument(
        "--allow-fallback-atmosphere",
        action="store_true",
        help="Allow the full simulation to run without an exact NRLMSISE-family backend.",
    )
    args = parser.parse_args()

    config = build_reference_config()
    reports = run_paper_regression(config, require_exact_atmosphere=not args.allow_fallback_atmosphere)
    for report in reports:
        print(format_report(report))
        print()
    save_report_text(Path(args.output), reports)


if __name__ == "__main__":
    main()
