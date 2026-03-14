# astro21-sim

Reproduction scaffold for the adaptive differential-drag rendezvous paper:

- `astro21_sim/` contains the dynamics, controller, atmosphere wrapper, and plotting code.
- `run_suite.py` runs the SS and full-dynamics reference scenarios and writes figure PNGs into `outputs/`.
- `astro21_gui.py` launches a Tk GUI with the paper defaults preloaded so you can change parameters interactively.
- `compare_to_paper.py` runs a paper-regression check against manual targets extracted from the paper screenshots.
- `calibrate_to_paper.py` searches over hidden scenario and adaptive-initialization parameters to reduce the mismatch against the paper targets.

## Usage

```bash
python3 run_suite.py
```

To require an installed NRLMSISE-00 backend for the full-dynamics case:

```bash
python3 run_suite.py --require-exact-atmosphere
```

## GUI

```bash
python3 astro21_gui.py
```

The GUI exposes the main orbit, spacecraft, controller, density-fit, scenario, atmosphere, and solver parameters. The default values are the paper values currently encoded in the simulator.

## Paper Comparison

```bash
python3 compare_to_paper.py
```

This writes a text report to `outputs/paper_regression_report.txt` and prints the current mismatches against the paper-derived regression targets.

## Calibration

```bash
python3 calibrate_to_paper.py --mode ss
python3 calibrate_to_paper.py --mode full
```

This does a bounded search over target perturbations and the leading adaptive initial estimates to find a closer paper-matching reference case.
