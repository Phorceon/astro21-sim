from __future__ import annotations

import os
import queue
import threading
import traceback
from dataclasses import replace
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parent
MPL_DIR = WORKSPACE_ROOT / ".mplconfig"
XDG_CACHE_DIR = WORKSPACE_ROOT / ".cache"
MPL_DIR.mkdir(exist_ok=True)
XDG_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLBACKEND", "TkAgg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from astro21_sim.constants import (
    AtmosphereConfig,
    ControllerConfig,
    DensityFitConfig,
    OrbitElements,
    ReferenceConfig,
    SimulationCase,
    SpacecraftConfig,
    TargetPerturbation,
    build_reference_config,
)
from astro21_sim.plots import (
    make_control_figure,
    make_estimates_figure,
    make_relative_states_figure,
)
from astro21_sim.simulations import simulate_full, simulate_ss


class ScrollableFrame(ttk.Frame):
    def __init__(self, master: tk.Misc, **kwargs):
        super().__init__(master, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure(window, width=event.width),
        )
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")


class Astro21Gui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("astro21 Simulation GUI")
        self.root.geometry("1600x980")
        self.default_config = build_reference_config()
        self.vars: dict[str, tk.Variable] = {}
        self.figure_canvases: dict[str, tuple[FigureCanvasTkAgg, NavigationToolbar2Tk]] = {}
        self.result_queue: queue.Queue = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.log_text: tk.Text | None = None
        self.status_var = tk.StringVar(value="Paper defaults loaded.")
        self.run_mode_var = tk.StringVar(value="both")
        self.exact_atmosphere_var = tk.BooleanVar(value=True)
        self.input_channel_sign_var = tk.StringVar(value="+1")

        self._build_layout()
        self._load_defaults()
        self.root.after(200, self._poll_result_queue)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ScrollableFrame(self.root)
        controls.grid(row=0, column=0, sticky="nsew")
        controls.configure(width=460)

        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_controls(controls.inner)
        self._build_results(right)

    def _section(self, master: ttk.Frame, title: str) -> ttk.LabelFrame:
        frame = ttk.LabelFrame(master, text=title, padding=(10, 8))
        frame.pack(fill="x", padx=8, pady=6)
        frame.columnconfigure(1, weight=1)
        return frame

    def _add_entry(self, master: ttk.LabelFrame, row: int, label: str, key: str, width: int = 16) -> None:
        ttk.Label(master, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        var = tk.StringVar()
        self.vars[key] = var
        ttk.Entry(master, textvariable=var, width=width).grid(row=row, column=1, sticky="ew", pady=2)

    def _build_controls(self, master: ttk.Frame) -> None:
        action_frame = self._section(master, "Actions")
        ttk.Label(action_frame, text="Simulation mode").grid(row=0, column=0, sticky="w")
        mode_box = ttk.Combobox(
            action_frame,
            textvariable=self.run_mode_var,
            values=("both", "ss", "full"),
            state="readonly",
            width=12,
        )
        mode_box.grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(
            action_frame,
            text="Require exact NRLMSISE-family atmosphere",
            variable=self.exact_atmosphere_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 2))
        ttk.Label(action_frame, text="Input channel sign").grid(row=2, column=0, sticky="w")
        sign_box = ttk.Combobox(
            action_frame,
            textvariable=self.input_channel_sign_var,
            values=("+1", "-1"),
            state="readonly",
            width=12,
        )
        sign_box.grid(row=2, column=1, sticky="w")
        buttons = ttk.Frame(action_frame)
        buttons.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        buttons.columnconfigure((0, 1, 2), weight=1)
        ttk.Button(buttons, text="Run Simulation", command=self._start_run).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(buttons, text="Reset Defaults", command=self._load_defaults).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(buttons, text="Save Figures", command=self._save_figures).grid(row=0, column=2, sticky="ew", padx=2)

        orbit = self._section(master, "Chaser Orbit")
        self._add_entry(orbit, 0, "a_c [km]", "orbit.a_km")
        self._add_entry(orbit, 1, "e_c", "orbit.e")
        self._add_entry(orbit, 2, "i_c [deg]", "orbit.i_deg")
        self._add_entry(orbit, 3, "RAAN_c [deg]", "orbit.raan_deg")
        self._add_entry(orbit, 4, "omega_c [deg]", "orbit.argp_deg")
        self._add_entry(orbit, 5, "nu_c [deg]", "orbit.nu_deg")

        spacecraft = self._section(master, "Spacecraft / Drag")
        self._add_entry(spacecraft, 0, "C_D", "sc.cd")
        self._add_entry(spacecraft, 1, "m_c [kg]", "sc.mc")
        self._add_entry(spacecraft, 2, "m_t [kg]", "sc.mt")
        self._add_entry(spacecraft, 3, "S_c,min [m^2]", "sc.sc_min")
        self._add_entry(spacecraft, 4, "S_c,max [m^2]", "sc.sc_max")
        self._add_entry(spacecraft, 5, "S_t [m^2]", "sc.st")

        controller = self._section(master, "Controller")
        self._add_entry(controller, 0, "Q11", "ctrl.q11")
        self._add_entry(controller, 1, "Q22", "ctrl.q22")
        self._add_entry(controller, 2, "Q33", "ctrl.q33")
        self._add_entry(controller, 3, "Q44", "ctrl.q44")
        self._add_entry(controller, 4, "R", "ctrl.r")
        self._add_entry(controller, 5, "Gamma1 diag", "ctrl.gamma1")
        self._add_entry(controller, 6, "Gamma2 diag", "ctrl.gamma2")
        self._add_entry(controller, 7, "min theta1 basis", "ctrl.min_theta1")

        theta_init = self._section(master, "Adaptive Initial Estimates")
        for idx in range(3):
            self._add_entry(theta_init, idx, f"Theta1_hat0[{idx}]", f"theta1_hat0.{idx}")
        for idx in range(3):
            self._add_entry(theta_init, idx + 3, f"Theta2_hat0[{idx}]", f"theta2_hat0.{idx}")

        theta_bounds = self._section(master, "Adaptive Bounds")
        row = 0
        for idx in range(3):
            self._add_entry(theta_bounds, row, f"Theta1 lower[{idx}]", f"theta1_lower.{idx}")
            row += 1
        for idx in range(3):
            self._add_entry(theta_bounds, row, f"Theta1 upper[{idx}]", f"theta1_upper.{idx}")
            row += 1
        for idx in range(3):
            self._add_entry(theta_bounds, row, f"Theta2 lower[{idx}]", f"theta2_lower.{idx}")
            row += 1
        for idx in range(3):
            self._add_entry(theta_bounds, row, f"Theta2 upper[{idx}]", f"theta2_upper.{idx}")
            row += 1

        density = self._section(master, "SS Density Fit")
        self._add_entry(density, 0, "D1", "density.d1")
        self._add_entry(density, 1, "D2", "density.d2")
        self._add_entry(density, 2, "D3", "density.d3")

        ss_case = self._section(master, "SS Scenario")
        self._add_entry(ss_case, 0, "delta a_t [m]", "ss.delta_a")
        self._add_entry(ss_case, 1, "e_t", "ss.e")
        self._add_entry(ss_case, 2, "delta nu_t [deg]", "ss.delta_nu_deg")
        self._add_entry(ss_case, 3, "duration [h]", "ss.duration_h")
        self._add_entry(ss_case, 4, "sample step [s]", "ss.sample_step_s")

        full_case = self._section(master, "Full Scenario")
        self._add_entry(full_case, 0, "delta a_t [m]", "full.delta_a")
        self._add_entry(full_case, 1, "e_t", "full.e")
        self._add_entry(full_case, 2, "delta nu_t [deg]", "full.delta_nu_deg")
        self._add_entry(full_case, 3, "duration [h]", "full.duration_h")
        self._add_entry(full_case, 4, "sample step [s]", "full.sample_step_s")
        self._add_entry(full_case, 5, "target tumble [RPM]", "full.tumble_rpm")
        self._add_entry(full_case, 6, "target tumble frac", "full.tumble_frac")

        atmosphere = self._section(master, "Atmosphere / Solver")
        self._add_entry(atmosphere, 0, "F10.7", "atm.f107")
        self._add_entry(atmosphere, 1, "F10.7A", "atm.f107a")
        self._add_entry(atmosphere, 2, "Ap", "atm.ap")
        self._add_entry(atmosphere, 3, "fallback H [m]", "atm.scale_height")
        self._add_entry(atmosphere, 4, "solver rtol", "solver.rtol")
        self._add_entry(atmosphere, 5, "solver atol", "solver.atol")

        derived = self._section(master, "Derived")
        ttk.Label(derived, textvariable=self.status_var, wraplength=390, justify="left").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

    def _build_results(self, master: ttk.Frame) -> None:
        top = ttk.Frame(master, padding=(10, 8))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(0, weight=1)
        ttk.Label(top, text="Results", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")

        notebook = ttk.Notebook(master)
        notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.plot_tabs: dict[str, ttk.Frame] = {}
        for key, title in [
            ("fig3", "SS Relative States"),
            ("fig4", "SS Control"),
            ("fig5", "SS Estimates"),
            ("fig6", "Full Relative States"),
            ("fig7", "Full Control"),
            ("fig8", "Full Estimates"),
            ("log", "Log"),
        ]:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=title)
            self.plot_tabs[key] = frame
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(self.plot_tabs["log"], wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        self.log_text.insert("end", "Ready.\n")
        self.log_text.configure(state="disabled")

    def _log(self, text: str) -> None:
        if self.log_text is None:
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text.rstrip() + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_var(self, key: str, value: float) -> None:
        self.vars[key].set(str(value))

    def _load_defaults(self) -> None:
        cfg = self.default_config
        self._set_var("orbit.a_km", cfg.chaser_orbit.semi_major_axis_m / 1e3)
        self._set_var("orbit.e", cfg.chaser_orbit.eccentricity)
        self._set_var("orbit.i_deg", np.degrees(cfg.chaser_orbit.inclination_rad))
        self._set_var("orbit.raan_deg", np.degrees(cfg.chaser_orbit.raan_rad))
        self._set_var("orbit.argp_deg", np.degrees(cfg.chaser_orbit.arg_perigee_rad))
        self._set_var("orbit.nu_deg", np.degrees(cfg.chaser_orbit.true_anomaly_rad))
        self._set_var("sc.cd", cfg.spacecraft.drag_coefficient)
        self._set_var("sc.mc", cfg.spacecraft.chaser_mass_kg)
        self._set_var("sc.mt", cfg.spacecraft.target_mass_kg)
        self._set_var("sc.sc_min", cfg.spacecraft.chaser_area_min_m2)
        self._set_var("sc.sc_max", cfg.spacecraft.chaser_area_max_m2)
        self._set_var("sc.st", cfg.spacecraft.target_area_m2)
        diag = np.diag(cfg.controller.q_matrix)
        self._set_var("ctrl.q11", diag[0])
        self._set_var("ctrl.q22", diag[1])
        self._set_var("ctrl.q33", diag[2])
        self._set_var("ctrl.q44", diag[3])
        self._set_var("ctrl.r", cfg.controller.r_scalar)
        self._set_var("ctrl.gamma1", cfg.controller.gamma1[0, 0])
        self._set_var("ctrl.gamma2", cfg.controller.gamma2[0, 0])
        self._set_var("ctrl.min_theta1", cfg.controller.min_theta1_basis_value)
        for idx, value in enumerate(cfg.controller.theta1_hat0):
            self._set_var(f"theta1_hat0.{idx}", value)
        for idx, value in enumerate(cfg.controller.theta2_hat0):
            self._set_var(f"theta2_hat0.{idx}", value)
        for idx, value in enumerate(cfg.controller.theta1_lower):
            self._set_var(f"theta1_lower.{idx}", value)
        for idx, value in enumerate(cfg.controller.theta1_upper):
            self._set_var(f"theta1_upper.{idx}", value)
        for idx, value in enumerate(cfg.controller.theta2_lower):
            self._set_var(f"theta2_lower.{idx}", value)
        for idx, value in enumerate(cfg.controller.theta2_upper):
            self._set_var(f"theta2_upper.{idx}", value)
        self._set_var("density.d1", cfg.density_fit.d1)
        self._set_var("density.d2", cfg.density_fit.d2)
        self._set_var("density.d3", cfg.density_fit.d3)
        self._set_var("ss.delta_a", cfg.ss_case.target_perturbation.delta_a_m)
        self._set_var("ss.e", cfg.ss_case.target_perturbation.eccentricity)
        self._set_var("ss.delta_nu_deg", np.degrees(cfg.ss_case.target_perturbation.delta_true_anomaly_rad))
        self._set_var("ss.duration_h", cfg.ss_case.duration_hours)
        self._set_var("ss.sample_step_s", cfg.ss_case.sample_step_s)
        self._set_var("full.delta_a", cfg.full_case.target_perturbation.delta_a_m)
        self._set_var("full.e", cfg.full_case.target_perturbation.eccentricity)
        self._set_var("full.delta_nu_deg", np.degrees(cfg.full_case.target_perturbation.delta_true_anomaly_rad))
        self._set_var("full.duration_h", cfg.full_case.duration_hours)
        self._set_var("full.sample_step_s", cfg.full_case.sample_step_s)
        self._set_var("full.tumble_rpm", cfg.full_target_tumble_rpm)
        self._set_var("full.tumble_frac", cfg.full_target_tumble_fraction)
        self._set_var("atm.f107", cfg.atmosphere.f107)
        self._set_var("atm.f107a", cfg.atmosphere.f107a)
        self._set_var("atm.ap", cfg.atmosphere.ap)
        self._set_var("atm.scale_height", cfg.atmosphere.fallback_scale_height_m)
        self._set_var("solver.rtol", cfg.solver_rtol)
        self._set_var("solver.atol", cfg.solver_atol)
        self.run_mode_var.set("both")
        self.exact_atmosphere_var.set(True)
        self.input_channel_sign_var.set("+1")
        self._set_status_from_config(cfg, prefix="Paper defaults loaded.")

    def _float(self, key: str) -> float:
        return float(self.vars[key].get().strip())

    def _build_config_from_form(self) -> ReferenceConfig:
        base = self.default_config
        orbit = OrbitElements(
            semi_major_axis_m=self._float("orbit.a_km") * 1e3,
            eccentricity=self._float("orbit.e"),
            inclination_rad=np.radians(self._float("orbit.i_deg")),
            raan_rad=np.radians(self._float("orbit.raan_deg")),
            arg_perigee_rad=np.radians(self._float("orbit.argp_deg")),
            true_anomaly_rad=np.radians(self._float("orbit.nu_deg")),
        )
        spacecraft = SpacecraftConfig(
            drag_coefficient=self._float("sc.cd"),
            chaser_mass_kg=self._float("sc.mc"),
            target_mass_kg=self._float("sc.mt"),
            chaser_area_min_m2=self._float("sc.sc_min"),
            chaser_area_max_m2=self._float("sc.sc_max"),
            target_area_m2=self._float("sc.st"),
        )
        controller = ControllerConfig(
            q_matrix=np.diag(
                [
                    self._float("ctrl.q11"),
                    self._float("ctrl.q22"),
                    self._float("ctrl.q33"),
                    self._float("ctrl.q44"),
                ]
            ),
            r_scalar=self._float("ctrl.r"),
            gamma1=np.eye(3) * self._float("ctrl.gamma1"),
            gamma2=np.eye(3) * self._float("ctrl.gamma2"),
            theta1_hat0=np.array([self._float(f"theta1_hat0.{idx}") for idx in range(3)], dtype=float),
            theta2_hat0=np.array([self._float(f"theta2_hat0.{idx}") for idx in range(3)], dtype=float),
            theta1_lower=np.array([self._float(f"theta1_lower.{idx}") for idx in range(3)], dtype=float),
            theta1_upper=np.array([self._float(f"theta1_upper.{idx}") for idx in range(3)], dtype=float),
            theta2_lower=np.array([self._float(f"theta2_lower.{idx}") for idx in range(3)], dtype=float),
            theta2_upper=np.array([self._float(f"theta2_upper.{idx}") for idx in range(3)], dtype=float),
            min_theta1_basis_value=self._float("ctrl.min_theta1"),
        )
        density_fit = DensityFitConfig(
            d1=self._float("density.d1"),
            d2=self._float("density.d2"),
            d3=self._float("density.d3"),
        )
        ss_case = SimulationCase(
            name="ss_gui",
            target_perturbation=TargetPerturbation(
                delta_a_m=self._float("ss.delta_a"),
                eccentricity=self._float("ss.e"),
                delta_true_anomaly_rad=np.radians(self._float("ss.delta_nu_deg")),
            ),
            duration_hours=self._float("ss.duration_h"),
            sample_step_s=self._float("ss.sample_step_s"),
        )
        full_case = SimulationCase(
            name="full_gui",
            target_perturbation=TargetPerturbation(
                delta_a_m=self._float("full.delta_a"),
                eccentricity=self._float("full.e"),
                delta_true_anomaly_rad=np.radians(self._float("full.delta_nu_deg")),
            ),
            duration_hours=self._float("full.duration_h"),
            sample_step_s=self._float("full.sample_step_s"),
        )
        atmosphere = AtmosphereConfig(
            epoch_utc=base.atmosphere.epoch_utc,
            f107=self._float("atm.f107"),
            f107a=self._float("atm.f107a"),
            ap=self._float("atm.ap"),
            fallback_scale_height_m=self._float("atm.scale_height"),
        )
        config = replace(
            base,
            chaser_orbit=orbit,
            spacecraft=spacecraft,
            controller=controller,
            density_fit=density_fit,
            ss_case=ss_case,
            full_case=full_case,
            atmosphere=atmosphere,
            input_channel_sign=float(self.input_channel_sign_var.get()),
            full_target_tumble_rpm=self._float("full.tumble_rpm"),
            full_target_tumble_fraction=self._float("full.tumble_frac"),
            solver_rtol=self._float("solver.rtol"),
            solver_atol=self._float("solver.atol"),
        )
        self._validate_config(config)
        return config

    def _validate_config(self, config: ReferenceConfig) -> None:
        if config.spacecraft.chaser_area_min_m2 <= 0.0 or config.spacecraft.chaser_area_max_m2 <= 0.0:
            raise ValueError("Chaser areas must be positive.")
        if config.spacecraft.chaser_area_min_m2 > config.spacecraft.chaser_area_max_m2:
            raise ValueError("S_c,min must be less than or equal to S_c,max.")
        if config.ss_case.duration_hours <= 0.0 or config.full_case.duration_hours <= 0.0:
            raise ValueError("Simulation durations must be positive.")
        if config.ss_case.sample_step_s <= 0.0 or config.full_case.sample_step_s <= 0.0:
            raise ValueError("Sample steps must be positive.")
        if config.solver_rtol <= 0.0 or config.solver_atol <= 0.0:
            raise ValueError("Solver tolerances must be positive.")
        if np.any(config.controller.theta1_lower > config.controller.theta1_upper):
            raise ValueError("Theta1 lower bounds must not exceed Theta1 upper bounds.")
        if np.any(config.controller.theta2_lower > config.controller.theta2_upper):
            raise ValueError("Theta2 lower bounds must not exceed Theta2 upper bounds.")
        if config.full_target_tumble_fraction < 0.0:
            raise ValueError("Target tumble fraction must be non-negative.")

    def _set_status_from_config(self, config: ReferenceConfig, prefix: str) -> None:
        self.status_var.set(
            (
                f"{prefix} "
                f"C_b,t={config.spacecraft.target_ballistic_coefficient:.6f}, "
                f"C_b,c∈[{config.spacecraft.chaser_ballistic_coefficient_min:.6f}, "
                f"{config.spacecraft.chaser_ballistic_coefficient_max:.6f}]"
            )
        )

    def _start_run(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Simulation running", "Wait for the current run to finish.")
            return
        try:
            config = self._build_config_from_form()
        except Exception as exc:
            messagebox.showerror("Invalid parameters", str(exc))
            return
        self._set_status_from_config(config, prefix="Running simulation.")
        self._log("Starting simulation run.")
        mode = self.run_mode_var.get()
        require_exact = self.exact_atmosphere_var.get()
        self.worker_thread = threading.Thread(
            target=self._run_worker,
            args=(config, mode, require_exact),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_worker(self, config: ReferenceConfig, mode: str, require_exact: bool) -> None:
        try:
            payload: dict[str, object] = {"mode": mode}
            if mode in {"both", "ss"}:
                payload["ss"] = simulate_ss(config)
            if mode in {"both", "full"}:
                payload["full"] = simulate_full(config, require_exact_atmosphere=require_exact)
            self.result_queue.put(("success", payload))
        except Exception:
            self.result_queue.put(("error", traceback.format_exc()))

    def _poll_result_queue(self) -> None:
        try:
            status, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(200, self._poll_result_queue)
            return

        if status == "error":
            self.status_var.set("Simulation failed.")
            self._log(payload)
            messagebox.showerror("Simulation failed", str(payload))
        else:
            self._handle_success(payload)
        self.root.after(200, self._poll_result_queue)

    def _handle_success(self, payload: dict[str, object]) -> None:
        mode = str(payload["mode"])
        if "ss" in payload:
            ss_result = payload["ss"]
            self._render_figure("fig3", make_relative_states_figure(ss_result, "Fig. 3 Recreation"))
            self._render_figure("fig4", make_control_figure(ss_result, "Fig. 4 Recreation"))
            self._render_figure("fig5", make_estimates_figure(ss_result, "Fig. 5 Recreation"))
            self._log(
                f"SS complete. atmosphere={ss_result.atmosphere_source}, final_state={np.array2string(ss_result.x_history[-1], precision=4)}"
            )
        if "full" in payload:
            full_result = payload["full"]
            self._render_figure("fig6", make_relative_states_figure(full_result, "Fig. 6 Recreation"))
            self._render_figure("fig7", make_control_figure(full_result, "Fig. 7 Recreation"))
            self._render_figure("fig8", make_estimates_figure(full_result, "Fig. 8 Recreation"))
            self._log(
                f"Full complete. atmosphere={full_result.atmosphere_source}, final_state={np.array2string(full_result.x_history[-1], precision=4)}"
            )
        self.status_var.set(f"Completed {mode} run.")

    def _render_figure(self, tab_key: str, fig) -> None:
        tab = self.plot_tabs[tab_key]
        if tab_key in self.figure_canvases:
            old_canvas, old_toolbar = self.figure_canvases.pop(tab_key)
            old_toolbar.destroy()
            old_canvas.get_tk_widget().destroy()
        canvas = FigureCanvasTkAgg(fig, master=tab)
        toolbar = NavigationToolbar2Tk(canvas, tab, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        canvas.draw()
        self.figure_canvases[tab_key] = (canvas, toolbar)

    def _save_figures(self) -> None:
        output_dir = WORKSPACE_ROOT / "outputs"
        saved = []
        for key, name in [
            ("fig3", "fig3_ss_relative_states_gui.png"),
            ("fig4", "fig4_ss_control_gui.png"),
            ("fig5", "fig5_ss_estimates_gui.png"),
            ("fig6", "fig6_full_relative_states_gui.png"),
            ("fig7", "fig7_full_control_gui.png"),
            ("fig8", "fig8_full_estimates_gui.png"),
        ]:
            if key not in self.figure_canvases:
                continue
            canvas, _toolbar = self.figure_canvases[key]
            path = output_dir / name
            path.parent.mkdir(exist_ok=True)
            canvas.figure.savefig(path, dpi=200)
            saved.append(path.name)
        if saved:
            self._log("Saved figures: " + ", ".join(saved))
            self.status_var.set(f"Saved {len(saved)} figure(s) to outputs/.")
        else:
            self.status_var.set("No figures available to save yet.")


def main() -> None:
    root = tk.Tk()
    app = Astro21Gui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
