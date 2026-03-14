from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta, timezone
import warnings

import numpy as np

from .constants import DensityFitConfig, EarthConstants, ReferenceConfig
from .orbits import eci_to_geocentric


@dataclass
class AtmosphereSample:
    density_kg_m3: float
    source: str


class SinusoidalDensityModel:
    def __init__(self, density_fit: DensityFitConfig):
        self.density_fit = density_fit

    def density(self, t_seconds: float, orbit_rate_rad_s: float) -> float:
        fit = self.density_fit
        return fit.d1 + fit.d2 * np.sin(orbit_rate_rad_s * t_seconds) + fit.d3 * np.cos(
            orbit_rate_rad_s * t_seconds
        )


class AtmosphereModel:
    def __init__(self, config: ReferenceConfig, require_exact: bool = False):
        self.config = config
        self.require_exact = require_exact
        self._exact_backend = self._load_exact_backend()
        if self._exact_backend is None and require_exact:
            raise RuntimeError(
                "No NRLMSISE-00 Python backend is available. Install one of "
                "`nrlmsise00`, `msise00`, or `pymsis` before running the full simulation."
            )
        if self._exact_backend is None:
            warnings.warn(
                "Falling back to an exponential atmosphere because no NRLMSISE-00 backend is "
                "installed. The full-dynamics runner remains usable, but it will not exactly "
                "match the paper's atmosphere model.",
                RuntimeWarning,
                stacklevel=2,
            )

    def _load_exact_backend(self):
        try:
            import nrlmsise00  # type: ignore

            return ("nrlmsise00", nrlmsise00)
        except ImportError:
            pass
        try:
            import msise00  # type: ignore

            return ("msise00", msise00)
        except ImportError:
            pass
        try:
            import pymsis  # type: ignore

            return ("pymsis", pymsis)
        except ImportError:
            return None

    def density(self, earth: EarthConstants, t_seconds: float, r_eci: np.ndarray) -> AtmosphereSample:
        if self._exact_backend is None:
            lat_rad, lon_rad, altitude_m = eci_to_geocentric(
                earth, self.config.atmosphere.epoch_utc, t_seconds, r_eci
            )
            rho_ref = self.config.density_fit.d1
            h_ref = self.config.chaser_orbit.semi_major_axis_m - earth.radius_m
            density = rho_ref * np.exp(-(altitude_m - h_ref) / self.config.atmosphere.fallback_scale_height_m)
            density *= 1.0 + 0.05 * np.sin(earth.omega_rad_s * t_seconds + lat_rad + lon_rad)
            return AtmosphereSample(float(max(density, 1.0e-15)), "exponential-fallback")

        backend_name, backend = self._exact_backend
        latitude_rad, longitude_rad, altitude_m = eci_to_geocentric(
            earth, self.config.atmosphere.epoch_utc, t_seconds, r_eci
        )
        when = self.config.atmosphere.epoch_utc + timedelta(seconds=float(t_seconds))
        altitude_km = altitude_m / 1_000.0
        latitude_deg = np.rad2deg(latitude_rad)
        longitude_deg = np.rad2deg(longitude_rad)

        if backend_name == "nrlmsise00":
            model = getattr(backend, "msise_model", None)
            if model is None:
                model = getattr(backend, "gtd7_flat", None)
            if model is None:
                raise RuntimeError("Unsupported nrlmsise00 package layout.")
            values = model(
                when,
                altitude_km,
                latitude_deg,
                longitude_deg,
                self.config.atmosphere.f107a,
                self.config.atmosphere.f107,
                self.config.atmosphere.ap,
            )
            density = float(values[0][5]) * 1e3  # g/cm³ → kg/m³
            return AtmosphereSample(density, backend_name)

        if backend_name == "msise00":
            values = backend.run(
                when,
                altitude_km,
                latitude_deg,
                longitude_deg,
                self.config.atmosphere.f107a,
                self.config.atmosphere.f107,
                self.config.atmosphere.ap,
            )
            density = float(values[0][5])
            return AtmosphereSample(density, backend_name)

        when_naive_utc = when.astimezone(timezone.utc).replace(tzinfo=None)
        result = backend.calculate(
            np.array([when_naive_utc], dtype="datetime64[ns]"),
            np.array([longitude_deg]),
            np.array([latitude_deg]),
            np.array([altitude_km]),
            f107s=np.array([self.config.atmosphere.f107]),
            f107as=np.array([self.config.atmosphere.f107a]),
            aps=np.array([self.config.atmosphere.ap]),
        )
        density = float(result.reshape(-1, result.shape[-1])[0, 0])
        return AtmosphereSample(density, backend_name)
