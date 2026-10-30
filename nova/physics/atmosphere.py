"""
nova.physics.atmosphere
=======================
Atmospheric state solver for Project NOVA.

Implements the International Standard Atmosphere (ISA) 1976 / US Standard
Atmosphere 1976 with the correct multi-layer temperature profile:

  Layer 0: Troposphere        0 – 11 000 m   L = −6.5  K/km
  Layer 1: Lower stratosphere 11 000 – 20 000 m  isothermal (216.65 K)
  Layer 2: Middle stratosphere 20 000 – 32 000 m  L = +1.0  K/km
  Layer 3: Upper stratosphere  32 000 – 47 000 m  L = +2.8  K/km
  Layer 4: Above 47 000 m     isothermal (270.65 K) + exponential decay

Each layer uses the correct hydrostatic pressure formula for its lapse rate:
  Constant lapse:    P = P_base * (T / T_base)^(g₀ / (-L · R_air))
  Isothermal (L=0):  P = P_base * exp(-g₀ · Δh / (R_air · T))

All base pressures and temperatures are ISA 1976 tabulated values, NOT
derived from lower layers, to prevent error accumulation.

Output quantities (strict SI):
  ρ   [kg m⁻³]   atmospheric density
  T   [K]         static temperature
  P   [Pa]        static pressure
  a   [m s⁻¹]    speed of sound
  μ   [Pa·s]      dynamic viscosity (Sutherland's law)

References
----------
- US Standard Atmosphere, 1976 (NOAA-S/T 76-1562)
- ICAO Doc 7488, Manual of the ICAO Standard Atmosphere, 3rd ed.
- Anderson, "Introduction to Flight", 8th ed., Appendix A
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nova.core.constants import (
    EARTH_T_SL,
    EARTH_P_SL,
    EARTH_RHO_SL,
    EARTH_A_SL,
    EARTH_RADIUS_MEAN,
    R_AIR,
    GAMMA_AIR,
    STD_GRAVITY,
    ISA_TROPOPAUSE_ALT,
    ISA_TROPOPAUSE_TEMP,
)


# ---------------------------------------------------------------------------
# ISA 1976 layer definitions
# ---------------------------------------------------------------------------
# Each tuple: (base_altitude_m, base_temp_K, base_pressure_Pa, lapse_K_per_m)
# Base pressures are ISA 1976 authoritative tabulated values — NOT derived.
# Lapse rate sign convention: negative = cooling with altitude.

_ISA_LAYERS = [
    (    0.0,  288.15, 101_325.0,   -6.5e-3),   # Layer 0: troposphere
    (11_000.0, 216.65,  22_632.1,    0.0   ),   # Layer 1: lower stratosphere (isothermal)
    (20_000.0, 216.65,   5_474.9,   +1.0e-3),   # Layer 2: middle stratosphere
    (32_000.0, 228.65,     868.019, +2.8e-3),   # Layer 3: upper stratosphere
    (47_000.0, 270.65,     110.906,  0.0   ),   # Layer 4: stratopause (isothermal)
]


def _isa_layer_state(altitude: float) -> tuple[float, float]:
    """
    Compute ISA temperature and pressure at the given altitude using the
    appropriate layer formula.

    Parameters
    ----------
    altitude : float
        Geodetic altitude [m]. Extrapolation below 0 m uses Layer 0 formula.

    Returns
    -------
    (T, P) : tuple of float
        Temperature [K] and pressure [Pa].
    """
    # Find the applicable layer (scan from top down)
    h_base, T_base, P_base, L = _ISA_LAYERS[0]
    for (h_b, T_b, P_b, lapse) in reversed(_ISA_LAYERS):
        if altitude >= h_b:
            h_base, T_base, P_base, L = h_b, T_b, P_b, lapse
            break

    dh = altitude - h_base

    if abs(L) < 1.0e-12:
        # Isothermal layer: P = P_base * exp(-g₀·Δh / (R·T))
        T = T_base
        P = P_base * math.exp(-STD_GRAVITY * dh / (R_AIR * T_base))
    else:
        # Linear lapse: T = T_base + L·Δh
        T = T_base + L * dh
        T = max(T, 1.0)   # numerical guard
        exponent = STD_GRAVITY / (-L * R_AIR)
        P = P_base * (T / T_base) ** exponent

    P = max(P, 0.0)
    return T, P


# ---------------------------------------------------------------------------
# Sutherland's law for dynamic viscosity
# ---------------------------------------------------------------------------

_SUTHERLAND_C1: float = 1.458e-6   # kg m⁻¹ s⁻¹ K⁻⁰·⁵
_SUTHERLAND_S:  float = 110.4      # K


def sutherland_viscosity(temperature: float) -> float:
    """
    Dynamic viscosity of air [Pa·s] via Sutherland's law.

    μ = C₁ T^(3/2) / (T + S)

    Valid for 170 K < T < 1900 K.
    """
    T = max(temperature, 1.0)
    return _SUTHERLAND_C1 * (T ** 1.5) / (T + _SUTHERLAND_S)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AtmosphericState:
    """
    Complete thermodynamic state of the atmosphere at a given altitude.

    All fields are in strict SI units.
    """
    altitude:          float   # [m]
    density:           float   # [kg m⁻³]
    temperature:       float   # [K]
    pressure:          float   # [Pa]
    speed_of_sound:    float   # [m s⁻¹]
    dynamic_viscosity: float   # [Pa·s]

    def mach(self, airspeed: float) -> float:
        """Mach number M = v / a for a given airspeed [m s⁻¹]."""
        if self.speed_of_sound < 1.0e-12:
            return 0.0
        return airspeed / self.speed_of_sound

    @property
    def dynamic_pressure_at_speed(self):
        """Return callable q(v) = ½ρv² [Pa] for a given airspeed v [m s⁻¹]."""
        rho = self.density
        return lambda v: 0.5 * rho * v * v


# ---------------------------------------------------------------------------
# Core atmosphere function
# ---------------------------------------------------------------------------

def atmosphere(altitude: float) -> AtmosphericState:
    """
    Compute the ISA 1976 atmospheric state at a given geodetic altitude.

    Parameters
    ----------
    altitude : float
        Geodetic altitude above mean sea level [m].
        Accepts negative values (extrapolation below MSL via Layer 0).

    Returns
    -------
    AtmosphericState
        Frozen snapshot of all atmospheric quantities at this altitude.
    """
    T, P  = _isa_layer_state(altitude)
    rho   = P / (R_AIR * T)
    a     = math.sqrt(GAMMA_AIR * R_AIR * T)
    mu    = sutherland_viscosity(T)

    return AtmosphericState(
        altitude=altitude,
        density=rho,
        temperature=T,
        pressure=P,
        speed_of_sound=a,
        dynamic_viscosity=mu,
    )


# ---------------------------------------------------------------------------
# Vector form — accepts ECI position
# ---------------------------------------------------------------------------

def atmosphere_from_eci(
    position_eci: np.ndarray,
    planet_radius: float,
) -> AtmosphericState:
    """
    Compute atmospheric state from an ECI position vector.

    Uses spherical-Earth approximation: altitude = ‖r_eci‖ − R_planet.

    Parameters
    ----------
    position_eci : ndarray, shape (3,)
        ECI position [m].
    planet_radius : float
        Reference planet radius [m].
    """
    r_mag    = float(np.linalg.norm(position_eci))
    altitude = r_mag - planet_radius
    return atmosphere(altitude)


# ---------------------------------------------------------------------------
# Standalone aerodynamic helpers
# ---------------------------------------------------------------------------

def dynamic_pressure(rho: float, airspeed: float) -> float:
    """q = ½ ρ v²  [Pa]."""
    return 0.5 * rho * airspeed * airspeed


def mach_number(airspeed: float, speed_of_sound: float) -> float:
    """
    M = v / a  (dimensionless).

    Raises
    ------
    ValueError if speed_of_sound ≤ 0.
    """
    if speed_of_sound <= 0.0:
        raise ValueError(
            f"speed_of_sound must be > 0, got {speed_of_sound!r}"
        )
    return airspeed / speed_of_sound


def prandtl_glauert_factor(mach: float) -> float:
    """
    Prandtl-Glauert compressibility correction β = 1 / √(1 − M²).

    Valid for 0 ≤ M < 1.

    Raises
    ------
    ValueError if mach not in [0, 1).
    """
    if not (0.0 <= mach < 1.0):
        raise ValueError(
            f"Prandtl-Glauert correction requires 0 ≤ M < 1, got M={mach:.4f}"
        )
    return 1.0 / math.sqrt(1.0 - mach * mach)