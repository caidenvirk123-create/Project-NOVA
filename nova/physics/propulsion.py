"""
nova.physics.propulsion
=======================
Propulsion model for Project NOVA.

Architecture role — Pipeline Stage 3 (physics engine input) + Stage 9
----------------------------------------------------------------------
Computes engine thrust forces and mass flow rate for one simulation tick.
Also provides the Tsiolkovsky rocket equation for Δv budget validation.

The propulsion model operates as a pure function: given engine configuration,
throttle setting, and current atmospheric conditions, it returns:
  1. Thrust vector in Body frame [N]            → ForceAccumulator
  2. Mass flow rate ṁ [kg s⁻¹]                 → Component update (Stage 9)
  3. Specific impulse Isp [s]                   → Telemetry / avionics HUD

Physics implemented
-------------------

Thrust
~~~~~~
  F_thrust = Isp · g₀ · ṁ                     (effective exhaust thrust)

where the effective Isp is a throttle- and altitude-corrected value:
  Isp_eff = Isp_vac − (P_atm / P_exit) · Isp_correction_factor

For a pressure-thrust correction (nozzle exit pressure not matched):
  F_pressure = (P_exit − P_atm) · A_exit        [N]
  F_total    = F_momentum + F_pressure

Mass flow
~~~~~~~~~
  ṁ = F_thrust / (Isp_vac · g₀)   [kg s⁻¹]

This is the vacuum mass flow; the pipeline uses it as an approximation
at all altitudes (error < 1% for high-expansion-ratio nozzles in the
atmosphere, acceptable for trajectory-level fidelity).

Tsiolkovsky validation
~~~~~~~~~~~~~~~~~~~~~~
Δv = Isp · g₀ · ln(m₀ / m_f)

This is verified in the unit test suite against the integrator's propellant
depletion model to within 1×10⁻⁶ m/s (architecture spec §7).

Gimbal model
~~~~~~~~~~~~
Thrust is assumed to point along the −Z_b direction by default (engine at
base of vehicle), but a gimbal angle (θ_gimbal, φ_gimbal) can deflect it
in the pitch and yaw planes. The gimbal deflection produces a moment arm
torque that is handled by TorqueAccumulator.add_moment_arm().

References
----------
- Sutton & Biblarz, "Rocket Propulsion Elements", 8th ed.
- Turner, "Rocket and Spacecraft Propulsion", 3rd ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nova.core.constants import STD_GRAVITY


# ---------------------------------------------------------------------------
# Engine configuration
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """
    Physical configuration of a single rocket engine.

    Parameters
    ----------
    name : str
        Human-readable engine label (e.g. ``"Merlin-1D"``).
    thrust_vac : float
        Vacuum thrust at full throttle [N].
    isp_vac : float
        Vacuum specific impulse [s].
    isp_sl : float
        Sea-level specific impulse [s]. Used for linear altitude interpolation.
    throttle_min : float
        Minimum throttle setting (0–1) below which the engine is off.
    throttle_max : float
        Maximum throttle setting (0–1). Normally 1.0.
    gimbal_max_rad : float
        Maximum gimbal deflection angle [rad] in pitch and yaw.
    exit_area : float
        Nozzle exit area [m²]. Used for pressure-thrust correction.
    exit_pressure : float
        Nozzle exit static pressure at design point [Pa].
        At vacuum: exit_pressure matches ambient. Provide the design-point
        value; correction is computed from (P_exit − P_atm) · A_exit.
    mount_point_body : ndarray, shape (3,), float64
        Position of engine nozzle exit relative to CoM in Body frame [m].
        Used for moment-arm torque calculation. Typically negative X (aft).
    nominal_direction_body : ndarray, shape (3,), float64
        Unit vector of nominal thrust direction in Body frame.
        Default: [1, 0, 0] (+X forward = engine pushes forward).
        For a tail engine pointing aft: [1, 0, 0] (exhaust exits aft, thrust forward).
    """
    name:                   str
    thrust_vac:             float          # [N]
    isp_vac:                float          # [s]
    isp_sl:                 float          # [s]
    throttle_min:           float = 0.0
    throttle_max:           float = 1.0
    gimbal_max_rad:         float = 0.0    # [rad]
    exit_area:              float = 0.0    # [m²]
    exit_pressure:          float = 0.0    # [Pa]
    mount_point_body:       np.ndarray = field(
        default_factory=lambda: np.array([-5.0, 0.0, 0.0], dtype=np.float64)
    )
    nominal_direction_body: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )

    def __post_init__(self) -> None:
        if self.thrust_vac < 0.0:
            raise ValueError(f"EngineConfig.thrust_vac must be ≥ 0, got {self.thrust_vac!r}")
        if self.isp_vac <= 0.0:
            raise ValueError(f"EngineConfig.isp_vac must be > 0, got {self.isp_vac!r}")
        if self.isp_sl <= 0.0:
            raise ValueError(f"EngineConfig.isp_sl must be > 0, got {self.isp_sl!r}")
        if not (0.0 <= self.throttle_min <= self.throttle_max <= 1.0):
            raise ValueError(
                f"Throttle range invalid: min={self.throttle_min}, max={self.throttle_max}"
            )
        # Coerce array fields
        object.__setattr__(self, "mount_point_body",
                           np.asarray(self.mount_point_body, dtype=np.float64))
        object.__setattr__(self, "nominal_direction_body",
                           np.asarray(self.nominal_direction_body, dtype=np.float64))
        # Normalise direction
        d = self.nominal_direction_body
        d_norm = float(np.linalg.norm(d))
        if d_norm < 1.0e-10:
            raise ValueError("EngineConfig.nominal_direction_body must be non-zero.")
        object.__setattr__(self, "nominal_direction_body", d / d_norm)


# ---------------------------------------------------------------------------
# Per-tick engine state output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropulsionState:
    """
    Propulsion output for one simulation tick.

    Attributes
    ----------
    thrust_body : ndarray, shape (3,), float64
        Net thrust vector in Body frame [N].
    mass_flow_rate : float
        Propellant consumption rate ṁ [kg s⁻¹]. Positive = mass decreasing.
    isp_effective : float
        Effective specific impulse this tick [s].
    throttle : float
        Actual throttle applied (clamped to [throttle_min, throttle_max]).
    gimbal_angle_pitch : float
        Pitch gimbal deflection applied [rad].
    gimbal_angle_yaw : float
        Yaw gimbal deflection applied [rad].
    is_active : bool
        True if the engine produced thrust this tick.
    """
    thrust_body:        np.ndarray
    mass_flow_rate:     float
    isp_effective:      float
    throttle:           float
    gimbal_angle_pitch: float
    gimbal_angle_yaw:   float
    is_active:          bool


# ---------------------------------------------------------------------------
# Gimbal rotation helper
# ---------------------------------------------------------------------------

def _gimbal_direction(
    nominal_dir: np.ndarray,
    pitch_rad:   float,
    yaw_rad:     float,
) -> np.ndarray:
    """
    Rotate the nominal thrust direction by gimbal angles.

    Small-angle approximation is NOT used; full Rodrigues rotation applied.
    Pitch rotates about +Y_body (nose-up); yaw rotates about +Z_body (nose-right).

    Parameters
    ----------
    nominal_dir : ndarray, shape (3,)
        Normalised nominal thrust direction in Body frame.
    pitch_rad : float
        Gimbal pitch deflection [rad]. Positive → thrust vector tilts downward
        (nose-up moment).
    yaw_rad : float
        Gimbal yaw deflection [rad]. Positive → thrust vector tilts right
        (nose-left moment).

    Returns
    -------
    ndarray, shape (3,)
        Normalised gimballed thrust direction.
    """
    # Rotation about Y (pitch): Ry
    cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
    Ry = np.array([
        [ cp, 0.0,  sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0,  cp],
    ], dtype=np.float64)

    # Rotation about Z (yaw): Rz
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    gimballed = Rz @ Ry @ nominal_dir
    norm = float(np.linalg.norm(gimballed))
    if norm < 1.0e-10:
        return nominal_dir.copy()
    return gimballed / norm


# ---------------------------------------------------------------------------
# Effective Isp interpolation (altitude correction)
# ---------------------------------------------------------------------------

def effective_isp(
    config:       EngineConfig,
    atm_pressure: float,
) -> float:
    """
    Compute effective specific impulse at the given atmospheric pressure.

    Linear interpolation between sea-level and vacuum Isp using the
    ratio of ambient pressure to sea-level pressure:
        Isp_eff = Isp_vac − (Isp_vac − Isp_sl) · (P_atm / P_sl)

    where P_sl = 101 325 Pa.

    Parameters
    ----------
    config : EngineConfig
    atm_pressure : float
        Ambient static pressure [Pa]. Pass 0.0 for vacuum.

    Returns
    -------
    float
        Effective Isp [s].
    """
    P_SL  = 101_325.0
    frac  = min(atm_pressure / P_SL, 1.0)   # clamp at 1.0 (below sea level)
    return config.isp_vac - (config.isp_vac - config.isp_sl) * frac


# ---------------------------------------------------------------------------
# Core propulsion computation
# ---------------------------------------------------------------------------

def compute_propulsion(
    config:       EngineConfig,
    throttle:     float,
    atm_pressure: float,
    gimbal_pitch: float = 0.0,
    gimbal_yaw:   float = 0.0,
    propellant_remaining: float = math.inf,
) -> PropulsionState:
    """
    Compute thrust and mass flow for one simulation tick.

    Parameters
    ----------
    config : EngineConfig
        Engine physical configuration.
    throttle : float
        Commanded throttle [0, 1]. Clamped to [throttle_min, throttle_max].
        Values below throttle_min are treated as engine-off (throttle = 0).
    atm_pressure : float
        Ambient static pressure [Pa]. From atmosphere solver (Stage 7).
    gimbal_pitch : float
        Pitch gimbal command [rad]. Clamped to ±gimbal_max_rad.
    gimbal_yaw : float
        Yaw gimbal command [rad]. Clamped to ±gimbal_max_rad.
    propellant_remaining : float
        Remaining propellant mass [kg]. Engine shuts off if ≤ 0.
        Defaults to infinity (unlimited — useful for unit tests).

    Returns
    -------
    PropulsionState
        Complete propulsion output snapshot (frozen).
    """
    _ZERO = np.zeros(3, dtype=np.float64)

    # --- Engine cutoff conditions ---
    if propellant_remaining <= 0.0:
        return PropulsionState(
            thrust_body=_ZERO.copy(), mass_flow_rate=0.0,
            isp_effective=config.isp_vac, throttle=0.0,
            gimbal_angle_pitch=0.0, gimbal_angle_yaw=0.0,
            is_active=False,
        )

    # --- Throttle clamping ---
    if throttle < config.throttle_min:
        actual_throttle = 0.0   # below minimum → engine off
    else:
        actual_throttle = min(max(throttle, config.throttle_min),
                              config.throttle_max)

    if actual_throttle <= 0.0:
        return PropulsionState(
            thrust_body=_ZERO.copy(), mass_flow_rate=0.0,
            isp_effective=config.isp_vac, throttle=0.0,
            gimbal_angle_pitch=0.0, gimbal_angle_yaw=0.0,
            is_active=False,
        )

    # --- Effective Isp at current altitude ---
    isp_eff = effective_isp(config, atm_pressure)

    # --- Thrust magnitude ---
    # Vacuum thrust scaled by throttle; pressure-thrust correction:
    # F_total = throttle * F_vac_momentum + (P_exit − P_atm) * A_exit
    F_momentum = actual_throttle * config.thrust_vac
    F_pressure = (config.exit_pressure - atm_pressure) * config.exit_area
    F_total    = F_momentum + F_pressure
    F_total    = max(F_total, 0.0)   # thrust cannot be negative

    # --- Mass flow rate ---
    # ṁ = F_total / (Isp_eff · g₀)
    mdot = F_total / (isp_eff * STD_GRAVITY)

    # --- Gimbal clamping ---
    gmax = config.gimbal_max_rad
    gp   = max(-gmax, min(gmax, gimbal_pitch)) if gmax > 0.0 else 0.0
    gy   = max(-gmax, min(gmax, gimbal_yaw  )) if gmax > 0.0 else 0.0

    # --- Thrust direction (Body frame) ---
    direction = _gimbal_direction(config.nominal_direction_body, gp, gy)
    thrust_body = F_total * direction

    return PropulsionState(
        thrust_body=thrust_body,
        mass_flow_rate=mdot,
        isp_effective=isp_eff,
        throttle=actual_throttle,
        gimbal_angle_pitch=gp,
        gimbal_angle_yaw=gy,
        is_active=True,
    )


# ---------------------------------------------------------------------------
# Tsiolkovsky rocket equation
# ---------------------------------------------------------------------------

def tsiolkovsky_delta_v(
    isp:            float,
    mass_initial:   float,
    mass_final:     float,
    g0:             float = STD_GRAVITY,
) -> float:
    """
    Tsiolkovsky rocket equation: ideal Δv for a propulsive manoeuvre.

    Δv = Isp · g₀ · ln(m₀ / m_f)

    Parameters
    ----------
    isp : float
        Effective specific impulse [s].
    mass_initial : float
        Vehicle mass at start of burn [kg].
    mass_final : float
        Vehicle mass at end of burn [kg]. Must be < mass_initial.
    g0 : float
        Standard gravity [m s⁻²]. Default = STD_GRAVITY.

    Returns
    -------
    float
        Ideal Δv [m s⁻¹].

    Raises
    ------
    ValueError
        If mass_final ≥ mass_initial or either mass ≤ 0.
    """
    if mass_initial <= 0.0:
        raise ValueError(f"mass_initial must be > 0, got {mass_initial!r}")
    if mass_final <= 0.0:
        raise ValueError(f"mass_final must be > 0, got {mass_final!r}")
    if mass_final >= mass_initial:
        raise ValueError(
            f"mass_final ({mass_final!r}) must be < mass_initial ({mass_initial!r})"
        )
    return isp * g0 * math.log(mass_initial / mass_final)


def tsiolkovsky_final_mass(
    isp:          float,
    delta_v:      float,
    mass_initial: float,
    g0:           float = STD_GRAVITY,
) -> float:
    """
    Inverse Tsiolkovsky: compute final mass given Δv budget.

    m_f = m₀ · exp(−Δv / (Isp · g₀))

    Parameters
    ----------
    isp : float
        Specific impulse [s].
    delta_v : float
        Required Δv [m s⁻¹]. Must be > 0.
    mass_initial : float
        Initial vehicle mass [kg].
    g0 : float
        Standard gravity [m s⁻²].

    Returns
    -------
    float
        Final vehicle mass [kg].
    """
    if delta_v < 0.0:
        raise ValueError(f"delta_v must be ≥ 0, got {delta_v!r}")
    if mass_initial <= 0.0:
        raise ValueError(f"mass_initial must be > 0, got {mass_initial!r}")
    return mass_initial * math.exp(-delta_v / (isp * g0))


def propellant_required(
    isp:          float,
    delta_v:      float,
    mass_payload: float,
    g0:           float = STD_GRAVITY,
) -> float:
    """
    Compute propellant mass required for a given Δv with a given payload.

    m_prop = m_payload · (exp(Δv / (Isp · g₀)) − 1)

    Parameters
    ----------
    isp : float
        Specific impulse [s].
    delta_v : float
        Required Δv [m s⁻¹].
    mass_payload : float
        Dry mass (payload + structure) [kg].
    g0 : float
        Standard gravity [m s⁻²].

    Returns
    -------
    float
        Required propellant mass [kg].
    """
    if delta_v < 0.0:
        raise ValueError(f"delta_v must be ≥ 0, got {delta_v!r}")
    if mass_payload <= 0.0:
        raise ValueError(f"mass_payload must be > 0, got {mass_payload!r}")
    return mass_payload * (math.exp(delta_v / (isp * g0)) - 1.0)