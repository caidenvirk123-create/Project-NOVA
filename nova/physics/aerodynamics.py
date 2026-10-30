"""
nova.physics.aerodynamics
=========================
6-DoF rigid-body aerodynamic model for Project NOVA.

Architecture role — Pipeline Stage 3 (physics engine input)
------------------------------------------------------------
Computes aerodynamic forces and moments acting on the vehicle at one
simulation tick, given the current atmospheric state, airspeed vector,
attitude, and control surface deflections.

Output is passed to:
  - ForceAccumulator.add_aerodynamic()   → Stage 3 force tensor
  - TorqueAccumulator.add_aerodynamic_moments() → Stage 3 torque tensor

Physics implemented
-------------------

Forces (stability axes, then rotated to Body)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  L = q_∞ · S · C_L(α, δ_e, M)         Lift       [N]  (+Z_stab → up)
  D = q_∞ · S · C_D(α, M)              Drag       [N]  (−X_stab → aft)
  Y = q_∞ · S · C_Y(β, δ_r)            Side force [N]  (+Y_stab)

where q_∞ = ½ρv² is dynamic pressure, S is reference wing area.

Moments (Body frame)
~~~~~~~~~~~~~~~~~~~~
  M_pitch = q_∞ · S · c̄ · C_m(α, q_rate, δ_e)    [N·m]
  M_yaw   = q_∞ · S · b  · C_n(β, r_rate, δ_r)    [N·m]
  M_roll  = q_∞ · S · b  · C_l(β, p_rate, δ_a)    [N·m]

Stability derivative model
~~~~~~~~~~~~~~~~~~~~~~~~~~
Coefficients are built from linear stability derivative expansions:
  C_L = C_L0 + C_Lα·α + C_Lδe·δ_e
  C_D = C_D0 + C_D_induced + C_D_wave(M)   (polar + wave drag)
  C_Y = C_Yβ·β + C_Yδr·δ_r
  C_m = C_m0 + C_mα·α + C_mq·(q·c̄/(2v)) + C_mδe·δ_e
  C_n = C_nβ·β + C_nr·(r·b/(2v)) + C_nδr·δ_r
  C_l = C_lβ·β + C_lp·(p·b/(2v)) + C_lδa·δ_a

Compressibility
~~~~~~~~~~~~~~~
Prandtl-Glauert correction applied for M < MACH_PG_UPPER:
  C_L_corrected = C_L / √(1 − M²)

Above MACH_DRAG_DIVERGENCE, a tabulated wave-drag increment ΔC_D_wave is
added. The table is defined as a piecewise-linear spline indexed by Mach.

Coordinate conventions
~~~~~~~~~~~~~~~~~~~~~~
  Stability axes: X_s → forward (into freestream), Z_s → up, Y_s → right
  Body axes:      X_b → forward, Y_b → right, Z_b → down

The rotation from stability to body axes is:
  [X_b]   [ cos α   0  −sin α ] [X_s]
  [Y_b] = [   0     1     0   ] [Y_s]
  [Z_b]   [ sin α   0   cos α ] [Z_s]

For small α this is approximately identity in the pitch plane.

References
----------
- Stevens & Lewis, "Aircraft Control and Simulation", 3rd ed., §2.3–§2.5
- Anderson, "Introduction to Flight", 8th ed., §4–§5
- Roskam, "Airplane Flight Dynamics & Automatic Flight Controls", Part I
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nova.core.constants import (
    MACH_PG_UPPER,
    MACH_DRAG_DIVERGENCE,
)
from nova.physics.atmosphere import (
    dynamic_pressure,
    prandtl_glauert_factor,
    AtmosphericState,
)


# ---------------------------------------------------------------------------
# Control surface deflection angles
# ---------------------------------------------------------------------------

@dataclass
class ControlDeflections:
    """
    Actuated control surface deflection angles [rad].

    All angles measured from neutral (trim) position.
    Positive deflection conventions follow standard aerospace sign rules:
      δ_e > 0 → trailing edge down (pitch-up moment)
      δ_a > 0 → right aileron down, left up (right-roll moment)
      δ_r > 0 → trailing edge left (yaw-left / nose-right moment)

    Attributes
    ----------
    elevator : float   δ_e [rad]
    aileron  : float   δ_a [rad]
    rudder   : float   δ_r [rad]
    """
    elevator: float = 0.0   # δ_e [rad]
    aileron:  float = 0.0   # δ_a [rad]
    rudder:   float = 0.0   # δ_r [rad]


# ---------------------------------------------------------------------------
# Vehicle aerodynamic configuration
# ---------------------------------------------------------------------------

@dataclass
class AeroConfig:
    """
    Vehicle-specific aerodynamic reference geometry and stability derivatives.

    All derivatives are per-radian unless noted as dimensionless.

    Parameters
    ----------
    reference_area : float
        S — wing reference area [m²].
    mean_chord : float
        c̄ — mean aerodynamic chord [m]. Used for pitch moment normalisation.
    span : float
        b — wing span [m]. Used for roll/yaw moment normalisation.

    Lift derivatives
    ----------------
    CL0 : float
        Zero-AoA lift coefficient [-].
    CLa : float
        Lift-curve slope dC_L/dα [rad⁻¹]. Typically ~2π for thin wings.
    CLde : float
        Elevator lift effectiveness dC_L/dδ_e [rad⁻¹].

    Drag polar
    ----------
    CD0 : float
        Parasite drag coefficient [-].
    k_induced : float
        Induced drag factor k such that C_Di = k·C_L² (= 1/(π·AR·e)).
    CD_wave_table : list of (Mach, ΔC_D) pairs, optional
        Piecewise-linear wave-drag increment table. Indexed by Mach number.
        Applied above MACH_DRAG_DIVERGENCE.

    Side force
    ----------
    CYb : float
        Side-force due to sideslip dC_Y/dβ [rad⁻¹]. Typically negative.
    CYdr : float
        Side-force due to rudder dC_Y/dδ_r [rad⁻¹].

    Pitching moment
    ---------------
    Cm0 : float
        Zero-lift pitching moment coefficient [-].
    Cma : float
        Pitch stiffness dC_m/dα [rad⁻¹]. Negative → statically stable.
    Cmq : float
        Pitch damping dC_m/d(q·c̄/(2v)) [rad⁻¹]. Typically negative.
    Cmde : float
        Elevator pitch effectiveness dC_m/dδ_e [rad⁻¹].

    Yawing moment
    -------------
    Cnb : float
        Yaw stiffness dC_n/dβ [rad⁻¹]. Positive → weathercock stable.
    Cnr : float
        Yaw damping dC_n/d(r·b/(2v)) [rad⁻¹]. Typically negative.
    Cndr : float
        Rudder yaw effectiveness dC_n/dδ_r [rad⁻¹].

    Rolling moment
    --------------
    Clb : float
        Roll-due-to-sideslip (dihedral effect) dC_l/dβ [rad⁻¹]. Neg → stable.
    Clp : float
        Roll damping dC_l/d(p·b/(2v)) [rad⁻¹]. Typically negative.
    Clda : float
        Aileron roll effectiveness dC_l/dδ_a [rad⁻¹].
    """
    # Reference geometry
    reference_area: float   # S [m²]
    mean_chord:     float   # c̄ [m]
    span:           float   # b [m]

    # Lift
    CL0:  float = 0.0
    CLa:  float = 5.730    # ≈ 2π·0.9 for a moderately cambered wing [rad⁻¹]
    CLde: float = 0.5

    # Drag polar
    CD0:         float = 0.020
    k_induced:   float = 0.050   # k = 1/(π·AR·e), typical clean wing

    # Wave drag: list of (Mach, delta_CD) breakpoints (piecewise linear)
    CD_wave_table: list = field(default_factory=lambda: [
        (0.85, 0.000),
        (0.90, 0.005),
        (0.95, 0.020),
        (1.00, 0.060),
        (1.05, 0.050),
        (1.20, 0.030),
        (2.00, 0.015),
        (5.00, 0.008),
    ])

    # Side force
    CYb:  float = -0.980   # [rad⁻¹]
    CYdr: float =  0.175

    # Pitching moment
    Cm0:  float =  0.000
    Cma:  float = -1.800   # [rad⁻¹] — statically stable
    Cmq:  float = -12.0    # [rad⁻¹]
    Cmde: float = -1.500   # [rad⁻¹]

    # Yawing moment
    Cnb:  float =  0.120   # [rad⁻¹]
    Cnr:  float = -0.150   # [rad⁻¹]
    Cndr: float = -0.100   # [rad⁻¹]

    # Rolling moment
    Clb:  float = -0.100   # [rad⁻¹]
    Clp:  float = -0.500   # [rad⁻¹]
    Clda: float =  0.220   # [rad⁻¹]

    def __post_init__(self) -> None:
        if self.reference_area <= 0.0:
            raise ValueError(f"AeroConfig.reference_area must be > 0, got {self.reference_area!r}")
        if self.mean_chord <= 0.0:
            raise ValueError(f"AeroConfig.mean_chord must be > 0, got {self.mean_chord!r}")
        if self.span <= 0.0:
            raise ValueError(f"AeroConfig.span must be > 0, got {self.span!r}")


# ---------------------------------------------------------------------------
# Aerodynamic state output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AeroState:
    """
    Complete aerodynamic output for one simulation tick.

    All forces in Body frame [N]; all moments in Body frame [N·m].

    Attributes
    ----------
    alpha : float
        Angle of attack [rad].
    beta : float
        Sideslip angle [rad].
    mach : float
        Freestream Mach number [-].
    dynamic_pressure : float
        q_∞ = ½ρv² [Pa].
    CL, CD, CY : float
        Lift, drag, side-force coefficients (corrected for compressibility).
    Cm, Cn, Cl : float
        Pitching, yawing, rolling moment coefficients.
    lift_body : ndarray, shape (3,), float64
        Lift force in Body frame [N].
    drag_body : ndarray, shape (3,), float64
        Drag force in Body frame [N].
    side_force_body : ndarray, shape (3,), float64
        Side force in Body frame [N].
    force_body : ndarray, shape (3,), float64
        Net aerodynamic force in Body frame [N] = lift + drag + side.
    pitching_moment : float   [N·m]
    yawing_moment   : float   [N·m]
    rolling_moment  : float   [N·m]
    """
    alpha:            float
    beta:             float
    mach:             float
    dynamic_pressure: float
    CL:               float
    CD:               float
    CY:               float
    Cm:               float
    Cn:               float
    Cl:               float
    lift_body:        np.ndarray
    drag_body:        np.ndarray
    side_force_body:  np.ndarray
    force_body:       np.ndarray
    pitching_moment:  float
    yawing_moment:    float
    rolling_moment:   float


# ---------------------------------------------------------------------------
# Wave drag interpolator
# ---------------------------------------------------------------------------

def _wave_drag_increment(mach: float, table: list) -> float:
    """
    Piecewise-linear interpolation of wave drag increment ΔC_D from
    a (Mach, ΔC_D) breakpoint table.

    Parameters
    ----------
    mach : float
        Freestream Mach number.
    table : list of (float, float)
        Sorted (Mach, ΔC_D) breakpoints.

    Returns
    -------
    float
        Wave drag increment ΔC_D [-].
    """
    if not table or mach <= table[0][0]:
        return 0.0
    if mach >= table[-1][0]:
        return float(table[-1][1])
    for i in range(len(table) - 1):
        m0, cd0 = table[i]
        m1, cd1 = table[i + 1]
        if m0 <= mach <= m1:
            t = (mach - m0) / (m1 - m0)
            return cd0 + t * (cd1 - cd0)
    return 0.0


# ---------------------------------------------------------------------------
# Angle-of-attack and sideslip extraction
# ---------------------------------------------------------------------------

def aero_angles(velocity_body: np.ndarray) -> tuple[float, float]:
    """
    Compute angle of attack (α) and sideslip angle (β) from the
    velocity vector expressed in the Body frame.

    Definitions (standard aerospace)
    ---------------------------------
    α = arctan2(w_b, u_b)     [rad]   pitch-plane angle of attack
    β = arcsin(v_b / |v|)     [rad]   yaw-plane sideslip

    where (u_b, v_b, w_b) = velocity components in Body frame:
      u_b → forward (+X_b)
      v_b → right   (+Y_b)
      w_b → down    (+Z_b)  ← positive w means downward relative wind → positive α

    Parameters
    ----------
    velocity_body : ndarray, shape (3,), float64
        Airspeed vector in Body frame [m s⁻¹].

    Returns
    -------
    (alpha, beta) : tuple of float, [rad]
        Returns (0, 0) if airspeed magnitude is below 0.01 m/s.
    """
    v = np.asarray(velocity_body, dtype=np.float64)
    v_mag = float(np.linalg.norm(v))
    if v_mag < 0.01:
        return 0.0, 0.0

    u_b, v_b, w_b = float(v[0]), float(v[1]), float(v[2])

    alpha = math.atan2(w_b, u_b)
    beta  = math.asin(max(-1.0, min(1.0, v_b / v_mag)))

    return alpha, beta


# ---------------------------------------------------------------------------
# Stability-to-body axis rotation
# ---------------------------------------------------------------------------

def stability_to_body(
    L_stab: float,
    D_stab: float,
    Y_stab: float,
    alpha:  float,
) -> np.ndarray:
    """
    Rotate aerodynamic forces from stability axes to Body axes.

    Stability axes: X_s = −drag direction, Z_s = −lift direction
    Body axes:      X_b = forward, Y_b = right, Z_b = down

    The rotation about the Y axis (pitch axis) by angle α gives:

        F_body = R_stab_to_body(α) · F_stab

    where:
        X_body = X_stab · cos α − Z_stab · sin α   (= −D·cosα + L·sinα)
        Y_body = Y_stab
        Z_body = X_stab · sin α + Z_stab · cos α   (= −D·sinα − L·cosα)

    Note: in stability axes, X_s is aligned with the velocity vector (−drag
    direction), so D is applied along −X_s and L along +Z_s (upward).
    After rotation:
      F_x_body = −D·cos α + L·sin α  (net axial force, usually negative = drag)
      F_y_body = Y                    (side force unchanged)
      F_z_body = −D·sin α − L·cos α  (net normal force; negative = upward lift)

    Parameters
    ----------
    L_stab : float   Lift [N]   (positive = up in stability axes)
    D_stab : float   Drag [N]   (positive = opposes velocity)
    Y_stab : float   Side force [N]
    alpha  : float   Angle of attack [rad]

    Returns
    -------
    ndarray, shape (3,), float64
        Net aerodynamic force in Body frame [N].
    """
    ca, sa = math.cos(alpha), math.sin(alpha)

    # In stability axes: F_stab = [−D, Y, −L]  (drag aft, lift up→ −Z_stab)
    # Rotate to body:
    Fx_body = -D_stab * ca + L_stab * sa   # axial  (aft = negative)
    Fy_body =  Y_stab                       # lateral
    Fz_body = -D_stab * sa - L_stab * ca   # normal (up = negative in body Z-down)

    return np.array([Fx_body, Fy_body, Fz_body], dtype=np.float64)


# ---------------------------------------------------------------------------
# Primary aerodynamics function
# ---------------------------------------------------------------------------

def compute_aero(
    velocity_body:  np.ndarray,
    omega_body:     np.ndarray,
    atm:            AtmosphericState,
    config:         AeroConfig,
    deflections:    ControlDeflections,
) -> AeroState:
    """
    Compute the full aerodynamic force and moment state for one tick.

    Parameters
    ----------
    velocity_body : ndarray, shape (3,), float64
        Vehicle airspeed vector in Body frame [m s⁻¹].
        Typically v_eci − v_wind, rotated to Body frame.
    omega_body : ndarray, shape (3,), float64
        Angular velocity in Body frame (p, q, r) [rad s⁻¹].
    atm : AtmosphericState
        Atmospheric state at the vehicle's current altitude.
    config : AeroConfig
        Vehicle aerodynamic configuration and stability derivatives.
    deflections : ControlDeflections
        Current control surface deflection angles [rad].

    Returns
    -------
    AeroState
        Complete aerodynamic output snapshot (frozen).

    Notes
    -----
    * If airspeed < 0.01 m/s, returns a zero-force AeroState (vacuum/hover).
    * No stall model is implemented beyond the linear CLa regime. AoA
      limiting (stall onset) is handled by the AI monitor advisory layer.
    * The dynamic pressure term uses true airspeed relative to the atmosphere;
      the caller must subtract wind velocity from ECI velocity before rotating.
    """
    v    = np.asarray(velocity_body, dtype=np.float64)
    omg  = np.asarray(omega_body,    dtype=np.float64)
    v_mag = float(np.linalg.norm(v))

    # --- Zero-airspeed guard ---
    _ZERO = np.zeros(3, dtype=np.float64)
    if v_mag < 0.01:
        return AeroState(
            alpha=0.0, beta=0.0, mach=0.0, dynamic_pressure=0.0,
            CL=0.0, CD=0.0, CY=0.0, Cm=0.0, Cn=0.0, Cl=0.0,
            lift_body=_ZERO.copy(), drag_body=_ZERO.copy(),
            side_force_body=_ZERO.copy(), force_body=_ZERO.copy(),
            pitching_moment=0.0, yawing_moment=0.0, rolling_moment=0.0,
        )

    # --- Aero angles ---
    alpha, beta = aero_angles(v)
    p_rate, q_rate, r_rate = float(omg[0]), float(omg[1]), float(omg[2])

    # --- Atmospheric quantities ---
    rho  = atm.density
    a_s  = atm.speed_of_sound
    q_inf = dynamic_pressure(rho, v_mag)
    M     = v_mag / a_s if a_s > 1.0e-6 else 0.0

    # --- Compressibility correction ---
    if M < MACH_PG_UPPER:
        beta_pg = prandtl_glauert_factor(M)   # β_PG = 1/√(1−M²)
    else:
        beta_pg = 1.0   # P-G breaks down near M=1; correction capped

    # --- Dimensionless rate terms (standard normalised rates) ---
    v_safe = max(v_mag, 0.01)
    q_hat = q_rate * config.mean_chord / (2.0 * v_safe)   # pitch rate term
    r_hat = r_rate * config.span       / (2.0 * v_safe)   # yaw rate term
    p_hat = p_rate * config.span       / (2.0 * v_safe)   # roll rate term

    de = deflections.elevator
    da = deflections.aileron
    dr = deflections.rudder

    # ---------------------------------------------------------------
    # Lift coefficient
    # ---------------------------------------------------------------
    CL_base = config.CL0 + config.CLa * alpha + config.CLde * de
    CL      = CL_base * beta_pg   # Prandtl-Glauert correction

    # ---------------------------------------------------------------
    # Drag coefficient (polar + induced + wave)
    # ---------------------------------------------------------------
    CD_induced = config.k_induced * CL_base**2
    CD_wave    = _wave_drag_increment(M, config.CD_wave_table)
    CD         = config.CD0 + CD_induced + CD_wave

    # ---------------------------------------------------------------
    # Side-force coefficient
    # ---------------------------------------------------------------
    CY = config.CYb * beta + config.CYdr * dr

    # ---------------------------------------------------------------
    # Pitching moment coefficient
    # ---------------------------------------------------------------
    Cm = (config.Cm0
          + config.Cma  * alpha
          + config.Cmq  * q_hat
          + config.Cmde * de)

    # ---------------------------------------------------------------
    # Yawing moment coefficient
    # ---------------------------------------------------------------
    Cn = (config.Cnb  * beta
          + config.Cnr  * r_hat
          + config.Cndr * dr)

    # ---------------------------------------------------------------
    # Rolling moment coefficient
    # ---------------------------------------------------------------
    Cl = (config.Clb  * beta
          + config.Clp  * p_hat
          + config.Clda * da)

    # ---------------------------------------------------------------
    # Dimensionalise forces
    # ---------------------------------------------------------------
    S = config.reference_area
    L_stab =  CL * q_inf * S   # [N] lift, positive up
    D_stab =  CD * q_inf * S   # [N] drag, always positive
    Y_stab =  CY * q_inf * S   # [N] side force

    # Rotate to Body frame
    aero_force_body = stability_to_body(L_stab, D_stab, Y_stab, alpha)

    # Decompose for telemetry
    ca, sa = math.cos(alpha), math.sin(alpha)
    lift_body = np.array([ L_stab * sa,  0.0, -L_stab * ca], dtype=np.float64)
    drag_body = np.array([-D_stab * ca,  0.0, -D_stab * sa], dtype=np.float64)
    side_body = np.array([0.0, Y_stab, 0.0],                  dtype=np.float64)

    # ---------------------------------------------------------------
    # Dimensionalise moments
    # ---------------------------------------------------------------
    c_bar = config.mean_chord
    b     = config.span

    M_pitch =  Cm * q_inf * S * c_bar   # [N·m]
    M_yaw   =  Cn * q_inf * S * b       # [N·m]
    M_roll  =  Cl * q_inf * S * b       # [N·m]

    return AeroState(
        alpha=alpha,
        beta=beta,
        mach=M,
        dynamic_pressure=q_inf,
        CL=CL,
        CD=CD,
        CY=CY,
        Cm=Cm,
        Cn=Cn,
        Cl=Cl,
        lift_body=lift_body,
        drag_body=drag_body,
        side_force_body=side_body,
        force_body=aero_force_body,
        pitching_moment=M_pitch,
        yawing_moment=M_yaw,
        rolling_moment=M_roll,
    )