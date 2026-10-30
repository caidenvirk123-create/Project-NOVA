"""
nova.physics.orbital
====================
Orbital mechanics solver for Project NOVA.

Provides two independent subsystems:

1. **Keplerian orbital elements** — extract and use the six classical orbital
   elements (a, e, i, Ω, ω, ν) from an ECI state vector, and compute
   closed-form Keplerian trajectories for single-body motion.

2. **N-body gravitational acceleration** — compute the gravitational
   acceleration on the vehicle from an arbitrary number of point-mass bodies
   (Earth, Moon, Sun, custom). This is the quantity injected into the force
   accumulator at pipeline Stage 6.

Physics contracts
-----------------
* Newtonian gravity only — no GR corrections.
* J2 oblateness perturbation is included for Earth (switchable).
* No atmospheric drag in this module — that belongs in aerodynamics.py.
* All vectors in ECI frame, SI units.

The orbital solver does NOT modify VehicleState. It returns force vectors
and orbital elements that are consumed by the force accumulator and the
orbital mechanics HUD panel respectively.

References
----------
- Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed.
- Bate, Mueller & White, "Fundamentals of Astrodynamics", Dover
- Montenbruck & Gill, "Satellite Orbits", Springer
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from nova.core.constants import (
    EARTH_MU,
    EARTH_RADIUS_EQ,
    EARTH_J2,
    G,
)


# ---------------------------------------------------------------------------
# Gravitational body descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GravBody:
    """
    Point-mass gravitational body for N-body computation.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "Earth", "Moon").
    mu : float
        Standard gravitational parameter μ = GM [m³ s⁻²].
    position_eci : ndarray, shape (3,)
        Position of the body's centre of mass in ECI [m].
        For the primary body (Earth), this is typically [0, 0, 0].
    """
    name: str
    mu: float
    position_eci: np.ndarray   # shape (3,), float64

    def __post_init__(self) -> None:
        pos = self.position_eci
        if not isinstance(pos, np.ndarray):
            object.__setattr__(
                self, "position_eci",
                np.asarray(pos, dtype=np.float64)
            )
        if self.mu <= 0.0:
            raise ValueError(f"GravBody.mu must be > 0, got {self.mu!r}")


# Pre-built Earth body centred at ECI origin
EARTH_BODY = GravBody(
    name="Earth",
    mu=EARTH_MU,
    position_eci=np.zeros(3, dtype=np.float64),
)


# ---------------------------------------------------------------------------
# Keplerian orbital elements
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeplerianElements:
    """
    Classical six Keplerian orbital elements computed from an ECI state.

    All angles in radians. Distances in metres.

    Attributes
    ----------
    semi_major_axis : float
        a [m]. Negative for hyperbolic trajectories (e > 1).
    eccentricity : float
        e [-]. 0 = circular, 0 < e < 1 = elliptic,
               e = 1 = parabolic, e > 1 = hyperbolic.
    inclination : float
        i [rad]. Angle between orbit plane and equatorial plane. [0, π].
    raan : float
        Ω [rad]. Right Ascension of the Ascending Node. [0, 2π).
    argument_of_periapsis : float
        ω [rad]. Angle from ascending node to periapsis. [0, 2π).
    true_anomaly : float
        ν [rad]. Current angular position from periapsis. [0, 2π).
    specific_angular_momentum : float
        h [m² s⁻¹]. ‖r × v‖.
    specific_orbital_energy : float
        ε [J kg⁻¹]. v²/2 − μ/r. Negative for bound orbits.
    period : float
        T [s]. Orbital period. Positive for elliptic; inf for parabolic/hyperbolic.
    apoapsis : float
        r_a [m]. Radius at apoapsis = a(1 + e). Inf if e ≥ 1.
    periapsis : float
        r_p [m]. Radius at periapsis = a(1 − e).
    """
    semi_major_axis:            float   # a [m]
    eccentricity:               float   # e [-]
    inclination:                float   # i [rad]
    raan:                       float   # Ω [rad]
    argument_of_periapsis:      float   # ω [rad]
    true_anomaly:               float   # ν [rad]
    specific_angular_momentum:  float   # h [m² s⁻¹]
    specific_orbital_energy:    float   # ε [J kg⁻¹]
    period:                     float   # T [s]
    apoapsis:                   float   # r_a [m]
    periapsis:                  float   # r_p [m]


def elements_from_state(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
    mu: float = EARTH_MU,
) -> KeplerianElements:
    """
    Convert an ECI state vector (r, v) to classical Keplerian orbital elements.

    Algorithm follows Bate, Mueller & White §2.4 ("The Determination of an
    Orbit from r and v"). All edge cases (circular, equatorial, hyperbolic)
    are handled numerically; true anomaly is always measured from periapsis.

    Parameters
    ----------
    position_eci : ndarray, shape (3,)
        ECI position [m].
    velocity_eci : ndarray, shape (3,)
        ECI velocity [m s⁻¹].
    mu : float
        Gravitational parameter [m³ s⁻²]. Default = EARTH_MU.

    Returns
    -------
    KeplerianElements

    Raises
    ------
    ValueError
        If the position vector is zero (spacecraft at body centre).
    """
    r_vec = np.asarray(position_eci, dtype=np.float64)
    v_vec = np.asarray(velocity_eci, dtype=np.float64)

    r_mag = float(np.linalg.norm(r_vec))
    v_mag = float(np.linalg.norm(v_vec))

    if r_mag < 1.0:
        raise ValueError(
            f"Position vector magnitude {r_mag:.3f} m is too small. "
            "Vehicle may be inside the planet."
        )

    # --- Specific angular momentum h = r × v ---
    h_vec = np.cross(r_vec, v_vec)
    h_mag = float(np.linalg.norm(h_vec))

    # --- Node vector N = k̂ × h (k̂ = ECI Z-axis) ---
    k_hat = np.array([0.0, 0.0, 1.0])
    N_vec = np.cross(k_hat, h_vec)
    N_mag = float(np.linalg.norm(N_vec))

    # --- Eccentricity vector e = (1/μ)[(v²−μ/r)r − (r·v)v] ---
    rdotv    = float(np.dot(r_vec, v_vec))
    e_vec    = ((v_mag**2 - mu / r_mag) * r_vec - rdotv * v_vec) / mu
    ecc      = float(np.linalg.norm(e_vec))

    # --- Specific orbital energy ε = v²/2 − μ/r ---
    energy = 0.5 * v_mag**2 - mu / r_mag

    # --- Semi-major axis a ---
    if abs(ecc - 1.0) < 1.0e-10:
        # Parabolic — semi-major axis undefined (use convention: ∞)
        a = math.inf
    else:
        a = -mu / (2.0 * energy)

    # --- Inclination i = arccos(h_z / |h|) ---
    inclination = math.acos(max(-1.0, min(1.0, h_vec[2] / h_mag))) if h_mag > 0.0 else 0.0

    # --- RAAN Ω = arccos(N_x / |N|) ---
    if N_mag < 1.0e-10:
        # Equatorial orbit — RAAN undefined; set to 0
        raan = 0.0
    else:
        raan = math.acos(max(-1.0, min(1.0, N_vec[0] / N_mag)))
        if N_vec[1] < 0.0:
            raan = 2.0 * math.pi - raan

    # --- Argument of periapsis ω = arccos((N · e) / (|N||e|)) ---
    if N_mag < 1.0e-10 or ecc < 1.0e-10:
        # Circular or equatorial — ω undefined; set to 0
        arg_pe = 0.0
    else:
        cos_w = float(np.dot(N_vec, e_vec)) / (N_mag * ecc)
        arg_pe = math.acos(max(-1.0, min(1.0, cos_w)))
        if e_vec[2] < 0.0:
            arg_pe = 2.0 * math.pi - arg_pe

    # --- True anomaly ν = arccos((e · r) / (|e||r|)) ---
    if ecc < 1.0e-10:
        # Circular — measure ν from ascending node
        if N_mag < 1.0e-10:
            # Circular equatorial — measure from +X
            cos_nu = r_vec[0] / r_mag
            nu = math.acos(max(-1.0, min(1.0, cos_nu)))
            if r_vec[1] < 0.0:
                nu = 2.0 * math.pi - nu
        else:
            cos_nu = float(np.dot(N_vec, r_vec)) / (N_mag * r_mag)
            nu = math.acos(max(-1.0, min(1.0, cos_nu)))
            if float(np.dot(N_vec, v_vec)) > 0.0:
                nu = 2.0 * math.pi - nu
    else:
        cos_nu = float(np.dot(e_vec, r_vec)) / (ecc * r_mag)
        nu = math.acos(max(-1.0, min(1.0, cos_nu)))
        if rdotv < 0.0:
            nu = 2.0 * math.pi - nu

    # --- Period T = 2π √(a³/μ) ---
    if a > 0.0 and ecc < 1.0:
        period = 2.0 * math.pi * math.sqrt(a**3 / mu)
    else:
        period = math.inf

    # --- Apoapsis and periapsis radii ---
    if ecc < 1.0:
        apoapsis  = a * (1.0 + ecc)
        periapsis = a * (1.0 - ecc)
    else:
        apoapsis  = math.inf
        periapsis = a * (1.0 - ecc) if not math.isinf(a) else math.inf

    return KeplerianElements(
        semi_major_axis=a,
        eccentricity=ecc,
        inclination=inclination,
        raan=raan,
        argument_of_periapsis=arg_pe,
        true_anomaly=nu,
        specific_angular_momentum=h_mag,
        specific_orbital_energy=energy,
        period=period,
        apoapsis=apoapsis,
        periapsis=periapsis,
    )


# ---------------------------------------------------------------------------
# Analytical Keplerian propagation (for validation only)
# ---------------------------------------------------------------------------

def kepler_period(semi_major_axis: float, mu: float = EARTH_MU) -> float:
    """
    Orbital period via Kepler's 3rd law.

    T = 2π √(a³ / μ)

    Parameters
    ----------
    semi_major_axis : float
        Semi-major axis [m]. Must be > 0 (elliptic orbit).
    mu : float
        Gravitational parameter [m³ s⁻²].

    Returns
    -------
    float
        Period [s].
    """
    if semi_major_axis <= 0.0:
        raise ValueError(f"semi_major_axis must be > 0, got {semi_major_axis!r}")
    return 2.0 * math.pi * math.sqrt(semi_major_axis**3 / mu)


def vis_viva_speed(radius: float, semi_major_axis: float, mu: float = EARTH_MU) -> float:
    """
    Orbital speed at a given radius via the vis-viva equation.

    v = √(μ · (2/r − 1/a))

    Parameters
    ----------
    radius : float
        Current orbital radius [m].
    semi_major_axis : float
        Semi-major axis [m]. Use r for circular orbit.
    mu : float
        Gravitational parameter [m³ s⁻²].

    Returns
    -------
    float
        Speed [m s⁻¹].
    """
    val = mu * (2.0 / radius - 1.0 / semi_major_axis)
    if val < 0.0:
        raise ValueError(
            f"Vis-viva argument is negative ({val:.3e}): "
            f"r={radius:.1f} m, a={semi_major_axis:.1f} m — "
            "check that r ≤ 2a for a bound orbit."
        )
    return math.sqrt(val)


def circular_orbit_speed(radius: float, mu: float = EARTH_MU) -> float:
    """
    Speed for a circular orbit at the given radius.

    v_c = √(μ / r)

    Parameters
    ----------
    radius : float
        Orbital radius from planet centre [m].
    mu : float
        Gravitational parameter [m³ s⁻²].

    Returns
    -------
    float
        Circular speed [m s⁻¹].
    """
    return math.sqrt(mu / radius)


def hohmann_delta_v(r1: float, r2: float, mu: float = EARTH_MU) -> tuple[float, float]:
    """
    Compute the two Δv burns for a Hohmann transfer between two circular orbits.

    Parameters
    ----------
    r1 : float
        Initial circular orbit radius [m].
    r2 : float
        Target circular orbit radius [m].
    mu : float
        Gravitational parameter [m³ s⁻²].

    Returns
    -------
    (dv1, dv2) : tuple of float
        First and second Δv burns [m s⁻¹]. Both positive (prograde).

    Derivation
    ----------
    Transfer ellipse semi-major axis: a_t = (r1 + r2) / 2

    v1_circ = √(μ/r1)         circular speed at r1
    v_pe    = √(μ(2/r1−1/aₜ)) transfer orbit speed at periapsis
    Δv₁     = v_pe − v1_circ  (prograde burn to enter transfer)

    v2_circ = √(μ/r2)
    v_ap    = √(μ(2/r2−1/aₜ)) transfer orbit speed at apoapsis
    Δv₂     = v2_circ − v_ap  (prograde burn to circularise at r2)
    """
    a_t   = 0.5 * (r1 + r2)
    v1c   = circular_orbit_speed(r1, mu)
    v2c   = circular_orbit_speed(r2, mu)
    v_pe  = vis_viva_speed(r1, a_t, mu)
    v_ap  = vis_viva_speed(r2, a_t, mu)
    dv1   = v_pe - v1c
    dv2   = v2c - v_ap
    return dv1, dv2


# ---------------------------------------------------------------------------
# N-body gravitational acceleration
# ---------------------------------------------------------------------------

def gravity_acceleration(
    position_eci: np.ndarray,
    bodies: Sequence[GravBody],
    include_j2: bool = True,
    planet_radius_eq: float = EARTH_RADIUS_EQ,
    j2_coefficient: float = EARTH_J2,
    primary_mu: float = EARTH_MU,
) -> np.ndarray:
    """
    Compute the total gravitational acceleration on the vehicle.

    Sums point-mass contributions from all provided bodies, plus an optional
    J2 oblateness perturbation from the primary body (Earth).

    Parameters
    ----------
    position_eci : ndarray, shape (3,)
        Vehicle position in ECI [m].
    bodies : sequence of GravBody
        List of gravitational bodies to include. Must contain at least one.
    include_j2 : bool
        If True, add the J2 zonal harmonic perturbation from the primary body.
        Only valid when the primary body is centred at ECI origin.
    planet_radius_eq : float
        Equatorial radius of the primary body [m]. Used in J2 term.
    j2_coefficient : float
        J2 coefficient (dimensionless). Default = EARTH_J2.
    primary_mu : float
        μ of the primary body [m³ s⁻²]. Used in J2 term.

    Returns
    -------
    ndarray, shape (3,), float64
        Total gravitational acceleration [m s⁻²] in ECI frame.

    Notes
    -----
    The J2 perturbation accelerations in ECI are:

        a_x = −(3/2) J2 (μ/r²)(R_eq/r)² (1 − 5(z/r)²) (x/r)
        a_y = −(3/2) J2 (μ/r²)(R_eq/r)² (1 − 5(z/r)²) (y/r)
        a_z = −(3/2) J2 (μ/r²)(R_eq/r)² (3 − 5(z/r)²) (z/r)

    Reference: Vallado §8.7.1
    """
    r_veh = np.asarray(position_eci, dtype=np.float64)
    a_total = np.zeros(3, dtype=np.float64)

    for body in bodies:
        r_body = np.asarray(body.position_eci, dtype=np.float64)
        r_rel  = r_veh - r_body           # vector from body to vehicle
        r_mag  = float(np.linalg.norm(r_rel))

        if r_mag < 1.0:
            # Vehicle is inside body — physically undefined; skip
            continue

        # Point-mass term: a = −μ / r³ · r_rel
        a_total += (-body.mu / r_mag**3) * r_rel

    if include_j2 and len(bodies) > 0:
        # J2 perturbation (primary body assumed at ECI origin)
        r_mag = float(np.linalg.norm(r_veh))
        if r_mag > 1.0:
            x, y, z = r_veh[0], r_veh[1], r_veh[2]
            ratio   = planet_radius_eq / r_mag
            coeff   = -1.5 * j2_coefficient * (primary_mu / r_mag**2) * ratio**2
            z_r     = z / r_mag
            five_z2 = 5.0 * z_r**2

            a_total[0] += coeff * (1.0 - five_z2) * (x / r_mag)
            a_total[1] += coeff * (1.0 - five_z2) * (y / r_mag)
            a_total[2] += coeff * (3.0 - five_z2) * (z / r_mag)

    return a_total


def gravity_force(
    position_eci: np.ndarray,
    mass: float,
    bodies: Sequence[GravBody] = (EARTH_BODY,),
    include_j2: bool = True,
) -> np.ndarray:
    """
    Gravitational force on a vehicle of given mass.

    F_grav = m · a_grav

    Parameters
    ----------
    position_eci : ndarray, shape (3,)
        Vehicle ECI position [m].
    mass : float
        Vehicle mass [kg].
    bodies : sequence of GravBody
        Gravitational bodies. Default: Earth only.
    include_j2 : bool
        Include J2 oblateness perturbation.

    Returns
    -------
    ndarray, shape (3,), float64
        Gravitational force [N] in ECI frame.
    """
    a_grav = gravity_acceleration(
        position_eci, bodies, include_j2=include_j2
    )
    return float(mass) * a_grav


# ---------------------------------------------------------------------------
# Maneuver Δv helper (HUD maneuver node support)
# ---------------------------------------------------------------------------

def delta_v_budget(
    position_eci: np.ndarray,
    velocity_eci: np.ndarray,
    target_velocity_eci: np.ndarray,
) -> float:
    """
    Compute the Δv magnitude required to go from the current velocity
    to a target velocity at the same position (instantaneous burn assumption).

    Δv = ‖v_target − v_current‖

    Parameters
    ----------
    position_eci, velocity_eci : ndarray, shape (3,)
        Current state in ECI.
    target_velocity_eci : ndarray, shape (3,)
        Desired velocity after burn in ECI [m s⁻¹].

    Returns
    -------
    float
        Δv magnitude [m s⁻¹].
    """
    dv_vec = np.asarray(target_velocity_eci) - np.asarray(velocity_eci)
    return float(np.linalg.norm(dv_vec))