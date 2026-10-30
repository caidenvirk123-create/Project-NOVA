"""
nova.frames.transforms
======================
All coordinate frame transformation matrices for Project NOVA.

Four reference frames are implemented, with explicit documented
transformations between every adjacent pair:

  ECI  ──T_ECI_to_ECEF──▶  ECEF  ──T_ECEF_to_ENU──▶  ENU  ──T_ENU_to_body──▶  Body

Every transformation function returns a (3, 3) float64 NumPy array
representing the rotation matrix **T** such that:

    v_B = T_A_to_B @ v_A

All inverse transformations are the matrix transpose (all rotation matrices
are orthogonal: T⁻¹ = Tᵀ), verified within TRANSFORM_IDENTITY_TOL.

Angle convention
----------------
All angles are in RADIANS. No function accepts degrees. If you have
degrees, multiply by ``nova.core.constants.DEG_TO_RAD`` before calling.

Quaternion convention
---------------------
q = (q0, q1, q2, q3) — scalar part first (Hamilton convention).
‖q‖ = 1 is assumed; no internal renormalisation is performed here.

Frame axis conventions
----------------------
ECI:   X → vernal equinox, Z → north celestial pole, Y completes RH
ECEF:  X → prime meridian/equator, Z → north pole, Y completes RH
ENU:   X → East, Y → North, Z → Up  (local tangent plane)
Body:  X → forward (longitudinal), Y → right (lateral), Z → down (normal)
       Right-hand NED-aligned body frame.

References
----------
- Vallado, "Fundamentals of Astrodynamics and Applications", 4th ed., §3
- Stevens & Lewis, "Aircraft Control and Simulation", §2
- Titterton & Weston, "Strapdown Inertial Navigation Technology", §3
"""

from __future__ import annotations

import math
import numpy as np

from nova.core.constants import (
    EARTH_OMEGA,
    TRANSFORM_IDENTITY_TOL,
)

# ---------------------------------------------------------------------------
# Type alias (for clarity in signatures)
# ---------------------------------------------------------------------------

DCM = np.ndarray   # Direction Cosine Matrix: shape (3, 3), dtype float64


# ---------------------------------------------------------------------------
# 1. ECI → ECEF   T(t)
# ---------------------------------------------------------------------------

def T_ECI_to_ECEF(time: float, omega: float = EARTH_OMEGA) -> DCM:
    """
    Rotation matrix from Earth-Centred Inertial (ECI) to
    Earth-Centred Earth-Fixed (ECEF) frame.

    The ECEF frame rotates about the ECI Z-axis with angular velocity
    ``omega`` [rad s⁻¹]. The rotation angle θ = omega · t increases
    eastward (right-hand rule about +Z).

    Parameters
    ----------
    time : float
        Elapsed simulation time [s]. t=0 assumes ECI and ECEF are aligned
        (Greenwich sidereal time = 0 at epoch; sufficient for relative
        trajectory analysis — absolute GAST alignment handled by the
        orbital initialiser).
    omega : float
        Planetary rotation rate [rad s⁻¹]. Defaults to Earth's sidereal
        rotation rate (7.292 115 × 10⁻⁵ rad s⁻¹).

    Returns
    -------
    DCM : ndarray, shape (3, 3), dtype float64

    Matrix definition
    -----------------
    θ = omega · t

    T_ECI→ECEF = ⎡  cos θ   sin θ   0 ⎤
                 ⎢ −sin θ   cos θ   0 ⎥
                 ⎣   0        0     1 ⎦

    Note: some texts define this as the *transpose* (ECI components
    expressed in ECEF). Here we follow the active rotation convention:
    apply this matrix to an ECI vector to get the ECEF vector.
    """
    theta = omega * time
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [ c,  s,  0.0],
        [-s,  c,  0.0],
        [ 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def T_ECEF_to_ECI(time: float, omega: float = EARTH_OMEGA) -> DCM:
    """
    Inverse of T_ECI_to_ECEF — rotate ECEF vector to ECI.

    Mathematically: T_ECEF→ECI = T_ECI→ECEF(t)ᵀ
    """
    return T_ECI_to_ECEF(time, omega).T


# ---------------------------------------------------------------------------
# 2. ECEF → ENU   T(λ, φ)
# ---------------------------------------------------------------------------

def T_ECEF_to_ENU(longitude_rad: float, latitude_rad: float) -> DCM:
    """
    Rotation matrix from ECEF to local East-North-Up (ENU) frame.

    The ENU frame is tangent to the planet ellipsoid at the point defined
    by (longitude, latitude). It is standard for launch-pad and
    atmospheric-flight operations.

    Parameters
    ----------
    longitude_rad : float
        Geodetic longitude λ [rad], positive east of prime meridian.
    latitude_rad : float
        Geodetic latitude φ [rad], positive north of equator.

    Returns
    -------
    DCM : ndarray, shape (3, 3), dtype float64

    Matrix definition
    -----------------
    λ = longitude,  φ = latitude

    T_ECEF→ENU = ⎡  −sin λ            cos λ            0      ⎤
                 ⎢  −sin φ cos λ      −sin φ sin λ     cos φ   ⎥
                 ⎣   cos φ cos λ       cos φ sin λ     sin φ   ⎦

    Row 0 = East unit vector expressed in ECEF
    Row 1 = North unit vector expressed in ECEF
    Row 2 = Up unit vector expressed in ECEF

    This transforms an ECEF column vector [x, y, z]ᵀ to ENU
    [East, North, Up]ᵀ.
    """
    lam = longitude_rad    # λ
    phi = latitude_rad     # φ

    sl, cl = math.sin(lam), math.cos(lam)
    sp, cp = math.sin(phi), math.cos(phi)

    return np.array([
        [      -sl,        cl,    0.0],
        [-sp * cl,  -sp * sl,    cp ],
        [ cp * cl,   cp * sl,    sp ],
    ], dtype=np.float64)


def T_ENU_to_ECEF(longitude_rad: float, latitude_rad: float) -> DCM:
    """
    Inverse of T_ECEF_to_ENU — rotate ENU vector to ECEF.

    Mathematically: T_ENU→ECEF = T_ECEF→ENU(λ, φ)ᵀ
    """
    return T_ECEF_to_ENU(longitude_rad, latitude_rad).T


# ---------------------------------------------------------------------------
# 3. ENU → Body   R(q)  via unit quaternion
# ---------------------------------------------------------------------------

def T_ENU_to_body(quaternion: np.ndarray) -> DCM:
    """
    Direction Cosine Matrix (DCM) rotating the ENU frame to the
    vehicle Body Frame, derived from a unit quaternion.

    The quaternion encodes the vehicle's attitude relative to the local
    ENU frame. At q = [1, 0, 0, 0] (identity), the body axes are aligned
    with ENU (X_body = East, Y_body = North, Z_body = Up — note that
    Z_body = Up means Z_body is inverted relative to the NED-down convention;
    the NED-to-body final flip is applied separately in full 6-DoF pipelines
    but is NOT included here to keep this function general).

    Parameters
    ----------
    quaternion : ndarray, shape (4,), dtype float64
        Unit quaternion (q0, q1, q2, q3) — scalar-first Hamilton convention.
        ‖q‖ must equal 1.0 (not enforced here; enforced by VehicleState).

    Returns
    -------
    DCM : ndarray, shape (3, 3), dtype float64

    Matrix definition
    -----------------
    q = (q0, q1, q2, q3)

    R(q) = ⎡ 1−2(q2²+q3²)    2(q1q2−q0q3)    2(q1q3+q0q2) ⎤
           ⎢ 2(q1q2+q0q3)   1−2(q1²+q3²)     2(q2q3−q0q1) ⎥
           ⎣ 2(q1q3−q0q2)    2(q2q3+q0q1)   1−2(q1²+q2²) ⎦

    This is the standard Shepperd/Diebel DCM from a Hamilton unit quaternion.
    """
    q = quaternion.astype(np.float64)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    q0q0, q1q1, q2q2, q3q3 = q0*q0, q1*q1, q2*q2, q3*q3
    q0q1, q0q2, q0q3       = q0*q1, q0*q2, q0*q3
    q1q2, q1q3, q2q3       = q1*q2, q1*q3, q2*q3

    return np.array([
        [1.0 - 2.0*(q2q2 + q3q3),  2.0*(q1q2 - q0q3),  2.0*(q1q3 + q0q2)],
        [      2.0*(q1q2 + q0q3),  1.0 - 2.0*(q1q1 + q3q3),  2.0*(q2q3 - q0q1)],
        [      2.0*(q1q3 - q0q2),  2.0*(q2q3 + q0q1),  1.0 - 2.0*(q1q1 + q2q2)],
    ], dtype=np.float64)


def T_body_to_ENU(quaternion: np.ndarray) -> DCM:
    """
    Inverse of T_ENU_to_body — rotate Body Frame vector to ENU.

    Mathematically: T_body→ENU = T_ENU→body(q)ᵀ = R(q)ᵀ = R(q⁻¹)
    """
    return T_ENU_to_body(quaternion).T


# ---------------------------------------------------------------------------
# 4. Composed transformations (ECI → Body and inverses)
# ---------------------------------------------------------------------------

def T_ECI_to_body(
    quaternion: np.ndarray,
    longitude_rad: float,
    latitude_rad: float,
    time: float,
    omega: float = EARTH_OMEGA,
) -> DCM:
    """
    Full ECI → Body frame transformation (composed product).

    T_ECI→Body = T_ENU→Body(q) @ T_ECEF→ENU(λ, φ) @ T_ECI→ECEF(t)

    Parameters
    ----------
    quaternion : ndarray, shape (4,)
        Vehicle attitude quaternion in ENU frame (scalar-first).
    longitude_rad : float
        Surface longitude of reference point [rad].
    latitude_rad : float
        Surface latitude of reference point [rad].
    time : float
        Elapsed simulation time [s].
    omega : float
        Planet rotation rate [rad s⁻¹].

    Returns
    -------
    DCM : ndarray, shape (3, 3), dtype float64
    """
    R_eci_to_ecef = T_ECI_to_ECEF(time, omega)
    R_ecef_to_enu = T_ECEF_to_ENU(longitude_rad, latitude_rad)
    R_enu_to_body = T_ENU_to_body(quaternion)
    return R_enu_to_body @ R_ecef_to_enu @ R_eci_to_ecef


def T_body_to_ECI(
    quaternion: np.ndarray,
    longitude_rad: float,
    latitude_rad: float,
    time: float,
    omega: float = EARTH_OMEGA,
) -> DCM:
    """
    Full Body → ECI frame transformation.

    T_Body→ECI = T_ECI→Body(...)ᵀ
    """
    return T_ECI_to_body(quaternion, longitude_rad, latitude_rad, time, omega).T


# ---------------------------------------------------------------------------
# 5. Quaternion kinematics matrix Ξ(q)
# ---------------------------------------------------------------------------

def xi_matrix(quaternion: np.ndarray) -> np.ndarray:
    """
    The 4×3 quaternion kinematic matrix Ξ(q) used in the attitude ODE:

        dq/dt = ½ · Ξ(q) · ω_body

    where ω_body = [p, q, r]ᵀ is the angular velocity in the Body Frame.

    Parameters
    ----------
    quaternion : ndarray, shape (4,)
        Unit quaternion (q0, q1, q2, q3), scalar-first.

    Returns
    -------
    Xi : ndarray, shape (4, 3), dtype float64

    Matrix definition
    -----------------
              ⎡ −q1   −q2   −q3 ⎤
    Ξ(q) =   ⎢  q0   −q3    q2 ⎥
              ⎢  q3    q0   −q1 ⎥
              ⎣ −q2    q1    q0 ⎦

    Reference: Diebel (2006), "Representing Attitude", §5.4
    """
    q = quaternion.astype(np.float64)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    return np.array([
        [-q1, -q2, -q3],
        [ q0, -q3,  q2],
        [ q3,  q0, -q1],
        [-q2,  q1,  q0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# 6. Euler angles ↔ quaternion (display use only — NOT used internally)
# ---------------------------------------------------------------------------

def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert ZYX Tait-Bryan Euler angles to a unit quaternion.

    Rotation order: yaw (ψ) first, then pitch (θ), then roll (φ).
    This is the standard aerospace ZYX 3-2-1 sequence.

    Parameters
    ----------
    roll : float   [rad]  φ — rotation about X_body
    pitch : float  [rad]  θ — rotation about Y_body
    yaw : float    [rad]  ψ — rotation about Z_body

    Returns
    -------
    q : ndarray, shape (4,), dtype float64, ‖q‖ = 1
        (q0, q1, q2, q3) — scalar-first.

    WARNING: Euler angles suffer from gimbal lock at pitch = ±90°.
    This function is provided for initial condition setup and display
    only. The integrator operates exclusively on quaternions.
    """
    cr, sr = math.cos(roll  * 0.5), math.sin(roll  * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw   * 0.5), math.sin(yaw   * 0.5)

    q0 = cr*cp*cy + sr*sp*sy
    q1 = sr*cp*cy - cr*sp*sy
    q2 = cr*sp*cy + sr*cp*sy
    q3 = cr*cp*sy - sr*sp*cy

    return np.array([q0, q1, q2, q3], dtype=np.float64)


def quaternion_to_euler(quaternion: np.ndarray) -> tuple[float, float, float]:
    """
    Convert a unit quaternion to ZYX Tait-Bryan Euler angles.

    Returns
    -------
    (roll, pitch, yaw) : tuple of float, all in [rad]

    WARNING: Gimbal lock occurs at pitch = ±π/2. Output is undefined there.
    For display use only — never feed these back into the integrator.
    """
    q = quaternion.astype(np.float64)
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

    # Roll (φ)
    sinr_cosp = 2.0 * (q0*q1 + q2*q3)
    cosr_cosp = 1.0 - 2.0 * (q1*q1 + q2*q2)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (θ) — clamp for numerical safety at gimbal lock
    sinp = 2.0 * (q0*q2 - q3*q1)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # Yaw (ψ)
    siny_cosp = 2.0 * (q0*q3 + q1*q2)
    cosy_cosp = 1.0 - 2.0 * (q2*q2 + q3*q3)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


# ---------------------------------------------------------------------------
# 7. Validation utility
# ---------------------------------------------------------------------------

def assert_dcm_orthogonal(dcm: DCM, name: str = "DCM") -> None:
    """
    Assert that a DCM is orthogonal: ‖T @ Tᵀ − I‖_F ≤ TRANSFORM_IDENTITY_TOL.

    Raises
    ------
    AssertionError if the Frobenius norm of (T @ Tᵀ − I₃) exceeds tolerance.
    """
    residual = dcm @ dcm.T - np.eye(3, dtype=np.float64)
    frob = float(np.linalg.norm(residual, "fro"))
    assert frob <= TRANSFORM_IDENTITY_TOL, (
        f"{name} orthogonality check failed: "
        f"‖T·Tᵀ − I‖_F = {frob:.3e} > {TRANSFORM_IDENTITY_TOL:.3e}"
    )