"""
nova.core.state_vector
======================
Defines the canonical, immutable VehicleState — the complete 13-element
state vector propagated by the RK4 integrator at every simulation tick.

Design constraints
------------------
* Frozen dataclass: no field may be mutated after construction. The
  integrator produces a *new* VehicleState each tick; it never modifies
  an existing one. This enforces the deterministic, race-condition-free
  pipeline defined in the architecture.

* All vectors are NumPy arrays with dtype=np.float64 (64-bit IEEE 754).
  This is non-negotiable: float32 accumulates ~1 ULP/step error that
  violates the 1×10⁻⁶ conservation tolerances at simulation timescales.

* Angles are stored in radians. The quaternion `q` encodes orientation
  as (q₀, q₁, q₂, q₃) = (scalar, i, j, k).  ‖q‖ = 1 is maintained
  by the integrator after every RK4 step.

* The state is frame-tagged. `position_eci` and `velocity_eci` live in
  the Earth-Centred Inertial frame. Body-frame quantities are derivable
  via the coordinate transforms in nova.frames.transforms.

State vector layout (13 scalars → stored as named array fields)
---------------------------------------------------------------
  [0:3]   position_eci     [m]          ECI Cartesian position
  [3:6]   velocity_eci     [m s⁻¹]      ECI Cartesian velocity
  [6:10]  quaternion       [-]          q = (q0, q1, q2, q3), ‖q‖ = 1
  [10:13] omega_body       [rad s⁻¹]    Angular velocity in Body Frame (p, q, r)

Additional scalar fields (not part of the ODE state, but carried in the
snapshot for telemetry):
  mass                     [kg]         Current total vehicle mass
  time                     [s]          Mission elapsed time
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from nova.core.constants import QUATERNION_NORM_TOL


# ---------------------------------------------------------------------------
# VehicleState — frozen, immutable simulation state snapshot
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleState:
    """
    Complete rigid-body state of the simulated vehicle at one instant in time.

    All vectors are stored as 1-D NumPy float64 arrays. The dataclass is
    frozen so any attempted mutation raises ``FrozenInstanceError`` at
    runtime — enforcing the immutable-snapshot contract required by the
    deterministic pipeline.

    Parameters
    ----------
    position_eci : ndarray, shape (3,), dtype float64
        Cartesian position in the ECI frame [m].
    velocity_eci : ndarray, shape (3,), dtype float64
        Cartesian velocity in the ECI frame [m s⁻¹].
    quaternion : ndarray, shape (4,), dtype float64
        Orientation quaternion (q0, q1, q2, q3) — scalar part first.
        Must satisfy ‖q‖ ≈ 1.0 within QUATERNION_NORM_TOL.
    omega_body : ndarray, shape (3,), dtype float64
        Angular velocity vector expressed in the Body Frame [rad s⁻¹].
        Components are (p, q, r) — roll rate, pitch rate, yaw rate.
    mass : float
        Total vehicle mass at this instant [kg]. Must be > 0.
    time : float
        Mission elapsed time (MET) at this state [s]. Must be ≥ 0.
    """

    position_eci: np.ndarray   # shape (3,), float64, [m]
    velocity_eci: np.ndarray   # shape (3,), float64, [m s⁻¹]
    quaternion:   np.ndarray   # shape (4,), float64, unit quaternion
    omega_body:   np.ndarray   # shape (3,), float64, [rad s⁻¹]
    mass:         float        # [kg], > 0
    time:         float        # [s], MET ≥ 0

    # ------------------------------------------------------------------
    # Post-init validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate shapes, dtypes, and physical constraints on construction."""

        # Enforce numpy array types and shapes
        _assert_vector(self.position_eci, shape=(3,), name="position_eci")
        _assert_vector(self.velocity_eci, shape=(3,), name="velocity_eci")
        _assert_vector(self.quaternion,   shape=(4,), name="quaternion")
        _assert_vector(self.omega_body,   shape=(3,), name="omega_body")

        # Physical constraints
        if self.mass <= 0.0:
            raise ValueError(
                f"VehicleState.mass must be > 0 kg, got {self.mass!r}"
            )
        if self.time < 0.0:
            raise ValueError(
                f"VehicleState.time must be ≥ 0 s (MET), got {self.time!r}"
            )

        # Quaternion unit-norm check
        q_norm = float(np.linalg.norm(self.quaternion))
        if abs(q_norm - 1.0) > QUATERNION_NORM_TOL:
            raise ValueError(
                f"VehicleState.quaternion must be a unit quaternion "
                f"(‖q‖ = 1.0 ± {QUATERNION_NORM_TOL}), "
                f"got ‖q‖ = {q_norm:.15f}"
            )

    # ------------------------------------------------------------------
    # Convenience properties (derived, zero-allocation)
    # ------------------------------------------------------------------

    @property
    def speed(self) -> float:
        """Magnitude of ECI velocity vector [m s⁻¹]."""
        return float(np.linalg.norm(self.velocity_eci))

    @property
    def altitude_eci(self) -> float:
        """
        Distance from ECI origin (planetary barycenter) [m].

        Note: this is the ECI radius, NOT geodetic altitude above the
        surface. Use the frame transforms module to obtain geodetic altitude.
        """
        return float(np.linalg.norm(self.position_eci))

    @property
    def kinetic_energy(self) -> float:
        """
        Translational kinetic energy in the ECI frame [J].

        E_k = ½ m v²

        Does not include rotational kinetic energy (requires inertia tensor
        from the vehicle component graph — not stored in the base state).
        """
        v_sq = float(np.dot(self.velocity_eci, self.velocity_eci))
        return 0.5 * self.mass * v_sq

    @property
    def angular_momentum_body(self) -> np.ndarray:
        """
        Angular momentum direction in Body Frame [rad s⁻¹].

        Returns omega_body directly. The full angular momentum vector
        H = I · ω requires the inertia tensor I, which lives in the
        mass model — this property returns the ω component only.
        """
        return self.omega_body.copy()

    # ------------------------------------------------------------------
    # Flat state array interface (for RK4 integrator)
    # ------------------------------------------------------------------

    def to_flat(self) -> np.ndarray:
        """
        Serialise to a flat 13-element float64 array for ODE integration.

        Layout: [pos(3), vel(3), quat(4), omega(3)]

        The integrator operates on this flat representation; the result is
        deserialised back to a VehicleState via ``VehicleState.from_flat``.
        """
        return np.concatenate([
            self.position_eci,
            self.velocity_eci,
            self.quaternion,
            self.omega_body,
        ], dtype=np.float64)

    @staticmethod
    def from_flat(
        flat: np.ndarray,
        mass: float,
        time: float,
        normalize_quaternion: bool = True,
    ) -> "VehicleState":
        """
        Deserialise from a flat 13-element array produced by ``to_flat``.

        Parameters
        ----------
        flat : ndarray, shape (13,)
            Flat state vector in the canonical layout.
        mass : float
            Vehicle mass at this instant [kg]. Must be supplied externally
            because mass is not part of the ODE state (it changes via the
            propulsion model, not by integration).
        time : float
            Mission elapsed time [s].
        normalize_quaternion : bool
            If True (default), re-normalise the quaternion before
            constructing the state. Always True during integration to
            prevent norm drift across RK4 steps.

        Returns
        -------
        VehicleState
        """
        if flat.shape != (13,):
            raise ValueError(
                f"from_flat expects shape (13,), got {flat.shape}"
            )

        pos  = flat[0:3].astype(np.float64)
        vel  = flat[3:6].astype(np.float64)
        quat = flat[6:10].astype(np.float64)
        omg  = flat[10:13].astype(np.float64)

        if normalize_quaternion:
            q_norm = np.linalg.norm(quat)
            if q_norm < 1.0e-15:
                raise ValueError(
                    "Quaternion norm collapsed to zero — state is degenerate."
                )
            quat = quat / q_norm

        return VehicleState(
            position_eci=pos,
            velocity_eci=vel,
            quaternion=quat,
            omega_body=omg,
            mass=mass,
            time=time,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def with_time(self, new_time: float) -> "VehicleState":
        """Return a new VehicleState identical to this one but at ``new_time``."""
        return VehicleState(
            position_eci=self.position_eci.copy(),
            velocity_eci=self.velocity_eci.copy(),
            quaternion=self.quaternion.copy(),
            omega_body=self.omega_body.copy(),
            mass=self.mass,
            time=new_time,
        )

    def __repr__(self) -> str:
        r = float(self.altitude_eci)
        v = self.speed
        return (
            f"VehicleState("
            f"t={self.time:.3f}s, "
            f"|r|={r:.1f}m, "
            f"|v|={v:.3f}m/s, "
            f"m={self.mass:.3f}kg)"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_state(
    *,
    position_eci: "ArrayLike",
    velocity_eci: "ArrayLike",
    quaternion:   "ArrayLike",
    omega_body:   "ArrayLike",
    mass:         float,
    time:         float = 0.0,
) -> VehicleState:
    """
    Convenience constructor that coerces inputs to float64 arrays.

    All positional arrays may be Python lists, tuples, or ndarrays;
    they are coerced to float64 and validated on construction.

    Example
    -------
    >>> from nova.core.state_vector import make_state
    >>> import numpy as np
    >>> state = make_state(
    ...     position_eci=[6_571_000.0, 0.0, 0.0],   # 200 km LEO
    ...     velocity_eci=[0.0, 7_784.0, 0.0],
    ...     quaternion=[1.0, 0.0, 0.0, 0.0],         # identity
    ...     omega_body=[0.0, 0.0, 0.0],
    ...     mass=1000.0,
    ... )
    """
    return VehicleState(
        position_eci=np.asarray(position_eci, dtype=np.float64),
        velocity_eci=np.asarray(velocity_eci, dtype=np.float64),
        quaternion=np.asarray(quaternion,   dtype=np.float64),
        omega_body=np.asarray(omega_body,   dtype=np.float64),
        mass=float(mass),
        time=float(time),
    )


def identity_state(
    position_eci: "ArrayLike" = (6_571_000.0, 0.0, 0.0),
    velocity_eci: "ArrayLike" = (0.0, 7_784.0, 0.0),
    mass: float = 1000.0,
) -> VehicleState:
    """
    Return a zero-rotation, zero-angular-velocity state at the given
    position and velocity. Useful for unit tests that only need to exercise
    one dimension of the state.

    Default position is a nominal 200 km LEO pass-through (not a valid
    circular orbit at these numbers — use OrbitalSolver for precise ICs).
    """
    return make_state(
        position_eci=position_eci,
        velocity_eci=velocity_eci,
        quaternion=[1.0, 0.0, 0.0, 0.0],
        omega_body=[0.0, 0.0, 0.0],
        mass=mass,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _assert_vector(arr: object, shape: tuple, name: str) -> None:
    """Raise TypeError/ValueError if ``arr`` is not a float64 array of ``shape``."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"VehicleState.{name} must be a numpy.ndarray, "
            f"got {type(arr).__name__}"
        )
    if arr.dtype != np.float64:
        raise TypeError(
            f"VehicleState.{name} must have dtype float64, "
            f"got {arr.dtype}"
        )
    if arr.shape != shape:
        raise ValueError(
            f"VehicleState.{name} must have shape {shape}, "
            f"got {arr.shape}"
        )