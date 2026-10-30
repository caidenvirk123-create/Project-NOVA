"""
nova.physics.forces
===================
Force tensor accumulator for Project NOVA.

Architecture role — Pipeline Stage 3 (partial)
-----------------------------------------------
The ForceAccumulator gathers every external force acting on the vehicle
during one simulation tick and produces a single net force vector in the
ECI frame for consumption by the RK4 integrator (Stage 4).

Design principles
-----------------
* **Single responsibility**: this module only *accumulates* forces. It does
  not integrate, does not modify state, and does not own physics models.
  Gravity, aerodynamics, and propulsion each live in their own modules;
  they produce force vectors that are registered here.

* **Frame contract**: all forces are converted to ECI before registration.
  The integrator receives one ECI vector — it never sees Body-frame forces
  directly. Coordinate transforms are applied inside each registration
  method, not at integration time.

* **Immutable per-tick snapshot**: once ``build()`` is called the result is
  a frozen (3,) float64 array. The accumulator object may then be discarded;
  a fresh one is constructed each tick.

Force taxonomy
--------------
  Category A — Gravitational  : point-mass + J2  (ECI; from orbital.py)
  Category B — Aerodynamic    : lift, drag, side  (Body → ECI via DCM)
  Category C — Propulsive     : thrust vector     (Body → ECI via DCM)
  Category D — Custom         : any additional ECI or Body-frame vector

All SI units: force in Newtons [N], vectors float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from nova.core.state_vector import VehicleState
from nova.frames.transforms import T_ENU_to_body


# ---------------------------------------------------------------------------
# Force contribution record (for telemetry / AI monitor inspection)
# ---------------------------------------------------------------------------

@dataclass
class ForceContribution:
    """
    A single named force contribution in the ECI frame.

    Stored alongside the accumulated total so the telemetry layer and AI
    monitor can inspect per-category force magnitudes without re-computing.

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"gravity"``, ``"drag"``, ``"thrust"``).
    vector_eci : ndarray, shape (3,), float64
        Force vector in ECI frame [N].
    """
    name:       str
    vector_eci: np.ndarray   # (3,) float64, [N]


# ---------------------------------------------------------------------------
# Force accumulator
# ---------------------------------------------------------------------------

class ForceAccumulator:
    """
    Mutable per-tick accumulator that collects all external forces acting on
    the vehicle and produces a net ECI force vector for the RK4 integrator.

    Usage pattern (called by the pipeline each tick)::

        acc = ForceAccumulator(state)
        acc.add_gravity(gravity_force(state.position_eci, state.mass))
        acc.add_body_force("thrust", thrust_vec_body, dcm_body_to_eci)
        acc.add_body_force("aero",   aero_vec_body,   dcm_body_to_eci)
        F_net_eci = acc.build()

    Parameters
    ----------
    state : VehicleState
        Current immutable state snapshot. Stored for reference (mass, DCM).
    """

    def __init__(self, state: VehicleState) -> None:
        self._state: VehicleState = state
        self._contributions: List[ForceContribution] = []
        self._total_eci: np.ndarray = np.zeros(3, dtype=np.float64)
        self._built: bool = False

    # ------------------------------------------------------------------
    # Registration methods
    # ------------------------------------------------------------------

    def add_eci_force(self, name: str, vector_eci: np.ndarray) -> None:
        """
        Register a force already expressed in the ECI frame.

        Parameters
        ----------
        name : str
            Contribution label for telemetry.
        vector_eci : ndarray, shape (3,), float64
            Force vector in ECI [N].
        """
        self._check_open()
        v = np.asarray(vector_eci, dtype=np.float64)
        _assert_vec3(v, name)
        self._contributions.append(ForceContribution(name, v.copy()))
        self._total_eci += v

    def add_body_force(
        self,
        name: str,
        vector_body: np.ndarray,
        dcm_body_to_eci: np.ndarray,
    ) -> None:
        """
        Register a force expressed in the Body frame.

        The provided DCM rotates the vector from Body to ECI before
        accumulation.  Typically ``dcm_body_to_eci = T_body_to_ECI(q, ...)``.

        Parameters
        ----------
        name : str
            Contribution label.
        vector_body : ndarray, shape (3,), float64
            Force vector in Body frame [N].
        dcm_body_to_eci : ndarray, shape (3, 3), float64
            Rotation matrix Body → ECI.
        """
        self._check_open()
        v_body = np.asarray(vector_body, dtype=np.float64)
        R      = np.asarray(dcm_body_to_eci, dtype=np.float64)
        _assert_vec3(v_body, name)
        _assert_dcm33(R, name)
        v_eci = R @ v_body
        self._contributions.append(ForceContribution(name, v_eci.copy()))
        self._total_eci += v_eci

    def add_gravity(self, gravity_force_eci: np.ndarray) -> None:
        """
        Convenience wrapper: register a gravitational force (ECI).

        Parameters
        ----------
        gravity_force_eci : ndarray, shape (3,), float64
            Output of ``nova.physics.orbital.gravity_force(...)`` [N].
        """
        self.add_eci_force("gravity", gravity_force_eci)

    def add_thrust(
        self,
        thrust_vector_body: np.ndarray,
        dcm_body_to_eci: np.ndarray,
    ) -> None:
        """
        Convenience wrapper: register engine thrust (Body frame).

        Parameters
        ----------
        thrust_vector_body : ndarray, shape (3,), float64
            Thrust vector in Body frame [N]. Typically aligned with +X_body
            for an axial engine; off-axis for gimbal or RCS.
        dcm_body_to_eci : ndarray, shape (3, 3), float64
            Rotation matrix Body → ECI.
        """
        self.add_body_force("thrust", thrust_vector_body, dcm_body_to_eci)

    def add_aerodynamic(
        self,
        aero_force_body: np.ndarray,
        dcm_body_to_eci: np.ndarray,
    ) -> None:
        """
        Convenience wrapper: register net aerodynamic force (Body frame).

        The aerodynamic force vector in the Body frame is the vector sum of
        lift, drag, and side-force already resolved from the stability axis
        by the aerodynamics module.

        Parameters
        ----------
        aero_force_body : ndarray, shape (3,), float64
            Net aero force in Body frame [N].
        dcm_body_to_eci : ndarray, shape (3, 3), float64
            Rotation matrix Body → ECI.
        """
        self.add_body_force("aerodynamic", aero_force_body, dcm_body_to_eci)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> np.ndarray:
        """
        Finalise and return the net force vector in ECI [N].

        Marks the accumulator as closed — further ``add_*`` calls will raise.

        Returns
        -------
        ndarray, shape (3,), float64
            Net external force in ECI frame [N].
        """
        self._built = True
        return self._total_eci.copy()

    # ------------------------------------------------------------------
    # Telemetry inspection
    # ------------------------------------------------------------------

    @property
    def contributions(self) -> List[ForceContribution]:
        """Read-only list of all registered force contributions."""
        return list(self._contributions)

    @property
    def total_magnitude(self) -> float:
        """‖F_net‖ [N]."""
        return float(np.linalg.norm(self._total_eci))

    def contribution_by_name(self, name: str) -> Optional[np.ndarray]:
        """
        Return the ECI force vector for a named contribution, or None.

        Parameters
        ----------
        name : str
            Contribution label as passed to ``add_*``.
        """
        for c in self._contributions:
            if c.name == name:
                return c.vector_eci.copy()
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._built:
            raise RuntimeError(
                "ForceAccumulator.build() has already been called. "
                "Create a new accumulator for the next tick."
            )

    def __repr__(self) -> str:
        n = len(self._contributions)
        return (
            f"ForceAccumulator("
            f"contributions={n}, "
            f"|F_net|={self.total_magnitude:.3f} N, "
            f"built={self._built})"
        )


# ---------------------------------------------------------------------------
# Functional helpers (stateless — used in tests and by the pipeline)
# ---------------------------------------------------------------------------

def net_force_eci(contributions: List[ForceContribution]) -> np.ndarray:
    """
    Sum a list of ForceContribution records into one ECI vector.

    Parameters
    ----------
    contributions : list of ForceContribution

    Returns
    -------
    ndarray, shape (3,), float64  [N]
    """
    total = np.zeros(3, dtype=np.float64)
    for c in contributions:
        total += c.vector_eci
    return total


def rotate_body_to_eci(
    vector_body: np.ndarray,
    quaternion: np.ndarray,
) -> np.ndarray:
    """
    Rotate a Body-frame vector to ECI using the vehicle's attitude quaternion.

    This is a convenience wrapper for the common pattern of converting a
    Body-frame force to ECI without constructing the full composed transform
    (which would also require longitude/latitude/time for ECEF→ECI).

    For the force accumulator, only the *attitude* rotation (ENU→Body) is
    reversed; the ECI→ECEF→ENU chain cancels because forces accumulate
    in ECI and the ECI axes are inertial.

    In practice: R_body→ECI = R_ENU→Body(q)ᵀ  (ignoring ECEF rotation,
    which is negligible for force magnitudes over one tick at dt=0.01 s).

    Parameters
    ----------
    vector_body : ndarray, shape (3,), float64
        Vector expressed in Body frame.
    quaternion : ndarray, shape (4,), float64
        Vehicle attitude quaternion (scalar-first, unit norm).

    Returns
    -------
    ndarray, shape (3,), float64
        Vector expressed in ECI frame (approx — ignores ECEF rotation).
    """
    R_enu_to_body = T_ENU_to_body(quaternion)
    R_body_to_enu = R_enu_to_body.T          # orthogonal → transpose = inverse
    # ENU ≈ ECI for force direction over one tick; full chain in pipeline.
    return R_body_to_enu @ np.asarray(vector_body, dtype=np.float64)


# ---------------------------------------------------------------------------
# Internal validation helpers
# ---------------------------------------------------------------------------

def _assert_vec3(v: np.ndarray, name: str) -> None:
    if v.shape != (3,):
        raise ValueError(
            f"Force '{name}' must be shape (3,), got {v.shape}"
        )


def _assert_dcm33(R: np.ndarray, name: str) -> None:
    if R.shape != (3, 3):
        raise ValueError(
            f"DCM for force '{name}' must be shape (3, 3), got {R.shape}"
        )