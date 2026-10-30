"""
nova.physics.torques
====================
Torque tensor accumulator for Project NOVA.

Architecture role — Pipeline Stage 3 (partial)
-----------------------------------------------
The TorqueAccumulator mirrors the ForceAccumulator but operates on torques
(moments), which drive the rotational equations of motion in the RK4
integrator via Euler's rotation equations:

    I ω̇ = τ_net − ω × (I ω)

All torques are accumulated in the **Body Frame** because the inertia
tensor I is most naturally expressed in body-fixed coordinates. The RK4
integrator receives the Body-frame τ_net directly.

Torque taxonomy
---------------
  Category A — Aerodynamic moments  : pitching, yawing, rolling moments
                                       applied at the Aerodynamic Centre (AC)
                                       relative to the Centre of Mass (CoM).
  Category B — Thrust gimbal        : moment arm × thrust force for gimballed
                                       engines or offset RCS thrusters.
  Category C — Gravity gradient     : tidal torque for large structures in LEO.
  Category D — Custom               : any additional Body-frame torque vector.

All moments are in Newton-metres [N·m], vectors float64.

Sign convention
---------------
Right-hand rule in the Body Frame:
  +τ_x  → rolling  moment (right-wing down for positive roll)
  +τ_y  → pitching moment (nose up for positive pitch)
  +τ_z  → yawing   moment (nose right for positive yaw)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Torque contribution record
# ---------------------------------------------------------------------------

@dataclass
class TorqueContribution:
    """
    A single named torque contribution in the Body frame [N·m].

    Attributes
    ----------
    name : str
        Human-readable label (e.g. ``"aero_pitch"``, ``"gimbal"``).
    vector_body : ndarray, shape (3,), float64
        Torque vector in Body frame [N·m]. Components: (roll, pitch, yaw).
    """
    name:        str
    vector_body: np.ndarray   # (3,) float64, [N·m]


# ---------------------------------------------------------------------------
# Torque accumulator
# ---------------------------------------------------------------------------

class TorqueAccumulator:
    """
    Mutable per-tick accumulator that collects all external torques acting on
    the vehicle and produces a net Body-frame torque vector for the RK4
    integrator.

    Usage pattern (called by the pipeline each tick)::

        tacc = TorqueAccumulator(state)
        tacc.add_body_torque("aero_pitch", np.array([0, M_pitch, 0]))
        tacc.add_moment_arm("gimbal", thrust_body, moment_arm_body)
        tau_net_body = tacc.build()

    Parameters
    ----------
    state : VehicleState
        Current immutable state snapshot (stored for reference).
    """

    def __init__(self, state) -> None:
        self._state = state
        self._contributions: List[TorqueContribution] = []
        self._total_body: np.ndarray = np.zeros(3, dtype=np.float64)
        self._built: bool = False

    # ------------------------------------------------------------------
    # Registration methods
    # ------------------------------------------------------------------

    def add_body_torque(self, name: str, vector_body: np.ndarray) -> None:
        """
        Register a torque already expressed in the Body frame.

        Parameters
        ----------
        name : str
            Contribution label for telemetry.
        vector_body : ndarray, shape (3,), float64
            Torque vector in Body frame [N·m].
            Components: (τ_roll, τ_pitch, τ_yaw).
        """
        self._check_open()
        v = np.asarray(vector_body, dtype=np.float64)
        _assert_vec3(v, name)
        self._contributions.append(TorqueContribution(name, v.copy()))
        self._total_body += v

    def add_moment_arm(
        self,
        name: str,
        force_body: np.ndarray,
        moment_arm_body: np.ndarray,
    ) -> None:
        """
        Register a torque from a force applied at an offset from the CoM.

        τ = r × F   (cross product in Body frame)

        Used for:
          - Gimballed engine thrust offset from CoM
          - RCS thruster moment arms
          - Aerodynamic force at AC offset from CoM

        Parameters
        ----------
        name : str
            Contribution label.
        force_body : ndarray, shape (3,), float64
            Applied force in Body frame [N].
        moment_arm_body : ndarray, shape (3,), float64
            Vector from CoM to the point of force application,
            expressed in Body frame [m].
            τ = moment_arm × force
        """
        self._check_open()
        F = np.asarray(force_body,      dtype=np.float64)
        r = np.asarray(moment_arm_body, dtype=np.float64)
        _assert_vec3(F, f"{name}/force")
        _assert_vec3(r, f"{name}/moment_arm")
        tau = np.cross(r, F)
        self._contributions.append(TorqueContribution(name, tau.copy()))
        self._total_body += tau

    def add_aerodynamic_moments(
        self,
        pitching_moment: float,
        yawing_moment:   float,
        rolling_moment:  float,
    ) -> None:
        """
        Register the three aerodynamic moment components.

        These are the dimensionalised moments (already multiplied by
        dynamic pressure, reference area, and reference length) computed
        by the aerodynamics module.

        Parameters
        ----------
        pitching_moment : float
            M_pitch [N·m] — positive nose-up.
        yawing_moment : float
            M_yaw [N·m] — positive nose-right.
        rolling_moment : float
            M_roll [N·m] — positive right-wing-down.
        """
        tau = np.array([rolling_moment, pitching_moment, yawing_moment],
                       dtype=np.float64)
        self.add_body_torque("aerodynamic", tau)

    def add_gravity_gradient(
        self,
        position_body: np.ndarray,
        inertia_tensor: np.ndarray,
        mu: float,
    ) -> None:
        """
        Gravity gradient torque for an extended body in a non-uniform
        gravitational field.

        τ_gg = (3μ/r⁵) · r_body × (I · r_body)

        where r_body is the position vector in Body frame and I is the
        inertia tensor.

        This effect is significant for large spacecraft (e.g. space stations,
        solar arrays) in LEO but negligible for compact launch vehicles.
        Include only when the vehicle's largest dimension exceeds ~10 m.

        Parameters
        ----------
        position_body : ndarray, shape (3,), float64
            Vehicle position vector expressed in the Body frame [m].
        inertia_tensor : ndarray, shape (3, 3), float64
            Inertia tensor in Body frame [kg·m²].
        mu : float
            Gravitational parameter [m³ s⁻²].
        """
        self._check_open()
        r = np.asarray(position_body,  dtype=np.float64)
        I = np.asarray(inertia_tensor, dtype=np.float64)
        r_mag = float(np.linalg.norm(r))
        if r_mag < 1.0:
            return   # degenerate — skip
        coeff = 3.0 * mu / r_mag**5
        tau   = coeff * np.cross(r, I @ r)
        self._contributions.append(TorqueContribution("gravity_gradient", tau.copy()))
        self._total_body += tau

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> np.ndarray:
        """
        Finalise and return the net torque vector in Body frame [N·m].

        Marks the accumulator as closed.

        Returns
        -------
        ndarray, shape (3,), float64
            Net external torque in Body frame [N·m].
            Components: (τ_roll, τ_pitch, τ_yaw).
        """
        self._built = True
        return self._total_body.copy()

    # ------------------------------------------------------------------
    # Telemetry inspection
    # ------------------------------------------------------------------

    @property
    def contributions(self) -> List[TorqueContribution]:
        """Read-only list of all registered torque contributions."""
        return list(self._contributions)

    @property
    def total_magnitude(self) -> float:
        """‖τ_net‖ [N·m]."""
        return float(np.linalg.norm(self._total_body))

    def contribution_by_name(self, name: str) -> Optional[np.ndarray]:
        """
        Return the Body-frame torque vector for a named contribution, or None.
        """
        for c in self._contributions:
            if c.name == name:
                return c.vector_body.copy()
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self._built:
            raise RuntimeError(
                "TorqueAccumulator.build() has already been called. "
                "Create a new accumulator for the next tick."
            )

    def __repr__(self) -> str:
        n = len(self._contributions)
        return (
            f"TorqueAccumulator("
            f"contributions={n}, "
            f"|τ_net|={self.total_magnitude:.3f} N·m, "
            f"built={self._built})"
        )


# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

def _assert_vec3(v: np.ndarray, name: str) -> None:
    if v.shape != (3,):
        raise ValueError(
            f"Torque vector '{name}' must be shape (3,), got {v.shape}"
        )