"""
nova.vehicle.mass_model
=======================
Dynamic mass and inertia model for Project NOVA.

Architecture role — Pipeline Stage 9 (Component Updates) + Stage 3 input
-------------------------------------------------------------------------
Tracks the vehicle's total mass, centre of mass (CoM) position in the Body
frame, and inertia tensor as propellant is depleted and structural components
change state.

The inertia tensor is required by the RK4 integrator (via ``build_deriv_fn``)
and by the torque accumulator (gravity gradient). The CoM position is required
by the force/torque accumulators to compute moment arms correctly.

Physics
-------
For N point-mass or solid-body components, each with mass mᵢ and CoM
position rᵢ in a shared reference frame:

  m_total = Σ mᵢ

  r_CoM   = (1/m_total) · Σ mᵢ rᵢ         (in Body Frame)

  I_total = Σ [Iᵢ_own + mᵢ · D(rᵢ − r_CoM)]

where D(r) is the parallel-axis (Huygens-Steiner) displacement tensor:
  D(r)ₙₙ = |r|² − rₙ²    (diagonal)
  D(r)ₘₙ = −rₘ rₙ        (off-diagonal)

This gives the full symmetric 3×3 inertia tensor with parallel-axis
correction applied component by component.

Design notes
------------
* MassComponent is a lightweight dataclass — no external imports needed.
* MassModel is rebuilt from scratch each tick by the pipeline (Component
  Updates, Stage 9) after applying propellant depletion. This avoids
  accumulated floating-point state drift.
* All positions are in the Body Frame, which is CoM-centred by definition.
  Therefore r_CoM should be near zero in a well-designed vehicle; any
  residual offset is a CoM shift due to asymmetric propellant depletion.

References
----------
- Goldstein, "Classical Mechanics", 3rd ed., §5.3
- Kane & Levinson, "Dynamics: Theory and Applications", §4.4
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Individual mass component
# ---------------------------------------------------------------------------

@dataclass
class MassComponent:
    """
    A single rigid-body mass component contributing to the vehicle mass model.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"LOX tank"``, ``"payload fairing"``).
    mass : float
        Current mass of this component [kg]. Must be ≥ 0.
    position_body : ndarray, shape (3,), float64
        Position of this component's own CoM in the Body Frame [m].
        Measured relative to the vehicle's nominal Body Frame origin
        (which coincides with the full-stack CoM at a reference configuration).
    inertia_own : ndarray, shape (3, 3), float64
        Inertia tensor of this component about its OWN CoM [kg·m²].
        Must be symmetric positive semi-definite. For a point mass, use zeros.
    is_active : bool
        If False, component has been jettisoned/separated — excluded from
        mass totals and inertia computation.
    """
    name:          str
    mass:          float
    position_body: np.ndarray   # (3,) float64 [m]
    inertia_own:   np.ndarray   # (3,3) float64 [kg·m²]
    is_active:     bool = True

    def __post_init__(self) -> None:
        if self.mass < 0.0:
            raise ValueError(f"MassComponent '{self.name}': mass must be ≥ 0, got {self.mass!r}")
        self.position_body = np.asarray(self.position_body, dtype=np.float64)
        self.inertia_own   = np.asarray(self.inertia_own,   dtype=np.float64)
        if self.position_body.shape != (3,):
            raise ValueError(
                f"MassComponent '{self.name}': position_body must be shape (3,), "
                f"got {self.position_body.shape}"
            )
        if self.inertia_own.shape != (3, 3):
            raise ValueError(
                f"MassComponent '{self.name}': inertia_own must be shape (3,3), "
                f"got {self.inertia_own.shape}"
            )


# ---------------------------------------------------------------------------
# Parallel-axis (Steiner) displacement tensor
# ---------------------------------------------------------------------------

def steiner_tensor(r: np.ndarray) -> np.ndarray:
    """
    Compute the parallel-axis displacement tensor for offset r.

    D(r)ᵢⱼ = |r|²·δᵢⱼ − rᵢ·rⱼ

    This is added to a component's own inertia tensor to shift it from
    the component's CoM to the vehicle's full CoM:

        I_about_CoM = I_own + m · D(r)

    where r = component_position − vehicle_CoM_position.

    Parameters
    ----------
    r : ndarray, shape (3,)
        Offset vector [m] from original CoM to new reference point.

    Returns
    -------
    ndarray, shape (3,3), float64
    """
    r = np.asarray(r, dtype=np.float64)
    r_sq = float(np.dot(r, r))
    D = r_sq * np.eye(3, dtype=np.float64) - np.outer(r, r)
    return D


# ---------------------------------------------------------------------------
# Full vehicle mass model
# ---------------------------------------------------------------------------

@dataclass
class MassModel:
    """
    Computed mass properties of the complete vehicle assembly.

    Built by ``compute_mass_properties`` from a list of MassComponents.
    This object is frozen after construction.

    Attributes
    ----------
    total_mass : float
        Sum of all active component masses [kg].
    com_body : ndarray, shape (3,), float64
        Centre of mass position in the Body Frame [m].
        For a well-trimmed vehicle this should be near [0, 0, 0].
    inertia_body : ndarray, shape (3,3), float64
        Full inertia tensor about the vehicle CoM in the Body Frame [kg·m²].
        Symmetric positive semi-definite.
    inertia_inv : ndarray, shape (3,3), float64
        Inverse of the inertia tensor (pre-computed for the integrator).
    """
    total_mass:   float
    com_body:     np.ndarray   # (3,) float64 [m]
    inertia_body: np.ndarray   # (3,3) float64 [kg·m²]
    inertia_inv:  np.ndarray   # (3,3) float64 [(kg·m²)⁻¹]

    @property
    def Ixx(self) -> float:
        """Roll inertia [kg·m²]."""
        return float(self.inertia_body[0, 0])

    @property
    def Iyy(self) -> float:
        """Pitch inertia [kg·m²]."""
        return float(self.inertia_body[1, 1])

    @property
    def Izz(self) -> float:
        """Yaw inertia [kg·m²]."""
        return float(self.inertia_body[2, 2])

    @property
    def is_symmetric(self) -> bool:
        """True if inertia tensor is symmetric (as it always should be)."""
        return bool(np.allclose(self.inertia_body, self.inertia_body.T, atol=1.0e-6))


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def compute_mass_properties(
    components: List[MassComponent],
    enforce_positive_definite: bool = True,
) -> MassModel:
    """
    Compute total mass, CoM, and inertia tensor from a list of components.

    Parameters
    ----------
    components : list of MassComponent
        All vehicle components. Inactive (jettisoned) ones are skipped.
    enforce_positive_definite : bool
        If True (default), add a small regularisation (1e-6·I₃) to the
        inertia tensor to prevent singular I in degenerate cases (e.g.
        a single point mass with no extent). This keeps the integrator stable.

    Returns
    -------
    MassModel

    Raises
    ------
    ValueError
        If total active mass is zero or negative (unphysical state).
    """
    active = [c for c in components if c.is_active]

    if not active:
        raise ValueError("compute_mass_properties: no active components — total mass is zero.")

    # --- Total mass ---
    m_total = sum(c.mass for c in active)
    if m_total <= 0.0:
        raise ValueError(
            f"compute_mass_properties: total mass = {m_total:.6f} kg ≤ 0. "
            "All active components have zero mass?"
        )

    # --- Centre of mass (Body Frame) ---
    r_com = np.zeros(3, dtype=np.float64)
    for c in active:
        r_com += c.mass * c.position_body
    r_com /= m_total

    # --- Inertia tensor about vehicle CoM (parallel-axis theorem) ---
    I_total = np.zeros((3, 3), dtype=np.float64)
    for c in active:
        if c.mass <= 0.0:
            continue   # zero-mass component contributes nothing
        r_offset = c.position_body - r_com   # vector from vehicle CoM to component CoM
        D        = steiner_tensor(r_offset)
        I_total += c.inertia_own + c.mass * D

    # --- Symmetrise (eliminate floating-point asymmetry) ---
    I_total = 0.5 * (I_total + I_total.T)

    # --- Regularisation ---
    if enforce_positive_definite:
        I_total += 1.0e-6 * np.eye(3, dtype=np.float64)

    # --- Inertia inverse ---
    I_inv = np.linalg.inv(I_total)

    return MassModel(
        total_mass=m_total,
        com_body=r_com,
        inertia_body=I_total,
        inertia_inv=I_inv,
    )


# ---------------------------------------------------------------------------
# Convenience constructors for common component shapes
# ---------------------------------------------------------------------------

def point_mass(name: str, mass: float, position_body: "ArrayLike") -> MassComponent:
    """
    Create a point-mass component (zero own inertia).

    Parameters
    ----------
    name : str
    mass : float   [kg]
    position_body : array-like, shape (3,)   [m]
    """
    return MassComponent(
        name=name,
        mass=mass,
        position_body=np.asarray(position_body, dtype=np.float64),
        inertia_own=np.zeros((3, 3), dtype=np.float64),
    )


def solid_cylinder(
    name:          str,
    mass:          float,
    radius:        float,
    length:        float,
    position_body: "ArrayLike",
    axis:          int = 0,
) -> MassComponent:
    """
    Create a solid uniform cylinder component.

    Parameters
    ----------
    name : str
    mass : float    [kg]
    radius : float  [m]  Cylinder radius.
    length : float  [m]  Cylinder length along ``axis``.
    position_body : array-like, shape (3,)   CoM position in Body frame [m].
    axis : int
        Body-frame axis along which the cylinder is oriented.
        0 = X (longitudinal), 1 = Y, 2 = Z.

    Returns
    -------
    MassComponent

    Inertia
    -------
    Axial moment:       I_axial     = (1/2) m r²
    Transverse moments: I_transverse = (1/12) m (3r² + L²)
    """
    I_axial     = 0.5  * mass * radius**2
    I_transverse = (1.0 / 12.0) * mass * (3.0 * radius**2 + length**2)

    # Place axial inertia along the correct diagonal
    diag = [I_transverse, I_transverse, I_transverse]
    diag[axis] = I_axial
    I_own = np.diag(diag).astype(np.float64)

    return MassComponent(
        name=name,
        mass=mass,
        position_body=np.asarray(position_body, dtype=np.float64),
        inertia_own=I_own,
    )


def solid_sphere(
    name:          str,
    mass:          float,
    radius:        float,
    position_body: "ArrayLike",
) -> MassComponent:
    """
    Create a solid uniform sphere component.

    I = (2/5) m r²  (isotropic — same for all axes)
    """
    I_val = (2.0 / 5.0) * mass * radius**2
    I_own = I_val * np.eye(3, dtype=np.float64)
    return MassComponent(
        name=name,
        mass=mass,
        position_body=np.asarray(position_body, dtype=np.float64),
        inertia_own=I_own,
    )


def hollow_cylinder(
    name:          str,
    mass:          float,
    inner_radius:  float,
    outer_radius:  float,
    length:        float,
    position_body: "ArrayLike",
    axis:          int = 0,
) -> MassComponent:
    """
    Create a hollow cylinder (thin-walled tank) component.

    Axial:      I_axial     = (1/2) m (r_i² + r_o²)
    Transverse: I_transverse = (1/12) m (3(r_i² + r_o²) + L²)
    """
    r_sq_sum    = inner_radius**2 + outer_radius**2
    I_axial     = 0.5 * mass * r_sq_sum
    I_transverse = (1.0 / 12.0) * mass * (3.0 * r_sq_sum + length**2)

    diag = [I_transverse, I_transverse, I_transverse]
    diag[axis] = I_axial
    I_own = np.diag(diag).astype(np.float64)

    return MassComponent(
        name=name,
        mass=mass,
        position_body=np.asarray(position_body, dtype=np.float64),
        inertia_own=I_own,
    )