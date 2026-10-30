"""
nova.physics.structural
=======================
Structural mechanics solver for Project NOVA.

Architecture role — Pipeline Stage 9 (Component Updates)
---------------------------------------------------------
Computes internal structural load tensors at each joint in the component
graph, then checks them against analytical failure criteria. Triggered
after the physics engine and integrator have determined the vehicle's
acceleration, angular acceleration, and dynamic pressure.

Internal loads computed per joint
----------------------------------
For each joint j connecting parent P and child C, the internal loads
represent the forces and moments that the parent must exert on the child
to maintain structural continuity. They are computed by summing all
external forces and inertial reactions on the child subtree:

  F_axial   [N]     Axial compressive/tensile force along joint axis (+X_body)
  V_shear   [N]     Resultant transverse shear (√(Vy²+Vz²))
  M_bend    [N·m]   Resultant bending moment (√(My²+Mz²))
  T_torsion [N·m]   Torsional moment about joint axis

Failure criteria
----------------
1. Euler buckling (compressive axial):
   F_axial_critical = π² E I / L_eff²
   Failure if F_a (compressive) > F_cr.

2. Shear yield (rectangular approx):
   V_max = (2/3) τ_yield A
   Failure if |V| > V_max.

3. Bending moment:
   M_yield = σ_yield I / c
   Failure if |M_b| > M_yield.

4. Damage accumulation (S-N):
   D += (n_cycles / N_failure) per tick.
   Failure if D ≥ 1.

All criteria are checked in sequence; the first exceeded triggers
joint.failed = True and joint.damage = 1.0.

Simplified loading model
------------------------
This module uses a simplified free-body diagram approach. Each child subtree
is treated as a rigid body under:
  - Weight of the subtree (gravity)
  - Aerodynamic drag on the subtree (from vehicle aero state)
  - Inertial reaction force = m_subtree · a_vehicle

The full multi-body structural FEM is beyond the scope of a trajectory
simulation; this model captures the dominant axial and shear loads that
drive real launch-vehicle structural failures at high Q and during max-Q.

References
----------
- Megson, "Aircraft Structures for Engineering Students", 5th ed., §2, §15
- Bruhn, "Analysis and Design of Flight Vehicle Structures", §C2, §D
- NASA SP-8028, "Buckling of Thin-Walled Circular Cylinders"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from nova.vehicle.component_graph import ComponentGraph, StructuralJoint
from nova.vehicle.mass_model import MassModel, compute_mass_properties


# ---------------------------------------------------------------------------
# Load state input (per-tick snapshot from physics engine)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VehicleLoadState:
    """
    Vehicle-level load quantities needed for structural analysis.

    Passed in from the physics pipeline after the integrator has run.

    Attributes
    ----------
    acceleration_body : ndarray, shape (3,), float64
        Vehicle linear acceleration in Body Frame [m s⁻²].
    alpha_body : ndarray, shape (3,), float64
        Vehicle angular acceleration in Body Frame [rad s⁻²].
    dynamic_pressure : float
        Freestream dynamic pressure q_∞ [Pa].
    axial_thrust : float
        Total engine thrust force along vehicle +X axis [N].
        Used to compute axial compression from thrust.
    aero_drag_body : ndarray, shape (3,), float64
        Net aerodynamic drag force in Body Frame [N].
    gravity_body : ndarray, shape (3,), float64
        Gravitational acceleration in Body Frame [m s⁻²].
        (Force = mass × gravity_body per subtree.)
    """
    acceleration_body: np.ndarray   # (3,) [m s⁻²]
    alpha_body:        np.ndarray   # (3,) [rad s⁻²]
    dynamic_pressure:  float        # [Pa]
    axial_thrust:      float        # [N]
    aero_drag_body:    np.ndarray   # (3,) [N]
    gravity_body:      np.ndarray   # (3,) [m s⁻²]


# ---------------------------------------------------------------------------
# Joint load result
# ---------------------------------------------------------------------------

@dataclass
class JointLoadResult:
    """
    Internal load tensor computed for a single structural joint.

    Attributes
    ----------
    joint_id : str
    axial_force : float       F_axial [N] — positive = tension
    shear_force : float       V_total [N] — resultant transverse shear
    bending_moment : float    M_bend  [N·m] — resultant bending moment
    torsion_moment : float    T       [N·m] — torsional moment
    failed : bool             True if any failure criterion exceeded this tick
    margin_axial : float      (F_cr − |F_axial|) / F_cr — buckling margin
    margin_shear : float      (V_max − |V|) / V_max — shear margin
    margin_bending : float    (M_yield − |M_b|) / M_yield — bending margin
    """
    joint_id:         str
    axial_force:      float
    shear_force:      float
    bending_moment:   float
    torsion_moment:   float
    failed:           bool
    margin_axial:     float
    margin_shear:     float
    margin_bending:   float


# ---------------------------------------------------------------------------
# Euler buckling limit (standalone — used in tests and HUD)
# ---------------------------------------------------------------------------

def euler_buckling_limit(
    elastic_modulus: float,
    second_moment:   float,
    effective_length: float,
) -> float:
    """
    Critical Euler column buckling load [N].

    F_cr = π² E I / L_eff²

    Parameters
    ----------
    elastic_modulus : float   E [Pa]
    second_moment : float     I [m⁴]
    effective_length : float  L_eff [m]

    Returns
    -------
    float   F_cr [N]
    """
    if effective_length <= 0.0:
        raise ValueError(f"effective_length must be > 0, got {effective_length!r}")
    return (math.pi**2 * elastic_modulus * second_moment) / effective_length**2


def shear_yield_limit(shear_yield_strength: float, area: float) -> float:
    """
    Shear yield force limit [N] (rectangular section approximation).

    V_max = (2/3) · τ_yield · A

    Parameters
    ----------
    shear_yield_strength : float   τ_yield [Pa]
    area : float                   A [m²]

    Returns
    -------
    float   V_max [N]
    """
    return (2.0 / 3.0) * shear_yield_strength * area


def bending_failure_moment(
    yield_strength: float,
    second_moment:  float,
    extreme_fibre:  float,
) -> float:
    """
    Bending moment at onset of yielding [N·m].

    M_yield = σ_yield · I / c

    Parameters
    ----------
    yield_strength : float   σ_yield [Pa]
    second_moment : float    I [m⁴]
    extreme_fibre : float    c [m]  distance to outermost fibre

    Returns
    -------
    float   M_yield [N·m]
    """
    if extreme_fibre <= 0.0:
        raise ValueError(f"extreme_fibre must be > 0, got {extreme_fibre!r}")
    return yield_strength * second_moment / extreme_fibre


# ---------------------------------------------------------------------------
# Subtree load computation
# ---------------------------------------------------------------------------

def _subtree_mass(graph: ComponentGraph, root_id: str) -> float:
    """Sum of active component masses in the subtree rooted at root_id."""
    total = 0.0
    stack = [root_id]
    while stack:
        nid  = stack.pop()
        node = graph._nodes.get(nid)
        if node is None or not node.is_active:
            continue
        total += node.mass_component.mass
        stack.extend(graph._children.get(nid, []))
    return total


def _subtree_com(graph: ComponentGraph, root_id: str) -> np.ndarray:
    """Mass-weighted CoM of subtree in Body Frame [m]."""
    m_total = 0.0
    r_total = np.zeros(3, dtype=np.float64)
    stack   = [root_id]
    while stack:
        nid  = stack.pop()
        node = graph._nodes.get(nid)
        if node is None or not node.is_active:
            continue
        m = node.mass_component.mass
        m_total += m
        r_total += m * node.mass_component.position_body
        stack.extend(graph._children.get(nid, []))
    if m_total <= 0.0:
        return np.zeros(3, dtype=np.float64)
    return r_total / m_total


# ---------------------------------------------------------------------------
# Per-joint load computation
# ---------------------------------------------------------------------------

def compute_joint_loads(
    joint:      StructuralJoint,
    graph:      ComponentGraph,
    load_state: VehicleLoadState,
) -> JointLoadResult:
    """
    Compute internal loads at a single joint using the free-body diagram
    of the child subtree.

    The joint transmits whatever forces and moments are needed to give the
    child subtree the same acceleration as the rest of the vehicle.

    Newton's 2nd law for the child subtree:
        F_joint + F_external = m_sub · a_vehicle
        → F_joint = m_sub · a_vehicle − F_external

    External forces on the subtree:
        F_gravity  = m_sub · g_body
        F_aero     = aero_drag_body (scaled by subtree mass fraction)
        F_thrust   = 0 (thrust is on the engine node itself, not the subtree)

    For simplicity, aero drag is distributed proportionally to mass.
    Full aerodynamic distribution requires a panel method (Phase 8).

    Bending moment is estimated as the axial load × lateral CoM offset
    from the joint axis (simplified beam model).

    Parameters
    ----------
    joint : StructuralJoint
    graph : ComponentGraph
    load_state : VehicleLoadState

    Returns
    -------
    JointLoadResult
    """
    child_id  = joint.child_id
    m_sub     = _subtree_mass(graph, child_id)
    r_com_sub = _subtree_com(graph, child_id)

    # Vehicle total mass for proportional distribution
    total_active = graph.active_nodes
    m_vehicle    = sum(n.mass_component.mass for n in total_active) or 1.0
    mass_frac    = m_sub / m_vehicle

    a    = np.asarray(load_state.acceleration_body, dtype=np.float64)
    g    = np.asarray(load_state.gravity_body,      dtype=np.float64)
    drag = np.asarray(load_state.aero_drag_body,    dtype=np.float64)

    # Inertial force needed to accelerate the subtree at vehicle acceleration
    F_inertia = m_sub * a

    # External forces on the subtree
    F_grav    = m_sub * g
    F_aero_sub = mass_frac * drag

    # Joint force = what the parent must supply (Newton reaction)
    F_joint = F_inertia - F_grav - F_aero_sub

    # Axial component (along vehicle +X_body)
    F_axial = float(F_joint[0])

    # Add thrust-induced compression on joints below the engine
    # (simplified: thrust compresses the stack above the engine)
    # For structural analysis: thrust axial compression acts downward on upper joints
    F_axial -= load_state.axial_thrust * mass_frac   # compression = negative

    # Shear force: transverse components (Y and Z)
    V_y = float(F_joint[1])
    V_z = float(F_joint[2])
    V_total = math.sqrt(V_y**2 + V_z**2)

    # Bending moment: F_shear × moment arm from joint to subtree CoM
    # Joint is at child node's body position; subtree CoM offset is r_com_sub
    # Moment arm = lateral offset of CoM from joint axial line
    child_node = graph._nodes.get(child_id)
    if child_node is not None:
        r_joint = child_node.mass_component.position_body
    else:
        r_joint = np.zeros(3, dtype=np.float64)

    r_arm     = r_com_sub - r_joint
    M_y       = V_z * abs(float(r_arm[0]))   # bending about Y from Z-shear × X-arm
    M_z       = V_y * abs(float(r_arm[0]))   # bending about Z from Y-shear × X-arm
    M_bend    = math.sqrt(M_y**2 + M_z**2)

    # Torsion: angular acceleration reaction on subtree (simplified)
    # T ≈ I_axial_sub × alpha_x  (rough estimate)
    T_torsion = abs(float(load_state.alpha_body[0])) * m_sub * 0.5   # rough

    # Update joint internal loads
    joint.axial_force    = F_axial
    joint.shear_force    = V_total
    joint.bending_moment = M_bend
    joint.torsion_moment = T_torsion

    # Failure check
    joint.check_failure()

    # Compute safety margins
    F_cr    = joint.euler_buckling_limit
    V_max   = joint.shear_yield_limit
    M_yield = joint.bending_failure_moment

    margin_axial   = (F_cr   - abs(F_axial)) / F_cr   if F_cr   > 0 else 1.0
    margin_shear   = (V_max  - V_total)      / V_max  if V_max  > 0 else 1.0
    margin_bending = (M_yield - M_bend)      / M_yield if M_yield > 0 else 1.0

    return JointLoadResult(
        joint_id=joint.joint_id,
        axial_force=F_axial,
        shear_force=V_total,
        bending_moment=M_bend,
        torsion_moment=T_torsion,
        failed=joint.failed,
        margin_axial=margin_axial,
        margin_shear=margin_shear,
        margin_bending=margin_bending,
    )


# ---------------------------------------------------------------------------
# Full structural analysis pass (called by pipeline Stage 9)
# ---------------------------------------------------------------------------

def analyse_structure(
    graph:      ComponentGraph,
    load_state: VehicleLoadState,
) -> List[JointLoadResult]:
    """
    Run the full structural analysis pass for all joints in the graph.

    Iterates over all joints, computes internal loads, and checks failure
    criteria. Updates joint.failed and joint.damage in-place.

    Parameters
    ----------
    graph : ComponentGraph
        Vehicle component graph (mutable — joints are updated in-place).
    load_state : VehicleLoadState
        Current tick vehicle load quantities.

    Returns
    -------
    list of JointLoadResult
        One result per joint, ordered arbitrarily.
    """
    results = []
    for joint in graph.joints:
        result = compute_joint_loads(joint, graph, load_state)
        results.append(result)
    return results


def critical_margin(results: List[JointLoadResult]) -> Optional[JointLoadResult]:
    """
    Return the joint result with the lowest (most critical) safety margin.

    The critical margin is min(margin_axial, margin_shear, margin_bending)
    across all joints.

    Parameters
    ----------
    results : list of JointLoadResult

    Returns
    -------
    JointLoadResult or None
        The most critical joint, or None if results is empty.
    """
    if not results:
        return None

    def _min_margin(r: JointLoadResult) -> float:
        return min(r.margin_axial, r.margin_shear, r.margin_bending)

    return min(results, key=_min_margin)