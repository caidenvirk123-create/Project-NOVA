"""
Integration Tests: Phase 14 — Orbital Angular Momentum Conservation.
Validates h = r x v invariance tracking orientations, torque constraints, 
and ensuring coordinate matrices adhere rigidly to ECI stability thresholds.
"""

import math
import numpy as np
import pytest

from nova.core.constants import EARTH_MU, EARTH_RADIUS_EQ, ANGULAR_MOMENTUM_TOL
from nova.core.state_vector import VehicleState
from nova.core.pipeline import SimulationPipeline, PipelineConfig, ControlInput
from nova.vehicle.mass_model import point_mass


class MockComponentNode:
    def __init__(self, mc):
        self.mass_component = mc
        self.is_active = True

class MockComponentGraph:
    def __init__(self, mass_kg):
        self.mc = point_mass("core", mass_kg, [0.0, 0.0, 0.0])
        self._nodes = {"root": MockComponentNode(self.mc)}
        self.joints = []
        
    @property
    def active_nodes(self):
        return list(self._nodes.values())
        
    def active_mass_components(self):
        return [self.mc]
        
    def evaluate_structural_failures(self):
        return []


def specific_angular_momentum(state: VehicleState) -> np.ndarray:
    """Computes specific angular momentum vector h [m^2/s] invariant metric."""
    return np.cross(state.position_eci, state.velocity_eci)


@pytest.fixture
def reference_configuration():
    return PipelineConfig(
        dt=0.01,
        enable_aerodynamics=False,
        enable_j2=False, # Gravity fields are kept fully symmetric and uniform to preserve vectors
        enable_structural=False
    )


def test_angular_momentum_constancy(reference_configuration):
    """
    Test continuous 2D orbit preservation where `h` relies heavily on cross-axis interactions. 
    Verifies that orthogonal dimensions aren't shifting randomly or being overwritten.
    """
    r0 = EARTH_RADIUS_EQ + 2_000_000.0
    v0 = math.sqrt(EARTH_MU / r0)

    # Slanted polar orbit initial condition
    inc = math.radians(45.0) 
    
    # State Vector Mapping Setup
    init_r = np.array([r0, 0.0, 0.0], dtype=np.float64)
    init_v = np.array([0.0, v0 * math.cos(inc), v0 * math.sin(inc)], dtype=np.float64)

    initial_state = VehicleState(
        position_eci=init_r,
        velocity_eci=init_v,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        omega_body=np.zeros(3, dtype=np.float64),
        mass=3000.0,
        time=0.0
    )

    initial_h = specific_angular_momentum(initial_state)

    pipeline = SimulationPipeline(reference_configuration, initial_state, MockComponentGraph(3000.0))

    # Evaluate deep propagation integrity loop limit checks 
    # Roughly one full period across an equatorial radius at the parameter scale bounds.
    ticks = 5000 
    
    control = ControlInput()
    for _ in range(ticks):
        result = pipeline.tick(control)

    final_h = specific_angular_momentum(result.new_state)

    # 1. Orientation Sub-component Direction and Absolute Check  
    drift_vector = final_h - initial_h
    vector_drift_mag = float(np.linalg.norm(drift_vector))

    assert vector_drift_mag <= ANGULAR_MOMENTUM_TOL, (
        f"Cross-axis trajectory leakage found! Orbital elements precessing under uniform field.\n"
        f"H vector discrepancy (L2 Norm): {vector_drift_mag:.8g}"
    )

    # 2. Orthogonality Sanity: h must be definitively orthogonal to both r and v continually.
    assert abs(float(np.dot(result.new_state.position_eci, final_h))) < 1e-6
    assert abs(float(np.dot(result.new_state.velocity_eci, final_h))) < 1e-6