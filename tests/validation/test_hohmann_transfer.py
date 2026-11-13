"""
Validation Tests: Phase 15 — Hohmann Transfer Delta-V Benchmarks.
Validates the propulsion scaling models and the orbital two-body tracking dynamics 
against strict analytical mathematical limits for impulsive energy transfer burns.
"""

import math
import numpy as np
import pytest

from nova.core.constants import EARTH_MU, EARTH_RADIUS_EQ
from nova.core.state_vector import VehicleState
from nova.core.pipeline import SimulationPipeline, PipelineConfig, ControlInput
from nova.physics.orbital import elements_from_state, KeplerianElements
from nova.vehicle.mass_model import solid_cylinder


class MockComponentNode:
    """Mock structural unit guaranteeing compliance with Component Updates Phase (Stage 9)."""
    def __init__(self, mc):
        self.mass_component = mc
        self.is_active = True

class MockComponentGraph:
    """Component matrix proxy establishing continuous uniform payload parameters globally."""
    def __init__(self, mass_kg):
        # 15,000 kg, 2 m radius, 20 m long rocket body, CoM at origin to prevent artificial torque
        self.mc = solid_cylinder("body", mass_kg, radius=1.0, length=20.0,
                                  position_body=np.array([0.0, 0.0, 0.0]))
        self._nodes = {"root": MockComponentNode(self.mc)}
        self.joints = []
        
    @property
    def active_nodes(self):
        return list(self._nodes.values())
        
    def active_mass_components(self):
        return [self.mc]
        
    def evaluate_structural_failures(self):
        return []


def test_analytical_deltav_closed_form_exactness():
    """
    Validates mathematical correctness directly analyzing classical impulsive Hohmann orbital 
    mechanics limits requested:
      dv1 = sqrt(mu / r1) * (sqrt(2*r2 / (r1 + r2)) - 1)
      dv2 = sqrt(mu / r2) * (1 - sqrt(2*r1 / (r1 + r2)))
    against library functional representations.
    """
    # 400km LEO to GEO Transfer scale
    r1 = EARTH_RADIUS_EQ + 400_000.0
    r2 = EARTH_RADIUS_EQ + 35_786_000.0
    
    # Precise numerical closed-form execution target derivations:
    analytical_dv1 = math.sqrt(EARTH_MU / r1) * (math.sqrt((2.0 * r2) / (r1 + r2)) - 1.0)
    analytical_dv2 = math.sqrt(EARTH_MU / r2) * (1.0 - math.sqrt((2.0 * r1) / (r1 + r2)))
    total_analytical = analytical_dv1 + analytical_dv2
    
    # Internal module representation validation tracking function parity 
    from nova.physics.orbital import hohmann_delta_v
    library_dv1, library_dv2 = hohmann_delta_v(r1, r2, EARTH_MU)
    
    # Tolerances held inside pure 64-bit IEEE scale errors:
    assert library_dv1 == pytest.approx(analytical_dv1, rel=1e-12)
    assert library_dv2 == pytest.approx(analytical_dv2, rel=1e-12)


def test_hohmann_insertion_mechanics_integration_bound():
    """
    Propagates a simulated rigid-body vehicle mathematically initiating and 
    completing a fully circular coplanar Hohmann ascent strictly over a continuous 
    ODE stepping environment simulating a numerical delta-v track matching elements precisely.
    """
    dt = 1.0 # 1 second steps to process extensive orbit transfers accurately inside time envelopes.
    r1 = EARTH_RADIUS_EQ + 500_000.0
    r2 = EARTH_RADIUS_EQ + 5_000_000.0
    v1_mag = math.sqrt(EARTH_MU / r1)
    
    initial_pos = np.array([r1, 0.0, 0.0], dtype=np.float64)
    initial_vel = np.array([0.0, v1_mag, 0.0], dtype=np.float64)
    mass_kg = 5000.0

    state = VehicleState(
        position_eci=initial_pos,
        velocity_eci=initial_vel,
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        omega_body=np.zeros(3, dtype=np.float64),
        mass=mass_kg,
        time=0.0
    )

    from nova.physics.orbital import hohmann_delta_v
    dv1, dv2 = hohmann_delta_v(r1, r2, EARTH_MU)
    
    # Stage 1: Periapsis Trajectory Shift (Target Burn Impulsive Override)
    velocity_unit_vec = state.velocity_eci / np.linalg.norm(state.velocity_eci)
    state = state.with_time(0.0) # Reset tracking base
    # Instantiate simulated delta V impulse across continuous pipeline wrapper explicitly natively
    augmented_velocity = state.velocity_eci + (velocity_unit_vec * dv1)
    
    # Build updated vehicle vector snapshot tracking immediate Kepler alterations directly
    transfer_state = VehicleState(
        position_eci=state.position_eci.copy(),
        velocity_eci=augmented_velocity,
        quaternion=state.quaternion.copy(),
        omega_body=state.omega_body.copy(),
        mass=mass_kg,
        time=0.0
    )

    # Establish orbit tracking to capture the vehicle coast period to intercept apogee dynamically
    config = PipelineConfig(dt=dt, enable_aerodynamics=False, enable_j2=False, enable_structural=False)
    graph = MockComponentGraph(mass_kg)
    pipeline = SimulationPipeline(config, transfer_state, graph)
    control = ControlInput(throttle=0.0)
    
    current_state = transfer_state
    
    # Coast dynamically to orbit Apoapsis evaluating radial velocity bounds across sequence bounds:
    # A true apogee identifies passing the `r dot v` root inflection point logically evaluating zero crossings.
    reached_apoapsis = False
    
    # Predict exact transfer coast window using semi-major mechanics representing transfer time boundaries
    a_transfer = 0.5 * (r1 + r2)
    predicted_half_period_sec = math.pi * math.sqrt((a_transfer ** 3) / EARTH_MU)
    
    # Integrate sequentially inside safe termination bounds (safety stop inside limits ~30 mins tracking)
    for _ in range(int((predicted_half_period_sec + 2000.0) / dt)):
        prev_r_dot_v = np.dot(current_state.position_eci, current_state.velocity_eci)
        result = pipeline.tick(control)
        current_state = result.new_state
        curr_r_dot_v = np.dot(current_state.position_eci, current_state.velocity_eci)
        
        # Apoapsis triggers uniquely at cross intercept transitions indicating outbound to return paths:
        if prev_r_dot_v > 0.0 and curr_r_dot_v <= 0.0:
            reached_apoapsis = True
            break
            
    assert reached_apoapsis, "Integration trajectory coast boundaries did not successfully intercept numerical target orbit limits!"
    
    # Assess apoapsis range bound accurately capturing transfer tolerance over extensive numeric tracks (rel 1e-4 covers step jumps)
    apoapsis_radius = np.linalg.norm(current_state.position_eci)
    assert apoapsis_radius == pytest.approx(r2, rel=5e-4)

    # Stage 2: Insertion Apoapsis Orbital Calibration Burn (Secondary Impulse Application Tracking)
    ap_velocity_unit_vec = current_state.velocity_eci / np.linalg.norm(current_state.velocity_eci)
    ap_augmented_velocity = current_state.velocity_eci + (ap_velocity_unit_vec * dv2)

    final_state = VehicleState(
        position_eci=current_state.position_eci.copy(),
        velocity_eci=ap_augmented_velocity,
        quaternion=current_state.quaternion.copy(),
        omega_body=current_state.omega_body.copy(),
        mass=mass_kg,
        time=current_state.time
    )

    final_orbit = elements_from_state(final_state.position_eci, final_state.velocity_eci)
    
    # Insertion Exactness Properties Assurance check against Hohmann circular stabilization constraints
    assert final_orbit.eccentricity == pytest.approx(0.0, abs=1e-3), "Orbit boundary tracking error! Burn insertion was non-circular inside analytical drift checks!"
    assert final_orbit.semi_major_axis == pytest.approx(r2, rel=1e-3)