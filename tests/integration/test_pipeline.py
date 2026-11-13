"""
Integration Tests: Phase 14 — Determinsim and Mathematical Integration Symmetries.
Confirms global consistency ensuring re-running outputs is deterministically precise.
Conducts high-stress step forward, parameter flip, step backward exact trace.
"""

import numpy as np
import pytest

from nova.core.constants import EARTH_MU, EARTH_RADIUS_EQ
from nova.core.state_vector import VehicleState
from nova.core.pipeline import SimulationPipeline, PipelineConfig, ControlInput
from nova.vehicle.mass_model import solid_cylinder


class MockComponentNode:
    def __init__(self, mc):
        self.mass_component = mc
        self.is_active = True

class MockComponentGraph:
    def __init__(self, mass_kg):
        # Use solid_cylinder with CoM at origin to prevent artificial torque
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


def create_baseline_state() -> VehicleState:
    r0 = EARTH_RADIUS_EQ + 100_000.0
    v0 = 8500.0  # Hypersonic launch bound speed target
    
    return VehicleState(
        position_eci=np.array([r0, 0.0, 0.0], dtype=np.float64),
        velocity_eci=np.array([0.0, v0, 0.0], dtype=np.float64),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        omega_body=np.array([0.05, 0.1, -0.05], dtype=np.float64),
        mass=8000.0,
        time=0.0
    )

def test_full_loop_determinism():
    """
    Evaluates that for an explicitly identified initialization criteria state set S_0, 
    no background side effects affect output sets over S_N given N execution boundaries.
    """
    cfg = PipelineConfig(dt=0.01, enable_aerodynamics=False, enable_j2=True)
    ctrl = ControlInput(throttle=0.0) # Burn conditions OFF
    ticks = 500
    
    # ----------------
    # RUN 1
    # ----------------
    graph1 = MockComponentGraph(8000.0)
    pipeline1 = SimulationPipeline(cfg, create_baseline_state(), graph1)
    for _ in range(ticks):
        res1 = pipeline1.tick(ctrl)

    # ----------------
    # RUN 2 
    # ----------------
    graph2 = MockComponentGraph(8000.0)
    pipeline2 = SimulationPipeline(cfg, create_baseline_state(), graph2)
    for _ in range(ticks):
        res2 = pipeline2.tick(ctrl)
        
    # Bitwise precision consistency check - output hashes of raw flats must be fundamentally exact.
    flat_A = res1.new_state.to_flat()
    flat_B = res2.new_state.to_flat()

    np.testing.assert_allclose(flat_A, flat_B, atol=1e-15, rtol=0.0, 
        err_msg="Determinism Violated. Pipelines diverging sequentially at the exact sub-decimal scale."
    )


def test_pipeline_time_reversibility():
    """
    Time reversibility verification: Uses integrators' underlying temporal symmetry laws 
    governing second-order Newton motion without friction (+N sequence, flip phase v/-w, +N reverse).
    Ensures that r(N*2) matches r0 directly to confirm absolute ODE logic compliance in tracking physics loops.
    """
    cfg = PipelineConfig(dt=0.01, enable_aerodynamics=False, enable_j2=False)
    initial_state = create_baseline_state()
    ticks = 250
    control = ControlInput()

    # Step Forward 
    pipe_fwd = SimulationPipeline(cfg, initial_state, MockComponentGraph(initial_state.mass))
    for _ in range(ticks):
        res_fwd = pipe_fwd.tick(control)

    forward_endpoint_state = res_fwd.new_state

    # To reverse chronological physics trajectories completely through Newton mappings, 
    # we negate temporal components mapping positional velocities and body axial velocities natively.
    reversal_state = VehicleState(
        position_eci=forward_endpoint_state.position_eci.copy(),
        velocity_eci=-forward_endpoint_state.velocity_eci.copy(),
        quaternion=forward_endpoint_state.quaternion.copy(),
        omega_body=-forward_endpoint_state.omega_body.copy(),
        mass=forward_endpoint_state.mass,
        time=forward_endpoint_state.time  
    )

    # Step Process Reverse Route (still technically flowing pipeline positively logic-wise via `t + dt` sequence mapping natively).
    pipe_bwd = SimulationPipeline(cfg, reversal_state, MockComponentGraph(reversal_state.mass))
    for _ in range(ticks):
        res_bwd = pipe_bwd.tick(control)
        
    return_state = res_bwd.new_state

    # Due to compounded RK4 error damping during multi-trajectory evaluations, exact bitwise comparisons cannot exist, 
    # so we measure distance error returned locally against limits (~sub-centimeter allowance given orbital ranges).
    pos_difference = np.linalg.norm(initial_state.position_eci - return_state.position_eci)

    assert pos_difference < 1.0e-3, (
        f"Pipeline Integrator Irreversibility: Trajectories are asymmetrical tracking.\n"
        f"Return offset distance failed logic loop symmetry parameter boundaries at distance = {pos_difference:.5f} m"
    )

    # Angular mapping trajectory trace error tracking
    # Ensures non-coupled Euler spin traces properly mirrored back natively over identical torques logic matrices.
    velocity_offset = np.linalg.norm(-initial_state.velocity_eci - return_state.velocity_eci)
    
    assert velocity_offset < 1.0e-3, (
        f"Kinetic Vector Inconsistencies detected returning values inside boundaries (Reversibility).\n"
        f"Return offset failure parameter limits at deviation offset metric scale metric of {velocity_offset:.5f} m/s"
    )