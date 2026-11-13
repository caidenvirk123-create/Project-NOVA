"""
Integration Tests: Phase 14 — Orbital Energy Conservation.
Verifies the integration scheme limits the variance of specific mechanical 
energy E = v^2/2 - mu/r under Newton's 2nd Law across long RK4 propagates.
"""

import math
import numpy as np
import pytest

from nova.core.constants import EARTH_MU, EARTH_RADIUS_EQ, ENERGY_CONSERVATION_TOL
from nova.core.state_vector import VehicleState
from nova.core.pipeline import SimulationPipeline, PipelineConfig, ControlInput
from nova.vehicle.mass_model import solid_cylinder


class MockComponentNode:
    """Mock tree-node component compliant with the pipeline architecture."""
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


def specific_mechanical_energy(state: VehicleState) -> float:
    """Calculate exact specific mechanical orbital energy [J kg⁻¹]."""
    v_mag_sq = float(np.dot(state.velocity_eci, state.velocity_eci))
    r_mag = float(np.linalg.norm(state.position_eci))
    return 0.5 * v_mag_sq - (EARTH_MU / r_mag)


@pytest.fixture
def base_vehicle_mass():
    return 5000.0


@pytest.mark.parametrize("eccentricity", [0.0, 0.5, 0.9])
@pytest.mark.parametrize("dt", [0.01, 0.05])
def test_orbital_energy_conservation(base_vehicle_mass, eccentricity, dt):
    """
    Simulates Keplerian orbital mechanics under the pipeline constraints.
    Tests E drift against rigid conservation bounds (1e-6 error permitted)
    for varying trajectory conditions and integration clock granularities.
    """
    # Safe periapsis threshold: 500km above standard radius.
    r_pe = EARTH_RADIUS_EQ + 500_000.0
    
    # Derivation for a valid ECI orbit at exactly the parameterized `e`
    a = r_pe / (1.0 - eccentricity)
    v_pe = math.sqrt(EARTH_MU * ((2.0 / r_pe) - (1.0 / a)))
    
    # Establish ECI parameters targeting +X axis orientation.
    initial_pos = np.array([r_pe, 0.0, 0.0], dtype=np.float64)
    initial_vel = np.array([0.0, v_pe, 0.0], dtype=np.float64)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    omega_body = np.zeros(3, dtype=np.float64)

    initial_state = VehicleState(
        position_eci=initial_pos,
        velocity_eci=initial_vel,
        quaternion=quaternion,
        omega_body=omega_body,
        mass=base_vehicle_mass,
        time=0.0
    )

    initial_energy = specific_mechanical_energy(initial_state)

    config = PipelineConfig(
        dt=dt,
        enable_aerodynamics=False,
        enable_j2=False,       # Oblateness disrupts pure two-body analytic constants.
        enable_structural=False
    )
    
    graph = MockComponentGraph(mass_kg=base_vehicle_mass)
    pipeline = SimulationPipeline(config, initial_state, graph)
    control = ControlInput()  # Zero actuation, deadstick free-fall 

    # Simulate 50 seconds worth of orbit time logic. RK4 should cap tracking drifts securely.
    steps = int(50.0 / dt)

    for _ in range(steps):
        result = pipeline.tick(control)

    final_energy = specific_mechanical_energy(result.new_state)
    energy_drift = abs(final_energy - initial_energy)

    # Invariants assertion matching our global numeric constants
    assert energy_drift <= ENERGY_CONSERVATION_TOL, (
        f"Energy Drift failed tolerance!\n"
        f"Eccentricity: {eccentricity}\n"
        f"Drift: {energy_drift:.8g} J kg⁻¹ (Max allowed: {ENERGY_CONSERVATION_TOL:.8g})"
    )