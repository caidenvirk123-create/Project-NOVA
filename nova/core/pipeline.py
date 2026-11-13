"""
nova.core.pipeline
==================
13-stage deterministic simulation pipeline for Project NOVA.

Architecture role — Master loop orchestrator
---------------------------------------------
The pipeline drives the complete simulation at a fixed timestep dt.
Each call to ``tick()`` executes all 13 stages in strict sequential order.
No stage reads mutable state written by a later stage in the same tick.

Stage execution order
---------------------
1.  Input Handler       Poll hardware control state vector.
2.  Vehicle Controller  Map inputs to actuator commands.
3.  Physics Engine      Accumulate global F and τ tensors.
4.  RK4 Integrator      Propagate state forward by dt.
5.  Collision Detection Bounding-volume and raycast checks.
6.  Orbital Solver      N-body gravity + Keplerian elements.
7.  Atmosphere Solver   Density, temperature, speed of sound.
8.  Thermodynamics      Convective/radiative skin heating.
9.  Component Updates   Mass flow, fatigue accumulation.
10. AI Monitor          Predictive anomaly detection.
11. Telemetry Layer     Serialise snapshot to registry.
12. Renderer            Variable-rate visual update (decoupled).
13. UI Engine           Refresh HUD panels.

Stages 8 (Thermodynamics), 12 (Renderer), and 13 (UI Engine) are stubbed
in Phase 5. They are designed as no-ops with the correct interface so Phase 8
and Phase 10 can drop in concrete implementations without touching pipeline.py.

Determinism contract
--------------------
The pipeline is deterministic: given identical initial state, control inputs,
and dt, it always produces identical output. This requires:
  - No wall-clock time in physics calculations.
  - No global mutable state outside the pipeline instance.
  - Forces and torques accumulated fresh every tick.
  - RK4 integrator operating on float64 flat state vector.

Configuration
-------------
PipelineConfig controls which optional subsystems are active. Inactive
subsystems (e.g. aerodynamics in vacuum) are bypassed at zero cost.
"""

from __future__ import annotations

import math
import time as _wall_clock
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from nova.core.constants import (
    EARTH_MU,
    EARTH_RADIUS_MEAN,
    EARTH_RADIUS_EQ,
    EARTH_J2,
    STD_GRAVITY,
)
from nova.core.state_vector import VehicleState
from nova.core.telemetry_registry import (
    TelemetryRegistry,
    TelemetrySnapshot,
    build_snapshot,
)
from nova.frames.transforms import (
    T_ENU_to_body,
    euler_to_quaternion,
    quaternion_to_euler,
)
from nova.physics.integrator import build_deriv_fn, integrate_state
from nova.physics.atmosphere import atmosphere, atmosphere_from_eci
from nova.physics.orbital import (
    gravity_acceleration,
    gravity_force,
    elements_from_state,
    EARTH_BODY,
    GravBody,
)
from nova.physics.forces import ForceAccumulator, rotate_body_to_eci
from nova.physics.torques import TorqueAccumulator
from nova.physics.aerodynamics import (
    AeroConfig,
    ControlDeflections,
    compute_aero,
    AeroState,
)
from nova.physics.propulsion import (
    EngineConfig,
    compute_propulsion,
    PropulsionState,
)
from nova.vehicle.mass_model import MassModel, compute_mass_properties
from nova.physics.structural import (
    VehicleLoadState,
    analyse_structure,
    critical_margin,
)


# ---------------------------------------------------------------------------
# Control input vector
# ---------------------------------------------------------------------------

@dataclass
class ControlInput:
    """
    Immutable hardware control state vector produced by Stage 1.

    All values normalised to [-1, 1] or [0, 1] as noted.

    Parameters
    ----------
    throttle : float
        Engine throttle command [0, 1].
    gimbal_pitch : float
        Engine gimbal pitch command [rad].
    gimbal_yaw : float
        Engine gimbal yaw command [rad].
    elevator : float
        Elevator deflection command [rad].
    aileron : float
        Aileron deflection command [rad].
    rudder : float
        Rudder deflection command [rad].
    staging : bool
        True on the tick a stage separation command is issued.
    """
    throttle:     float = 0.0
    gimbal_pitch: float = 0.0
    gimbal_yaw:   float = 0.0
    elevator:     float = 0.0
    aileron:      float = 0.0
    rudder:       float = 0.0
    staging:      bool  = False


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Runtime configuration for the simulation pipeline.

    Parameters
    ----------
    dt : float
        Fixed simulation timestep [s]. Default 0.01 s.
    enable_aerodynamics : bool
        If False, aerodynamic forces/moments are zeroed (rocket above 80 km).
    enable_j2 : bool
        If True, include J2 oblateness perturbation in gravity.
    enable_structural : bool
        If True, run structural analysis each tick.
    gravity_bodies : list of GravBody
        Gravitational bodies to include in N-body computation.
    aero_config : AeroConfig or None
        Vehicle aerodynamic configuration. Required if enable_aerodynamics=True.
    engine_config : EngineConfig or None
        Engine configuration. If None, thrust is always zero.
    aero_vacuum_altitude : float
        Altitude [m] above which aerodynamics is disabled regardless of flag.
    """
    dt:                    float  = 0.01
    enable_aerodynamics:   bool   = True
    enable_j2:             bool   = True
    enable_structural:     bool   = False
    gravity_bodies:        list   = field(default_factory=lambda: [EARTH_BODY])
    aero_config:           Optional[AeroConfig]   = None
    engine_config:         Optional[EngineConfig] = None
    aero_vacuum_altitude:  float  = 80_000.0   # [m]


# ---------------------------------------------------------------------------
# Per-tick pipeline result
# ---------------------------------------------------------------------------

@dataclass
class TickResult:
    """
    Complete output of one pipeline tick.

    Attributes
    ----------
    new_state : VehicleState
        State after RK4 integration.
    snapshot : TelemetrySnapshot
        Telemetry snapshot published to the registry.
    dt : float
        Timestep used [s].
    structural_failures : list of str
        Joint IDs that failed this tick. Empty if no failures.
    """
    new_state:            VehicleState
    snapshot:             TelemetrySnapshot
    dt:                   float
    structural_failures:  List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------

class SimulationPipeline:
    """
    13-stage deterministic simulation pipeline.

    Usage
    -----
    ::

        config  = PipelineConfig(dt=0.01, engine_config=my_engine)
        pipe    = SimulationPipeline(config, initial_state, component_graph)
        registry = pipe.registry

        for _ in range(n_ticks):
            control = ControlInput(throttle=1.0)
            result  = pipe.tick(control)
            # result.new_state is the next VehicleState
            # registry.latest   is the telemetry snapshot

    Parameters
    ----------
    config : PipelineConfig
        Simulation configuration.
    initial_state : VehicleState
        Starting vehicle state.
    component_graph : MassModel
        Vehicle component graph (mutable — structural solver updates joints).
    registry : TelemetryRegistry, optional
        External registry to publish to. Creates a new one if not supplied.
    propellant_mass : float
        Initial propellant mass [kg]. Depleted by the engine each tick.
    """

    def __init__(
        self,
        config:          PipelineConfig,
        initial_state:   VehicleState,
        component_graph: MassModel,
        registry:        Optional[TelemetryRegistry] = None,
        propellant_mass: float = math.inf,
    ) -> None:
        self._config       = config
        self._state        = initial_state
        self._graph        = component_graph
        self._registry     = registry if registry is not None else TelemetryRegistry()
        self._propellant   = propellant_mass
        self._downrange    = 0.0   # cumulative [m]
        self._tick_count   = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> VehicleState:
        """Current vehicle state (latest integrated snapshot)."""
        return self._state

    @property
    def registry(self) -> TelemetryRegistry:
        """Read-only telemetry registry."""
        return self._registry

    @property
    def tick_count(self) -> int:
        """Number of ticks executed so far."""
        return self._tick_count

    def tick(self, control: ControlInput = ControlInput()) -> TickResult:
        """
        Execute one complete simulation tick (all 13 stages).

        Parameters
        ----------
        control : ControlInput
            Current hardware control state vector.

        Returns
        -------
        TickResult
        """
        dt  = self._config.dt
        cfg = self._config
        s   = self._state

        # ── Stage 1: Input Handler ──────────────────────────────────
        # Control vector already consumed as parameter; no hardware poll
        # needed in headless mode. In full Pygame mode this stage calls
        # pygame.event.get() and maps keyboard/joystick to ControlInput.

        # ── Stage 2: Vehicle Controller ─────────────────────────────
        deflections = ControlDeflections(
            elevator=control.elevator,
            aileron=control.aileron,
            rudder=control.rudder,
        )

        # ── Stage 3: Physics Engine (force + torque accumulation) ────
        # Mass model from component graph
        active_comps = self._graph.active_mass_components()
        mass_model   = compute_mass_properties(active_comps)

        # DCM: body → ECI (approximate: use attitude quaternion directly)
        R_body_to_eci = T_ENU_to_body(s.quaternion).T   # body→ENU ≈ body→ECI

        # --- 3a. Forces ---
        facc = ForceAccumulator(s)

        # Gravity (Stage 6 feeds back here; pre-evaluated for current pos)
        a_grav = gravity_acceleration(
            s.position_eci,
            cfg.gravity_bodies,
            include_j2=cfg.enable_j2,
        )
        F_grav = s.mass * a_grav
        facc.add_gravity(F_grav)

        # Propulsion
        prop_state: Optional[PropulsionState] = None
        if cfg.engine_config is not None:
            atm_for_isp = atmosphere_from_eci(s.position_eci, EARTH_RADIUS_MEAN)
            prop_state  = compute_propulsion(
                cfg.engine_config,
                throttle=control.throttle,
                atm_pressure=atm_for_isp.pressure,
                gimbal_pitch=control.gimbal_pitch,
                gimbal_yaw=control.gimbal_yaw,
                propellant_remaining=self._propellant,
            )
            if prop_state.is_active:
                facc.add_thrust(prop_state.thrust_body, R_body_to_eci)

        # Aerodynamics
        aero_state: Optional[AeroState] = None
        atm_state   = atmosphere_from_eci(s.position_eci, EARTH_RADIUS_MEAN)
        aero_active = (
            cfg.enable_aerodynamics
            and cfg.aero_config is not None
            and atm_state.altitude < cfg.aero_vacuum_altitude
            and atm_state.density > 1.0e-6
        )
        if aero_active and cfg.aero_config is not None:
            v_body = T_ENU_to_body(s.quaternion) @ s.velocity_eci
            aero_state = compute_aero(
                v_body, s.omega_body, atm_state, cfg.aero_config, deflections
            )
            facc.add_aerodynamic(aero_state.force_body, R_body_to_eci)

        F_net_eci = facc.build()

        # --- 3b. Torques ---
        tacc = TorqueAccumulator(s)
        if aero_state is not None:
            tacc.add_aerodynamic_moments(
                pitching_moment=aero_state.pitching_moment,
                yawing_moment=aero_state.yawing_moment,
                rolling_moment=aero_state.rolling_moment,
            )
        if prop_state is not None and prop_state.is_active and cfg.engine_config is not None:
            if (cfg.engine_config.gimbal_max_rad > 0.0
                    and (prop_state.gimbal_angle_pitch != 0.0
                         or prop_state.gimbal_angle_yaw != 0.0)):
                tacc.add_moment_arm(
                    "gimbal",
                    prop_state.thrust_body,
                    cfg.engine_config.mount_point_body,
                )
        tau_net_body = tacc.build()

        # ── Stage 4: RK4 Integrator ──────────────────────────────────
        new_mass = s.mass
        if prop_state is not None and prop_state.is_active:
            dm = prop_state.mass_flow_rate * dt
            self._propellant = max(0.0, self._propellant - dm)
            new_mass         = max(s.mass - dm, 0.001)

        # Gravity is position-dependent, so re-evaluate it at every RK4 sub-stage.
        # Thrust/aero stay frozen for the tick, as before.
        F_nongrav = F_net_eci - F_grav
        deriv_fn  = build_deriv_fn(
            F_nongrav, tau_net_body, mass_model.inertia_body, s.mass,
            accel_fn=lambda r: gravity_acceleration(
                r, cfg.gravity_bodies, include_j2=cfg.enable_j2),
        )
                                   
        new_state = integrate_state(deriv_fn, s, dt, new_mass=new_mass)

        # ── Stage 5: Collision Detection ─────────────────────────────
        # Stubbed: bounding-volume/raycast against terrain mesh.
        # Phase 7 will insert CollisionDetector here.

        # ── Stage 6: Orbital Solver ───────────────────────────────────
        try:
            orb_elements = elements_from_state(
                new_state.position_eci, new_state.velocity_eci
            )
        except (ValueError, ZeroDivisionError):
            orb_elements = None

        # ── Stage 7: Atmosphere Solver ────────────────────────────────
        atm_new = atmosphere_from_eci(new_state.position_eci, EARTH_RADIUS_MEAN)

        # ── Stage 8: Thermodynamics Engine ───────────────────────────
        # Stubbed. Phase 8 inserts skin-heating model here.

        # ── Stage 9: Component Updates ────────────────────────────────
        # Structural analysis (if enabled)
        structural_failures: List[str] = []
        worst_margin  = 1.0
        critical_jid  = ""
        any_failure   = False

        if cfg.enable_structural:
            accel_body = T_ENU_to_body(new_state.quaternion) @ (
                F_net_eci / s.mass
            )
            thrust_mag = (float(np.linalg.norm(prop_state.thrust_body))
                          if prop_state and prop_state.is_active else 0.0)
            drag_body  = (aero_state.force_body.copy()
                          if aero_state else np.zeros(3, dtype=np.float64))
            grav_body  = T_ENU_to_body(new_state.quaternion) @ a_grav

            load_state = VehicleLoadState(
                acceleration_body=accel_body.astype(np.float64),
                alpha_body=np.zeros(3, dtype=np.float64),
                dynamic_pressure=atm_new.dynamic_pressure_at_speed(
                    float(np.linalg.norm(new_state.velocity_eci))),
                axial_thrust=thrust_mag,
                aero_drag_body=drag_body.astype(np.float64),
                gravity_body=grav_body.astype(np.float64),
            )
            results = analyse_structure(self._graph, load_state)
            cm      = critical_margin(results)
            if cm:
                worst_margin = min(cm.margin_axial, cm.margin_shear, cm.margin_bending)
                critical_jid = cm.joint_id
            failed_joints = self._graph.evaluate_structural_failures()
            structural_failures = [j.joint_id for j in failed_joints]
            any_failure = len(structural_failures) > 0

        # Staging event (component jettison)
        if control.staging:
            # Future: identify which stage to jettison from graph topology.
            # For now this is a no-op placeholder.
            pass

        # ── Stage 10: AI Monitor ──────────────────────────────────────
        # Stubbed. Phase 6 inserts AIMonitor here.

        # ── Stage 11: Telemetry Layer ─────────────────────────────────
        speed  = new_state.speed
        r_prev = float(np.linalg.norm(s.position_eci))
        r_now  = float(np.linalg.norm(new_state.position_eci))
        v_speed = (r_now - r_prev) / dt if dt > 0 else 0.0

        # Cumulative downrange (approximate arc-length on sphere)
        d_pos = float(np.linalg.norm(new_state.position_eci - s.position_eci))
        self._downrange += d_pos

        _z3 = np.zeros(3, dtype=np.float64)
        _g = facc.contribution_by_name("gravity");      F_grav_snap   = _g   if _g   is not None else _z3.copy()
        _t = facc.contribution_by_name("thrust");       F_thrust_snap = _t   if _t   is not None else _z3.copy()
        _a = facc.contribution_by_name("aerodynamic");  F_aero_snap   = _a   if _a   is not None else _z3.copy()
        _ta = tacc.contribution_by_name("aerodynamic"); tau_aero_snap   = _ta  if _ta  is not None else _z3.copy()
        _tg = tacc.contribution_by_name("gimbal");      tau_gimbal_snap = _tg  if _tg  is not None else _z3.copy()

        snapshot = build_snapshot(
            new_state,
            altitude=atm_new.altitude,
            density=atm_new.density,
            pressure=atm_new.pressure,
            speed_of_sound=atm_new.speed_of_sound,
            mach=atm_new.mach(speed),
            dynamic_pressure=atm_new.dynamic_pressure_at_speed(speed),
            alpha=aero_state.alpha if aero_state else 0.0,
            beta=aero_state.beta  if aero_state else 0.0,
            CL=aero_state.CL     if aero_state else 0.0,
            CD=aero_state.CD     if aero_state else 0.0,
            lift_force=float(np.linalg.norm(aero_state.lift_body)) if aero_state else 0.0,
            drag_force=float(np.linalg.norm(aero_state.drag_body)) if aero_state else 0.0,
            thrust_magnitude=float(np.linalg.norm(prop_state.thrust_body)) if prop_state else 0.0,
            mass_flow_rate=prop_state.mass_flow_rate if prop_state else 0.0,
            isp_effective=prop_state.isp_effective   if prop_state else 0.0,
            throttle=prop_state.throttle             if prop_state else 0.0,
            semi_major_axis=orb_elements.semi_major_axis           if orb_elements else 0.0,
            eccentricity=orb_elements.eccentricity                 if orb_elements else 0.0,
            inclination=orb_elements.inclination                   if orb_elements else 0.0,
            raan=orb_elements.raan                                 if orb_elements else 0.0,
            argument_of_periapsis=orb_elements.argument_of_periapsis if orb_elements else 0.0,
            true_anomaly=orb_elements.true_anomaly                 if orb_elements else 0.0,
            orbital_period=orb_elements.period                     if orb_elements else 0.0,
            apoapsis=orb_elements.apoapsis                         if orb_elements else 0.0,
            periapsis=orb_elements.periapsis                       if orb_elements else 0.0,
            force_gravity=F_grav_snap.astype(np.float64),
            force_thrust=F_thrust_snap.astype(np.float64),
            force_aero=F_aero_snap.astype(np.float64),
            force_net=F_net_eci.astype(np.float64),
            torque_aero=tau_aero_snap.astype(np.float64),
            torque_gimbal=tau_gimbal_snap.astype(np.float64),
            torque_net=tau_net_body.astype(np.float64),
            worst_structural_margin=worst_margin,
            critical_joint_id=critical_jid,
            any_structural_failure=any_failure,
            speed=speed,
            vertical_speed=v_speed,
            downrange_distance=self._downrange,
        )
        self._registry.publish(snapshot)

        # ── Stage 12: Renderer ────────────────────────────────────────
        # Decoupled from sim frequency. Phase 10 inserts Renderer here.

        # ── Stage 13: UI Engine ───────────────────────────────────────
        # Phase 10 inserts HUD refresh here.

        # Advance state
        self._state      = new_state
        self._tick_count += 1

        return TickResult(
            new_state=new_state,
            snapshot=snapshot,
            dt=dt,
            structural_failures=structural_failures,
        )

    def run(
        self,
        n_ticks: int,
        control_fn: Optional[Callable[[int, VehicleState], ControlInput]] = None,
    ) -> List[TickResult]:
        """
        Run the pipeline for ``n_ticks`` steps.

        Parameters
        ----------
        n_ticks : int
            Number of ticks to execute.
        control_fn : callable (tick_index, state) → ControlInput, optional
            Function that returns the control input for each tick.
            If None, uses zero control (ControlInput()) throughout.

        Returns
        -------
        list of TickResult
            One result per tick.
        """
        results = []
        for i in range(n_ticks):
            ctrl = control_fn(i, self._state) if control_fn else ControlInput()
            results.append(self.tick(ctrl))
        return results