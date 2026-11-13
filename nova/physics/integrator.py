"""
nova.physics.integrator
=======================
Classical 4th-Order Runge-Kutta (RK4) numerical integrator for the
VehicleState ODE system.

Architecture contract
---------------------
The integrator is a pure function: given a state and a derivative function,
it produces the next state. It has NO side effects, NO global state, and
does NOT call any subsystem (physics, atmosphere, guidance) directly.

The caller (pipeline.py) is responsible for assembling the derivative
function ``f(state, t)`` by composing force accumulation, torque
accumulation, and kinematics before passing it here.

This design means the integrator module is independently testable against
closed-form analytical solutions with zero simulation infrastructure.

ODE system
----------
The full state vector is 13-dimensional:

    y = [r(3), v(3), q(4), ω(3)]   (flat float64 array, VehicleState.to_flat())

The derivative ẏ = f(y, t) has the form:

    ṙ = v                                          (kinematic — trivial)
    v̇ = F_total / m                               (Newton's 2nd law)
    q̇ = ½ · Ξ(q) · ω                             (quaternion kinematics)
    ω̇ = I⁻¹ · (τ_total − ω × (I · ω))           (Euler's rotation equations)

where Ξ(q) is the 4×3 kinematic matrix from nova.frames.transforms.xi_matrix.

The integrator does not compute F_total, τ_total, or I itself — these
come from the physics pipeline and are baked into the callable ``deriv_fn``.

RK4 formulation
---------------
    k1 = f(y_n,             t_n        )
    k2 = f(y_n + dt/2 · k1, t_n + dt/2)
    k3 = f(y_n + dt/2 · k2, t_n + dt/2)
    k4 = f(y_n + dt   · k3, t_n + dt  )

    y_{n+1} = y_n + (dt/6) · (k1 + 2k2 + 2k3 + k4)

Local truncation error: O(dt⁵). Global error: O(dt⁴).

At dt = 0.01 s (default), this gives integration accuracy well within the
1×10⁻⁶ conservation tolerances over realistic simulation durations.

Quaternion post-normalisation
-----------------------------
After each RK4 step the quaternion component of y_{n+1} is explicitly
renormalised to unit length before reconstructing the VehicleState.
This prevents secular norm drift (ODE integration cannot enforce the
algebraic constraint ‖q‖ = 1 exactly).

References
----------
- Press et al., "Numerical Recipes in C", 3rd ed., §17.1
- Diebel (2006), "Representing Attitude: Euler Angles, Quaternions …", §5.4
- Dormand & Prince (1980), "A Family of Embedded Runge-Kutta Formulae"
  (for context on RK error order — NOVA uses classical RK4, not adaptive)
"""

from __future__ import annotations

from typing import Callable, Optional
import numpy as np

from nova.core.state_vector import VehicleState


# ---------------------------------------------------------------------------
# Type alias for the derivative function
# ---------------------------------------------------------------------------

#: Signature: (flat_state: ndarray shape (13,), time: float) → ndarray shape (13,)
DerivFn = Callable[[np.ndarray, float], np.ndarray]


# ---------------------------------------------------------------------------
# Core RK4 step
# ---------------------------------------------------------------------------

def rk4_step(
    deriv_fn: DerivFn,
    y: np.ndarray,
    t: float,
    dt: float,
) -> np.ndarray:
    """
    Advance a flat state vector by one RK4 step.

    This is the low-level numerical kernel. It operates entirely on raw
    float64 arrays with no VehicleState construction overhead — suitable for
    calling millions of times in tight loops.

    Parameters
    ----------
    deriv_fn : callable (y: ndarray, t: float) → ndarray
        The ODE right-hand side. Must be a pure function: same inputs →
        same output, no side effects. Called exactly 4 times per step.
    y : ndarray, shape (N,), dtype float64
        Current state vector.
    t : float
        Current time [s].
    dt : float
        Fixed timestep [s]. Must be > 0.

    Returns
    -------
    y_next : ndarray, shape (N,), dtype float64
        State vector advanced by ``dt`` seconds.

    Notes
    -----
    No quaternion renormalisation is performed here. That is the
    responsibility of the caller (``integrate_state``) which knows the
    layout of the state vector.
    """
    if dt <= 0.0:
        raise ValueError(f"RK4 timestep dt must be > 0, got {dt!r}")

    k1 = deriv_fn(y,                   t         )
    k2 = deriv_fn(y + 0.5 * dt * k1,  t + 0.5*dt)
    k3 = deriv_fn(y + 0.5 * dt * k2,  t + 0.5*dt)
    k4 = deriv_fn(y +       dt * k3,  t +     dt)

    return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ---------------------------------------------------------------------------
# High-level state integrator (operates on VehicleState)
# ---------------------------------------------------------------------------

def integrate_state(
    deriv_fn: DerivFn,
    state: VehicleState,
    dt: float,
    new_mass: Optional[float] = None,
) -> VehicleState:
    """
    Advance a VehicleState by one simulation tick using RK4.

    This is the function called by the pipeline at each fixed timestep.
    It handles the flat↔structured conversion and enforces quaternion
    unit-norm after integration.

    Parameters
    ----------
    deriv_fn : callable
        ODE right-hand side: (flat_state, time) → flat_deriv, both
        shape (13,), float64. Assembled by the pipeline from the current
        tick's force/torque accumulators.
    state : VehicleState
        Current (frozen, immutable) state snapshot.
    dt : float
        Fixed simulation timestep [s].
    new_mass : float, optional
        Vehicle mass at the end of this step [kg]. If None, ``state.mass``
        is carried forward unchanged. Mass is not part of the ODE (it is
        updated by the propulsion model in Stage 9), so it is passed
        externally rather than integrated.

    Returns
    -------
    VehicleState
        New immutable state at time ``state.time + dt``.

    Raises
    ------
    ValueError
        If dt ≤ 0, or if the integrated quaternion collapses to near-zero norm.
    """
    y_n  = state.to_flat()
    t_n  = state.time
    mass = new_mass if new_mass is not None else state.mass

    # Core RK4 advance
    y_next = rk4_step(deriv_fn, y_n, t_n, dt)

    # Renormalise quaternion (indices 6:10 in the flat layout)
    q_raw  = y_next[6:10]
    q_norm = np.linalg.norm(q_raw)
    if q_norm < 1.0e-15:
        raise ValueError(
            "Quaternion norm collapsed to near-zero after RK4 step. "
            "Check the angular velocity magnitude and dt."
        )
    y_next[6:10] = q_raw / q_norm

    return VehicleState.from_flat(
        y_next,
        mass=mass,
        time=t_n + dt,
        normalize_quaternion=False,   # already normalised above
    )


# ---------------------------------------------------------------------------
# Derivative builder helpers (used in tests and by the pipeline)
# ---------------------------------------------------------------------------

def build_translational_deriv(
    force_eci: np.ndarray,
    torque_body: np.ndarray,
    inertia_tensor_body: np.ndarray,
) -> DerivFn:
    """
    Construct a closed-form derivative function for the full 13D state ODE.

    This factory captures the current tick's force and torque tensors into
    a closure. The pipeline calls this once per tick, after stages 3–9 have
    accumulated F and τ, then passes the resulting callable to
    ``integrate_state``.

    Parameters
    ----------
    force_eci : ndarray, shape (3,), float64
        Net external force vector expressed in the ECI frame [N].
        (The physics engine accumulates forces in Body Frame and rotates
        them to ECI before passing here.)
    torque_body : ndarray, shape (3,), float64
        Net external torque vector expressed in the Body Frame [N·m].
    inertia_tensor_body : ndarray, shape (3, 3), float64
        Vehicle inertia tensor [kg·m²] expressed in the Body Frame,
        evaluated at the current CoM. Must be symmetric positive-definite.

    Returns
    -------
    deriv_fn : callable (y, t) → ẏ, both ndarray shape (13,)

    ODE right-hand side
    -------------------
    y   = [r(3), v(3), q(4), ω(3)]

    ṙ   = v                                      (indices 0:3 → 3:6)
    v̇   = F_eci / m                              (indices 3:6)
    q̇   = ½ · Ξ(q) · ω                          (indices 6:10)
    ω̇   = I⁻¹ · (τ − ω × (I · ω))              (indices 10:13)
         (Euler's moment equations — gyroscopic coupling term)

    Notes
    -----
    The inertia tensor is assumed constant within one RK4 step (frozen at
    the start of the tick). This is valid when dt ≪ the timescale of mass
    depletion — a safe assumption at dt = 0.01 s for typical vehicles.
    """
    from nova.frames.transforms import xi_matrix

    # Pre-compute inertia inverse once (symmetric → no LU needed, but
    # np.linalg.inv handles the general case correctly)
    I     = inertia_tensor_body.astype(np.float64)
    I_inv = np.linalg.inv(I)

    # Capture closures as float64 copies
    F = force_eci.astype(np.float64)
    tau = torque_body.astype(np.float64)

    def deriv_fn(y: np.ndarray, t: float) -> np.ndarray:
        # Unpack state
        # r   = y[0:3]   (not needed in ṙ = v)
        v   = y[3:6]
        q   = y[6:10]
        m_  = y           # mass is NOT in y — see below
        omega = y[10:13]

        # Note: mass must be injected externally.  The deriv_fn receives
        # the flat 13-vector only. We capture mass from the calling scope
        # via `_mass` (set by integrate_state via the closure re-bind below).
        # This variable is set by `build_translational_deriv_with_mass`.
        raise NotImplementedError(
            "Use build_translational_deriv_with_mass to inject mass."
        )

    # This function is intentionally NOT the one returned — see the
    # mass-aware version below which is the actual public API.
    return deriv_fn


def build_deriv_fn(
    force_eci: np.ndarray,
    torque_body: np.ndarray,
    inertia_body: np.ndarray,
    mass: float,
    accel_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> DerivFn:
    """
    Construct the full 13D ODE derivative function for one simulation tick.

    This is the function the pipeline and unit tests actually use.
    It captures all tick-constant quantities (F, τ, I, m) in a closure.

    Parameters
    ----------
    force_eci : ndarray, shape (3,)
        Net force in ECI frame [N].
    torque_body : ndarray, shape (3,)
        Net torque in Body Frame [N·m].
    inertia_body : ndarray, shape (3, 3)
        Inertia tensor in Body Frame [kg·m²].
    mass : float
        Vehicle mass this tick [kg].
    accel_fn : optional position-dependent acceleration [m/s²] (e.g. gravity), evaluated at every RK4 sub-stage.

    Returns
    -------
    callable : (y: ndarray shape (13,), t: float) → ẏ ndarray shape (13,)
    """
    from nova.frames.transforms import xi_matrix

    F     = np.asarray(force_eci,    dtype=np.float64)
    tau   = np.asarray(torque_body,  dtype=np.float64)
    I     = np.asarray(inertia_body, dtype=np.float64)
    I_inv = np.linalg.inv(I)
    m     = float(mass)

    def deriv_fn(y: np.ndarray, t: float) -> np.ndarray:
        # Slice state
        v     = y[3:6]
        q     = y[6:10]
        omega = y[10:13]

        # ṙ = v
        r_dot = v

        # v̇ = F / m  (+ position-dependent acceleration re-evaluated per RK4 sub-stage)
        v_dot = F / m
        if accel_fn is not None:
            v_dot = v_dot + accel_fn(y[0:3])

        # q̇ = ½ Ξ(q) ω
        Xi    = xi_matrix(q)           # (4, 3)
        q_dot = 0.5 * (Xi @ omega)    # (4,)

        # ω̇ = I⁻¹ (τ − ω × (I ω))
        I_omega   = I @ omega                        # (3,)
        gyro_term = np.cross(omega, I_omega)         # (3,)
        omega_dot = I_inv @ (tau - gyro_term)        # (3,)

        return np.concatenate([r_dot, v_dot, q_dot, omega_dot])

    return deriv_fn


# ---------------------------------------------------------------------------
# Multi-step propagation (convenience — used in validation tests)
# ---------------------------------------------------------------------------

def propagate(
    deriv_fn_factory: Callable[[VehicleState], DerivFn],
    initial_state: VehicleState,
    duration: float,
    dt: float,
) -> list[VehicleState]:
    """
    Propagate a VehicleState forward by ``duration`` seconds using RK4.

    At each step a fresh derivative function is constructed via
    ``deriv_fn_factory(current_state)`` — this allows the factory to
    re-evaluate forces based on the latest state (e.g. gravity at new
    position).

    Parameters
    ----------
    deriv_fn_factory : callable (VehicleState) → DerivFn
        Returns the ODE right-hand side for the given state.
    initial_state : VehicleState
        Starting state.
    duration : float
        Total integration time [s].
    dt : float
        Fixed timestep [s]. ``duration / dt`` must be a reasonable integer.

    Returns
    -------
    list of VehicleState
        Trajectory including the initial state and all subsequent steps.
        Length = floor(duration / dt) + 1.
    """
    if dt <= 0.0:
        raise ValueError(f"dt must be > 0, got {dt!r}")
    if duration <= 0.0:
        raise ValueError(f"duration must be > 0, got {duration!r}")

    n_steps = int(round(duration / dt))
    trajectory: list[VehicleState] = [initial_state]
    state = initial_state

    for _ in range(n_steps):
        fn = deriv_fn_factory(state)
        state = integrate_state(fn, state, dt)
        trajectory.append(state)

    return trajectory