"""
nova.ai.monitor
===============
Predictive anomaly detection engine for Project NOVA.

Architecture role — Pipeline Stage 10
--------------------------------------
The AI Monitor reads the immutable TelemetryRegistry sequentially. It
never reads or writes VehicleState directly. It computes time derivatives
of incoming telemetry vectors and generates predictive warnings before
structural or flight-envelope limits are reached.

Design contract
---------------
* Input: TelemetryRegistry (read-only snapshot buffer from Stage 11).
* Output: List[AlertMessage] — consumed by flight_logger and the HUD.
* No side effects. assess() is a pure function of the registry contents.
* The monitor is called once per tick from pipeline Stage 10 but operates
  entirely on immutable data — it cannot affect the physics outcome of the
  current tick.

Alert categories
----------------
  STRUCTURAL   — predicted joint failure from load trend
  AERODYNAMIC  — AoA/sideslip exceedance, max-Q approach
  PROPULSION   — propellant depletion, Isp degradation
  ORBITAL      — periapsis decay, eccentricity drift
  THERMAL      — dynamic pressure rate (proxy for heating)
  INFO         — nominal milestones (orbit insertion, MECO)

Alert severity
--------------
  CRITICAL  — limit predicted in < 5 s
  WARNING   — limit predicted in 5–30 s
  CAUTION   — limit predicted in 30–120 s
  INFO      — informational, no limit approach

Output format (matches architecture spec §5)
--------------------------------------------
``"ALERT: AoA = 19.3°. Dynamic pressure escalating at +45.2 kPa/s.
   Predicted structural shear failure of Interstage Joint in 4.8s.
   RECOMMENDATION: Reduce pitch deflection below 10° and throttle to 60%."``

References
----------
- Muñoz-Garcia et al., "Real-Time Monitoring of Aerospace Structures",
  Aerospace 2021, 8, 156.
- FAA AC 25.1309-1A, "System Design and Analysis".
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

from nova.core.telemetry_registry import TelemetryRegistry, TelemetrySnapshot
from nova.ai.derivatives import (
    scalar_derivative,
    scalar_derivative_at,
    time_to_limit,
    predict_value,
    compute_derivative_pack,
)


# ---------------------------------------------------------------------------
# Alert data types
# ---------------------------------------------------------------------------

class AlertSeverity(Enum):
    CRITICAL = "CRITICAL"   # < 5 s to limit
    WARNING  = "WARNING"    # 5–30 s
    CAUTION  = "CAUTION"    # 30–120 s
    INFO     = "INFO"       # informational


class AlertCategory(Enum):
    STRUCTURAL   = "STRUCTURAL"
    AERODYNAMIC  = "AERODYNAMIC"
    PROPULSION   = "PROPULSION"
    ORBITAL      = "ORBITAL"
    THERMAL      = "THERMAL"
    INFO         = "INFO"


@dataclass(frozen=True)
class AlertMessage:
    """
    A single structured alert produced by the AI Monitor.

    Attributes
    ----------
    severity : AlertSeverity
    category : AlertCategory
    message : str
        Human-readable description of the anomaly.
    recommendation : str
        Actionable corrective recommendation for the pilot.
    time_to_limit_s : float
        Estimated seconds until the predicted limit is breached.
        math.inf if no imminent limit.
    mission_time : float
        Simulation time at which this alert was generated [s].
    parameter : str
        Name of the telemetry attribute that triggered the alert.
    current_value : float
        Current value of the triggering parameter.
    rate : float
        Estimated derivative [units/s].
    limit : float
        The threshold value being approached.
    """
    severity:         AlertSeverity
    category:         AlertCategory
    message:          str
    recommendation:   str
    time_to_limit_s:  float
    mission_time:     float
    parameter:        str
    current_value:    float
    rate:             float
    limit:            float

    def format(self) -> str:
        """
        Format the alert as the architecture-spec output string.

        Returns
        -------
        str
            Multi-line alert string in the format:
            SEVERITY [CATEGORY] message. RECOMMENDATION: ...
        """
        ttl = (f"{self.time_to_limit_s:.1f}s" if math.isfinite(self.time_to_limit_s)
               else "∞")
        lines = [
            f"{self.severity.value} [{self.category.value}]  "
            f"(t={self.mission_time:.2f}s, t_to_limit={ttl})",
            f"  {self.message}",
            f"  RECOMMENDATION: {self.recommendation}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.format()


# ---------------------------------------------------------------------------
# Monitor configuration
# ---------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    """
    Limit thresholds and tuning parameters for the AI Monitor.

    All limits are in SI units (angles in radians).

    Parameters
    ----------
    aoa_warning_rad : float
        AoA at which a WARNING is issued [rad]. Default 15°.
    aoa_critical_rad : float
        AoA at which a CRITICAL is issued [rad]. Default 20°.
    sideslip_warning_rad : float
        Sideslip warning threshold [rad]. Default 10°.
    max_q_limit_pa : float
        Maximum dynamic pressure limit [Pa]. Default 50 kPa.
    mach_transonic_lower : float
        Lower Mach bound for transonic caution [–]. Default 0.85.
    mach_transonic_upper : float
        Upper Mach bound for transonic caution [–]. Default 1.10.
    structural_margin_warning : float
        Safety margin below which a WARNING is issued [–]. Default 0.25.
    structural_margin_critical : float
        Safety margin below which CRITICAL is issued [–]. Default 0.10.
    propellant_warning_s : float
        Seconds of propellant remaining at which WARNING issued. Default 30 s.
    propellant_critical_s : float
        Seconds remaining for CRITICAL. Default 10 s.
    periapsis_warning_alt_m : float
        Periapsis altitude below which reentry WARNING issued [m]. Default 120 km.
    prediction_horizon_s : float
        Maximum prediction horizon for rate extrapolation [s]. Default 120 s.
    n_points : int
        Number of telemetry snapshots used for derivative estimation. Default 3.
    """
    aoa_warning_rad:          float = math.radians(15.0)
    aoa_critical_rad:         float = math.radians(20.0)
    sideslip_warning_rad:     float = math.radians(10.0)
    max_q_limit_pa:           float = 50_000.0
    mach_transonic_lower:     float = 0.85
    mach_transonic_upper:     float = 1.10
    structural_margin_warning: float = 0.25
    structural_margin_critical: float = 0.10
    propellant_warning_s:     float = 30.0
    propellant_critical_s:    float = 10.0
    periapsis_warning_alt_m:  float = 120_000.0
    prediction_horizon_s:     float = 120.0
    n_points:                 int   = 3


# ---------------------------------------------------------------------------
# Severity helper
# ---------------------------------------------------------------------------

def _severity_from_ttl(ttl: float) -> AlertSeverity:
    """Map time-to-limit [s] to severity level."""
    if ttl <= 5.0:
        return AlertSeverity.CRITICAL
    elif ttl <= 30.0:
        return AlertSeverity.WARNING
    elif ttl <= 120.0:
        return AlertSeverity.CAUTION
    else:
        return AlertSeverity.INFO


# ---------------------------------------------------------------------------
# Individual check functions
# ---------------------------------------------------------------------------

def _check_aoa(
    snap:   TelemetrySnapshot,
    d_aoa:  Optional[float],
    cfg:    MonitorConfig,
) -> Optional[AlertMessage]:
    """Check angle of attack against limits with predictive extrapolation."""
    aoa   = snap.alpha
    t     = snap.mission_time
    rate  = d_aoa if d_aoa is not None else 0.0

    # Choose the tighter active limit
    if abs(aoa) >= cfg.aoa_critical_rad:
        limit = cfg.aoa_critical_rad
    elif abs(aoa) >= cfg.aoa_warning_rad:
        limit = cfg.aoa_critical_rad   # warn and predict to critical
    else:
        # Below warning — check if rate is driving toward warning
        ttl = time_to_limit(abs(aoa), abs(rate) if rate * aoa >= 0 else -abs(rate),
                             cfg.aoa_warning_rad)
        if ttl > cfg.prediction_horizon_s:
            return None
        limit = cfg.aoa_warning_rad

    aoa_deg = math.degrees(abs(aoa))
    ttl     = time_to_limit(abs(aoa), abs(rate), limit)
    sev     = _severity_from_ttl(ttl)

    if sev == AlertSeverity.INFO and abs(aoa) < cfg.aoa_warning_rad:
        return None

    return AlertMessage(
        severity=sev,
        category=AlertCategory.AERODYNAMIC,
        message=(
            f"AoA = {aoa_deg:.1f}°. "
            f"{'Increasing' if rate > 0 else 'Decreasing'} at "
            f"{abs(math.degrees(rate)):.2f}°/s. "
            f"Limit = {math.degrees(limit):.1f}°."
        ),
        recommendation=(
            "Reduce pitch deflection. Throttle back to reduce dynamic pressure. "
            f"Target AoA < {math.degrees(cfg.aoa_warning_rad):.0f}°."
        ),
        time_to_limit_s=ttl,
        mission_time=t,
        parameter="alpha",
        current_value=aoa,
        rate=rate,
        limit=limit,
    )


def _check_dynamic_pressure(
    snap:    TelemetrySnapshot,
    d_q_inf: Optional[float],
    cfg:     MonitorConfig,
) -> Optional[AlertMessage]:
    """Check dynamic pressure against max-Q limit."""
    q_inf = snap.dynamic_pressure
    t     = snap.mission_time
    rate  = d_q_inf if d_q_inf is not None else 0.0

    ttl = time_to_limit(q_inf, rate, cfg.max_q_limit_pa)

    if q_inf < cfg.max_q_limit_pa * 0.70 and ttl > cfg.prediction_horizon_s:
        return None

    sev = _severity_from_ttl(ttl)
    if sev == AlertSeverity.INFO and q_inf < cfg.max_q_limit_pa * 0.85:
        return None

    return AlertMessage(
        severity=sev,
        category=AlertCategory.AERODYNAMIC,
        message=(
            f"Dynamic pressure = {q_inf/1000:.1f} kPa. "
            f"Rate = {rate/1000:+.2f} kPa/s. "
            f"Limit = {cfg.max_q_limit_pa/1000:.1f} kPa."
        ),
        recommendation=(
            "Reduce throttle or increase pitch angle to limit "
            "dynamic pressure buildup. Monitor structural margins."
        ),
        time_to_limit_s=ttl,
        mission_time=t,
        parameter="dynamic_pressure",
        current_value=q_inf,
        rate=rate,
        limit=cfg.max_q_limit_pa,
    )


def _check_structural_margin(
    snap: TelemetrySnapshot,
    d_margin: Optional[float],
    cfg:  MonitorConfig,
) -> Optional[AlertMessage]:
    """Check worst structural safety margin."""
    margin = snap.worst_structural_margin
    t      = snap.mission_time

    if snap.any_structural_failure:
        return AlertMessage(
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.STRUCTURAL,
            message=(
                f"STRUCTURAL FAILURE: Joint '{snap.critical_joint_id}' has failed. "
                f"Margin = {margin:.3f}."
            ),
            recommendation=(
                "Immediately reduce thrust and aerodynamic loading. "
                "Vehicle structural integrity compromised."
            ),
            time_to_limit_s=0.0,
            mission_time=t,
            parameter="worst_structural_margin",
            current_value=margin,
            rate=d_margin if d_margin is not None else 0.0,
            limit=0.0,
        )

    # Predict margin approaching zero.
    # rate (dM/dt) is negative when margin is decreasing toward failure.
    rate = d_margin if d_margin is not None else 0.0
    # rate_toward_zero: how fast the margin is shrinking (positive = shrinking fast)
    rate_toward_zero = max(-rate, 0.0)   # only counts when margin is decreasing
    # Direct time-to-failure: margin / rate_toward_zero
    if rate_toward_zero < 1.0e-12:
        ttl = math.inf
    else:
        ttl = margin / rate_toward_zero

    if margin > cfg.structural_margin_warning and ttl > cfg.prediction_horizon_s:
        return None

    if margin <= cfg.structural_margin_critical:
        sev = AlertSeverity.CRITICAL
    elif margin <= cfg.structural_margin_warning:
        sev = AlertSeverity.WARNING
    else:
        sev = _severity_from_ttl(ttl)

    if sev == AlertSeverity.INFO:
        return None

    return AlertMessage(
        severity=sev,
        category=AlertCategory.STRUCTURAL,
        message=(
            f"Structural margin at '{snap.critical_joint_id}' = {margin:.3f}. "
            f"Trend = {rate:+.4f}/s. "
            f"Predicted failure in {ttl:.1f}s." if math.isfinite(ttl) else
            f"Structural margin at '{snap.critical_joint_id}' = {margin:.3f}."
        ),
        recommendation=(
            "Reduce dynamic pressure and axial loading. "
            "Consider reducing throttle below 70% and pitch below 10°."
        ),
        time_to_limit_s=ttl,
        mission_time=t,
        parameter="worst_structural_margin",
        current_value=margin,
        rate=rate,
        limit=0.0,
    )


def _check_propellant(
    snap:          TelemetrySnapshot,
    cfg:           MonitorConfig,
) -> Optional[AlertMessage]:
    """Check propellant depletion via mass flow rate."""
    mdot   = snap.mass_flow_rate
    mass   = snap.mass
    t      = snap.mission_time

    if mdot <= 0.0 or mass <= 0.0:
        return None

    # Time remaining = current_mass / mdot (assume all remaining mass is propellant
    # for this estimate — the pipeline tracks actual propellant separately)
    t_remaining = mass / mdot

    if t_remaining > cfg.propellant_warning_s:
        return None

    sev = (AlertSeverity.CRITICAL if t_remaining <= cfg.propellant_critical_s
           else AlertSeverity.WARNING)

    return AlertMessage(
        severity=sev,
        category=AlertCategory.PROPULSION,
        message=(
            f"Propellant estimate: {t_remaining:.1f}s remaining at current flow rate "
            f"({mdot:.3f} kg/s). Mass = {mass:.1f} kg."
        ),
        recommendation=(
            "Prepare for MECO. Verify staging sequence. "
            "If upper stage, confirm orbit insertion burn window."
        ),
        time_to_limit_s=t_remaining,
        mission_time=t,
        parameter="mass_flow_rate",
        current_value=mdot,
        rate=0.0,
        limit=0.0,
    )


def _check_periapsis(
    snap: TelemetrySnapshot,
    cfg:  MonitorConfig,
) -> Optional[AlertMessage]:
    """Warn if periapsis altitude is below reentry threshold."""
    peri_alt = snap.periapsis - 6_371_000.0   # approximate altitude above mean surface
    t        = snap.mission_time

    if peri_alt >= cfg.periapsis_warning_alt_m or snap.eccentricity >= 1.0:
        return None

    sev = (AlertSeverity.CRITICAL if peri_alt < 80_000.0 else AlertSeverity.WARNING)

    return AlertMessage(
        severity=sev,
        category=AlertCategory.ORBITAL,
        message=(
            f"Periapsis altitude = {peri_alt/1000:.1f} km. "
            f"Orbit will intersect atmosphere."
        ),
        recommendation=(
            "Execute apoapsis burn to raise periapsis above 150 km. "
            f"Required Δv ≈ positive prograde burn at apoapsis."
        ),
        time_to_limit_s=math.inf,
        mission_time=t,
        parameter="periapsis",
        current_value=peri_alt,
        rate=0.0,
        limit=cfg.periapsis_warning_alt_m,
    )


def _check_mach_transonic(
    snap: TelemetrySnapshot,
    cfg:  MonitorConfig,
) -> Optional[AlertMessage]:
    """Caution when entering or exiting the transonic drag-divergence regime."""
    M = snap.mach
    t = snap.mission_time

    in_transonic = cfg.mach_transonic_lower <= M <= cfg.mach_transonic_upper
    if not in_transonic:
        return None

    return AlertMessage(
        severity=AlertSeverity.CAUTION,
        category=AlertCategory.AERODYNAMIC,
        message=(
            f"Transonic regime: M = {M:.3f}. "
            f"Wave drag active. Structural loads elevated."
        ),
        recommendation=(
            "Monitor dynamic pressure and structural margins closely. "
            "Avoid high AoA manoeuvres in transonic band."
        ),
        time_to_limit_s=math.inf,
        mission_time=t,
        parameter="mach",
        current_value=M,
        rate=0.0,
        limit=cfg.mach_transonic_upper,
    )


# ---------------------------------------------------------------------------
# Primary monitor function
# ---------------------------------------------------------------------------

def assess(
    registry: TelemetryRegistry,
    config:   MonitorConfig = MonitorConfig(),
) -> List[AlertMessage]:
    """
    Run all anomaly checks against the latest telemetry.

    Reads the immutable TelemetryRegistry. Never modifies any state.
    Called once per tick from pipeline Stage 10.

    Parameters
    ----------
    registry : TelemetryRegistry
        The live telemetry buffer (read-only access).
    config : MonitorConfig
        Alert thresholds and tuning parameters.

    Returns
    -------
    list of AlertMessage
        All active alerts, sorted by severity (CRITICAL first).
        Empty list if no anomalies detected.
    """
    snap = registry.latest
    if snap is None:
        return []

    # Compute derivative pack for all monitored scalar channels
    channels = [
        "alpha", "dynamic_pressure", "worst_structural_margin",
        "mach", "altitude", "mass",
    ]
    deriv = compute_derivative_pack(registry, channels, n_points=config.n_points)

    alerts: List[AlertMessage] = []

    # Aerodynamic checks
    a = _check_aoa(snap, deriv.get("alpha"), config)
    if a:
        alerts.append(a)

    a = _check_dynamic_pressure(snap, deriv.get("dynamic_pressure"), config)
    if a:
        alerts.append(a)

    a = _check_mach_transonic(snap, config)
    if a:
        alerts.append(a)

    # Structural checks
    a = _check_structural_margin(snap, deriv.get("worst_structural_margin"), config)
    if a:
        alerts.append(a)

    # Propulsion checks
    a = _check_propellant(snap, config)
    if a:
        alerts.append(a)

    # Orbital checks
    a = _check_periapsis(snap, config)
    if a:
        alerts.append(a)

    # Sort: CRITICAL first, then WARNING, CAUTION, INFO
    _ORDER = {
        AlertSeverity.CRITICAL: 0,
        AlertSeverity.WARNING:  1,
        AlertSeverity.CAUTION:  2,
        AlertSeverity.INFO:     3,
    }
    alerts.sort(key=lambda a: _ORDER[a.severity])

    return alerts


# ---------------------------------------------------------------------------
# Convenience: format all alerts as a single log string
# ---------------------------------------------------------------------------

def format_alerts(alerts: List[AlertMessage]) -> str:
    """
    Format a list of alerts as a multi-line string for the flight logger or HUD.

    Parameters
    ----------
    alerts : list of AlertMessage

    Returns
    -------
    str
        Empty string if no alerts.
    """
    if not alerts:
        return ""
    return "\n\n".join(a.format() for a in alerts)