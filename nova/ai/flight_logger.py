"""
nova.ai.flight_logger
=====================
Structured event log writer for Project NOVA.

Architecture role — Pipeline Stage 10 (alongside AI Monitor)
-------------------------------------------------------------
The FlightLogger consumes TelemetrySnapshots from the registry and writes
two output artefacts:

1. **Telemetry CSV** — one row per published snapshot, one column per
   monitored scalar field. Suitable for post-flight analysis in NumPy,
   pandas, or any spreadsheet tool.

2. **Event JSON** — discrete flight events detected from telemetry
   transitions (engine ignition, MECO, max-Q, staging, orbit insertion,
   structural failure). Each event carries its mission time, type,
   description, and relevant parameter values.

Design contract
---------------
* The logger is purely append-only. It never reads back its own output.
* All writes are to in-memory buffers (list of rows / list of events).
  File I/O is explicit via ``save_csv()`` and ``save_json()``.
* The logger is stateful (it tracks previous snapshot for transition
  detection) but never modifies TelemetrySnapshot or VehicleState.
* Thread safety is NOT guaranteed — the logger runs synchronously in the
  single-threaded pipeline.
"""

from __future__ import annotations

import csv
import io
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from nova.core.telemetry_registry import TelemetrySnapshot
from nova.ai.monitor import AlertMessage, AlertSeverity


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

class EventType(Enum):
    ENGINE_IGNITION    = "ENGINE_IGNITION"
    ENGINE_CUTOFF      = "ENGINE_CUTOFF"
    MAX_Q              = "MAX_Q"
    STAGE_SEPARATION   = "STAGE_SEPARATION"
    ORBIT_INSERTION    = "ORBIT_INSERTION"
    REENTRY_INTERFACE  = "REENTRY_INTERFACE"
    STRUCTURAL_FAILURE = "STRUCTURAL_FAILURE"
    ALERT_CRITICAL     = "ALERT_CRITICAL"
    ALERT_WARNING      = "ALERT_WARNING"
    MILESTONE          = "MILESTONE"


@dataclass(frozen=True)
class FlightEvent:
    """
    A single discrete flight event detected from telemetry transitions.

    Attributes
    ----------
    event_type : EventType
    mission_time : float      Mission elapsed time [s].
    description : str         Human-readable event description.
    parameters : dict         Relevant numerical parameters at event time.
    """
    event_type:   EventType
    mission_time: float
    description:  str
    parameters:   Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type":   self.event_type.value,
            "mission_time": self.mission_time,
            "description":  self.description,
            "parameters":   self.parameters,
        }


# ---------------------------------------------------------------------------
# Telemetry CSV column schema
# ---------------------------------------------------------------------------

CSV_COLUMNS: List[str] = [
    "mission_time", "mass", "altitude", "speed", "vertical_speed",
    "mach", "dynamic_pressure", "alpha", "beta", "CL", "CD",
    "thrust_magnitude", "mass_flow_rate", "isp_effective", "throttle",
    "semi_major_axis", "eccentricity", "inclination", "orbital_period",
    "apoapsis", "periapsis", "worst_structural_margin", "any_structural_failure",
    "lift_force", "drag_force", "density", "pressure", "speed_of_sound",
    "downrange_distance",
]


def _snap_to_row(snap: TelemetrySnapshot) -> Dict[str, float]:
    row: Dict[str, float] = {}
    for col in CSV_COLUMNS:
        val = getattr(snap, col, None)
        if val is None:
            row[col] = float("nan")
        elif isinstance(val, bool):
            row[col] = 1.0 if val else 0.0
        else:
            try:
                row[col] = float(val)
            except (TypeError, ValueError):
                row[col] = float("nan")
    return row


# ---------------------------------------------------------------------------
# FlightLogger
# ---------------------------------------------------------------------------

class FlightLogger:
    """
    Stateful flight data recorder.

    Accumulates telemetry rows and discrete events in memory.
    Exports to CSV and JSON on demand.

    Parameters
    ----------
    max_rows : int
        Maximum telemetry rows to buffer. Default 100 000.
    """

    def __init__(self, max_rows: int = 100_000) -> None:
        if max_rows < 1:
            raise ValueError(f"max_rows must be >= 1, got {max_rows!r}")
        self._max_rows = max_rows
        self._rows:   List[Dict[str, float]] = []
        self._events: List[FlightEvent]      = []

        # Transition-detection state
        self._prev_snap: Optional[TelemetrySnapshot] = None
        self._max_q_seen:        float = 0.0
        self._max_q_logged:      bool  = False
        self._engine_was_active: bool  = False
        self._was_in_orbit:      bool  = False
        self._was_descending:    bool  = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, snap: TelemetrySnapshot) -> None:
        """Append a snapshot to the telemetry CSV buffer."""
        row = _snap_to_row(snap)
        if len(self._rows) >= self._max_rows:
            self._rows.pop(0)
        self._rows.append(row)

    def detect_events(
        self,
        snap:   TelemetrySnapshot,
        alerts: Optional[List[AlertMessage]] = None,
    ) -> List[FlightEvent]:
        """
        Detect discrete flight events from telemetry transitions.

        Parameters
        ----------
        snap : TelemetrySnapshot
            Current tick snapshot.
        alerts : list of AlertMessage, optional
            Active alerts from the AI monitor.

        Returns
        -------
        list of FlightEvent
            New events detected this tick (may be empty).
        """
        new_events: List[FlightEvent] = []
        t    = snap.mission_time
        prev = self._prev_snap

        # ── Engine ignition / cutoff ────────────────────────────────
        engine_active = snap.thrust_magnitude > 1.0
        if engine_active and not self._engine_was_active:
            new_events.append(FlightEvent(
                event_type=EventType.ENGINE_IGNITION,
                mission_time=t,
                description=f"Engine ignition. Throttle={snap.throttle:.2f}.",
                parameters={"thrust_N": snap.thrust_magnitude,
                            "throttle": snap.throttle,
                            "mass_kg":  snap.mass},
            ))
        elif not engine_active and self._engine_was_active:
            new_events.append(FlightEvent(
                event_type=EventType.ENGINE_CUTOFF,
                mission_time=t,
                description="Main engine cutoff (MECO).",
                parameters={"mass_kg":    snap.mass,
                            "speed_ms":   snap.speed,
                            "altitude_m": snap.altitude},
            ))
        self._engine_was_active = engine_active

        # ── Max-Q detection ─────────────────────────────────────────
        q_inf = snap.dynamic_pressure
        if q_inf > self._max_q_seen:
            self._max_q_seen   = q_inf
            self._max_q_logged = False
        elif (not self._max_q_logged
              and prev is not None
              and q_inf < prev.dynamic_pressure
              and self._max_q_seen > 1_000.0):
            new_events.append(FlightEvent(
                event_type=EventType.MAX_Q,
                mission_time=t,
                description=(
                    f"Max-Q: {self._max_q_seen/1000:.2f} kPa at "
                    f"alt={snap.altitude/1000:.1f} km, M={snap.mach:.3f}."
                ),
                parameters={"max_q_pa":   self._max_q_seen,
                            "altitude_m": snap.altitude,
                            "mach":       snap.mach,
                            "mass_kg":    snap.mass},
            ))
            self._max_q_logged = True

        # ── Orbit insertion ─────────────────────────────────────────
        in_orbit = (
            snap.eccentricity < 0.05
            and snap.periapsis > 6_371_000.0 + 80_000.0
            and snap.altitude > 80_000.0
        )
        if in_orbit and not self._was_in_orbit:
            new_events.append(FlightEvent(
                event_type=EventType.ORBIT_INSERTION,
                mission_time=t,
                description=(
                    f"Orbit insertion confirmed. "
                    f"a={snap.semi_major_axis/1000:.1f}km, "
                    f"e={snap.eccentricity:.4f}, "
                    f"i={math.degrees(snap.inclination):.2f}deg."
                ),
                parameters={"semi_major_axis_km": snap.semi_major_axis / 1000.0,
                            "eccentricity":       snap.eccentricity,
                            "inclination_deg":    math.degrees(snap.inclination),
                            "period_s":           snap.orbital_period},
            ))
        self._was_in_orbit = in_orbit

        # ── Reentry interface ───────────────────────────────────────
        descending  = prev is not None and snap.altitude < prev.altitude
        at_reentry  = 0.0 < snap.altitude < 120_000.0
        if at_reentry and descending and not self._was_descending:
            new_events.append(FlightEvent(
                event_type=EventType.REENTRY_INTERFACE,
                mission_time=t,
                description=(
                    f"Reentry interface. "
                    f"alt={snap.altitude/1000:.1f}km, v={snap.speed:.0f}m/s."
                ),
                parameters={"altitude_m": snap.altitude,
                            "speed_ms":   snap.speed,
                            "mach":       snap.mach},
            ))
        self._was_descending = at_reentry and descending

        # ── Structural failure ──────────────────────────────────────
        if snap.any_structural_failure:
            already = any(
                e.event_type == EventType.STRUCTURAL_FAILURE
                and abs(e.mission_time - t) < 0.1
                for e in self._events
            )
            if not already:
                new_events.append(FlightEvent(
                    event_type=EventType.STRUCTURAL_FAILURE,
                    mission_time=t,
                    description=(
                        f"Structural failure: joint '{snap.critical_joint_id}'. "
                        f"Margin={snap.worst_structural_margin:.3f}."
                    ),
                    parameters={"margin": snap.worst_structural_margin,
                                "q_inf":  snap.dynamic_pressure},
                ))

        # ── Critical alerts ─────────────────────────────────────────
        if alerts:
            for alert in alerts:
                if alert.severity == AlertSeverity.CRITICAL:
                    new_events.append(FlightEvent(
                        event_type=EventType.ALERT_CRITICAL,
                        mission_time=t,
                        description=alert.message,
                        parameters={
                            "current_value": alert.current_value,
                            "rate":          alert.rate,
                            "time_to_limit": (alert.time_to_limit_s
                                              if math.isfinite(alert.time_to_limit_s)
                                              else -1.0),
                        },
                    ))

        self._events.extend(new_events)
        self._prev_snap = snap
        return new_events

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_csv(self, path: str) -> None:
        """Write telemetry buffer to a CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self._rows)

    def save_json(self, path: str) -> None:
        """Write event log to a JSON file."""
        payload = {"event_count": len(self._events),
                   "events": [e.to_dict() for e in self._events]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def csv_string(self) -> str:
        """Return the CSV as an in-memory string (for testing)."""
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(self._rows)
        return buf.getvalue()

    def events_json_string(self) -> str:
        """Return the event log as an in-memory JSON string."""
        payload = {"event_count": len(self._events),
                   "events": [e.to_dict() for e in self._events]}
        return json.dumps(payload, indent=2)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def row_count(self) -> int:
        return len(self._rows)

    @property
    def event_count(self) -> int:
        return len(self._events)

    @property
    def events(self) -> List[FlightEvent]:
        return list(self._events)

    def events_of_type(self, event_type: EventType) -> List[FlightEvent]:
        return [e for e in self._events if e.event_type == event_type]

    def clear(self) -> None:
        """Discard all buffered rows and events."""
        self._rows.clear()
        self._events.clear()
        self._prev_snap        = None
        self._max_q_seen       = 0.0
        self._max_q_logged     = False
        self._engine_was_active = False
        self._was_in_orbit      = False
        self._was_descending    = False

    def __repr__(self) -> str:
        return f"FlightLogger(rows={self.row_count}, events={self.event_count})"