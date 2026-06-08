"""Emergency green corridor simulation.

This module consumes enriched lane analysis results and produces rule-based
emergency corridor recommendations. It is a simulation layer only and does not
control real traffic signals or hardware.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


INACTIVE = "INACTIVE"
READY = "READY"
ACTIVE_SIMULATION = "ACTIVE_SIMULATION"

PRIORITIZE_GREEN = "PRIORITIZE_GREEN"
HOLD_TRAFFIC = "HOLD_TRAFFIC"
MONITOR = "MONITOR"

CONFIDENCE_NOTE = "Rule-based corridor simulation"
DEFAULT_EMERGENCY_GREEN_SECONDS = 75
HOLD_TRAFFIC_GREEN_SECONDS = 20
DEFAULT_LOG_PATH = Path("data/logs/green_corridor_log.csv")
DEFAULT_SEQUENCE_LOG_PATH = Path("data/logs/green_corridor_sequence_log.csv")

GREEN_CORRIDOR_LOG_COLUMNS = [
    "timestamp",
    "corridor_active",
    "emergency_lane_id",
    "emergency_lane_name",
    "corridor_status",
    "estimated_clearance_window_seconds",
    "reason",
]

GREEN_CORRIDOR_SEQUENCE_LOG_COLUMNS = [
    "timestamp",
    "lane_id",
    "lane_name",
    "action",
    "green_seconds",
    "reason",
]


@dataclass(frozen=True)
class CorridorLaneAction:
    """One lane action in a simulated corridor sequence."""

    lane_id: str
    lane_name: str
    action: str
    green_seconds: int
    reason: str


@dataclass(frozen=True)
class GreenCorridorResult:
    """Structured green corridor simulation result."""

    corridor_active: bool
    emergency_lane_id: str
    emergency_lane_name: str
    corridor_status: str
    recommended_sequence: list[dict[str, int | str]]
    estimated_clearance_window_seconds: int
    reason: str
    confidence_note: str


@dataclass(frozen=True)
class GreenCorridorLogRecord:
    """CSV-compatible corridor summary log record."""

    timestamp: str
    corridor_active: bool
    emergency_lane_id: str
    emergency_lane_name: str
    corridor_status: str
    estimated_clearance_window_seconds: int
    reason: str


@dataclass(frozen=True)
class GreenCorridorSequenceLogRecord:
    """CSV-compatible corridor sequence log record."""

    timestamp: str
    lane_id: str
    lane_name: str
    action: str
    green_seconds: int
    reason: str


def _safe_bool(value: Any) -> bool:
    """Normalize bool-like values without raising."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "detected"}


def _safe_int(value: Any, default: int = 0) -> int:
    """Normalize integer-like values without raising."""

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Normalize float-like values without raising."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, default: str = "") -> str:
    """Normalize text values without raising."""

    if value is None:
        return default
    try:
        text = str(value).strip()
    except Exception:
        return default
    return text or default


def _inactive_result(reason: str = "No emergency vehicle detected") -> dict[str, Any]:
    """Return a dictionary for an inactive corridor simulation."""

    return asdict(
        GreenCorridorResult(
            corridor_active=False,
            emergency_lane_id="N/A",
            emergency_lane_name="N/A",
            corridor_status=INACTIVE,
            recommended_sequence=[],
            estimated_clearance_window_seconds=0,
            reason=reason,
            confidence_note=CONFIDENCE_NOTE,
        )
    )


def _normalize_lane(lane: Any) -> dict[str, Any] | None:
    """Normalize a lane result dictionary and reject malformed entries safely."""

    if not isinstance(lane, dict):
        logger.warning("Skipping malformed lane result: %s", lane)
        return None
    lane_id = _safe_text(lane.get("lane_id"))
    lane_name = _safe_text(lane.get("lane_name"), lane_id)
    if not lane_id:
        logger.warning("Skipping lane result without lane_id: %s", lane)
        return None
    return {
        "lane_id": lane_id,
        "lane_name": lane_name,
        "total_vehicles": _safe_int(lane.get("total_vehicles"), 0),
        "lane_utilization_percent": _safe_float(lane.get("lane_utilization_percent"), 0.0),
        "emergency_present": _safe_bool(lane.get("emergency_present")),
        "recommended_green_seconds": _safe_int(
            lane.get("recommended_green_seconds"),
            DEFAULT_EMERGENCY_GREEN_SECONDS,
        ),
        "congestion": _safe_text(lane.get("congestion"), "LOW_CONGESTION").upper(),
    }


def find_emergency_lane(lane_results: Iterable[Any]) -> dict[str, Any] | None:
    """Select the emergency lane using utilization, count, then lane ID."""

    normalized_lanes = [
        normalized
        for lane in lane_results or []
        if (normalized := _normalize_lane(lane)) is not None
    ]
    emergency_lanes = [lane for lane in normalized_lanes if lane["emergency_present"]]
    if not emergency_lanes:
        return None
    return sorted(
        emergency_lanes,
        key=lambda lane: (
            -float(lane["lane_utilization_percent"]),
            -int(lane["total_vehicles"]),
            str(lane["lane_id"]),
        ),
    )[0]


def build_corridor_sequence(lane_results: Iterable[Any], emergency_lane: dict[str, Any]) -> list[dict[str, int | str]]:
    """Build a deterministic lane action sequence for the simulated corridor."""

    normalized_lanes = [
        normalized
        for lane in lane_results or []
        if (normalized := _normalize_lane(lane)) is not None
    ]
    actions: list[CorridorLaneAction] = []

    for lane in sorted(normalized_lanes, key=lambda item: str(item["lane_id"])):
        if lane["lane_id"] == emergency_lane["lane_id"]:
            actions.append(
                CorridorLaneAction(
                    lane_id=lane["lane_id"],
                    lane_name=lane["lane_name"],
                    action=PRIORITIZE_GREEN,
                    green_seconds=_safe_int(
                        lane.get("recommended_green_seconds"),
                        DEFAULT_EMERGENCY_GREEN_SECONDS,
                    ),
                    reason="Emergency lane",
                )
            )
        elif lane["congestion"] == "LOW_CONGESTION":
            actions.append(
                CorridorLaneAction(
                    lane_id=lane["lane_id"],
                    lane_name=lane["lane_name"],
                    action=MONITOR,
                    green_seconds=HOLD_TRAFFIC_GREEN_SECONDS,
                    reason="Low congestion lane",
                )
            )
        else:
            actions.append(
                CorridorLaneAction(
                    lane_id=lane["lane_id"],
                    lane_name=lane["lane_name"],
                    action=HOLD_TRAFFIC,
                    green_seconds=HOLD_TRAFFIC_GREEN_SECONDS,
                    reason="Hold non-emergency lane",
                )
            )

    return [asdict(action) for action in sorted(actions, key=lambda action: 0 if action.action == PRIORITIZE_GREEN else 1)]


def simulate_green_corridor(
    lane_results: Iterable[Any],
    global_priority_result: dict[str, Any] | None = None,
    global_signal_timing_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate an emergency green corridor from enriched lane results."""

    try:
        lanes = list(lane_results or [])
        if not lanes:
            return _inactive_result()

        emergency_lane = find_emergency_lane(lanes)
        if emergency_lane is None:
            return _inactive_result()

        sequence = build_corridor_sequence(lanes, emergency_lane)
        emergency_action = next(
            (action for action in sequence if action["lane_id"] == emergency_lane["lane_id"]),
            None,
        )
        clearance_window = _safe_int(
            emergency_action.get("green_seconds") if emergency_action else None,
            _safe_int(
                (global_signal_timing_result or {}).get("recommended_green_seconds"),
                DEFAULT_EMERGENCY_GREEN_SECONDS,
            ),
        )
        reason = f"Emergency vehicle detected in {emergency_lane['lane_name']}"

        return asdict(
            GreenCorridorResult(
                corridor_active=True,
                emergency_lane_id=str(emergency_lane["lane_id"]),
                emergency_lane_name=str(emergency_lane["lane_name"]),
                corridor_status=ACTIVE_SIMULATION,
                recommended_sequence=sequence,
                estimated_clearance_window_seconds=clearance_window,
                reason=reason,
                confidence_note=CONFIDENCE_NOTE,
            )
        )
    except Exception as error:
        logger.warning("Green corridor simulation fell back to inactive state: %s", error)
        return _inactive_result("Malformed lane results")


def build_green_corridor_log_record(
    corridor_result: dict[str, Any],
    timestamp: str | None = None,
) -> GreenCorridorLogRecord:
    """Build a corridor summary log record."""

    return GreenCorridorLogRecord(
        timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
        corridor_active=_safe_bool(corridor_result.get("corridor_active")),
        emergency_lane_id=_safe_text(corridor_result.get("emergency_lane_id"), "N/A"),
        emergency_lane_name=_safe_text(corridor_result.get("emergency_lane_name"), "N/A"),
        corridor_status=_safe_text(corridor_result.get("corridor_status"), INACTIVE),
        estimated_clearance_window_seconds=_safe_int(corridor_result.get("estimated_clearance_window_seconds"), 0),
        reason=_safe_text(corridor_result.get("reason"), "No emergency vehicle detected"),
    )


def build_green_corridor_sequence_log_records(
    corridor_result: dict[str, Any],
    timestamp: str | None = None,
) -> list[GreenCorridorSequenceLogRecord]:
    """Build corridor sequence log records."""

    record_timestamp = timestamp or datetime.now().isoformat(timespec="seconds")
    records: list[GreenCorridorSequenceLogRecord] = []
    for action in corridor_result.get("recommended_sequence", []) or []:
        if not isinstance(action, dict):
            logger.warning("Skipping malformed corridor sequence action: %s", action)
            continue
        records.append(
            GreenCorridorSequenceLogRecord(
                timestamp=record_timestamp,
                lane_id=_safe_text(action.get("lane_id"), "N/A"),
                lane_name=_safe_text(action.get("lane_name"), "N/A"),
                action=_safe_text(action.get("action"), MONITOR),
                green_seconds=_safe_int(action.get("green_seconds"), 0),
                reason=_safe_text(action.get("reason"), ""),
            )
        )
    return records


def write_green_corridor_log(
    records: Iterable[GreenCorridorLogRecord],
    log_path: str | Path = DEFAULT_LOG_PATH,
) -> bool:
    """Write green corridor summary records to CSV without crashing callers."""

    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=GREEN_CORRIDOR_LOG_COLUMNS)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
        return True
    except Exception as error:
        logger.error("Failed to write green corridor log to %s: %s", path, error)
        return False


def write_green_corridor_sequence_log(
    records: Iterable[GreenCorridorSequenceLogRecord],
    log_path: str | Path = DEFAULT_SEQUENCE_LOG_PATH,
) -> bool:
    """Write green corridor sequence records to CSV without crashing callers."""

    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=GREEN_CORRIDOR_SEQUENCE_LOG_COLUMNS)
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
        return True
    except Exception as error:
        logger.error("Failed to write green corridor sequence log to %s: %s", path, error)
        return False
