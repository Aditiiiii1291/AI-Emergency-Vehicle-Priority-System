"""Phase 7 emergency priority recommendation engine.

This module consumes ambulance detection status, density analysis, and
congestion classification results. It produces simulated recommendation
actions only; it does not control traffic signals or hardware.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2


NORMAL_OPERATION = "NORMAL_OPERATION"
EXTEND_GREEN = "EXTEND_GREEN"
HIGH_TRAFFIC_WARNING = "HIGH_TRAFFIC_WARNING"
EMERGENCY_PRIORITY = "EMERGENCY_PRIORITY"

PRIORITY_RULES = {
    "emergency_present": EMERGENCY_PRIORITY,
    "HIGH_CONGESTION": HIGH_TRAFFIC_WARNING,
    "MEDIUM_CONGESTION": EXTEND_GREEN,
    "LOW_CONGESTION": NORMAL_OPERATION,
}

ACTION_COLORS = {
    NORMAL_OPERATION: (80, 180, 80),
    EXTEND_GREEN: (40, 180, 255),
    HIGH_TRAFFIC_WARNING: (40, 120, 255),
    EMERGENCY_PRIORITY: (40, 40, 255),
}

PRIORITY_LOG_COLUMNS = [
    "timestamp",
    "emergency_present",
    "density",
    "congestion",
    "recommended_action",
]


@dataclass(frozen=True)
class PriorityLogRecord:
    """CSV-compatible priority recommendation log record."""

    timestamp: str
    emergency_present: bool
    density: str
    congestion: str
    recommended_action: str


def _normalize_bool(value: bool | str | int) -> bool:
    """Normalize common bool-like values from CLI and CSV inputs."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0

    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y", "detected"}


def normalize_congestion(congestion: str) -> str:
    """Normalize congestion labels before applying priority rules."""

    normalized = str(congestion).strip().upper()
    allowed = {"LOW_CONGESTION", "MEDIUM_CONGESTION", "HIGH_CONGESTION"}
    if normalized not in allowed:
        raise ValueError(f"Unsupported congestion '{congestion}'. Expected one of: {', '.join(sorted(allowed))}")
    return normalized


def generate_priority_action(
    emergency_present: bool | str | int,
    density_result: dict[str, Any],
    congestion_result: dict[str, Any],
) -> dict[str, bool | str]:
    """Generate a simulated emergency priority recommendation."""

    emergency_detected = _normalize_bool(emergency_present)
    density = str(density_result.get("density", congestion_result.get("density", ""))).strip().upper()
    congestion = normalize_congestion(str(congestion_result.get("congestion", "")))

    if emergency_detected:
        recommended_action = PRIORITY_RULES["emergency_present"]
    else:
        recommended_action = PRIORITY_RULES[congestion]

    return {
        "emergency_present": emergency_detected,
        "density": density,
        "congestion": congestion,
        "recommended_action": recommended_action,
    }


def draw_priority_overlay(
    frame: Any,
    action_result: dict[str, bool | str],
    position: tuple[int, int] = (16, 304),
) -> Any:
    """Draw the recommended action on an annotated frame."""

    annotated = frame.copy()
    action = str(action_result["recommended_action"])
    color = ACTION_COLORS.get(action, (255, 255, 255))
    x, y = position
    label = f"Recommended Action: {action}"

    cv2.rectangle(annotated, (x, y), (x + 560, y + 44), color, -1)
    cv2.putText(
        annotated,
        label,
        (x + 12, y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def build_priority_log_record(
    action_result: dict[str, bool | str],
    timestamp: str | None = None,
) -> PriorityLogRecord:
    """Build one priority recommendation log record."""

    return PriorityLogRecord(
        timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
        emergency_present=bool(action_result["emergency_present"]),
        density=str(action_result["density"]),
        congestion=str(action_result["congestion"]),
        recommended_action=str(action_result["recommended_action"]),
    )


def write_priority_log(log_path: str | Path, records: Iterable[PriorityLogRecord]) -> None:
    """Write priority recommendation records to CSV."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=PRIORITY_LOG_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_congestion_log(log_path: str | Path) -> list[dict[str, str]]:
    """Read Phase 6 congestion CSV records."""

    with Path(log_path).open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def generate_actions_from_congestion_log(
    congestion_log_path: str | Path,
    output_log_path: str | Path = "data/logs/priority_actions.csv",
    emergency_present: bool | str | int = False,
) -> list[PriorityLogRecord]:
    """Create priority recommendation records from a Phase 6 congestion log."""

    congestion_rows = read_congestion_log(congestion_log_path)
    priority_records: list[PriorityLogRecord] = []

    for row in congestion_rows:
        density_result = {"density": row.get("density", "")}
        congestion_result = {
            "density": row.get("density", ""),
            "congestion": row.get("congestion", ""),
        }
        action_result = generate_priority_action(
            emergency_present=emergency_present,
            density_result=density_result,
            congestion_result=congestion_result,
        )
        priority_records.append(
            build_priority_log_record(
                action_result=action_result,
                timestamp=row.get("timestamp") or None,
            )
        )

    write_priority_log(output_log_path, priority_records)
    return priority_records


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for Phase 7 priority recommendations."""

    parser = argparse.ArgumentParser(description="Generate priority actions from a congestion CSV log.")
    parser.add_argument(
        "--congestion-log",
        required=True,
        help="Path to a Phase 6 congestion analysis CSV log.",
    )
    parser.add_argument(
        "--output-log",
        default="data/logs/priority_actions.csv",
        help="Path where the priority action CSV log will be saved.",
    )
    parser.add_argument(
        "--emergency-present",
        action="store_true",
        help="Treat every congestion row as having an emergency vehicle present.",
    )
    return parser


def main() -> None:
    """Run priority recommendation generation from a congestion CSV log."""

    args = build_arg_parser().parse_args()
    records = generate_actions_from_congestion_log(
        congestion_log_path=args.congestion_log,
        output_log_path=args.output_log,
        emergency_present=args.emergency_present,
    )

    print("Phase 7 priority action generation complete")
    print(f"records_written: {len(records)}")
    print(f"output_log: {args.output_log}")


if __name__ == "__main__":
    main()
