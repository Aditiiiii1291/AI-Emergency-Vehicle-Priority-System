"""Phase 5 traffic density analysis.

This module consumes vehicle detection results and classifies traffic density.
It is intentionally independent of Streamlit, prediction, and priority logic.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2


LOW_VEHICLE_THRESHOLD = 10
MEDIUM_VEHICLE_THRESHOLD = 25

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")

DENSITY_LOG_COLUMNS = [
    "source_video",
    "frame_index",
    "timestamp_seconds",
    "total_vehicles",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "density",
]

DENSITY_COLORS = {
    "LOW": (80, 180, 80),
    "MEDIUM": (40, 180, 255),
    "HIGH": (40, 40, 255),
}


@dataclass(frozen=True)
class VehicleCounts:
    """Vehicle counts for a single frame or aggregation window."""

    total_vehicles: int
    car_count: int
    motorcycle_count: int
    bus_count: int
    truck_count: int


@dataclass(frozen=True)
class DensityLogRecord:
    """CSV-compatible density log record."""

    source_video: str
    frame_index: int
    timestamp_seconds: float
    total_vehicles: int
    car_count: int
    motorcycle_count: int
    bus_count: int
    truck_count: int
    density: str


def _get_class_name(detection: Any) -> str | None:
    """Read class_name from a dataclass/object or dict detection record."""

    if isinstance(detection, dict):
        class_name = detection.get("class_name")
    else:
        class_name = getattr(detection, "class_name", None)

    return str(class_name).lower() if class_name is not None else None


def count_vehicle_detections(detections: Iterable[Any]) -> VehicleCounts:
    """Count supported vehicle classes from Phase 3 detection results."""

    class_counts = Counter(
        class_name
        for detection in detections
        if (class_name := _get_class_name(detection)) in VEHICLE_CLASSES
    )

    car_count = class_counts.get("car", 0)
    motorcycle_count = class_counts.get("motorcycle", 0)
    bus_count = class_counts.get("bus", 0)
    truck_count = class_counts.get("truck", 0)
    total_vehicles = car_count + motorcycle_count + bus_count + truck_count

    return VehicleCounts(
        total_vehicles=total_vehicles,
        car_count=car_count,
        motorcycle_count=motorcycle_count,
        bus_count=bus_count,
        truck_count=truck_count,
    )


def normalize_vehicle_counts(vehicle_counts: VehicleCounts | dict[str, int]) -> VehicleCounts:
    """Convert dict or VehicleCounts input into a VehicleCounts object."""

    if isinstance(vehicle_counts, VehicleCounts):
        return vehicle_counts

    car_count = int(vehicle_counts.get("car_count", 0))
    motorcycle_count = int(vehicle_counts.get("motorcycle_count", 0))
    bus_count = int(vehicle_counts.get("bus_count", 0))
    truck_count = int(vehicle_counts.get("truck_count", 0))
    total_vehicles = int(
        vehicle_counts.get(
            "total_vehicles",
            car_count + motorcycle_count + bus_count + truck_count,
        )
    )

    return VehicleCounts(
        total_vehicles=total_vehicles,
        car_count=car_count,
        motorcycle_count=motorcycle_count,
        bus_count=bus_count,
        truck_count=truck_count,
    )


def classify_density(total_vehicles: int) -> str:
    """Classify vehicle volume into LOW, MEDIUM, or HIGH density."""

    if total_vehicles < LOW_VEHICLE_THRESHOLD:
        return "LOW"
    if total_vehicles < MEDIUM_VEHICLE_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


def analyze_density(vehicle_counts: VehicleCounts | dict[str, int]) -> dict[str, int | str]:
    """Analyze density from vehicle counts.

    Example:
        {
            "total_vehicles": 18,
            "car_count": 10,
            "motorcycle_count": 4,
            "bus_count": 2,
            "truck_count": 2,
            "density": "MEDIUM",
        }
    """

    counts = normalize_vehicle_counts(vehicle_counts)
    return {
        "total_vehicles": counts.total_vehicles,
        "car_count": counts.car_count,
        "motorcycle_count": counts.motorcycle_count,
        "bus_count": counts.bus_count,
        "truck_count": counts.truck_count,
        "density": classify_density(counts.total_vehicles),
    }


def draw_density_overlay(
    frame: Any,
    density_result: dict[str, int | str],
    position: tuple[int, int] = (16, 156),
) -> Any:
    """Draw density status on an annotated frame."""

    annotated = frame.copy()
    density = str(density_result["density"])
    total = int(density_result["total_vehicles"])
    color = DENSITY_COLORS.get(density, (255, 255, 255))
    x, y = position

    lines = [
        f"Density: {density}",
        f"Total Vehicles: {total}",
    ]

    cv2.rectangle(annotated, (x, y), (x + 300, y + 76), color, -1)
    text_y = y + 30
    for line in lines:
        cv2.putText(
            annotated,
            line,
            (x + 12, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        text_y += 30

    return annotated


def build_density_log_record(
    density_result: dict[str, int | str],
    source_video: str = "",
    frame_index: int = 0,
    timestamp_seconds: float = 0.0,
) -> DensityLogRecord:
    """Build one density log record from a density result."""

    return DensityLogRecord(
        source_video=source_video,
        frame_index=frame_index,
        timestamp_seconds=round(timestamp_seconds, 3),
        total_vehicles=int(density_result["total_vehicles"]),
        car_count=int(density_result["car_count"]),
        motorcycle_count=int(density_result["motorcycle_count"]),
        bus_count=int(density_result["bus_count"]),
        truck_count=int(density_result["truck_count"]),
        density=str(density_result["density"]),
    )


def write_density_log(log_path: str | Path, records: Iterable[DensityLogRecord]) -> None:
    """Write density analysis records to CSV."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DENSITY_LOG_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def read_detection_log(log_path: str | Path) -> list[dict[str, str]]:
    """Read Phase 3-compatible detection CSV records."""

    with Path(log_path).open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def analyze_density_from_detection_log(
    detection_log_path: str | Path,
    density_log_path: str | Path = "data/logs/density_analysis.csv",
) -> list[DensityLogRecord]:
    """Create per-frame density records from a Phase 3 detection log CSV."""

    detection_rows = read_detection_log(detection_log_path)
    grouped_rows: dict[tuple[str, int, float], list[dict[str, str]]] = {}

    for row in detection_rows:
        source_video = row.get("source_video", "")
        frame_index = int(row.get("frame_index", 0))
        timestamp_seconds = float(row.get("timestamp_seconds", 0.0))
        grouped_rows.setdefault((source_video, frame_index, timestamp_seconds), []).append(row)

    density_records: list[DensityLogRecord] = []
    for (source_video, frame_index, timestamp_seconds), detections in sorted(grouped_rows.items()):
        vehicle_counts = count_vehicle_detections(detections)
        density_result = analyze_density(vehicle_counts)
        density_records.append(
            build_density_log_record(
                density_result=density_result,
                source_video=source_video,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )
        )

    write_density_log(density_log_path, density_records)
    return density_records


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for Phase 5 density log analysis."""

    parser = argparse.ArgumentParser(description="Analyze traffic density from a detection CSV log.")
    parser.add_argument(
        "--detections-log",
        required=True,
        help="Path to a Phase 3 vehicle detection CSV log.",
    )
    parser.add_argument(
        "--output-log",
        default="data/logs/density_analysis.csv",
        help="Path where the density analysis CSV log will be saved.",
    )
    return parser


def main() -> None:
    """Run density analysis from a detection CSV log."""

    args = build_arg_parser().parse_args()
    records = analyze_density_from_detection_log(
        detection_log_path=args.detections_log,
        density_log_path=args.output_log,
    )

    print("Phase 5 density analysis complete")
    print(f"records_written: {len(records)}")
    print(f"output_log: {args.output_log}")


if __name__ == "__main__":
    main()
