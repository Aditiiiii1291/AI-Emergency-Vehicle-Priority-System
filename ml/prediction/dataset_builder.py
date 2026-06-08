"""Build training-ready congestion datasets from historical project logs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DATASET_COLUMNS = [
    "timestamp",
    "total_vehicles",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "density",
    "congestion",
    "emergency_present",
    "recommended_action",
]

DEFAULT_OUTPUT_PATH = Path("data/datasets/congestion_training_dataset.csv")
MINIMUM_TRAINING_ROWS = 9


@dataclass(frozen=True)
class TrainingDatasetRecord:
    """One training-ready traffic history row."""

    timestamp: str
    total_vehicles: int
    car_count: int
    motorcycle_count: int
    bus_count: int
    truck_count: int
    density: str
    congestion: str
    emergency_present: bool
    recommended_action: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file if it exists."""

    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _write_dataset(output_path: str | Path, records: Iterable[TrainingDatasetRecord]) -> None:
    """Write training dataset rows to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def _to_bool(value: str | bool | int | None) -> bool:
    """Normalize emergency status values from logs."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "detected"}


def _timestamp_from_density(row: dict[str, str]) -> str:
    """Prefer timestamp_seconds for frame history when no wall-clock timestamp exists."""

    return row.get("timestamp") or row.get("timestamp_seconds") or ""


def _build_from_density_rows(
    density_rows: list[dict[str, str]],
    congestion_rows: list[dict[str, str]],
    priority_rows: list[dict[str, str]],
) -> list[TrainingDatasetRecord]:
    """Merge density, congestion, and priority logs into dataset rows."""

    congestion_by_timestamp = {row.get("timestamp", ""): row for row in congestion_rows}
    priority_by_timestamp = {row.get("timestamp", ""): row for row in priority_rows}
    records: list[TrainingDatasetRecord] = []

    for density_row in density_rows:
        timestamp = _timestamp_from_density(density_row)
        congestion_row = congestion_by_timestamp.get(timestamp, {})
        priority_row = priority_by_timestamp.get(timestamp, {})
        density = (density_row.get("density") or congestion_row.get("density") or "LOW").upper()
        congestion = (congestion_row.get("congestion") or _congestion_from_density(density)).upper()

        records.append(
            TrainingDatasetRecord(
                timestamp=timestamp,
                total_vehicles=int(float(density_row.get("total_vehicles", 0) or 0)),
                car_count=int(float(density_row.get("car_count", 0) or 0)),
                motorcycle_count=int(float(density_row.get("motorcycle_count", 0) or 0)),
                bus_count=int(float(density_row.get("bus_count", 0) or 0)),
                truck_count=int(float(density_row.get("truck_count", 0) or 0)),
                density=density,
                congestion=congestion,
                emergency_present=_to_bool(priority_row.get("emergency_present")),
                recommended_action=priority_row.get("recommended_action") or _action_from_congestion(False, congestion),
            )
        )

    return records


def _build_from_detection_rows(detection_rows: list[dict[str, str]]) -> list[TrainingDatasetRecord]:
    """Create dataset rows directly from Phase 3 detection logs."""

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in detection_rows:
        grouped[(row.get("source_video", ""), row.get("timestamp_seconds", ""))].append(row)

    records: list[TrainingDatasetRecord] = []
    for index, ((_source_video, timestamp), rows) in enumerate(sorted(grouped.items())):
        counts = {
            "car": 0,
            "motorcycle": 0,
            "bus": 0,
            "truck": 0,
        }
        emergency_present = False
        for row in rows:
            class_name = row.get("class_name", "").lower()
            if class_name in counts:
                counts[class_name] += 1
            if class_name == "ambulance":
                emergency_present = True

        total_vehicles = sum(counts.values())
        density = _density_from_total(total_vehicles)
        congestion = _congestion_from_density(density)
        records.append(
            TrainingDatasetRecord(
                timestamp=timestamp or str(index),
                total_vehicles=total_vehicles,
                car_count=counts["car"],
                motorcycle_count=counts["motorcycle"],
                bus_count=counts["bus"],
                truck_count=counts["truck"],
                density=density,
                congestion=congestion,
                emergency_present=emergency_present,
                recommended_action=_action_from_congestion(emergency_present, congestion),
            )
        )

    return records


def _density_from_total(total_vehicles: int) -> str:
    """Apply Phase 5 density thresholds."""

    if total_vehicles < 10:
        return "LOW"
    if total_vehicles < 25:
        return "MEDIUM"
    return "HIGH"


def _congestion_from_density(density: str) -> str:
    """Map density to Phase 6 congestion labels."""

    return {
        "LOW": "LOW_CONGESTION",
        "MEDIUM": "MEDIUM_CONGESTION",
        "HIGH": "HIGH_CONGESTION",
    }.get(density.upper(), "LOW_CONGESTION")


def _action_from_congestion(emergency_present: bool, congestion: str) -> str:
    """Map congestion and emergency status to Phase 7 action labels."""

    if emergency_present:
        return "EMERGENCY_PRIORITY"
    return {
        "LOW_CONGESTION": "NORMAL_OPERATION",
        "MEDIUM_CONGESTION": "EXTEND_GREEN",
        "HIGH_CONGESTION": "HIGH_TRAFFIC_WARNING",
    }.get(congestion.upper(), "NORMAL_OPERATION")


def generate_sample_training_records() -> list[TrainingDatasetRecord]:
    """Generate a small rule-consistent sample when historical logs are insufficient."""

    samples = [
        ("0", 4, 3, 1, 0, 0, False),
        ("1", 8, 5, 2, 1, 0, False),
        ("2", 11, 7, 2, 1, 1, False),
        ("3", 18, 10, 4, 2, 2, False),
        ("4", 23, 14, 5, 2, 2, True),
        ("5", 25, 15, 4, 3, 3, False),
        ("6", 31, 19, 5, 3, 4, False),
        ("7", 36, 21, 6, 4, 5, True),
        ("8", 15, 8, 4, 1, 2, True),
    ]

    records: list[TrainingDatasetRecord] = []
    for timestamp, total, cars, motorcycles, buses, trucks, emergency_present in samples:
        density = _density_from_total(total)
        congestion = _congestion_from_density(density)
        records.append(
            TrainingDatasetRecord(
                timestamp=timestamp,
                total_vehicles=total,
                car_count=cars,
                motorcycle_count=motorcycles,
                bus_count=buses,
                truck_count=trucks,
                density=density,
                congestion=congestion,
                emergency_present=emergency_present,
                recommended_action=_action_from_congestion(emergency_present, congestion),
            )
        )
    return records


def build_training_dataset(
    logs_dir: str | Path = "data/logs",
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    allow_sample_fallback: bool = True,
) -> list[TrainingDatasetRecord]:
    """Build and save a training dataset from historical logs."""

    logs = Path(logs_dir)
    density_rows = _read_csv(logs / "density_analysis.csv")
    congestion_rows = _read_csv(logs / "congestion_analysis.csv")
    priority_rows = _read_csv(logs / "priority_actions.csv")
    detection_rows = _read_csv(logs / "vehicle_detections.csv")

    records = _build_from_density_rows(density_rows, congestion_rows, priority_rows)
    if not records:
        records = _build_from_detection_rows(detection_rows)
    if allow_sample_fallback and len(records) < MINIMUM_TRAINING_ROWS:
        records = generate_sample_training_records()

    _write_dataset(output_path, records)
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the dataset builder CLI."""

    parser = argparse.ArgumentParser(description="Build a congestion training dataset from historical logs.")
    parser.add_argument("--logs", default="data/logs", help="Directory containing historical CSV logs.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Training dataset CSV output path.",
    )
    parser.add_argument(
        "--no-sample-fallback",
        action="store_true",
        help="Disable sample dataset generation when logs are insufficient.",
    )
    return parser


def main() -> None:
    """Run dataset generation from the command line."""

    args = build_arg_parser().parse_args()
    records = build_training_dataset(
        logs_dir=args.logs,
        output_path=args.output,
        allow_sample_fallback=not args.no_sample_fallback,
    )
    print("Training dataset generation complete")
    print(f"records_written: {len(records)}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
