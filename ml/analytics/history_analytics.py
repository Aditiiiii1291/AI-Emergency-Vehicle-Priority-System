"""Historical analytics over detection, density, congestion, and priority logs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CONGESTION_LEVELS = ["LOW_CONGESTION", "MEDIUM_CONGESTION", "HIGH_CONGESTION"]
RECOMMENDATION_ACTIONS = ["NORMAL_OPERATION", "EXTEND_GREEN", "HIGH_TRAFFIC_WARNING", "EMERGENCY_PRIORITY"]


@dataclass(frozen=True)
class HistoricalRecord:
    """One historical analytics record."""

    timestamp: str
    total_vehicles: int
    density: str
    congestion: str
    emergency_present: bool
    recommended_action: str


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file when present; return an empty list when missing."""

    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def _to_bool(value: str | bool | int | None) -> bool:
    """Normalize bool-like log values."""

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "detected"}


def _to_int(value: str | int | float | None) -> int:
    """Normalize numeric values from CSV logs."""

    if value in (None, ""):
        return 0
    return int(float(value))


def _timestamp(row: dict[str, str]) -> str:
    """Extract the best timestamp-like field from a log row."""

    return row.get("timestamp") or row.get("timestamp_seconds") or ""


def _density_from_total(total_vehicles: int) -> str:
    """Fallback density classification for detection-only logs."""

    if total_vehicles < 10:
        return "LOW"
    if total_vehicles < 25:
        return "MEDIUM"
    return "HIGH"


def _congestion_from_density(density: str) -> str:
    """Fallback congestion mapping."""

    return {
        "LOW": "LOW_CONGESTION",
        "MEDIUM": "MEDIUM_CONGESTION",
        "HIGH": "HIGH_CONGESTION",
    }.get(density.upper(), "LOW_CONGESTION")


def _recommendation_from_congestion(emergency_present: bool, congestion: str) -> str:
    """Fallback recommendation mapping."""

    if emergency_present:
        return "EMERGENCY_PRIORITY"
    return {
        "LOW_CONGESTION": "NORMAL_OPERATION",
        "MEDIUM_CONGESTION": "EXTEND_GREEN",
        "HIGH_CONGESTION": "HIGH_TRAFFIC_WARNING",
    }.get(congestion.upper(), "NORMAL_OPERATION")


def load_historical_records(logs_dir: str | Path = "data/logs") -> list[HistoricalRecord]:
    """Load and merge available historical records from project logs."""

    logs = Path(logs_dir)
    density_rows = _read_csv(logs / "density_analysis.csv")
    congestion_rows = _read_csv(logs / "congestion_analysis.csv")
    priority_rows = _read_csv(logs / "priority_actions.csv")
    detection_rows = _read_csv(logs / "vehicle_detections.csv")
    emergency_rows = _read_csv(logs / "emergency_detections.csv")

    records_by_timestamp: dict[str, dict[str, object]] = {}

    for row in density_rows:
        timestamp = _timestamp(row)
        records_by_timestamp[timestamp] = {
            "timestamp": timestamp,
            "total_vehicles": _to_int(row.get("total_vehicles")),
            "density": (row.get("density") or "LOW").upper(),
            "congestion": _congestion_from_density(row.get("density") or "LOW"),
            "emergency_present": False,
            "recommended_action": "NORMAL_OPERATION",
        }

    for row in congestion_rows:
        timestamp = _timestamp(row)
        record = records_by_timestamp.setdefault(
            timestamp,
            {
                "timestamp": timestamp,
                "total_vehicles": _to_int(row.get("total_vehicles")),
                "density": (row.get("density") or "LOW").upper(),
                "emergency_present": False,
                "recommended_action": "NORMAL_OPERATION",
            },
        )
        record["total_vehicles"] = _to_int(row.get("total_vehicles")) or int(record.get("total_vehicles", 0))
        record["density"] = (row.get("density") or str(record.get("density", "LOW"))).upper()
        record["congestion"] = (row.get("congestion") or _congestion_from_density(str(record["density"]))).upper()

    for row in priority_rows:
        timestamp = _timestamp(row)
        record = records_by_timestamp.setdefault(
            timestamp,
            {
                "timestamp": timestamp,
                "total_vehicles": 0,
                "density": (row.get("density") or "LOW").upper(),
                "congestion": (row.get("congestion") or "LOW_CONGESTION").upper(),
            },
        )
        record["density"] = (row.get("density") or str(record.get("density", "LOW"))).upper()
        record["congestion"] = (row.get("congestion") or str(record.get("congestion", "LOW_CONGESTION"))).upper()
        record["emergency_present"] = _to_bool(row.get("emergency_present"))
        record["recommended_action"] = row.get("recommended_action") or _recommendation_from_congestion(
            bool(record["emergency_present"]),
            str(record["congestion"]),
        )

    if not records_by_timestamp and detection_rows:
        grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in detection_rows:
            grouped_rows[_timestamp(row)].append(row)
        for timestamp, rows in grouped_rows.items():
            total_vehicles = sum(1 for row in rows if row.get("class_name", "").lower() in {"car", "motorcycle", "bus", "truck"})
            emergency_present = any(row.get("class_name", "").lower() == "ambulance" for row in rows)
            density = _density_from_total(total_vehicles)
            congestion = _congestion_from_density(density)
            records_by_timestamp[timestamp] = {
                "timestamp": timestamp,
                "total_vehicles": total_vehicles,
                "density": density,
                "congestion": congestion,
                "emergency_present": emergency_present,
                "recommended_action": _recommendation_from_congestion(emergency_present, congestion),
            }

    emergency_timestamps = {
        _timestamp(row)
        for row in emergency_rows
        if row.get("class_name", "").lower() == "ambulance"
    }
    for timestamp in emergency_timestamps:
        if timestamp in records_by_timestamp:
            records_by_timestamp[timestamp]["emergency_present"] = True
            records_by_timestamp[timestamp]["recommended_action"] = "EMERGENCY_PRIORITY"

    return [
        HistoricalRecord(
            timestamp=str(record.get("timestamp", "")),
            total_vehicles=int(record.get("total_vehicles", 0)),
            density=str(record.get("density", "LOW")).upper(),
            congestion=str(record.get("congestion", "LOW_CONGESTION")).upper(),
            emergency_present=bool(record.get("emergency_present", False)),
            recommended_action=str(record.get("recommended_action", "NORMAL_OPERATION")).upper(),
        )
        for record in sorted(records_by_timestamp.values(), key=lambda item: str(item.get("timestamp", "")))
    ]


def filter_records(
    records: Iterable[HistoricalRecord],
    date_filter: str | None = None,
    congestion_level: str | None = None,
    recommendation: str | None = None,
) -> list[HistoricalRecord]:
    """Filter historical records by timestamp prefix, congestion, and recommendation."""

    filtered = list(records)
    if date_filter:
        filtered = [record for record in filtered if record.timestamp.startswith(date_filter)]
    if congestion_level and congestion_level != "ALL":
        filtered = [record for record in filtered if record.congestion == congestion_level]
    if recommendation and recommendation != "ALL":
        filtered = [record for record in filtered if record.recommended_action == recommendation]
    return filtered


def _most_common(counter: Counter[str]) -> str:
    """Return the most common label or N/A."""

    return counter.most_common(1)[0][0] if counter else "N/A"


def generate_summary(records: Iterable[HistoricalRecord]) -> dict[str, int | str]:
    """Generate high-level historical summary metrics."""

    record_list = list(records)
    congestion_counts = Counter(record.congestion for record in record_list)
    recommendation_counts = Counter(record.recommended_action for record in record_list)
    return {
        "total_analyzed_records": len(record_list),
        "total_emergency_events": sum(1 for record in record_list if record.emergency_present),
        "most_common_congestion_level": _most_common(congestion_counts),
        "most_common_recommendation": _most_common(recommendation_counts),
    }


def generate_trend_data(records: Iterable[HistoricalRecord]) -> dict[str, object]:
    """Generate chart-friendly trend and distribution data."""

    record_list = list(records)
    return {
        "vehicle_count_over_time": [
            {"timestamp": record.timestamp, "total_vehicles": record.total_vehicles}
            for record in record_list
        ],
        "density_distribution": dict(Counter(record.density for record in record_list)),
        "congestion_distribution": dict(Counter(record.congestion for record in record_list)),
        "recommendation_distribution": dict(Counter(record.recommended_action for record in record_list)),
    }


def generate_event_statistics(records: Iterable[HistoricalRecord]) -> dict[str, object]:
    """Generate emergency and recommendation event statistics."""

    record_list = list(records)
    emergency_events = [record for record in record_list if record.emergency_present]
    return {
        "emergency_event_frequency": len(emergency_events),
        "recommendation_frequency": dict(Counter(record.recommended_action for record in record_list)),
        "emergency_rate": round(len(emergency_events) / len(record_list), 4) if record_list else 0.0,
    }


def export_history_summary(records: Iterable[HistoricalRecord], output_path: str | Path) -> None:
    """Write merged historical records to CSV for inspection."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(HistoricalRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the historical analytics CLI."""

    parser = argparse.ArgumentParser(description="Generate historical analytics from project logs.")
    parser.add_argument("--logs", default="data/logs", help="Directory containing historical CSV logs.")
    parser.add_argument("--date", default=None, help="Optional timestamp/date prefix filter.")
    parser.add_argument("--congestion", default="ALL", choices=["ALL", *CONGESTION_LEVELS])
    parser.add_argument("--recommendation", default="ALL", choices=["ALL", *RECOMMENDATION_ACTIONS])
    parser.add_argument("--export", default=None, help="Optional CSV path to export merged historical records.")
    return parser


def main() -> None:
    """Run historical analytics from the command line."""

    args = build_arg_parser().parse_args()
    records = filter_records(
        load_historical_records(args.logs),
        date_filter=args.date,
        congestion_level=args.congestion,
        recommendation=args.recommendation,
    )
    print("Historical analytics summary")
    print(generate_summary(records))
    print(generate_event_statistics(records))
    if args.export:
        export_history_summary(records, args.export)
        print(f"export: {args.export}")


if __name__ == "__main__":
    main()
