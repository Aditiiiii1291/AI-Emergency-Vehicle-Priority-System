from __future__ import annotations

import csv

from ml.analytics.green_corridor import (
    ACTIVE_SIMULATION,
    CONFIDENCE_NOTE,
    HOLD_TRAFFIC,
    INACTIVE,
    MONITOR,
    PRIORITIZE_GREEN,
    build_green_corridor_log_record,
    build_green_corridor_sequence_log_records,
    find_emergency_lane,
    simulate_green_corridor,
    write_green_corridor_log,
    write_green_corridor_sequence_log,
)


def _lane(
    lane_id: str,
    emergency: bool = False,
    utilization: float = 0.0,
    total: int = 0,
    congestion: str = "MEDIUM_CONGESTION",
    green: int = 55,
) -> dict[str, object]:
    return {
        "lane_id": lane_id,
        "lane_name": lane_id.replace("_", " ").title(),
        "total_vehicles": total,
        "lane_utilization_percent": utilization,
        "emergency_present": emergency,
        "congestion": congestion,
        "recommended_green_seconds": green,
    }


def test_no_emergency_returns_inactive_corridor() -> None:
    result = simulate_green_corridor([_lane("LANE_1"), _lane("LANE_2")])

    assert result["corridor_active"] is False
    assert result["corridor_status"] == INACTIVE
    assert result["reason"] == "No emergency vehicle detected"


def test_emergency_activates_corridor() -> None:
    result = simulate_green_corridor([_lane("LANE_1", emergency=True, green=90)])

    assert result["corridor_active"] is True
    assert result["corridor_status"] == ACTIVE_SIMULATION


def test_correct_emergency_lane_selected() -> None:
    lanes = [
        _lane("LANE_1", emergency=True, utilization=30.0, total=10),
        _lane("LANE_2", emergency=True, utilization=60.0, total=8),
        _lane("LANE_3", emergency=True, utilization=60.0, total=12),
    ]

    selected = find_emergency_lane(lanes)

    assert selected is not None
    assert selected["lane_id"] == "LANE_3"


def test_tied_emergency_lane_selection_stable_by_lane_id() -> None:
    lanes = [
        _lane("LANE_2", emergency=True, utilization=50.0, total=10),
        _lane("LANE_1", emergency=True, utilization=50.0, total=10),
    ]

    selected = find_emergency_lane(lanes)

    assert selected is not None
    assert selected["lane_id"] == "LANE_1"


def test_emergency_lane_gets_prioritize_green() -> None:
    result = simulate_green_corridor([_lane("LANE_2", emergency=True, green=90)])

    assert result["recommended_sequence"][0]["lane_id"] == "LANE_2"
    assert result["recommended_sequence"][0]["action"] == PRIORITIZE_GREEN
    assert result["recommended_sequence"][0]["green_seconds"] == 90


def test_non_emergency_lanes_get_hold_traffic() -> None:
    result = simulate_green_corridor(
        [
            _lane("LANE_1", emergency=True, green=90),
            _lane("LANE_2", congestion="HIGH_CONGESTION"),
        ]
    )

    actions = {item["lane_id"]: item for item in result["recommended_sequence"]}
    assert actions["LANE_2"]["action"] == HOLD_TRAFFIC
    assert actions["LANE_2"]["green_seconds"] == 20


def test_low_congestion_lane_becomes_monitor() -> None:
    result = simulate_green_corridor(
        [
            _lane("LANE_1", emergency=True, green=90),
            _lane("LANE_2", congestion="LOW_CONGESTION"),
        ]
    )

    actions = {item["lane_id"]: item for item in result["recommended_sequence"]}
    assert actions["LANE_2"]["action"] == MONITOR


def test_clearance_window_equals_emergency_lane_timing() -> None:
    result = simulate_green_corridor([_lane("LANE_2", emergency=True, green=88)])

    assert result["estimated_clearance_window_seconds"] == 88


def test_empty_lane_list_handled_safely() -> None:
    result = simulate_green_corridor([])

    assert result["corridor_active"] is False
    assert result["corridor_status"] == INACTIVE


def test_malformed_input_handled_safely() -> None:
    result = simulate_green_corridor(["bad", {"lane_name": "Missing ID"}])

    assert result["corridor_active"] is False
    assert result["corridor_status"] == INACTIVE


def test_output_schema_validation() -> None:
    result = simulate_green_corridor([_lane("LANE_1", emergency=True)])

    assert set(result) == {
        "corridor_active",
        "emergency_lane_id",
        "emergency_lane_name",
        "corridor_status",
        "recommended_sequence",
        "estimated_clearance_window_seconds",
        "reason",
        "confidence_note",
    }
    assert result["confidence_note"] == CONFIDENCE_NOTE


def test_csv_logging_validation(tmp_path) -> None:
    result = simulate_green_corridor([_lane("LANE_1", emergency=True, green=90)])
    summary_log = tmp_path / "green_corridor_log.csv"
    sequence_log = tmp_path / "green_corridor_sequence_log.csv"

    summary_record = build_green_corridor_log_record(result, timestamp="2026-06-08T12:00:00")
    sequence_records = build_green_corridor_sequence_log_records(result, timestamp="2026-06-08T12:00:00")

    assert write_green_corridor_log([summary_record], summary_log) is True
    assert write_green_corridor_sequence_log(sequence_records, sequence_log) is True

    with summary_log.open("r", newline="", encoding="utf-8") as csv_file:
        summary_rows = list(csv.DictReader(csv_file))
    with sequence_log.open("r", newline="", encoding="utf-8") as csv_file:
        sequence_rows = list(csv.DictReader(csv_file))

    assert summary_rows[0]["corridor_active"] == "True"
    assert summary_rows[0]["corridor_status"] == ACTIVE_SIMULATION
    assert sequence_rows[0]["action"] == PRIORITIZE_GREEN
