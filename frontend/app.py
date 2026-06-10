"""Streamlit dashboard for the emergency vehicle priority demo."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UPLOAD_DIR = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class DashboardImports:
    """Project modules needed by the dashboard."""

    cv2: Any
    analyze_density: Any
    classify_congestion: Any
    count_vehicle_detections: Any
    detect_ambulances: Any
    detect_vehicles: Any
    draw_congestion_overlay: Any
    draw_density_overlay: Any
    draw_priority_overlay: Any
    generate_signal_timing_recommendation: Any
    generate_priority_action: Any
    generate_event_statistics: Any
    generate_summary: Any
    generate_trend_data: Any
    analyze_lanes: Any
    default_lane_config: Any
    draw_lane_overlay: Any
    simulate_green_corridor: Any
    get_video_metadata: Any
    is_emergency_present: Any
    iter_frames: Any
    load_congestion_model: Any
    load_historical_records: Any
    load_ambulance_model: Any
    load_yolo_model: Any
    predict_congestion: Any
    resolve_ambulance_model_path: Any
    save_uploaded_video: Any
    should_process_frame: Any
    filter_records: Any
    congestion_levels: list[str]
    recommendation_actions: list[str]


def load_dashboard_imports() -> DashboardImports | None:
    """Import project modules lazily so missing dependencies can be reported cleanly."""

    try:
        import cv2

        from ml.analytics.congestion_classifier import classify_congestion, draw_congestion_overlay
        from ml.analytics.density_analyzer import (
            analyze_density,
            count_vehicle_detections,
            draw_density_overlay,
        )
        from ml.analytics.priority_engine import draw_priority_overlay, generate_priority_action
        from ml.analytics.signal_timing_engine import generate_signal_timing_recommendation
        from ml.analytics.lane_analyzer import analyze_lanes, default_lane_config, draw_lane_overlay
        from ml.analytics.green_corridor import simulate_green_corridor
        from ml.analytics.history_analytics import (
            CONGESTION_LEVELS,
            RECOMMENDATION_ACTIONS,
            filter_records,
            generate_event_statistics,
            generate_summary,
            generate_trend_data,
            load_historical_records,
        )
        from ml.cv_pipeline import get_video_metadata, iter_frames, save_uploaded_video
        from ml.detectors.ambulance_detector import (
            detect_ambulances,
            is_emergency_present,
            load_ambulance_model,
            resolve_ambulance_model_path,
        )
        from ml.detectors.vehicle_detector import detect_vehicles, load_yolo_model, should_process_frame
        from ml.prediction.congestion_predictor import (
            load_model as load_congestion_model,
            predict_congestion,
        )
    except ModuleNotFoundError as error:
        st.error(f"Missing dependency: {error.name}. Install dependencies with pip install -r requirements.txt.")
        return None
    except Exception as error:  # pragma: no cover - Streamlit displays this path interactively.
        st.error(f"Dashboard modules could not be loaded: {error}")
        return None

    return DashboardImports(
        cv2=cv2,
        analyze_density=analyze_density,
        classify_congestion=classify_congestion,
        count_vehicle_detections=count_vehicle_detections,
        detect_ambulances=detect_ambulances,
        detect_vehicles=detect_vehicles,
        draw_congestion_overlay=draw_congestion_overlay,
        draw_density_overlay=draw_density_overlay,
        draw_priority_overlay=draw_priority_overlay,
        generate_signal_timing_recommendation=generate_signal_timing_recommendation,
        generate_priority_action=generate_priority_action,
        generate_event_statistics=generate_event_statistics,
        generate_summary=generate_summary,
        generate_trend_data=generate_trend_data,
        analyze_lanes=analyze_lanes,
        default_lane_config=default_lane_config,
        draw_lane_overlay=draw_lane_overlay,
        simulate_green_corridor=simulate_green_corridor,
        get_video_metadata=get_video_metadata,
        is_emergency_present=is_emergency_present,
        iter_frames=iter_frames,
        load_congestion_model=load_congestion_model,
        load_historical_records=load_historical_records,
        load_ambulance_model=load_ambulance_model,
        load_yolo_model=load_yolo_model,
        predict_congestion=predict_congestion,
        resolve_ambulance_model_path=resolve_ambulance_model_path,
        save_uploaded_video=save_uploaded_video,
        should_process_frame=should_process_frame,
        filter_records=filter_records,
        congestion_levels=CONGESTION_LEVELS,
        recommendation_actions=RECOMMENDATION_ACTIONS,
    )


def render_status_card(label: str, value: str | int | bool) -> None:
    """Render a compact metric-style status card."""

    st.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{label}</div>
            <div class="status-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def configure_page() -> None:
    """Apply page configuration and lightweight dashboard styling."""

    st.set_page_config(
        page_title="Emergency Vehicle Priority Dashboard",
        layout="wide",
    )

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@600&display=swap');

            html, body, [class*="css"], .stApp {
                font-family: 'Calibri', 'Carlito', 'Trebuchet MS', sans-serif !important;
            }

            h1, h2, h3, h4,
            .stApp h1, .stApp h2, .stApp h3, .stApp h4,
            [data-testid="stHeading"],
            div[data-testid="stMarkdownContainer"] h1,
            div[data-testid="stMarkdownContainer"] h2,
            div[data-testid="stMarkdownContainer"] h3 {
                font-family: 'Franklin Gothic Demi Cond', 'Franklin Gothic Medium Cond',
                             'Oswald', 'Arial Narrow', sans-serif !important;
            }

            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            .status-card {
                border: 1px solid #d8dee9;
                border-radius: 8px;
                padding: 14px 16px;
                min-height: 92px;
                background: #ffffff;
                margin-bottom: 12px;
            }
            .status-label {
                font-family: 'Franklin Gothic Demi Cond', 'Franklin Gothic Medium Cond',
                             'Oswald', 'Arial Narrow', sans-serif !important;
                color: #4b5563;
                font-size: 0.82rem;
                font-weight: 600;
                text-transform: uppercase;
            }
            .status-value {
                font-family: 'Calibri', 'Carlito', 'Trebuchet MS', sans-serif !important;
                color: #111827;
                font-size: 1.35rem;
                font-weight: 700;
                margin-top: 10px;
                overflow-wrap: anywhere;
            }
            .section-note {
                font-family: 'Calibri', 'Carlito', 'Trebuchet MS', sans-serif !important;
                color: #4b5563;
                font-size: 0.95rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initial_results() -> dict[str, Any]:
    """Return default dashboard results before analysis runs."""

    return {
        "total_vehicles": 0,
        "car_count": 0,
        "motorcycle_count": 0,
        "bus_count": 0,
        "truck_count": 0,
        "density": "LOW",
        "congestion": "LOW_CONGESTION",
        "predicted_congestion": "Model not trained",
        "emergency_present": False,
        "recommended_action": "NORMAL_OPERATION",
        "recommended_green_seconds": 35,
        "signal_priority_action": "NORMAL_OPERATION",
        "signal_severity": "NORMAL",
        "signal_reason": "Low congestion detected",
        "signal_confidence_note": "Rule-based recommendation",
        "lane_results": [],
        "green_corridor_result": {
            "corridor_active": False,
            "emergency_lane_id": "N/A",
            "emergency_lane_name": "N/A",
            "corridor_status": "INACTIVE",
            "recommended_sequence": [],
            "estimated_clearance_window_seconds": 0,
            "reason": "No emergency vehicle detected",
            "confidence_note": "Rule-based corridor simulation",
        },
        "processed_frames": 0,
        "latest_frame": None,
        "ambulance_message": "Multi-Signal Ambulance Detection Pipeline Active.",
        "ambulance_detected": False,
        "ambulance_confidence": 0.0,
        "emergency_light_score": 0.0,
        "ambulance_reason": "No analysis run",
    }


def analyze_uploaded_video(
    imports: DashboardImports,
    video_path: Path,
    max_frames: int,
    skip_frames: int,
    vehicle_confidence: float,
    ambulance_confidence: float,
) -> dict[str, Any]:
    """Run a bounded dashboard analysis pass over an uploaded video."""

    import csv
    from datetime import datetime, timedelta
    from dataclasses import asdict

    metadata = imports.get_video_metadata(video_path)
    vehicle_model = imports.load_yolo_model()
    prediction_model = None
    model_path = PROJECT_ROOT / "data" / "models" / "congestion_predictor.pkl"
    if model_path.exists():
        prediction_model = imports.load_congestion_model(model_path)

    ambulance_model = None
    ambulance_model_path = imports.resolve_ambulance_model_path()
    if ambulance_model_path.exists():
        try:
            ambulance_model = imports.load_ambulance_model(ambulance_model_path)
            ambulance_message = f"Ambulance model loaded: {ambulance_model_path}"
        except Exception as error:
            ambulance_message = f"Error loading custom ambulance model: {error}. Using fallback."
    else:
        ambulance_message = "Multi-Signal Pipeline Active (using generic vehicle detections + HSV light detection)."

    results = initial_results()

    # Containers for logs
    run_start_time = datetime.now()
    logs_dir = PROJECT_ROOT / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    vehicle_det_rows = []
    emergency_det_rows = []
    density_rows = []
    congestion_rows = []
    priority_rows = []

    with st.spinner("Analyzing uploaded video..."):
        for frame_index, frame in imports.iter_frames(video_path):
            if not imports.should_process_frame(frame_index, skip_frames):
                continue
            if results["processed_frames"] >= max_frames:
                break

            annotated, vehicle_detections = imports.detect_vehicles(
                frame=frame,
                model=vehicle_model,
                confidence_threshold=vehicle_confidence,
                frame_index=frame_index,
                fps=metadata.fps,
                source_video=str(video_path),
            )

            # Always call detect_ambulances and pass vehicle_detections for candidate extraction
            annotated, ambulance_detections = imports.detect_ambulances(
                frame=frame,
                model=ambulance_model,
                confidence_threshold=ambulance_confidence,
                base_frame=annotated,
                vehicle_detections=vehicle_detections,
            )

            vehicle_counts = imports.count_vehicle_detections(vehicle_detections)
            density_result = imports.analyze_density(vehicle_counts)
            congestion_result = imports.classify_congestion(density_result)
            emergency_present = imports.is_emergency_present(ambulance_detections)
            action_result = imports.generate_priority_action(
                emergency_present=emergency_present,
                density_result=density_result,
                congestion_result=congestion_result,
            )
            signal_timing_result = imports.generate_signal_timing_recommendation(
                density_result=density_result,
                congestion_result=congestion_result,
                priority_result=action_result,
            )
            predicted_congestion = "Model not trained"
            if prediction_model is not None:
                predicted_congestion = imports.predict_congestion(
                    prediction_model,
                    {
                        **density_result,
                        "emergency_present": emergency_present,
                    },
                )

            frame_height, frame_width = frame.shape[:2]
            lane_regions = imports.default_lane_config(frame_width=frame_width, frame_height=frame_height)
            lane_results = imports.analyze_lanes(
                vehicle_detections=vehicle_detections,
                ambulance_detections=ambulance_detections,
                frame_width=frame_width,
                frame_height=frame_height,
                lane_regions=lane_regions,
            )
            enriched_lane_results = []
            for lane_result in lane_results:
                lane_density_result = imports.analyze_density(
                    {
                        "total_vehicles": lane_result["total_vehicles"],
                        "car_count": lane_result["car_count"],
                        "motorcycle_count": lane_result["motorcycle_count"],
                        "bus_count": lane_result["bus_count"],
                        "truck_count": lane_result["truck_count"],
                    }
                )
                lane_congestion_result = imports.classify_congestion(lane_density_result)
                lane_action_result = imports.generate_priority_action(
                    emergency_present=lane_result["emergency_present"],
                    density_result=lane_density_result,
                    congestion_result=lane_congestion_result,
                )
                lane_signal_result = imports.generate_signal_timing_recommendation(
                    density_result=lane_density_result,
                    congestion_result=lane_congestion_result,
                    priority_result=lane_action_result,
                )
                enriched_lane_results.append(
                    {
                        **lane_result,
                        "density": lane_density_result["density"],
                        "congestion": lane_congestion_result["congestion"],
                        "recommended_action": lane_action_result["recommended_action"],
                        "recommended_green_seconds": lane_signal_result["recommended_green_seconds"],
                    }
                )
            green_corridor_result = imports.simulate_green_corridor(
                enriched_lane_results,
                action_result,
                signal_timing_result,
            )

            annotated = imports.draw_density_overlay(annotated, density_result)
            annotated = imports.draw_congestion_overlay(annotated, congestion_result)
            annotated = imports.draw_priority_overlay(annotated, action_result)
            annotated = imports.draw_lane_overlay(annotated, lane_regions, enriched_lane_results)
            latest_frame = imports.cv2.cvtColor(annotated, imports.cv2.COLOR_BGR2RGB)

            frame_summary = getattr(ambulance_detections, "frame_summary", {})
            results = {
                **density_result,
                **congestion_result,
                **action_result,
                "recommended_green_seconds": signal_timing_result["recommended_green_seconds"],
                "signal_priority_action": signal_timing_result["priority_action"],
                "signal_severity": signal_timing_result["severity"],
                "signal_reason": signal_timing_result["reason"],
                "signal_confidence_note": signal_timing_result["confidence_note"],
                "lane_results": enriched_lane_results,
                "green_corridor_result": green_corridor_result,
                "predicted_congestion": predicted_congestion,
                "processed_frames": results["processed_frames"] + 1,
                "latest_frame": latest_frame,
                "ambulance_message": ambulance_message,
                "ambulance_detected": frame_summary.get("ambulance_detected", False),
                "ambulance_confidence": frame_summary.get("confidence", 0.0),
                "emergency_light_score": frame_summary.get("emergency_light_score", 0.0),
                "ambulance_reason": "; ".join(frame_summary.get("reason", [])) if frame_summary.get("reason") else "No vehicle candidates detected",
            }

            # Collect log rows
            fps = metadata.fps if metadata.fps > 0 else 30.0
            frame_offset_seconds = frame_index / fps
            frame_timestamp = (run_start_time + timedelta(seconds=frame_offset_seconds)).isoformat(timespec="milliseconds")

            for det in vehicle_detections:
                det_dict = asdict(det)
                det_dict["timestamp"] = frame_timestamp
                vehicle_det_rows.append(det_dict)

            for det in ambulance_detections:
                xmin, ymin, xmax, ymax = [int(value) for value in det["bbox"]]
                emergency_det_rows.append({
                    "source_video": str(video_path),
                    "frame_index": frame_index,
                    "timestamp_seconds": round(frame_offset_seconds, 3),
                    "class_id": 0,
                    "class_name": "ambulance",
                    "confidence": det["confidence"],
                    "xmin": xmin,
                    "ymin": ymin,
                    "xmax": xmax,
                    "ymax": ymax,
                    "box_width": max(0, xmax - xmin),
                    "box_height": max(0, ymax - ymin),
                    "timestamp": frame_timestamp,
                })

            density_rows.append({
                "source_video": str(video_path),
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_offset_seconds, 3),
                "total_vehicles": density_result["total_vehicles"],
                "car_count": density_result["car_count"],
                "motorcycle_count": density_result["motorcycle_count"],
                "bus_count": density_result["bus_count"],
                "truck_count": density_result["truck_count"],
                "density": density_result["density"],
                "timestamp": frame_timestamp,
            })

            congestion_rows.append({
                "timestamp": frame_timestamp,
                "total_vehicles": congestion_result["total_vehicles"],
                "density": congestion_result["density"],
                "congestion": congestion_result["congestion"],
            })

            priority_rows.append({
                "timestamp": frame_timestamp,
                "emergency_present": action_result["emergency_present"],
                "density": action_result["density"],
                "congestion": action_result["congestion"],
                "recommended_action": action_result["recommended_action"],
            })

    # Append collected logs to disk
    vehicle_det_fields = [
        "source_video", "frame_index", "timestamp_seconds", "class_id",
        "class_name", "confidence", "xmin", "ymin", "xmax", "ymax",
        "box_width", "box_height", "timestamp"
    ]
    density_fields = [
        "source_video", "frame_index", "timestamp_seconds", "total_vehicles",
        "car_count", "motorcycle_count", "bus_count", "truck_count", "density", "timestamp"
    ]
    congestion_fields = ["timestamp", "total_vehicles", "density", "congestion"]
    priority_fields = ["timestamp", "emergency_present", "density", "congestion", "recommended_action"]

    def append_rows_to_csv(file_name: str, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        file_path = logs_dir / file_name
        file_exists = file_path.exists()
        with file_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    append_rows_to_csv("vehicle_detections.csv", vehicle_det_fields, vehicle_det_rows)
    append_rows_to_csv("emergency_detections.csv", vehicle_det_fields, emergency_det_rows)
    append_rows_to_csv("density_analysis.csv", density_fields, density_rows)
    append_rows_to_csv("congestion_analysis.csv", congestion_fields, congestion_rows)
    append_rows_to_csv("priority_actions.csv", priority_fields, priority_rows)

    return results


def render_dashboard(results: dict[str, Any]) -> None:
    """Render dashboard sections and status cards."""

    st.subheader("Traffic Overview")
    traffic_cols = st.columns(5)
    with traffic_cols[0]:
        render_status_card("Vehicle Count", results["total_vehicles"])
    with traffic_cols[1]:
        render_status_card("Cars", results["car_count"])
    with traffic_cols[2]:
        render_status_card("Motorcycles", results["motorcycle_count"])
    with traffic_cols[3]:
        render_status_card("Buses", results["bus_count"])
    with traffic_cols[4]:
        render_status_card("Trucks", results["truck_count"])

    st.subheader("Emergency Status")
    emergency_cols = st.columns(2)
    with emergency_cols[0]:
        render_status_card("Emergency Status", "DETECTED" if results["emergency_present"] else "NONE")
    with emergency_cols[1]:
        st.info(results["ambulance_message"])

    st.subheader("Ambulance Detection Details")
    detail_cols = st.columns(4)
    with detail_cols[0]:
        render_status_card("Ambulance Detected", "True" if results.get("ambulance_detected", False) else "False")
    with detail_cols[1]:
        render_status_card("Confidence", f"{results.get('ambulance_confidence', 0.0):.4f}")
    with detail_cols[2]:
        render_status_card("Emergency Light Score", f"{results.get('emergency_light_score', 0.0):.4f}")
    with detail_cols[3]:
        render_status_card("Detection Reason", results.get("ambulance_reason", "N/A"))

    st.subheader("Congestion Analysis")
    congestion_cols = st.columns(2)
    with congestion_cols[0]:
        render_status_card("Density", results["density"])
    with congestion_cols[1]:
        render_status_card("Congestion", results["congestion"])
    render_status_card("Predicted Congestion", results["predicted_congestion"])

    st.subheader("Priority Recommendation")
    render_status_card("Recommended Action", results["recommended_action"])

    st.subheader("Smart Signal Timing")
    signal_cols = st.columns(3)
    with signal_cols[0]:
        render_status_card("Recommended Green Time", f"{results['recommended_green_seconds']} sec")
    with signal_cols[1]:
        render_status_card("Priority Action", results["signal_priority_action"])
    with signal_cols[2]:
        render_status_card("Severity", results["signal_severity"])
    signal_cols = st.columns(2)
    with signal_cols[0]:
        render_status_card("Reason", results["signal_reason"])
    with signal_cols[1]:
        render_status_card("Confidence Note", results["signal_confidence_note"])

    st.subheader("Multi-Lane Traffic Analysis")
    lane_results = results.get("lane_results", [])
    if lane_results:
        st.dataframe(
            [
                {
                    "Lane": lane["lane_name"],
                    "Vehicles": lane["total_vehicles"],
                    "Utilization": f"{lane['lane_utilization_percent']:.2f}%",
                    "Density": lane["density"],
                    "Congestion": lane["congestion"],
                    "Emergency": "DETECTED" if lane["emergency_present"] else "NONE",
                    "Action": lane["recommended_action"],
                    "Green Time": f"{lane['recommended_green_seconds']} sec",
                }
                for lane in lane_results
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown('<p class="section-note">Run analysis to display lane-level traffic results.</p>', unsafe_allow_html=True)

    st.subheader("Emergency Green Corridor Simulation")
    corridor_result = results.get("green_corridor_result", {})
    corridor_cols = st.columns(3)
    with corridor_cols[0]:
        render_status_card("Corridor Status", corridor_result.get("corridor_status", "INACTIVE"))
    with corridor_cols[1]:
        render_status_card("Emergency Lane", corridor_result.get("emergency_lane_name", "N/A"))
    with corridor_cols[2]:
        render_status_card(
            "Clearance Window",
            f"{corridor_result.get('estimated_clearance_window_seconds', 0)} sec",
        )
    corridor_cols = st.columns(2)
    with corridor_cols[0]:
        render_status_card("Reason", corridor_result.get("reason", "No emergency vehicle detected"))
    with corridor_cols[1]:
        render_status_card("Confidence Note", corridor_result.get("confidence_note", "Rule-based corridor simulation"))

    sequence = corridor_result.get("recommended_sequence", [])
    if sequence:
        st.dataframe(
            [
                {
                    "Lane": item["lane_name"],
                    "Action": item["action"],
                    "Green Time": f"{item['green_seconds']} sec",
                    "Reason": item["reason"],
                }
                for item in sequence
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.markdown('<p class="section-note">No emergency corridor sequence is active.</p>', unsafe_allow_html=True)

    if results["latest_frame"] is not None:
        st.image(results["latest_frame"], caption="Latest analyzed frame", use_container_width=True)
    else:
        st.markdown('<p class="section-note">Run analysis to display the latest processed frame.</p>', unsafe_allow_html=True)


def _distribution_chart(distribution: dict[str, int]) -> Any:
    """Build a small dataframe for Streamlit bar charts."""

    import pandas as pd

    if not distribution:
        return pd.DataFrame({"count": []})
    return pd.DataFrame(
        [{"label": label, "count": count} for label, count in distribution.items()]
    ).set_index("label")


def render_historical_analytics(
    imports: DashboardImports,
    date_filter: str | None,
    congestion_filter: str,
    recommendation_filter: str,
) -> None:
    """Render historical analytics filters, metrics, and charts."""

    import pandas as pd

    st.subheader("Historical Analytics")
    records = imports.load_historical_records(PROJECT_ROOT / "data" / "logs")
    filtered_records = imports.filter_records(
        records,
        date_filter=date_filter or None,
        congestion_level=congestion_filter,
        recommendation=recommendation_filter,
    )

    if not records:
        st.info("No historical logs found yet. Run the detection, density, congestion, and priority pipelines to generate analytics.")
        return

    summary = imports.generate_summary(filtered_records)
    trend_data = imports.generate_trend_data(filtered_records)
    event_stats = imports.generate_event_statistics(filtered_records)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_status_card("Analyzed Records", summary["total_analyzed_records"])
    with metric_cols[1]:
        render_status_card("Emergency Events", summary["total_emergency_events"])
    with metric_cols[2]:
        render_status_card("Common Congestion", summary["most_common_congestion_level"])
    with metric_cols[3]:
        render_status_card("Common Recommendation", summary["most_common_recommendation"])

    st.caption(f"Emergency event rate: {event_stats['emergency_rate']}")

    vehicle_trend = pd.DataFrame(trend_data["vehicle_count_over_time"])
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("Vehicle Count Over Time")
        if vehicle_trend.empty:
            st.info("No vehicle trend records match the selected filters.")
        else:
            st.line_chart(vehicle_trend.set_index("timestamp")["total_vehicles"])

    with chart_cols[1]:
        st.markdown("Density Distribution")
        st.bar_chart(_distribution_chart(trend_data["density_distribution"]))

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.markdown("Congestion Distribution")
        st.bar_chart(_distribution_chart(trend_data["congestion_distribution"]))

    with chart_cols[1]:
        st.markdown("Recommendation Distribution")
        st.bar_chart(_distribution_chart(trend_data["recommendation_distribution"]))


def main() -> None:
    """Run the Streamlit dashboard."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    configure_page()

    st.title("AI Emergency Vehicle Priority Dashboard")
    st.caption("Video-based traffic monitoring and simulated priority recommendations.")

    imports = load_dashboard_imports()

    with st.sidebar:
        st.header("Analysis Settings")
        max_frames = st.slider("Frames to analyze", min_value=1, max_value=100, value=10, step=1)
        skip_frames = st.slider("Skip frames", min_value=0, max_value=10, value=2, step=1)
        vehicle_confidence = st.slider("Vehicle confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
        ambulance_confidence = st.slider("Ambulance confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
        st.header("Historical Filters")
        history_date_filter = st.text_input("Date/timestamp prefix", value="")
        history_congestion_filter = st.selectbox("Congestion level", ["ALL", "LOW_CONGESTION", "MEDIUM_CONGESTION", "HIGH_CONGESTION"])
        history_recommendation_filter = st.selectbox(
            "Recommendation",
            ["ALL", "NORMAL_OPERATION", "EXTEND_GREEN", "HIGH_TRAFFIC_WARNING", "EMERGENCY_PRIORITY"],
        )

    uploaded_file = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])
    results = initial_results()

    if uploaded_file is not None:
        st.video(uploaded_file.getvalue())

        if imports is not None:
            uploaded_file.seek(0)
            video_path = imports.save_uploaded_video(uploaded_file, UPLOAD_DIR)
            st.caption(f"Uploaded file saved for analysis: {video_path.name}")

            if st.button("Analyze Video", type="primary"):
                try:
                    results = analyze_uploaded_video(
                        imports=imports,
                        video_path=video_path,
                        max_frames=max_frames,
                        skip_frames=skip_frames,
                        vehicle_confidence=vehicle_confidence,
                        ambulance_confidence=ambulance_confidence,
                    )
                    st.success(f"Analyzed {results['processed_frames']} frame(s).")
                except FileNotFoundError as error:
                    st.warning(str(error))
                except Exception as error:  # pragma: no cover - displayed interactively.
                    st.error(f"Analysis failed: {error}")
    else:
        st.info("Upload an mp4, avi, or mov traffic video to begin.")

    render_dashboard(results)
    if imports is not None:
        render_historical_analytics(
            imports=imports,
            date_filter=history_date_filter,
            congestion_filter=history_congestion_filter,
            recommendation_filter=history_recommendation_filter,
        )


if __name__ == "__main__":
    main()
