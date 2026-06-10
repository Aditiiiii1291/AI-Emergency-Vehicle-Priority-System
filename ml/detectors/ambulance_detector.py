"""Phase 4 ambulance detection infrastructure.

This module keeps ambulance detection separate from the generic vehicle
detector. It expects a future custom YOLO model trained for ambulance detection
and does not claim real-world accuracy.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.cv_pipeline import get_video_metadata, iter_frames, validate_video_path
from ml.detectors.vehicle_detector import (
    Detection,
    LOG_COLUMNS,
    detect_vehicles,
    load_yolo_model,
    should_process_frame,
)


DEFAULT_AMBULANCE_MODEL_PATH = Path("data/models/ambulance_detector.pt")
AMBULANCE_CLASS_NAMES = {"ambulance", "emergency vehicle", "emergency_vehicle"}
AMBULANCE_COLOR = (40, 40, 255)
AMBULANCE_MODEL_ENV_VAR = "AMBULANCE_MODEL_PATH"
AMBULANCE_KEYWORDS = ["ambulance", "emergency", "ems"]
AMBULANCE_CANDIDATE_CLASSES = {"car", "truck", "bus", "van", "ambulance"}
MIN_AMBULANCE_CONFIDENCE = 0.55
DEFAULT_AMBULANCE_LOG_PATH = Path("data/logs/ambulance_detection_log.csv")
AMBULANCE_LOG_COLUMNS = [
    "timestamp",
    "ambulance_detected",
    "confidence",
    "emergency_light_score",
    "reason",
]

CONFIDENCE_WEIGHTS = {
    "vehicle": 0.4,
    "emergency_light": 0.3,
    "visual_heuristic": 0.3,
}

_AMBULANCE_MODEL_CACHE: dict[str, YOLO] = {}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmergencyDetectionSummary:
    """Summary returned after combined vehicle and ambulance detection."""

    input_path: str
    output_path: str
    log_path: str
    vehicle_model_path: str
    ambulance_model_path: str
    vehicle_confidence_threshold: float
    ambulance_confidence_threshold: float
    skip_frames: int
    total_frames: int
    processed_frames: int
    total_vehicle_detections: int
    total_ambulance_detections: int
    emergency_detected: bool


@dataclass(frozen=True)
class AmbulanceDetectionLogRecord:
    """CSV-compatible ambulance detection summary record."""

    timestamp: str
    ambulance_detected: bool
    confidence: float
    emergency_light_score: float
    reason: str


class AmbulanceDetections(list):
    """A list of ambulance detections with frame-level metadata attached."""

    def __init__(self, detections: list[dict[str, Any]], frame_summary: dict[str, Any]):
        super().__init__(detections)
        self.frame_summary = frame_summary


def resolve_ambulance_model_path(model_path: str | Path | None = None) -> Path:
    """Resolve the ambulance model path from an argument, env var, or default."""

    if model_path is not None:
        return Path(model_path)

    configured_path = os.getenv(AMBULANCE_MODEL_ENV_VAR)
    if configured_path:
        return Path(configured_path)

    return DEFAULT_AMBULANCE_MODEL_PATH


def load_ambulance_model(model_path: str | Path | None = None) -> YOLO:
    """Load and cache the ambulance detector model.

    The default path is `data/models/ambulance_detector.pt`, but callers can
    pass a different path or set AMBULANCE_MODEL_PATH. This keeps model loading
    isolated for future fine-tuned weights.
    
    If the resolved model file does not exist and it is the default path, 
    we fall back to loading the generic YOLO vehicle model.
    """

    resolved_path = resolve_ambulance_model_path(model_path)
    key = str(resolved_path)

    # Fallback to yolov8n.pt if default path is not found
    if not resolved_path.exists() and key == str(DEFAULT_AMBULANCE_MODEL_PATH):
        logger.info(f"Ambulance model path {key} not found. Falling back to yolov8n.pt")
        key = "yolov8n.pt"

    if key not in _AMBULANCE_MODEL_CACHE:
        logger.info(f"Loading YOLO ambulance model from: {key}")
        try:
            _AMBULANCE_MODEL_CACHE[key] = YOLO(key)
        except Exception as error:
            logger.error(f"Failed to load ambulance model from {key}: {error}")
            raise RuntimeError(f"Failed to load ambulance model from {key}: {error}") from error
    return _AMBULANCE_MODEL_CACHE[key]


def get_ambulance_class_ids(model: YOLO) -> list[int]:
    """Find ambulance class IDs from a YOLO model's class-name metadata."""

    names = getattr(model, "names", {})
    if isinstance(names, dict):
        matches = [
            int(class_id)
            for class_id, class_name in names.items()
            if str(class_name).strip().lower() in AMBULANCE_CLASS_NAMES
        ]
    else:
        matches = [
            class_id
            for class_id, class_name in enumerate(names)
            if str(class_name).strip().lower() in AMBULANCE_CLASS_NAMES
        ]

    return matches or [0]


def _read_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dictionary or object detection."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Normalize float-like values without raising."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_bbox(detection: Any) -> list[int] | None:
    """Extract a bbox from a dict or dataclass detection."""

    try:
        bbox = _read_attr_or_key(detection, "bbox")
        if bbox is not None and len(bbox) == 4:
            return [int(value) for value in bbox]
        return [
            int(_read_attr_or_key(detection, "xmin")),
            int(_read_attr_or_key(detection, "ymin")),
            int(_read_attr_or_key(detection, "xmax")),
            int(_read_attr_or_key(detection, "ymax")),
        ]
    except (TypeError, ValueError):
        logger.warning(f"Skipping malformed ambulance candidate bbox: {detection}")
        return None


def _clip_bbox_to_frame(bbox: list[int], frame: Any) -> list[int]:
    """Clamp bbox coordinates to frame bounds."""

    height, width = frame.shape[:2]
    xmin, ymin, xmax, ymax = bbox
    return [
        max(0, min(width, xmin)),
        max(0, min(height, ymin)),
        max(0, min(width, xmax)),
        max(0, min(height, ymax)),
    ]


def _crop_frame(frame: Any, bbox: list[int]) -> Any:
    """Return a safe frame crop for a bbox."""

    xmin, ymin, xmax, ymax = _clip_bbox_to_frame(bbox, frame)
    if xmax <= xmin or ymax <= ymin:
        return None
    return frame[ymin:ymax, xmin:xmax]


def _count_color_regions(mask: Any, min_area: int = 2) -> int:
    """Count color regions from a binary mask. Minimal area is 2 pixels for vehicle crops."""

    if mask is None or not hasattr(mask, "shape") or mask.size == 0:
        return 0
    crop_height = mask.shape[0]
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    count = 0
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            # Filter out regions located in the very bottom 25% of the vehicle crop
            # (which are highly likely to be standard taillights, headlights, or bumper reflections)
            x, y, w, h_rect = cv2.boundingRect(contour)
            cy = y + h_rect / 2
            if cy < crop_height * 0.75:
                count += 1
    return count


def detect_emergency_lights(frame: Any) -> dict[str, int | float]:
    """Detect red and blue emergency-light-like regions using HSV thresholds."""

    try:
        if frame is None or not hasattr(frame, "shape") or frame.size == 0:
            return {"red_regions": 0, "blue_regions": 0, "emergency_light_score": 0.0}

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Saturated, bright red ranges
        red_mask_low = cv2.inRange(hsv, (0, 100, 150), (10, 255, 255))
        red_mask_high = cv2.inRange(hsv, (170, 100, 150), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask_low, red_mask_high)

        # Saturated, bright blue range
        blue_mask = cv2.inRange(hsv, (100, 100, 150), (140, 255, 255))

        red_regions = _count_color_regions(red_mask)
        blue_regions = _count_color_regions(blue_mask)

        # Base scoring logic: either red or blue or both are active in the upper roof/grille area
        if red_regions > 0 and blue_regions > 0:
            score = min(1.0, 0.6 + 0.1 * (red_regions + blue_regions))
        elif red_regions > 0:
            score = min(0.40, 0.08 * red_regions)
        elif blue_regions > 0:
            score = min(0.50, 0.15 * blue_regions)
        else:
            score = 0.0

        return {
            "red_regions": int(red_regions),
            "blue_regions": int(blue_regions),
            "emergency_light_score": round(float(score), 4),
        }
    except Exception as error:
        logger.warning(f"Emergency light detection failed: {error}")
        return {"red_regions": 0, "blue_regions": 0, "emergency_light_score": 0.0}


def _normalize_bbox(box: Any) -> list[int]:
    """Convert a YOLO box into [xmin, ymin, xmax, ymax]."""

    return [int(value) for value in box.xyxy[0].tolist()]


def _extract_ambulance_detections(results: list[Any]) -> list[dict[str, object]]:
    """Convert raw YOLO results into the public ambulance detection format."""

    detections: list[dict[str, object]] = []

    for result in results:
        for box in result.boxes:
            detections.append(
                {
                    "class_name": "ambulance",
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": _normalize_bbox(box),
                    "ambulance_detected": True,
                    "reason": ["YOLO ambulance class detected"],
                    "emergency_light_score": 0.0,
                }
            )

    return detections


def _candidate_vehicle_score(detection: Any) -> float:
    """Score a detected vehicle class as an ambulance candidate."""

    class_name = str(_read_attr_or_key(detection, "class_name", "")).strip().lower()
    confidence = _safe_float(_read_attr_or_key(detection, "confidence"), 0.0)
    
    if class_name in {"ambulance", "emergency vehicle", "emergency_vehicle"}:
        return max(confidence, 0.85)
    if class_name == "van":
        return max(confidence, 0.75)
    if class_name in {"car", "truck", "bus"}:
        return confidence
    return 0.0


def _visual_heuristic_score(detection: Any) -> tuple[float, list[str]]:
    """Score lightweight visual/candidate heuristics from detection metadata."""

    class_name = str(_read_attr_or_key(detection, "class_name", "")).strip().lower()
    bbox = _safe_bbox(detection)
    reasons: list[str] = []
    score = 0.0

    if any(keyword in class_name for keyword in AMBULANCE_KEYWORDS):
        score += 0.8
        reasons.append("Ambulance keyword detected")
    elif class_name == "van":
        score += 0.35
        reasons.append("Vehicle classified as van (potential ambulance candidate)")
    
    # Generic vehicles (car, truck, bus) do not get default class boost

    if bbox is not None:
        xmin, ymin, xmax, ymax = bbox
        width = max(0, xmax - xmin)
        height = max(0, ymax - ymin)
        aspect_ratio = width / height if height else 0.0
        # Ambulances typically have aspect ratio closer to box/van shape
        if 1.2 <= aspect_ratio <= 2.2:
            score += 0.25
            reasons.append("Emergency vehicle-like shape detected")

    return min(1.0, score), reasons


def calculate_ambulance_confidence(
    vehicle_confidence: float,
    emergency_light_score: float,
    visual_heuristic_score: float,
    class_name: str | None = None,
) -> float:
    """Calculate final ambulance confidence from weighted rule scores."""

    c_name = str(class_name or "").strip().lower()
    if c_name in {"ambulance", "emergency vehicle", "emergency_vehicle", "van"}:
        w_vehicle = 0.50
        w_light = 0.25
        w_visual = 0.25
    elif c_name in {"car", "bus", "truck", "motorcycle"}:
        # Generic vehicle classes require active emergency lights to be classified as ambulance
        w_vehicle = 0.10
        w_light = 0.60
        w_visual = 0.30
    else:
        # Fallback to default weights for backward compatibility
        w_vehicle = CONFIDENCE_WEIGHTS["vehicle"]
        w_light = CONFIDENCE_WEIGHTS["emergency_light"]
        w_visual = CONFIDENCE_WEIGHTS["visual_heuristic"]

    confidence = (
        w_vehicle * max(0.0, min(1.0, vehicle_confidence))
        + w_light * max(0.0, min(1.0, emergency_light_score))
        + w_visual * max(0.0, min(1.0, visual_heuristic_score))
    )
    return round(float(max(0.0, min(1.0, confidence))), 4)


def evaluate_ambulance_confidence(
    vehicle_confidence: float,
    emergency_light_score: float,
    visual_heuristic_score: float,
    reasons: list[str] | None = None,
    min_confidence_threshold: float = MIN_AMBULANCE_CONFIDENCE,
    class_name: str | None = None,
) -> dict[str, Any]:
    """Combine YOLO detection confidence, emergency light score, and visual heuristics.

    Returns:
        dict: {
            "ambulance_detected": bool,
            "confidence": float,
            "reason": list[str]
        }
    """
    confidence = calculate_ambulance_confidence(
        vehicle_confidence,
        emergency_light_score,
        visual_heuristic_score,
        class_name=class_name,
    )
    
    detected = confidence >= min_confidence_threshold
    
    out_reasons = list(reasons) if reasons is not None else []
    if detected:
        out_reasons.append(f"Ambulance verified: combined confidence {confidence:.4f} >= threshold {min_confidence_threshold:.4f}")
    else:
        out_reasons.append(f"Not an ambulance: combined confidence {confidence:.4f} < threshold {min_confidence_threshold:.4f}")
        
    return {
        "ambulance_detected": detected,
        "confidence": confidence,
        "reason": out_reasons,
    }


def evaluate_ambulance_candidate(frame: Any, detection: Any) -> dict[str, object] | None:
    """Evaluate one vehicle detection as a possible ambulance."""

    bbox = _safe_bbox(detection)
    if bbox is None:
        return None

    class_name = str(_read_attr_or_key(detection, "class_name", "")).strip().lower()

    vehicle_score = _candidate_vehicle_score(detection)
    if vehicle_score <= 0:
        return None

    crop = _crop_frame(frame, bbox)
    light_result = detect_emergency_lights(crop)
    light_score = float(light_result["emergency_light_score"])
    visual_score, reasons = _visual_heuristic_score(detection)

    if light_score > 0:
        reasons.insert(0, "Emergency light pattern detected")

    confidence_result = evaluate_ambulance_confidence(
        vehicle_confidence=vehicle_score,
        emergency_light_score=light_score,
        visual_heuristic_score=visual_score,
        reasons=reasons,
        class_name=class_name,
    )

    return {
        "class_name": "ambulance",
        "ambulance_detected": confidence_result["ambulance_detected"],
        "confidence": confidence_result["confidence"],
        "bbox": bbox,
        "reason": confidence_result["reason"],
        "emergency_light_score": round(light_score, 4),
        "vehicle_score": round(vehicle_score, 4),
        "visual_heuristic_score": round(visual_score, 4),
    }


def evaluate_ambulance_detections(
    frame: Any,
    vehicle_detections: list[Any] | None = None,
) -> list[dict[str, object]]:
    """Evaluate all vehicle detections as possible ambulances."""

    if frame is None or not hasattr(frame, "shape"):
        return []

    detections: list[dict[str, object]] = []
    for detection in vehicle_detections or []:
        candidate = evaluate_ambulance_candidate(frame, detection)
        if candidate is not None and candidate["ambulance_detected"]:
            detections.append(candidate)
    return detections


def draw_ambulance_detections(frame: Any, detections: list[dict[str, object]]) -> Any:
    """Draw ambulance boxes with visually distinct emergency annotations."""

    annotated = frame.copy()

    for detection in detections:
        xmin, ymin, xmax, ymax = [int(value) for value in detection["bbox"]]
        confidence = float(detection["confidence"])
        label = f"AMBULANCE {confidence:.2f}"

        cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), AMBULANCE_COLOR, 3)

        label_y = max(28, ymin - 10)
        cv2.rectangle(
            annotated,
            (xmin, label_y - 26),
            (xmin + max(165, len(label) * 13), label_y + 6),
            AMBULANCE_COLOR,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (xmin + 8, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def draw_emergency_status(frame: Any, detections: list[dict[str, object]]) -> Any:
    """Add frame-level emergency status overlay."""

    annotated = frame.copy()
    status = "DETECTED" if is_emergency_present(detections) else "NONE"
    label = f"Emergency Vehicle: {status}"
    color = AMBULANCE_COLOR if status == "DETECTED" else (80, 180, 80)

    cv2.rectangle(annotated, (16, 108), (430, 148), color, -1)
    cv2.putText(
        annotated,
        label,
        (28, 136),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def detect_ambulances(
    frame: Any,
    model: YOLO | None = None,
    confidence_threshold: float = 0.35,
    base_frame: Any | None = None,
    vehicle_detections: list[Any] | None = None,
) -> tuple[Any, list[dict[str, object]]]:
    """Detect ambulances in one frame and return annotated frame plus detections."""

    detections: list[dict[str, object]] = []

    # If vehicle_detections is not provided, run fallback generic vehicle detector
    if vehicle_detections is None:
        try:
            vehicle_model = load_yolo_model()
            _, vehicle_detections = detect_vehicles(
                frame=frame,
                model=vehicle_model,
                confidence_threshold=confidence_threshold,
            )
        except Exception as error:
            logger.warning(f"Fallback vehicle candidate detection failed: {error}")
            vehicle_detections = []

    if model is not None:
        try:
            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                classes=get_ambulance_class_ids(model),
                verbose=False,
            )
            detections.extend(_extract_ambulance_detections(results))
        except Exception as error:
            logger.warning(f"YOLO ambulance inference failed; falling back to heuristic detection: {error}")
    else:
        logger.info("No ambulance YOLO model supplied; using heuristic ambulance detection.")

    # Evaluate all vehicle candidates
    all_candidates = []
    for detection in vehicle_detections or []:
        candidate = evaluate_ambulance_candidate(frame, detection)
        if candidate is not None:
            all_candidates.append(candidate)

    # Detections that actually pass the threshold
    passing_detections = [c for c in all_candidates if c["ambulance_detected"]]
    detections.extend(passing_detections)

    # Calculate frame summary (incorporating candidates that didn't pass, for detailed reporting)
    summary = summarize_ambulance_detection(passing_detections, all_candidates)

    # Log results to CSV
    record = build_ambulance_detection_log_record(summary)
    write_ambulance_detection_log(record)

    annotated = base_frame.copy() if base_frame is not None else frame.copy()
    annotated = draw_ambulance_detections(annotated, detections)
    annotated = draw_emergency_status(annotated, detections)

    return annotated, AmbulanceDetections(detections, summary)


def is_emergency_present(detections: list[dict[str, object]]) -> bool:
    """Return True when at least one ambulance detection is present."""

    return any(bool(detection.get("ambulance_detected", True)) for detection in detections)


def summarize_ambulance_detection(
    detections: list[dict[str, object]],
    all_candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Summarize ambulance detections for dashboard display and logging."""

    if detections:
        best_detection = max(detections, key=lambda item: float(item.get("confidence", 0.0)))
        return {
            "ambulance_detected": True,
            "confidence": round(float(best_detection.get("confidence", 0.0)), 4),
            "reason": list(best_detection.get("reason", [])) or ["Ambulance candidate detected"],
            "emergency_light_score": round(float(best_detection.get("emergency_light_score", 0.0)), 4),
        }

    if all_candidates:
        best_candidate = max(all_candidates, key=lambda item: float(item.get("confidence", 0.0)))
        return {
            "ambulance_detected": False,
            "confidence": round(float(best_candidate.get("confidence", 0.0)), 4),
            "reason": list(best_candidate.get("reason", [])) or ["Vehicle candidate evaluated"],
            "emergency_light_score": round(float(best_candidate.get("emergency_light_score", 0.0)), 4),
        }

    return {
        "ambulance_detected": False,
        "confidence": 0.0,
        "reason": ["No vehicle candidates detected"],
        "emergency_light_score": 0.0,
    }


def build_ambulance_detection_log_record(
    detection_summary: dict[str, object],
    timestamp: str | None = None,
) -> AmbulanceDetectionLogRecord:
    """Build a CSV-compatible ambulance detection summary record."""

    reasons = detection_summary.get("reason", [])
    if isinstance(reasons, list):
        reason_text = "; ".join(str(reason) for reason in reasons)
    else:
        reason_text = str(reasons)

    return AmbulanceDetectionLogRecord(
        timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
        ambulance_detected=bool(detection_summary.get("ambulance_detected", False)),
        confidence=round(float(detection_summary.get("confidence", 0.0)), 4),
        emergency_light_score=round(float(detection_summary.get("emergency_light_score", 0.0)), 4),
        reason=reason_text,
    )


def write_ambulance_detection_log(
    record: AmbulanceDetectionLogRecord,
    log_path: str | Path = DEFAULT_AMBULANCE_LOG_PATH,
) -> bool:
    """Write ambulance detection summary logs without crashing callers."""

    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=AMBULANCE_LOG_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(asdict(record))
        return True
    except Exception as error:
        logger.error(f"Failed to write ambulance detection log to {path}: {error}")
        return False


def _ambulance_log_rows(
    detections: list[dict[str, object]],
    source_video: str,
    frame_index: int,
    fps: float,
) -> list[dict[str, object]]:
    """Convert public ambulance detections into CSV-compatible log rows."""

    timestamp_seconds = frame_index / fps if fps > 0 else 0.0
    rows: list[dict[str, object]] = []

    for detection in detections:
        xmin, ymin, xmax, ymax = [int(value) for value in detection["bbox"]]
        rows.append(
            {
                "source_video": source_video,
                "frame_index": frame_index,
                "timestamp_seconds": round(timestamp_seconds, 3),
                "class_id": 0,
                "class_name": "ambulance",
                "confidence": detection["confidence"],
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "box_width": max(0, xmax - xmin),
                "box_height": max(0, ymax - ymin),
            }
        )

    return rows


def write_combined_detection_log(log_path: str | Path, rows: list[dict[str, object]]) -> None:
    """Write vehicle and ambulance detections using the shared CSV format."""

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def process_video_with_emergency_detection(
    input_path: str | Path,
    output_path: str | Path = "data/processed/emergency_detection_output.mp4",
    log_path: str | Path = "data/logs/emergency_detections.csv",
    vehicle_model_path: str | Path = "yolov8n.pt",
    ambulance_model_path: str | Path | None = None,
    vehicle_confidence_threshold: float = 0.35,
    ambulance_confidence_threshold: float = 0.35,
    display: bool = False,
    skip_frames: int = 0,
    max_frames: int | None = None,
) -> EmergencyDetectionSummary:
    """Run vehicle detection followed by ambulance detection on a video."""

    if skip_frames < 0:
        raise ValueError("skip_frames must be 0 or greater")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be None or 1 or greater")

    source = validate_video_path(input_path)
    metadata = get_video_metadata(source)
    vehicle_model = load_yolo_model(vehicle_model_path)
    resolved_ambulance_model_path = resolve_ambulance_model_path(ambulance_model_path)
    
    # Load model using load_ambulance_model, which handles default PT check and fallback internally
    ambulance_model = load_ambulance_model(resolved_ambulance_model_path)
    
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    source_fps = metadata.fps if metadata.fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (metadata.width, metadata.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create output video: {output}")

    log_rows: list[dict[str, object]] = []
    processed_frames = 0
    total_vehicle_detections = 0
    total_ambulance_detections = 0

    try:
        for frame_index, frame in iter_frames(source):
            if not should_process_frame(frame_index, skip_frames):
                continue
            if max_frames is not None and processed_frames >= max_frames:
                break

            vehicle_annotated, vehicle_detections = detect_vehicles(
                frame=frame,
                model=vehicle_model,
                confidence_threshold=vehicle_confidence_threshold,
                frame_index=frame_index,
                fps=metadata.fps,
                source_video=str(source),
            )
            annotated, ambulance_detections = detect_ambulances(
                frame=frame,
                model=ambulance_model,
                confidence_threshold=ambulance_confidence_threshold,
                base_frame=vehicle_annotated,
                vehicle_detections=vehicle_detections,
            )

            log_rows.extend(asdict(detection) for detection in vehicle_detections)
            log_rows.extend(
                _ambulance_log_rows(
                    detections=ambulance_detections,
                    source_video=str(source),
                    frame_index=frame_index,
                    fps=metadata.fps,
                )
            )

            total_vehicle_detections += len(vehicle_detections)
            total_ambulance_detections += len(ambulance_detections)

            writer.write(annotated)
            processed_frames += 1

            if display:
                cv2.imshow("Emergency Vehicle Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        writer.release()
        if display:
            cv2.destroyAllWindows()

    write_combined_detection_log(log_path, log_rows)

    return EmergencyDetectionSummary(
        input_path=str(source),
        output_path=str(output),
        log_path=str(log_path),
        vehicle_model_path=str(vehicle_model_path),
        ambulance_model_path=str(resolved_ambulance_model_path),
        vehicle_confidence_threshold=vehicle_confidence_threshold,
        ambulance_confidence_threshold=ambulance_confidence_threshold,
        skip_frames=skip_frames,
        total_frames=metadata.frame_count,
        processed_frames=processed_frames,
        total_vehicle_detections=total_vehicle_detections,
        total_ambulance_detections=total_ambulance_detections,
        emergency_detected=total_ambulance_detections > 0,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for Phase 4 testing."""

    parser = argparse.ArgumentParser(description="Run vehicle plus ambulance detection on a video.")
    parser.add_argument("--input", required=True, help="Path to the source video file.")
    parser.add_argument(
        "--output",
        default="data/processed/emergency_detection_output.mp4",
        help="Path where the annotated output video will be saved.",
    )
    parser.add_argument(
        "--log",
        default="data/logs/emergency_detections.csv",
        help="Path where the combined detection CSV log will be saved.",
    )
    parser.add_argument(
        "--vehicle-model",
        default="yolov8n.pt",
        help="YOLO model path for generic vehicle detection.",
    )
    parser.add_argument(
        "--ambulance-model",
        default=None,
        help="Custom ambulance YOLO model path. Defaults to data/models/ambulance_detector.pt.",
    )
    parser.add_argument(
        "--vehicle-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence threshold for vehicle detections.",
    )
    parser.add_argument(
        "--ambulance-confidence",
        type=float,
        default=0.35,
        help="Minimum confidence threshold for ambulance detections.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display annotated frames in an OpenCV window. Press q to stop.",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help="Skip N frames after each processed frame. Example: 2 processes every third frame.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of processed frames for quick tests.",
    )
    return parser


def main() -> None:
    """Run combined vehicle and ambulance detection from the command line."""

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    args = build_arg_parser().parse_args()
    summary = process_video_with_emergency_detection(
        input_path=args.input,
        output_path=args.output,
        log_path=args.log,
        vehicle_model_path=args.vehicle_model,
        ambulance_model_path=args.ambulance_model,
        vehicle_confidence_threshold=args.vehicle_confidence,
        ambulance_confidence_threshold=args.ambulance_confidence,
        display=args.display,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames,
    )

    print("Phase 4 emergency vehicle detection infrastructure complete")
    for key, value in asdict(summary).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
