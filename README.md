# AI-Powered Emergency Vehicle Priority System

A resume-focused proof-of-concept that uses computer vision and machine learning to analyze uploaded traffic videos, detect vehicles and ambulances, estimate congestion, and recommend emergency vehicle priority actions.

This project simulates traffic priority decisions. It does not control real traffic lights, IoT hardware, GPS systems, or emergency infrastructure.

## Current Status

Phase 5 - Traffic Density Analysis

Completed:

- Streamlit-first architecture selected
- CPU-only development target
- YOLOv8n selected as the baseline detector
- CSV selected for initial historical storage
- Video upload mode selected as the first input workflow
- FastAPI deferred
- OpenCV video loading foundation added
- Frame extraction and basic frame annotation added
- Annotated output video saving added
- YOLOv8n vehicle detection added for car, motorcycle, bus, and truck
- Bounding boxes and confidence labels added
- CSV detection logs added
- Separate ambulance detector module added
- Emergency presence helper added
- Combined vehicle plus ambulance detection pipeline added
- Traffic density analysis module added
- Configurable LOW, MEDIUM, and HIGH density thresholds added
- Density CSV logging support added

## Technology Stack

- Python
- Streamlit
- OpenCV
- YOLOv8n through Ultralytics
- Scikit-Learn
- Pandas
- NumPy
- Plotly / Matplotlib
- CSV-based storage
- Pytest

## Project Structure

```text
AI-Emergency-Vehicle-Priority-System/
  backend/
    .gitkeep
  data/
    raw/
      .gitkeep
    processed/
      .gitkeep
    logs/
      .gitkeep
    models/
      .gitkeep
  docs/
    .gitkeep
  frontend/
    .gitkeep
  ml/
    __init__.py
    .gitkeep
    analytics/
      __init__.py
      density_analyzer.py
    cv_pipeline.py
    detectors/
      __init__.py
      ambulance_detector.py
      vehicle_detector.py
  tests/
    .gitkeep
  .gitignore
  PRD.md
  PROJECT_SETUP.md
  README.md
  requirements.txt
```

## Folder Responsibilities

- `frontend/`: Streamlit dashboard and user-facing app files.
- `ml/`: Computer vision, vehicle counting, congestion prediction, and priority engine modules.
- `data/raw/`: Uploaded or sample input videos.
- `data/processed/`: Processed frames, derived datasets, or intermediate outputs.
- `data/logs/`: CSV detection logs, congestion history, and dashboard history.
- `data/models/`: Downloaded YOLO weights or trained model artifacts.
- `backend/`: Reserved for optional helper services or API work if approved later.
- `docs/`: Architecture notes, screenshots, diagrams, and dataset documentation.
- `tests/`: Unit tests for density classification, counting, prediction, and recommendation logic.

## Setup Instructions

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify the environment

```powershell
python --version
pip list
```

### 5. Run the Phase 2 OpenCV pipeline

Place a sample traffic video in `data/raw/`, then run:

```powershell
python ml/cv_pipeline.py --input data/raw/sample_traffic.mp4 --output data/processed/annotated_output.mp4
```

To export processed frames as JPG files:

```powershell
python ml/cv_pipeline.py --input data/raw/sample_traffic.mp4 --output data/processed/annotated_output.mp4 --frames-dir data/processed/frames
```

For a quick test on only the first 100 frames:

```powershell
python ml/cv_pipeline.py --input data/raw/sample_traffic.mp4 --output data/processed/annotated_output.mp4 --max-frames 100
```

To display processed frames in an OpenCV window:

```powershell
python ml/cv_pipeline.py --input data/raw/sample_traffic.mp4 --display
```

Press `q` to close the display window early.

### 6. Expected Streamlit command after Phase 7

The Streamlit app will be added in a later phase. Once created, the expected command will be:

```powershell
streamlit run frontend/app.py
```

## Vehicle Detection

Phase 3 uses YOLOv8n to detect the following COCO vehicle classes:

- `car`
- `motorcycle`
- `bus`
- `truck`

Ambulance-specific detection is handled separately by the Phase 4 ambulance detector so the generic vehicle detector stays focused on COCO vehicle classes.

### Run YOLOv8n Vehicle Detection

Place a sample traffic video in `data/raw/`, then run:

```powershell
python ml/detectors/vehicle_detector.py --input data/raw/sample_traffic.mp4 --output data/processed/vehicle_detection_output.mp4 --log data/logs/vehicle_detections.csv
```

For a quick CPU-friendly test:

```powershell
python ml/detectors/vehicle_detector.py --input data/raw/sample_traffic.mp4 --output data/processed/vehicle_detection_output.mp4 --log data/logs/vehicle_detections.csv --max-frames 100
```

To skip two frames after each processed frame:

```powershell
python ml/detectors/vehicle_detector.py --input data/raw/sample_traffic.mp4 --skip-frames 2
```

To adjust the confidence threshold:

```powershell
python ml/detectors/vehicle_detector.py --input data/raw/sample_traffic.mp4 --confidence 0.45
```

To display annotated frames:

```powershell
python ml/detectors/vehicle_detector.py --input data/raw/sample_traffic.mp4 --display
```

The reusable frame-level detector API is:

```python
annotated_frame, detections = detect_vehicles(frame)
```

The first YOLOv8n run may download `yolov8n.pt` if the weight file is not already available locally.

### Detection Log Format

Vehicle detections are written to CSV with these columns:

```text
source_video,frame_index,timestamp_seconds,class_id,class_name,confidence,xmin,ymin,xmax,ymax,box_width,box_height
```

Each row represents one detected vehicle in one processed frame.

## Ambulance Detection

Phase 4 adds ambulance detection infrastructure in `ml/detectors/ambulance_detector.py`.

This module is separate from the generic vehicle detector. It does not change the Phase 3 vehicle-only pipeline, and it does not claim ambulance detection accuracy until a real ambulance model and test dataset are evaluated.

### Ambulance Model Requirements

The ambulance detector expects a custom YOLO model trained or configured to detect ambulances.

Default model path:

```text
data/models/ambulance_detector.pt
```

You can override the model path with:

```powershell
python ml/detectors/ambulance_detector.py --ambulance-model data/models/custom_ambulance.pt --input data/raw/sample_traffic.mp4
```

Or set:

```powershell
$env:AMBULANCE_MODEL_PATH="data/models/custom_ambulance.pt"
```

If the model class names include `ambulance`, `emergency vehicle`, or `emergency_vehicle`, that class is used. If class metadata is unavailable, the detector assumes class ID `0`, which is common for single-class custom YOLO models.

### Reusable Ambulance API

```python
annotated_frame, ambulance_detections = detect_ambulances(frame)
```

Each ambulance detection uses this format:

```python
{
    "class_name": "ambulance",
    "confidence": 0.0,
    "bbox": [xmin, ymin, xmax, ymax],
}
```

Emergency presence helper:

```python
is_emergency_present(ambulance_detections)
```

Returns `True` when at least one ambulance detection exists, otherwise `False`.

### Run Combined Vehicle and Ambulance Detection

```powershell
python ml/detectors/ambulance_detector.py --input data/raw/sample_traffic.mp4 --output data/processed/emergency_detection_output.mp4 --log data/logs/emergency_detections.csv --max-frames 100
```

To skip two frames after each processed frame:

```powershell
python ml/detectors/ambulance_detector.py --input data/raw/sample_traffic.mp4 --skip-frames 2
```

Expected output:

- Annotated output video at `data/processed/emergency_detection_output.mp4`
- Combined CSV log at `data/logs/emergency_detections.csv`
- Vehicle boxes from the generic vehicle detector
- Distinct ambulance boxes labeled `AMBULANCE`
- Frame-level status overlay showing `Emergency Vehicle: DETECTED` or `Emergency Vehicle: NONE`

### Combined CSV Log Format

Vehicle and ambulance detections use the same CSV columns:

```text
source_video,frame_index,timestamp_seconds,class_id,class_name,confidence,xmin,ymin,xmax,ymax,box_width,box_height
```

Ambulance rows use `class_name` set to `ambulance`.

## Traffic Density Analysis

Phase 5 adds traffic density analysis in `ml/analytics/density_analyzer.py`.

The analyzer consumes Phase 3 vehicle detections or count dictionaries and calculates:

- Total vehicle count
- Car count
- Motorcycle count
- Bus count
- Truck count
- Density level: `LOW`, `MEDIUM`, or `HIGH`

It is independent of Streamlit and does not implement congestion prediction, emergency priority decisions, or traffic signal control.

### Density Thresholds

Default configurable constants:

```python
LOW_VEHICLE_THRESHOLD = 10
MEDIUM_VEHICLE_THRESHOLD = 25
```

Classification rules:

```text
LOW: total vehicles < 10
MEDIUM: total vehicles < 25
HIGH: total vehicles >= 25
```

### Reusable Density API

```python
density_result = analyze_density(vehicle_counts)
```

Example input:

```python
vehicle_counts = {
    "car_count": 10,
    "motorcycle_count": 4,
    "bus_count": 2,
    "truck_count": 2,
}
```

Example output:

```python
{
    "total_vehicles": 18,
    "car_count": 10,
    "motorcycle_count": 4,
    "bus_count": 2,
    "truck_count": 2,
    "density": "MEDIUM",
}
```

### Density Overlay

Use `draw_density_overlay(frame, density_result)` to add:

```text
Density: LOW
Density: MEDIUM
Density: HIGH
```

The overlay also displays the total vehicle count.

### Density Logging

Density logs are written with these columns:

```text
source_video,frame_index,timestamp_seconds,total_vehicles,car_count,motorcycle_count,bus_count,truck_count,density
```

Analyze an existing Phase 3 detection log:

```powershell
python ml/analytics/density_analyzer.py --detections-log data/logs/vehicle_detections.csv --output-log data/logs/density_analysis.csv
```

## Development Phases

1. Project Setup
2. Computer Vision Foundation
3. Vehicle Detection
4. Ambulance Detection
5. Vehicle Counting
6. Congestion Classification
7. Dashboard
8. Machine Learning
9. Priority Engine
10. Analytics
11. Production Readiness

## Phase 2 Testing Steps

1. Install dependencies with `pip install -r requirements.txt`.
2. Add a short video file to `data/raw/`.
3. Run `python ml/cv_pipeline.py --input data/raw/sample_traffic.mp4 --output data/processed/annotated_output.mp4 --max-frames 100`.
4. Confirm the terminal output shows FPS, frame count, dimensions, and processed frame count.
5. Confirm `data/processed/annotated_output.mp4` is created.
6. Run again with `--frames-dir data/processed/frames` and confirm JPG frames are created.
7. Optionally run with `--display` and confirm processed frames appear with frame number, FPS, and timestamp overlays.

## Phase 3 Testing Steps

1. Install dependencies with `pip install -r requirements.txt`.
2. Add a short traffic video to `data/raw/`.
3. Run `python ml/detectors/vehicle_detector.py --input data/raw/sample_traffic.mp4 --max-frames 100`.
4. Confirm `data/processed/vehicle_detection_output.mp4` is created.
5. Confirm the output video shows bounding boxes and confidence labels for cars, motorcycles, buses, and trucks.
6. Confirm `data/logs/vehicle_detections.csv` is created.
7. Open the CSV and confirm rows use the documented detection log format.
8. Re-run with `--confidence 0.45` and compare detection counts.
9. Re-run with `--skip-frames 2` and confirm fewer frames are processed.

## Phase 4 Testing Steps

1. Install dependencies with `pip install -r requirements.txt`.
2. Add a custom ambulance YOLO model to `data/models/ambulance_detector.pt` or pass `--ambulance-model`.
3. Run `python ml/detectors/ambulance_detector.py --input data/raw/sample_traffic.mp4 --max-frames 100`.
4. Confirm the script creates `data/processed/emergency_detection_output.mp4`.
5. Confirm the script creates `data/logs/emergency_detections.csv`.
6. Confirm ambulance rows, when present, use the shared CSV log format.
7. Confirm the frame overlay shows either `Emergency Vehicle: DETECTED` or `Emergency Vehicle: NONE`.

## Phase 5 Testing Steps

1. Install dependencies with `pip install -r requirements.txt`.
2. Generate a Phase 3 vehicle detection log at `data/logs/vehicle_detections.csv`.
3. Run `python ml/analytics/density_analyzer.py --detections-log data/logs/vehicle_detections.csv --output-log data/logs/density_analysis.csv`.
4. Confirm `data/logs/density_analysis.csv` is created.
5. Confirm each row includes total count, per-class counts, and `LOW`, `MEDIUM`, or `HIGH`.
6. Import `analyze_density` in Python and verify it returns the documented dictionary format.

## Phase 6 Preview

Phase 6 will add congestion classification history:

- Store density history over time
- Prepare low, medium, and high congestion records
- Keep prediction logic for a later ML phase
