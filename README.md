# AI-Powered Emergency Vehicle Priority System

A resume-focused proof-of-concept that uses computer vision and machine learning to analyze uploaded traffic videos, detect vehicles and ambulances, estimate congestion, and recommend emergency vehicle priority actions.

This project simulates traffic priority decisions. It does not control real traffic lights, IoT hardware, GPS systems, or emergency infrastructure.

## Current Status

Phase 1 - Project Setup

Completed:

- Streamlit-first architecture selected
- CPU-only development target
- YOLOv8n selected as the baseline detector
- CSV selected for initial historical storage
- Video upload mode selected as the first input workflow
- FastAPI deferred

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
    .gitkeep
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

### 5. Expected first app command after Phase 2

The Streamlit app will be added in a later phase. Once created, the expected command will be:

```powershell
streamlit run frontend/app.py
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

## Phase 2 Preview

Phase 2 will add the OpenCV foundation:

- Video upload/loading workflow
- Frame reading
- Basic frame preprocessing
- Early verification with a sample video or uploaded file
