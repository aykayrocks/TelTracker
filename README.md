# TelTracker
### Intelligent Warehouse Safety & Visual Monitoring System

TelTracker is an ongoing Computer Vision and Applied Machine Learning project focused on developing an intelligent visual monitoring system for industrial and warehouse environments.

The project initially serves as a learning-oriented implementation of object detection and multi-object tracking, with the long-term goal of evolving into a system capable of monitoring workplace activity, identifying potential safety risks, and analysing worker movement in warehouse environments.

**Current status:** Early development — Milestone 1 (Object Detection) completed locally; further development and testing are in progress.

---

## Project Overview

Warehouses and industrial facilities involve the continuous movement of workers, forklifts, machinery, and other equipment. Monitoring these environments manually can make it difficult to identify unsafe interactions, restricted-zone violations, and potentially hazardous situations.

TelTracker aims to explore how Computer Vision can help address these challenges through automated object detection, tracking, spatial analysis, and visual monitoring.

The initial development focuses on building the fundamental components of a multi-object tracking system before extending it into an industry-oriented safety monitoring application.

## Current Implementation

The first milestone establishes a basic object detection pipeline using YOLO and OpenCV.

Current capabilities:

- **Person Detection:** Detects people in video footage using a pretrained YOLOv8 Nano model.
- **Bounding Box Visualisation:** Displays detected people using bounding boxes.
- **Confidence Scores:** Displays the model's confidence for each detection.
- **FPS Monitoring:** Measures and displays approximate processing speed.
- **Video Processing:** Reads and processes video files frame by frame.

The current implementation focuses exclusively on person detection. Warehouse-specific safety monitoring features have not yet been implemented.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| OpenCV | Video processing and visualisation |
| YOLOv8 (Ultralytics) | Object detection |
| NumPy | Numerical operations and array processing |
| PyTorch | Deep learning framework underlying YOLO |

## Project Structure

```text
TelTracker/
│
├── videos/                 # Input test videos
│
├── src/
│   ├── main.py             # Main video processing loop
│   ├── detector.py         # YOLO detection implementation
│   ├── visualize.py        # Bounding box and FPS visualisation
│   └── iou.py              # Planned: IoU implementation
│
├── tests/                  # Planned: Unit tests
│
├── requirements.txt        # Project dependencies
├── README.md
└── .gitignore
```

*Note: Some files and directories listed above are planned additions and may not exist in the current version.*

## Setup and Installation

### Prerequisites

- Python 3.12
- Git
- A video file containing people for testing

### 1. Clone the repository

```bash
git clone https://github.com/aykayrocks/TelTracker.git
cd TelTracker
```

### 2. Create a virtual environment

```bash
py -3.12 -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add a test video

Place a video containing people inside the `videos/` directory.

For example:

```text
videos/
└── sample.mp4
```

### 5. Run the application

```bash
python src/main.py --source videos/sample.mp4
```

The first execution may automatically download the pretrained YOLOv8 Nano weights.

Press `q` to close the video window.

## Development Roadmap

TelTracker is being developed incrementally, with each milestone introducing new Computer Vision concepts and gradually expanding the system's capabilities.

| Milestone | Planned Development | Status |
|---|---|---|
| 1 | Video-based person detection using YOLO | Initial implementation |
| 2 | Intersection over Union (IoU) calculation and unit testing | Planned |
| 3 | Custom IoU-based multi-object tracking with persistent IDs | Planned |



