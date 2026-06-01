# Real-Time Object Detection and Audio Feedback System for Visually Impaired Navigation

Graduation thesis project by **İbrahim Emre YILDIZ** (Çukurova University,
Department of Computer Engineering).

## Overview

A local, offline navigation assistance system that detects obstacles through
a webcam, estimates their distance, and delivers spoken feedback to the user.

## Components

| Module | Responsibility |
| --- | --- |
| `config.py` | Centralized, typed configuration |
| `detector.py` | YOLOv8 wrapper and structured detection output |
| `distance_estimator.py` | Pinhole-model distance and directional zone |
| `audio_feedback.py` | Non-blocking Text-to-Speech announcer |
| `visualizer.py` | Debug bounding-box overlay |
| `main.py` | Entry point and real-time loop |

## Installation

```bash
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Run

```bash
python -m src.main
```

Press `q` to exit.
