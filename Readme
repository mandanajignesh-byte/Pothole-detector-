# 🚧 RoadScan v2 — Pothole Severity Detection System

[![HuggingFace](https://img.shields.io/badge/🤗%20Demo-Live-blue)](https://huggingface.co/spaces/Jignesh2619/pothole-detector)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)](https://opencv.org)

A road damage assessment pipeline that goes beyond basic detection — it classifies potholes by severity, scores them 1–10, tags them with GPS coordinates, and plots them on an interactive map.

**[→ Try the live demo](https://huggingface.co/spaces/Jignesh2619/pothole-detector)**

---

## What it does

Most YOLO pothole projects stop at "there is a pothole here." This pipeline outputs:

```
major_pothole | severity: 8.4/10 | darkness_delta: 36.79 | GPS: 18.5204, 73.8567
```

That's actionable data a municipality can actually use.

---

## Pipeline

```
Input Image / Video Frame
        │
        ▼
   CLAHE Enhancement          ← contrast boost in LAB color space
        │                        helps model on low-contrast road images
        ▼
   YOLOv8 Detection           ← fine-tuned on severity-labeled dataset
        │                        classes: minor / medium / major / pothole
        ▼
   Darkness Delta Scorer      ← compares pothole brightness vs road surface
        │                        deeper hole = darker = higher score
        ▼
   GPS Interpolation          ← matches frame timestamp to GPS coordinates
        │                        linear interpolation between 1-second GPS logs
        ▼
   Deduplication Filter       ← KDTree spatial index, 5m minimum distance
        │                        prevents same pothole logged multiple times
        ▼
   Output: annotated image + CSV report + interactive Folium map
```

---

## Model Performance

Trained on [Roboflow severity dataset](https://universe.roboflow.com/detection-mnkvw/severity-7s2um/dataset/1) — 813 training images, 4 severity classes.

| Class | mAP50 | Precision | Recall |
|---|---|---|---|
| major_pothole | 0.522 | 0.492 | 0.577 |
| medium_pothole | 0.408 | 0.398 | 0.481 |
| minor_pothole | 0.208 | 0.376 | 0.206 |
| pothole | 0.519 | 0.536 | 0.641 |
| **overall** | **0.414** | 0.327 | 0.322 |

**Key insight from confusion matrix:** 53% of minor potholes are classified as background — the model detects them but confidence falls below threshold. Optimal confidence for minor potholes is 0.15, not the default 0.25. Pipeline uses `conf=0.15` accordingly.

---

## Why darkness delta instead of MiDaS

The original design used MiDaS for monocular depth estimation. It failed on this dataset because road images are top-down, grayscale, and uniform in texture — MiDaS needs color variation and perspective cues to estimate depth.

The replacement: a pothole blocks light from reaching its bottom. Deeper hole → darker interior → larger brightness difference vs surrounding road.

```python
darkness_delta = road_surface_brightness - pothole_interior_brightness
# positive = real pothole, higher = deeper
```

Simpler, faster, and more reliable on this specific dataset.

---

## Severity Scoring Formula

```python
darkness_score = min(darkness_delta / 80.0, 1.0)   # normalized 0-1
area_score     = min(bbox_area / 50000, 1.0)         # normalized 0-1

base_score = (0.7 × darkness_score) + (0.3 × area_score)
# depth weighted 70% — a small deep pothole is more dangerous than a large shallow one

adjusted = base_score + (class_weight - 1.0) × 0.2 × base_score
# class label adds max 10% bonus — physical measurement dominates

final_score = adjusted × 10  # scale to 1-10
```

Class weights: `major=1.5, medium=1.2, minor=1.0, pothole=1.1`

---

## GPS Pipeline

For video input, each frame is matched to GPS coordinates via linear interpolation:

```python
# GPS logs at 1Hz, video runs at 30fps
# Frame 45 at t=1.5s → interpolate between t=1.0s and t=2.0s GPS points
ratio = (timestamp - t0) / (t1 - t0)
lat   = lat0 + ratio × (lat1 - lat0)
lng   = lng0 + ratio × (lng1 - lng0)
```

Duplicate detections filtered with a KDTree spatial index — any detection within 5 meters of an already-logged pothole is skipped.

---

## Results

| | Before (YOLO only) | After (full pipeline) |
|---|---|---|
| Output | "pothole detected" | severity 8.4/10, GPS tagged |
| Depth signal | none | darkness delta |
| Location | none | GPS interpolated |
| Map | none | interactive Folium map |
| Duplicates | N/A | deduplicated via KDTree |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/pothole-detector
cd pothole-detector
pip install ultralytics opencv-python folium pandas gradio
```

**Single image:**
```python
from pipeline import run_pipeline
detections = run_pipeline('road_image.jpg')
```

**Video / image folder with GPS map:**
```python
from pipeline import process_video, generate_map
df = process_video('path/to/images/')
generate_map()  # outputs pothole_map.html
```

---

## Project Structure

```
pothole-detector/
├── app.py                  ← Gradio demo (HuggingFace Space)
├── pipeline.py             ← full detection + scoring + GPS pipeline
├── pothole_best.pt         ← fine-tuned YOLOv8s weights
├── requirements.txt
└── notebooks/
    └── Pothole_Detection.ipynb   ← training + experiments
```

---

## Known Limitations

- **Water-filled potholes** score low — water reflects light, making interior brighter than road surface. Darkness delta gives near-zero or negative result.
- **Minor pothole recall is 0.206** — model misses 8 out of 10 minor potholes. Primary fix would be collecting more labeled minor pothole data (dataset had only 43 instances).
- **GPS is simulated** in the current demo — real deployment requires a dashcam + GPX logger running simultaneously.

---

## What I learned building this

- MiDaS fails on top-down low-contrast road images — domain mismatch between training data and real use case
- Over-augmentation hurts more than it helps — 22x augmentation degraded mAP across all classes
- Confusion matrix tells you *where* a model fails, mAP only tells you *how much*
- Simpler signals (brightness comparison) often outperform complex models (depth estimation) when the domain is constrained

---

## Dataset

[Pothole Severity Dataset — Roboflow Universe](https://universe.roboflow.com/detection-mnkvw/severity-7s2um/dataset/1)

813 training images, 216 validation images, 4 severity classes, YOLOv8 format.

---

## Tech Stack

- **Detection:** YOLOv8s (Ultralytics)
- **Preprocessing:** OpenCV CLAHE (LAB color space)
- **Mapping:** Folium + OpenStreetMap
- **Spatial indexing:** scipy KDTree
- **Demo:** Gradio + HuggingFace Spaces
- **Training:** Google Colab T4 GPU
