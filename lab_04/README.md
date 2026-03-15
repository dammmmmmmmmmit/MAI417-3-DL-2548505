# Lab 04 — Object Detection on VisDrone Dataset (YOLO + Custom CNN)

## About This Program

This lab implements **object detection** on the **VisDrone dataset** — a challenging real-world dataset captured by drones containing dense, small-scale objects like pedestrians, cars, and bikes. Two detection approaches are compared: a **pretrained YOLOv5/YOLOv8 model** (transfer learning) and a **custom CNN built from scratch** using CuPy for GPU acceleration.

---

## Dataset

| Property      | Detail                                                                              |
|---------------|-------------------------------------------------------------------------------------|
| Dataset       | VisDrone (drone-captured imagery)                                                   |
| Objects       | Pedestrians, people, bicycles, cars, vans, trucks, tricycles, awning-tricycles, buses, motors |
| Annotation    | Bounding box format converted to YOLO format (class x_center y_center width height) |
| Config file   | `visdrone.yaml` — specifies dataset paths, number of classes (10), and class names  |

---

## Approach

### 1. YOLOv5 / YOLOv8 (Pretrained — Transfer Learning)
- Used pretrained weights (`yolov5s.pt`, `yolov8n.pt`) as a starting point
- Fine-tuned on the VisDrone dataset
- YOLO (You Only Look Once) processes the entire image in a single forward pass, making it fast and accurate

### 2. Custom CNN from Scratch
- Implemented using **CuPy** (GPU-accelerated NumPy) for manual matrix operations
- Trained without pretrained weights to understand the fundamentals of detection
- Training curves and predictions saved as images (`scratch_training_curves.png`, `scratch_predictions.png`)

### Training Outputs
- `full_comparison.png` — side-by-side comparison of YOLOv5, YOLOv8, and custom CNN detections
- `sample_annotation.png` — visualisation of ground truth bounding boxes
- `ensemble_weights/` — saved ensemble model weights
- `runs/` — YOLO training run logs and results

---

## Output Interpretation

- **YOLO models** (pretrained) typically achieve higher mAP (mean Average Precision) on VisDrone because they leverage millions of pre-learned features from COCO
- **Custom CNN** shows how detection can be built from first principles — useful for understanding architecture design
- `full_comparison.png` visually shows how each model draws bounding boxes around detected objects; more boxes closer to ground truth = better model
- Training curves (`scratch_training_curves.png`) show loss decreasing over epochs, confirming learning is occurring

---

## Files in This Branch

| File / Folder              | Description                                      |
|----------------------------|--------------------------------------------------|
| `2548505_lab4.ipynb`       | Main notebook                                    |
| `visdrone.yaml`            | Dataset config (classes, paths)                    |
| `yolov5s.pt`               | Pretrained YOLOv5s weights                       |
| `yolov8n.pt`               | Pretrained YOLOv8n weights                       |
| `scratch_cnn_best.npy`     | Best custom CNN weights (NumPy format)           |
| `full_comparison.png`      | Detection comparison image                       |
| `sample_annotation.png`    | Ground truth annotation visualisation           |
| `scratch_predictions.png`  | Custom CNN prediction results                    |
| `scratch_training_curves.png` | Loss/accuracy curves for scratch CNN         |
| `ensemble_weights/`        | Ensemble model weights directory                |
| `runs/`                    | YOLO training run outputs                       |

---

## Technologies Used

- Python 3
- PyTorch, Ultralytics (YOLOv5 / YOLOv8)
- CuPy (GPU-accelerated NumPy)
- CUDA (GPU acceleration)
- NumPy, Matplotlib
- YAML (dataset configuration)
