# Results & Figures Guide

> This folder will contain visualization outputs from the trained model.
> You need to export these from Ultralytics' `runs/` directory after training.

## What to Include

### 1. Training Curves (`V16_curves.png`)
- **Source**: `<project>/runs/aerial_train/yolo26s_v16/results.png`
- **What it shows**: mAP50, mAP50-95, Precision, Recall, box loss, cls loss, dfl loss
  over 200 epochs
- **How to get**: Ultralytics generates this automatically during training with `plots=True`.
  Copy it from the training output directory.

### 2. Confusion Matrix (`confusion_matrix.png`)
- **Source**: `<project>/runs/aerial_train/yolo26s_v16/confusion_matrix.png`
- **What it shows**: Normalized confusion matrix on the validation set (7 classes + background)
- **How to get**: Ultralytics generates this during validation at the end of training.

### 3. Detection Samples (`detection_samples.png`)
Take 3–4 representative screenshots or use Ultralytics' validation batch visualizations:

| Scene | Altitude | What to look for |
|-------|----------|------------------|
| Dense urban street | Low (~50m) | Multiple classes: person, cycle, car in frame |
| Suburb / highway | Medium (~100m) | Car, truck detection at moderate scale |
| Open area | High (~150m+) | Small pedestrians, cyclists (challenging targets) |
| Night / low-light | Any | (Optional) Edge-case performance |

**How to get**: After validation, check the `val_batch0_pred.jpg` or `val_batch1_pred.jpg`
files in `<project>/runs/aerial_train/yolo26s_v16/`. Alternatively, run:
```python
from ultralytics import YOLO
model = YOLO("path/to/best.pt")
model.predict(source="path/to/sample_images/", save=True, save_txt=False)
```

### Expected File Structure
```
results/
├── V16_curves.png           (required — training convergence)
├── confusion_matrix.png     (required — per-class accuracy)
└── detection_samples.png    (required — 3–4 representative detections)
```
