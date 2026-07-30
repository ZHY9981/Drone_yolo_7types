# Patches to Ultralytics 8.4.12 for Aerial Small-Object Detection

This directory documents every custom modification made to the base Ultralytics 8.4.12 codebase.
These patches enable:

1. **Coordinate Attention (CoordAtt)** in the detection neck
2. **WIoU v3 loss** for small-object gradient enhancement
3. **TAL 4 px threshold** in TaskAlignedAssigner for tiny targets
4. **Class-weighted loss** for long-tail class imbalance

## Patch Index

| File | Modification | Purpose |
|:---|:---|:---|
| `coordatt.py` | New module: CoordAtt coordinate attention | Inject 2D positional encoding into channel attention |
| `loss.py` | Class weights in v8DetectionLoss | Address person/car/cycle imbalance |
| `loss.py` | WIoU v3 replacement for CIoU | Box regression loss with adaptive attention |
| `tal.py` | TAL positive-sample threshold: 8 px → 4 px | Improve small-object assignment |

## 1. CoordAtt (Coordinate Attention)

**Paper:** Hou et al., "Coordinate Attention for Efficient Mobile Network Design," CVPR 2021.

**Where:** Injected at P3 and P4 neck positions, immediately before the Detect head.

**Why here and not backbone:** Per-scale placement avoids cross-FPN degradation.
A CoordAtt at P5 backbone would require two nearest-neighbor upsample steps to reach
P3 detection features, losing spatial precision. Per-scale CoordAtt at P3 and P4
directly enhances the 120×120 and 60×60 feature maps respectively.

**V16 architecture (best model):**
- P3 CoordAtt[32] — enhances 120×120 features for person/cycle/small-bus
- P4 CoordAtt[32] — enhances 60×60 features for car/truck/bus
- No CoordAtt on P5 backbone (removed — dual-head has no P5 detector)

**Code:** See `coordatt.py` in this directory.

---

## 2. WIoU v3 Loss

**Paper:** Tong et al., "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism," arXiv 2023.

**Change:** Replaced the default CIoU loss in `ultralytics/utils/loss.py` with WIoU v3.

**Why:** Standard CIoU penalizes large and small objects equally in box regression.
Small aerial targets (person < 30 px, cycle < 20 px) get disproportionately
weak box regression gradients. WIoU v3 uses an attention-based outlier suppression
mechanism that adaptively focuses on small targets without penalizing large ones.

**Impact:** +1.78% mAP@0.5 (V8.0 → V8.1), zero VRAM cost.

**Modification location:** `ultralytics/utils/loss.py`, in the `bbox_iou` call within
`BboxLoss.forward()`. Replace the default `iou=CIoU` with `iou=WIoUv3`.

---

## 3. TAL 4 px Threshold

**Paper:** Feng et al., "TOOD: Task-aligned One-stage Object Detection," ICCV 2021 (original TAL);

**Change:** Modified `ultralytics/utils/tal.py` — changed `self.stride_tensor` minimum
from 8 px to 4 px in the positive-sample assignment logic.

**Why:** The default TAL positive-sample threshold of 8 px means any anchor point
more than 8 pixels from a ground-truth center is classified as negative. For small
aerial targets (e.g., a person at 20×15 px), this excludes >80% of valid anchor
points from positive training signals.

At 4 px, small targets get ~4× more positive anchors per instance.

**Impact:** Person +6%, cycle +7% (V8 → V10 comparison, holding CoordAtt constant).

**Modification location:** `ultralytics/utils/tal.py`, in the `TaskAlignedAssigner`
initialization or `get_pos_mask` method — reduce the stride floor from 8 to 4.

---

## 4. Class-Weighted Loss

**Why:** The aerial_v9 dataset has extreme class imbalance:
`person: 73,770 | car: 69,840 | cycle: 14,356 | truck: 13,772 | bus: 6,994 | freight: 1,306 | small-bus: 1,117`

Standard equal-weight training causes the model to optimize for person/car at the
expense of freight (< 2% of instances).

**V16 weights (in `ultralytics/utils/loss.py`, `v8DetectionLoss.__init__`):**

```python
self.class_weight = torch.tensor([2.0, 3.0, 1.8, 1.5, 1.0, 1.0, 1.0])
#                                person cycle bus small-bus car truck freight
```

Multiplied into the classification loss for each class. Cycle gets the highest
weight (3.0×) because it is both rare and small. Freight and small-bus use
base weight because they are large and easy to detect despite low instance counts.

---

## Reproducing from Scratch

1. Install base ultralytics: `pip install ultralytics==8.4.12`
2. Apply the above patches to the installed package
3. Use the model configs in `configs/` and training scripts in `scripts/`

See `configs/` for the YOLO26s model architecture definitions used in each version.
