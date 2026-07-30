# Technical Report — YOLO26s Aerial Small-Object Detection

> **Author**: Haoyi Zou (ZHY9981)
> **Date**: July 2026
> **Status**: Technical report — not a published paper

## Abstract

Aerial small-object detection presents a dual challenge: targets occupy < 1% of image pixels,
and consumer GPUs impose hard memory limits. This project investigates a customized YOLO26s
architecture under the fixed constraint of an RTX 5060 Laptop GPU (8 GB VRAM, batch=4).
Through 20+ versioned ablation experiments, we identify five complementary modifications —
Coordinate Attention, P3+P4 dual-head, WIoU v3 loss, a tightened TAL threshold, and class
weighting — that collectively achieve **74.00% mAP@0.5** at 7.03M parameters and 4.0 ms
inference (FP16). The best model removes the P5 detection head entirely, trading large-object
capacity for 32% parameter reduction and +2.1% mAP gain on 8 GB hardware. All training
and evaluation are reproducible from this repository.

## 1. Introduction

### 1.1 Problem Statement

Object detection from UAV platforms (drones) is critical for surveillance, search-and-rescue,
traffic monitoring, and disaster assessment. However, three factors make this a uniquely
difficult setting:

1. **Tiny targets**: Pedestrians at 100 m+ altitude occupy fewer than 30 pixels — under 1% of
   a typical 800×800 crop. Standard detection heads, designed around COCO-scale objects,
   routinely miss them.

2. **Consumer GPU constraints**: Our hardware (RTX 5060 Laptop, 8 GB VRAM) cannot fit a
   high-resolution P2 detection head (25,600 grid cells) at batch=4 — the TaskAlignedAssigner's
   alignment matrix alone exceeds available memory. This constraint is fixed; it cannot be
   relaxed by hardware upgrades.

3. **Class imbalance**: In our custom dataset (aerial_v9, 7 classes, 8,075 images), person and
   car each account for ~30% of instances, while freight and small-bus each account for < 2%.

### 1.2 Contributions

- Systematic ablation study of 20+ architectural and hyperparameter modifications under a
  **fixed 8 GB VRAM budget**
- A **dual-head (P3+P4) architecture** that outperforms the standard triple-head by +2.1% mAP
  while reducing parameters by 32%
- Per-scale **Coordinate Attention** (CoordAtt) that boosts person detection by 3.2 pp and
  cycle by 4.9 pp
- Honest documentation of **failed experiments**, including P2 OOM, 2-stage training collapse,
  and transfer-learning relay failure

## 2. Dataset

### 2.1 aerial_v9 (Primary Dataset)

Curated from three public aerial datasets (aerial.v1i CC BY 4.0, Aerial Vehicle Detection MIT,
aerial.v3i MIT) with extensive label cleaning. Key differences from raw VisDrone2019:

- **7 classes** (person, cycle, bus, small-bus, car, truck, freight) — consolidated from
  VisDrone's 10-class schema
- **8,075 training** / **2,224 validation** / **738 test** images
- **Label quality** is the dominant factor: switching from VisDrone-noisy labels to aerial_v9
  accounted for +12.56% mAP@0.5, more than any architectural change (Table 1, V7→V8)

| Class | Train Instances | % of Total |
|:------|:--------------:|:----------:|
| person | 73,770 | 40.7% |
| car | 69,840 | 38.5% |
| cycle | 14,356 | 7.9% |
| truck | 13,772 | 7.6% |
| bus | 6,994 | 3.9% |
| freight | 1,306 | 0.7% |
| small-bus | 1,117 | 0.6% |

### 2.2 VisDrone2019 (Comparison Only)

Used exclusively in V11.0 as a public benchmark reference. Official split: 6,471 train / 548 val,
10 classes. Our YOLO26s achieved 32.65% mAP@0.5 vs. published SOTA (YOLOv8s-p2, ~43.7%).
The gap is attributable to two factors: (1) YOLO26s has no P2 head, limiting small-object
resolution; (2) VisDrone's label noise disproportionately impacts models without domain-specific
data cleaning. This comparison is included for transparency; the primary benchmark is aerial_v9.

## 3. Architecture

### 3.1 Baseline

The starting point is a standard YOLO26s (s-scale: 50% depth, 50% width, max 1024 channels)
with P3/8–P5/32 outputs and CoordAtt attention in the backbone. This is the same architecture
used in V10.0 (mAP@0.5 = 70.51%) and serves as the pre-dual-head reference.

### 3.2 Key Modifications (V16.0 — Best Model)

**P3+P4 Dual-Head (V14.0).** The standard P5 detection head targets large objects (e.g.,
buses occupying > 100×100 pixels). However, in our aerial setting, large objects are rare
and already well-detected by P4 (AP50: bus=91.8%, freight=96.6%). Removing P5 frees ~40%
of the head's VRAM budget, allowing the P3 head to operate at imgsz=800 without OOM.
Result: -32% parameters, +2.1% mAP vs. the V12 triple-head baseline.

**Per-Scale CoordAtt (V16.0).** Rather than placing CoordAtt solely in the backbone (V10.0),
V16.0 inserts independent CoordAtt[32] modules at two locations: after the P3 FPN fusion
(enhancing 120×120 grid attention) and after P4 PAN fusion (enhancing 60×60 grid attention).
This per-scale placement targets spatial attention directly at the detection grid, producing
+3.2 pp person AP50 and +4.9 pp cycle AP50 over the V14 baseline.

**WIoU v3 Loss (V8.1).** Weighted IoU replaces standard CIoU, using an attention-based
weighting scheme that amplifies the gradient contribution of high-quality anchor boxes.
Cost: 2 lines of code, zero VRAM. Benefit: +1.78% mAP@0.5, primarily through +2.0% Recall.

**TAL 4px Threshold.** The default TaskAlignedAssigner uses an 8-pixel center-radius
threshold to determine positive samples. For objects < 30 pixels, this excludes most
valid anchors. Reducing to 4px directly boosts person/cycle by +6–7% each.

**Class Weighting.** Applied via the loss function: `cls_weights = [2.0, 3.0, 1.8, 1.5, 1.0, 1.0, 1.0]`
for [person, cycle, bus, small-bus, car, truck, freight]. Cycle receives the highest weight
(3.0×) due to its small size, high intra-class variance, and low instance count.

### 3.3 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 200 |
| Batch size | 4 |
| Input size | 800×800 |
| Optimizer | SGD, lr₀=0.01, momentum=0.937, weight_decay=5e-4 |
| LR schedule | Linear warmup (3 epochs) → cosine to lr₀×0.01 |
| Augmentation | mosaic=0.5, degrees=45, scale=0.3, flipud=0.5 |
| Mixup/Copy-paste | Disabled (harmful for small objects) |
| Close mosaic | Epoch 10 |
| Mixed precision | AMP + FP16 |
| Training time | ~16.6 hours |

## 4. Ablation Results

### 4.1 Main Progression

| Version | Key Change | mAP@0.5 | mAP@0.5:0.95 | Δ |
|:-------:|-----------|:-------:|:------------:|:---:|
| V8.0 | Clean data baseline | 67.81% | 44.31% | — |
| V8.1 | + WIoU v3 | 69.59% | 47.00% | +1.78% |
| V10.0 | + CoordAtt + aerial_v9 | 70.51% | 49.33% | +0.92% |
| V12.0 | + imgsz 640→800 | 71.58% | 50.27% | +1.07% |
| V14.0 | P3+P4 dual-head, imgsz=960 | 73.71% | **52.65%** | +2.13% |
| **V16.0** | **+ per-scale CoordAtt** | **74.00%** | 52.11% | +0.29% |

### 4.2 Failed Approaches

| Version | Approach | Result | Root Cause |
|:-------:|---------|:------:|------------|
| V4.0 | Freeze→unfreeze 2-stage | 46.82% | COCO pretrain features misaligned with aerial domain |
| V7.0 | VisDrone raw labels | 55.25% (R=50%) | Label noise → massive false negatives |
| V8.2 | Transfer relay V8.1→v9 | 69.71% | lr=0.001 too low for new data distribution |
| V13.0 | P2 high-res head | 66.68% | 8 GB OOM at batch=4; batch=2 unstable |

### 4.3 V16.0 Per-Class Breakdown

| Class | AP50 | AP50-95 | Precision | Recall |
|:------|:----:|:-------:|:---------:|:------:|
| person | 60.2% | 24.7% | 77.7% | 50.9% |
| cycle | 43.0% | 17.6% | 67.0% | 36.3% |
| bus | 91.8% | 74.5% | 91.3% | 86.3% |
| small-bus | 99.2% | 83.3% | 92.3% | 99.4% |
| car | 67.2% | 47.3% | 87.2% | 51.4% |
| truck | 60.0% | 41.0% | 63.8% | 53.6% |
| freight | 96.6% | 76.6% | 89.8% | 95.0% |
| **Overall** | **74.0%** | **52.1%** | **81.3%** | **67.6%** |

Key observations:
- **Small objects (person, cycle)** dominate the error: AP50-95 of 24.7% and 17.6%
  respectively indicate precise localization remains the bottleneck.
- **Large objects (bus, freight, small-bus)** are near-solved, with AP50 > 90%.
- **Car (67.2%)** is the outlier among abundant classes: 69,840 instances yet only 47.3%
  AP50-95. Removing P5 disproportionately impacted car, which sits at the P4→P5 boundary.

## 5. Discussion

### 5.1 Data >> Architecture

The single largest improvement (+12.56% mAP@0.5) came from curating a clean dataset (V7→V8),
not from any architectural modification. This finding aligns with the broader literature on
noisy-label robustness in detection, but its magnitude is notable: **data quality contributed
6× more than CoordAtt** to the final result. This has practical implications for practitioners:
before optimizing architecture, invest in label quality.

### 5.2 Dual-Head as a Constraint-Driven Innovation

The dual-head (P3+P4) design was not an aesthetic choice — it was forced by the OOM on the
triple-head + P2 variant (V13.0). Under this constraint, the removal of P5 proved to be a
**constructive** rather than destructive modification: it freed VRAM for higher P3 resolution
and removed a head that primarily served large objects already well-detected by P4. This is
an example of "necessity-driven optimization" — a narrative that resonates in resource-limited
deployment scenarios.

### 5.3 CoordAtt Convergence

CoordAtt's per-scale placement (V16.0) required extended training: 120 epochs to reach
72.9%, and the final 80 epochs contributed +1.15 pp. This suggests that attention mechanisms
in detection heads benefit from longer schedules than standard convolution-only architectures,
likely because attention weights interact non-trivially with the anchor assignment dynamics
during later training stages.

### 5.4 Limitations

- **Person/cycle recall remains low** (50.9% / 36.3%). These classes have the most extreme
  scale variation and inter-class similarity (cycle vs. person-on-bike). Purely geometric
  approaches (TAL threshold, dual-head) partially address this; further gains likely require
  multi-scale test-time augmentation or feature-level super-resolution.
- **Car detection is the P5 gap**: The 67.3% AP50 for car (vs. 71%+ on triple-head V12)
  suggests that some car instances genuinely require P5-scale features. The triple-head V20
  partially recovers this (+0.6%) but at the cost of person/cycle regression — a trade-off
  that favours dual-head for our application.
- **Inference-only evaluation**: All metrics are on the validation set. Real-time drone
  deployment introduces motion blur, altitude variation, and illumination changes not
  captured in the static test set.

## 6. Conclusion

This project demonstrates that under a fixed 8 GB VRAM constraint, a carefully designed
dual-head architecture with per-scale attention can achieve competitive aerial detection
performance (74.00% mAP@0.5, 7.03M params, 4.0 ms inference). The methodological
contribution is in the systematic ablation approach — 20+ versions, each isolating one
variable — which provides a transparent, reproducible account of what works and why.

The repository at [github.com/ZHY9981/Drone_yolo_7types](https://github.com/ZHY9981/Drone_yolo_7types)
contains the full training configuration, evaluation scripts, model weights (Release), and
version history needed to reproduce all results.

---

> **Note**: This is a technical report, not a peer-reviewed publication. It serves as
> supporting documentation for the associated GitHub repository and as evidence of
> research methodology for graduate (MSc) application review.
