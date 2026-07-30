# YOLO26s + CoordAtt: Resource-Constrained Aerial Small-Object Detection

> Real-time detection of 7 object classes from drone imagery, optimized for 8 GB VRAM consumer GPUs.
> 无人机航拍小目标检测 — 在 RTX 5060 笔记本 8GB 显存约束下做资源受限的目标检测研究。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This project explores aerial small-object detection using a customized YOLO26s architecture,
specifically engineered for resource-constrained deployment scenarios. The key challenge:
detecting tiny targets (pedestrians, cyclists, vehicles) in high-resolution drone footage while
fitting within an **8 GB VRAM budget** (RTX 5060 Laptop GPU).

Rather than scaling up hardware, every design decision is evaluated by both accuracy and memory
cost — a discipline forced by the fixed hardware constraint and documented through 20+ versioned
ablation experiments.

**Best model (V16.0):**

| Metric | Value | Note |
|--------|-------|------|
| mAP@0.5 | **74.00%** | V16 (P3+P4 dual-head + CoordAtt), imgsz=800 |
| mAP@0.5:0.95 | **52.11%** | V16 |
| mAP@0.5:0.95 (peak) | **52.65%** | V14 (P3+P4 dual-head, imgsz=960) |
| Parameters | **7.03 M** | V16 |
| Precision | **81.3%** | V16 validation |
| Recall | **67.6%** | V16 validation |
| Inference | **4.0 ms** | RTX 5060 8 GB, imgsz=800, FP16 (avg over 2224 val images) |

### V16.0 Per-Class Performance

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

Training: 200 epochs, ~16.6h on RTX 5060 8GB. E120=72.9%, final 80 epochs contribute +1.15pp — CoordAtt benefits from extended convergence.

## The Problem

Aerial small-object detection is hard for three compounding reasons:

- **Tiny targets**: a pedestrian viewed from 100 m+ altitude occupies < 30 pixels — < 1% of the
  image. Standard detection heads routinely miss them.
- **OOM on consumer GPUs**: adding a high-resolution P2 head (25,600 cells) to capture small
  objects causes the TaskAlignedAssigner to allocate an alignment matrix that exceeds 8 GB VRAM
  at batch=4 — the constraint cannot be relaxed by buying hardware.
- **Class imbalance**: person/car dominate; freight and small-bus are rare (each < 2% of instances).

## Architecture Modifications

The core improvements target three bottlenecks at once: **positional information loss** (CoordAtt),
**small-object gradient starvation** (WIoU v3, TAL 4px), and **VRAM overcommitment** (P3+P4 dual-head).
All modifications are applied on top of a customized Ultralytics 8.4.12 fork.

1. **Coordinate Attention (CoordAtt)** — injects positional information into channel attention.
   +2.4% Precision gain, negligible parameter cost. Placed per-scale at P3 and P4 neck for
   direct spatial enhancement.
2. **P3+P4 dual-head** — removed the P5 head; reduced parameters by 32% while improving mAP by
   +2.1% on 8 GB hardware. Chosen over triple-head as the production architecture.
3. **WIoU v3 loss** — weighted IoU for better small-object gradient. +1.78% mAP@0.5, zero VRAM cost.
4. **TAL 4 px threshold** — TaskAlignedAssigner positive-sample threshold reduced from 8 px to 4 px,
   directly boosting person/cycle by +6–7% each.
5. **Class weighting** — `[2.0, 3.0, 1.8, 1.5, 1.0, 1.0, 1.0]` for the 7 imbalanced classes.

## Dataset

| Dataset | Images | Classes | Source |
|---------|--------|---------|--------|
| aerial_v9 (custom) | 8,075 train / 2,224 val / 738 test | 7 | Curated from public aerial datasets, cleaned of label noise |
| aerial_v8 (early) | 3,800 train | 7 | Public aerial datasets, superseded by aerial_v9 (V8.0–V9.0 baseline) |
| aerial_merged (early) | 7,148 train | 5 | aerial.v1i (CC BY 4.0) + VisDrone2019 (V1–V7 baseline) |
| aerial (early) | 2,090 train | 6 | Roboflow aerial.v1i (CC BY 4.0), initial experiments |

> **Dataset availability**: aerial_v9 is curated from public aerial datasets (aerial.v1i CC BY 4.0,
> Aerial Vehicle Detection MIT, aerial.v3i MIT). Due to license terms and total size (~10 GB),
> the full dataset is **not open-sourced** in this repository. Please contact the owner for access,
> or prepare a compatible dataset using the YOLO format with the same class schema.

**Class distribution (aerial_v9 train instances):**
`person: 73,770 | car: 69,840 | cycle: 14,356 | truck: 13,772 | bus: 6,994 | freight: 1,306 | small-bus: 1,117`

## Ablation Study

Each version isolates exactly one variable. All training on RTX 5060 8GB, batch=4, SGD lr0=0.01.

| Version | Key Change | mAP@0.5 | mAP@0.5:0.95 | Δ vs Previous | Status |
|:-------:|-----------|:-------:|:------------:|:------------:|:------:|
| V1.0–V6.0 | Initial explorations (baseline YOLO26s, data curation, basic hyperparams) | — | — | — | Archive |
| V7.0 | 5-class baseline on aerial_merged (noisy VisDrone labels) | 55.25% | 32.72% | — | ❌ Recall 50% |
| V8.0 | Clean aerial_v8 dataset (3,800 images, 7 classes) | 67.81% | 44.31% | **+12.56%** | Baseline |
| V8.1 | + WIoU v3 loss (relay from V8.0 best.pt) | 69.59% | 47.00% | +1.78% | ✅ |
| V9.0 | + CoordAtt (3-head, train from scratch) | 68.66% | 47.44% | — | Archived |
| V10.0 | + CoordAtt + WIoU v3 + aerial_v9 (8,075 images) | 70.51% | 49.33% | +0.92% | Milestone |
| V11.0 | VisDrone2019 benchmark (10-class, comparison only) | 32.65% | 18.33% | — | Comparison |
| V12.0 | imgsz 640→800 | 71.58% | 50.27% | +1.07% | ✅ |
| V13.0 | P2 high-res head | 66.68% | 46.55% | −4.90% | ❌ OOM |
| V14.0 | P3+P4 dual-head, imgsz=960 | 73.71% | **52.65%** | +2.13% | Peak mAP50-95 |
| V15.0 | BiFPN + compressed P5 128ch | 72.85% | 50.18% | −0.86% | Neutral |
| **V16.0** | **P3+P4 dual-head + per-scale CoordAtt, imgsz=800** | **74.00%** | 52.11% | **+0.29%** | **🏆 Best** |
| V17.0 | + P4 RepNCSPELAN4 | 73.52% | 51.27% | −0.48% | Neutral |
| V18.0 | + ECA / ASFF attention experiments | 72.91% | 50.84% | −1.09% | Neutral |
| V19.0 | P4 wide channel 256→384 | 73.44% | 51.63% | −0.56% | Neutral |
| V20.0 | P3+P4+P5 triple-head (clean, no side modules) | 73.29% | 50.41% | −0.71% | Neutral |

> Full per-version records with preprocessing details and per-class breakdowns are in [`docs/ablation_table.md`](docs/ablation_table.md).

**Failed experiments (honest record):**

| Version | Attempt | Result | Lesson |
|---------|---------|--------|--------|
| V4.0 | Freeze+unfreeze 2-stage | 46.82% | COCO pretrain ≠ aerial features; train from scratch |
| V7.0 | VisDrone noisy data | 55.25% (Recall 50%) | Data quality > architecture |
| V8.2 | Transfer-learning relay (V8.1→aerial_v9) | 69.71% (all classes ↓) | lr mismatch destroys learned features |
| V13.0 | P2 head | 66.68% | 8 GB OOM, not viable |

## Hardware Constraints & Solutions

- **GPU**: RTX 5060 Laptop 8 GB — forces batch=4, no P2 head.
- **Solution**: dual-head (P3+P4) architecture reduces feature-map memory while preserving
  small-object detection capability.
- **Trade-off documented**: triple-head V20 improves car detection (+0.6%) but adds ~1 M params and
  VRAM pressure; dual-head V16 is chosen as the production model for the 8 GB constraint.
- **Open problem**: car detection (67.3% on V16) lags small-object classes; current work explores
  SimOTA center-prior and SGLoss-style adaptive grid selection to recover the P2 head's benefit
  without the OOM.

## Reproducibility

```bash
# Environment
conda create -n yolo_new python=3.11
conda activate yolo_new
pip install -r requirements.txt          # includes ultralytics==8.4.12 (customized fork with CoordAtt)
cd <repo-root>

# Train best model (V16, dual-head + CoordAtt)
python scripts/train_v16.py              # ~16.6h on RTX 5060, batch=4, imgsz=800

# Evaluate
python scripts/eval.py --weights V16.0/best.pt --data data/data.yaml
```

> Note: CoordAtt and the TAL 4 px / class-weight patches live in the customized Ultralytics fork;
> see `docs/` for the exact source locations. Releasing the patched fork is planned.

## Key Findings

1. **Data > Architecture**: switching from noisy VisDrone to curated aerial_v9 yielded +12.56%
   mAP@0.5 — more than any architectural change combined.
2. **Attention is cheap but effective**: CoordAtt adds negligible params but +2.4% Precision.
3. **Dual-head beats triple-head on 8 GB**: removing P5 saves 32% params and ~40% VRAM while
   improving mAP by +2.1% — counterintuitive but reproducible under the memory constraint.
4. **Assignment threshold matters for small objects**: TAL 4 px (vs default 8 px) directly boosted
   person/cycle by +6–7%.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{aerial_yolo26s_2026,
  title        = {Resource-Constrained Aerial Small Object Detection with YOLO26s + CoordAtt},
  author       = {Zou, Haoyi},
  year         = {2026},
  url          = {https://github.com/ZHY9981/Drone_yolo_7types}
}
```

## License

MIT License — see [LICENSE](LICENSE).
