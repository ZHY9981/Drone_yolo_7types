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
| Inference | TBD | RTX 5060 8 GB, imgsz=800 (benchmark pending) |

## The Problem

Aerial small-object detection is hard for four compounding reasons:

- **Tiny targets**: a pedestrian viewed from 100 m+ altitude occupies < 30 pixels — < 1% of the
  image. Standard detection heads routinely miss them.
- **OOM on consumer GPUs**: adding a high-resolution P2 head (25,600 cells) to capture small
  objects causes the TaskAlignedAssigner to allocate an alignment matrix that exceeds 8 GB VRAM
  at batch=4 — the constraint cannot be relaxed by buying hardware.
- **Label noise**: public aerial datasets (e.g. VisDrone2019) contain inconsistent annotations that
  silently degrade recall.
- **Class imbalance**: person/car dominate; freight and small-bus are rare (each < 2% of instances).

## Architecture Modifications

All modifications are applied on top of a customized Ultralytics 8.4.12 fork with a CoordAtt module.

1. **Coordinate Attention (CoordAtt)** — injects positional information into channel attention.
   +2.4% Precision gain, negligible parameter cost.
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
| VisDrone2019 (benchmark) | 6,471 train / 548 val | 10 | Official benchmark (comparison only) |

> **Dataset availability**: aerial_v9 is curated from public aerial datasets (aerial.v1i CC BY 4.0,
> Aerial Vehicle Detection MIT, aerial.v3i MIT). Due to license terms and total size (~10 GB),
> the full dataset is **not open-sourced** in this repository. Please contact the owner for access,
> or prepare a compatible dataset using the YOLO format with the same class schema.

**Class distribution (aerial_v9 train instances):**
`person: 73,770 | car: 69,840 | cycle: 14,356 | truck: 13,772 | bus: 6,994 | freight: 1,306 | small-bus: 1,117`

## Ablation Study

Key versions (mAP@0.5 on aerial_v9):

| Version | Key change | mAP@0.5 | mAP@0.5:0.95 | Params | Status |
|---------|-----------|---------|--------------|--------|--------|
| V8.0 | Baseline (clean data) | 67.81% | 44.31% | — | Baseline |
| V8.1 | + WIoU v3 | 69.59% | 47.00% | — | +1.78% |
| V10.0 | + CoordAtt + aerial_v9 | 70.51% | 49.33% | — | +2.4% Prec |
| V12.0 | + imgsz 640→800 | 71.58% | 50.27% | — | +1.07% |
| V14.0 | P3+P4 dual-head, imgsz=960 | 73.71% | **52.65%** | 6.82 M | peak mAP@0.5:0.95 |
| **V16.0** | **P3+P4 + CoordAtt, imgsz=800** | **74.00%** | 52.11% | **7.03 M** | **Best (production)** |
| V20.0 | P3+P4+P5 triple-head | 73.29% | 50.41% | 8.06 M | Neutral |

> Full per-version table with precision/recall per class is in [`docs/ablation_table.md`](docs/ablation_table.md).

**Failed experiments (honest record):**

| Version | Attempt | Result | Lesson |
|---------|---------|--------|--------|
| V4.0 | Freeze+unfreeze 2-stage | 46.82% | COCO pretrain ≠ aerial features; train from scratch |
| V7.0 | VisDrone noisy data | 55.25% (Recall 50%) | Data quality > architecture |
| V8.2 | Transfer-learning接力 | 69.71% (all classes↓) | lr mismatch destroys learned features |
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
  url          = {https://github.com/ZHY9981/aerial-yolo26s-drone-detection}
}
```

## License

MIT License — see [LICENSE](LICENSE).
