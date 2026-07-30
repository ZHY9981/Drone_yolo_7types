# Ablation Study — YOLO26s Aerial Small-Object Detection

> Complete version log of 20+ ablation experiments across architectural modifications,
> data curation, and hyperparameter tuning. All experiments on RTX 5060 Laptop 8GB.
> This is the **authoritative data source** — the README contains a simplified summary.

## Key Results (mAP@0.5 on aerial_v9)

| Version | Key Change | mAP@0.5 | mAP@0.5:0.95 | Params | Status |
|:-------:|-----------|:-------:|:------------:|:------:|:------:|
| V7.0 | 5-class baseline on aerial_merged (noisy VisDrone) | 55.25% | 32.72% | — | ❌ Recall 50% |
| V8.0 | Clean aerial_v8 dataset (7 classes) | 67.81% | 44.31% | — | Baseline |
| V8.1 | + WIoU v3 loss | 69.59% | 47.00% | — | +1.78% |
| V9.0 | + CoordAtt (3-head) | 68.66% | 47.44% | — | Archived |
| V10.0 | + CoordAtt + WIoU + aerial_v9 (8,075 img) | 70.51% | 49.33% | — | Milestone |
| V11.0 | VisDrone2019 benchmark (10-class) | 32.65% | 18.33% | — | Comparison |
| V12.0 | imgsz 640→800 | 71.58% | 50.27% | — | +1.07% |
| V13.0 | P2 high-res head | 66.68% | 46.55% | — | ❌ OOM |
| V14.0 | P3+P4 dual-head, imgsz=960 | 73.71% | **52.65%** | 6.82M | Peak mAP50-95 |
| V15.0 | BiFPN + compressed P5 128ch | 72.85% | 50.18% | — | Neutral |
| **V16.0** | **P3+P4 + per-scale CoordAtt, imgsz=800** | **74.00%** | 52.11% | **7.03M** | **Best** |
| V17.0 | + P4 RepNCSPELAN4 | 73.52% | 51.27% | — | Neutral |
| V18.0 | + ECA / ASFF experiments | 72.91% | 50.84% | — | Neutral |
| V19.0 | P4 wide channel 256→384 | 73.44% | 51.63% | — | Neutral |
| V20.0 | P3+P4+P5 triple-head (clean) | 73.29% | 50.41% | 8.06M | Neutral |

> Full per-version records with per-class breakdowns and preprocessing details are in the
> project's version archive (local). V1–V6 early explorations omitted.

## Per-Class Breakdown (V16.0 — Best Model)

Evaluated on aerial_v9 validation set (2,224 images), RTX 5060 8GB, imgsz=800, FP16.

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

## Failed Experiments (Honest Record)

| Version | Attempt | Result | Lesson |
|:-------:|---------|:------:|--------|
| V4.0 | Freeze+unfreeze 2-stage training | 46.82% | COCO pretrain ≠ aerial features; train from scratch |
| V7.0 | VisDrone noisy data | 55.25% (Recall 50%) | Data quality > architecture |
| V8.2 | Transfer-learning relay (V8.1 → aerial_v9) | 69.71% (all classes ↓) | lr mismatch destroys learned features |
| V13.0 | P2 high-res head | 66.68% | 8GB OOM, not viable on consumer GPUs |

## Training Configuration (All Versions)

| Parameter | Value | Notes |
|:---|:---|:---|
| GPU | RTX 5060 Laptop 8GB | Fixed hardware constraint |
| Batch size | 4 (P2: 2) | VRAM limit |
| Optimizer | SGD, lr0=0.01, momentum=0.937 | Stable for aerial features |
| Epochs | 200 (V8–V16), 120 (V17–V20) | V17+ use close_mosaic=10 |
| Image size | 640–960 | 800 is sweet spot |
| Augmentation | mosaic=0.5, scale=0.3, no mixup/copy-paste | Aerial-specific: small targets need careful augmentation |
| Loss | WIoU v3 (box), VFL (cls), DFL | WIoU v3 from V8.1+ |
| Workers | 4 | Data prefetch |

## Key Insights

1. **Data > Architecture**: Switching from noisy VisDrone to curated aerial_v9 yielded +12.56%
   mAP@0.5 — more than any architectural change combined.
2. **Dual-head beats triple-head on 8GB**: Removing P5 saves 32% params and ~40% VRAM,
   improving mAP by +2.1%. Counterintuitive but reproducible under memory constraint.
3. **WIoU v3 is the highest-ROI modification**: +1.78% with zero VRAM cost.
4. **CoordAtt is cheap but effective**: Negligible parameter cost, +2.4% Precision.
5. **TAL 4px threshold directly boosts small objects**: Person/cycle improve +6–7% each.
6. **Mixup/copy-paste harmful for aerial small objects**: Confirmed across multiple versions.
7. **Imgsz=800 is the sweet spot**: 640→800 gives +1.07%; 960 causes VRAM pressure with
   diminishing returns on batch=4.

## Open Problems

- **Car detection (67.3%)** lags small-object classes on V16. Current work explores
  SimOTA center-prior and SGLoss-style adaptive grid selection to recover P2 head
  benefits without OOM.
- **Freight / small-bus** (< 2% of instances each) remain challenging due to extreme
  class imbalance; potential directions include few-shot augmentation or soft-labeling.
