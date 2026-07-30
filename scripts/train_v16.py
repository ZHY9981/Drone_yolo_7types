"""
YOLO26s V16 — Train Best Production Model (P3+P4 Dual-Head + CoordAtt)
======================================================================
Architecture: P3+P4 dual-head + per-scale CoordAtt, imgsz=800
Hardware:     RTX 5060 Laptop 8GB VRAM, batch=4
Training:     ~16.6 hours, 200 epochs

Usage:
    conda activate yolo_new
    python scripts/train_v16.py

Requires:
    - data.yaml pointing to your aerial_v9 dataset
    - Ultralytics fork with CoordAtt module (see patches/)
    - Model config: configs/yolo26s-v16-p34-coordatt.yaml
"""
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("configs/yolo26s-v16-p34-coordatt.yaml").load("yolo26s.pt")

    results = model.train(
        # Data
        data="data/data.yaml",
        device="0",
        epochs=200,
        batch=4,
        imgsz=800,
        workers=4,
        # Optimizer (SGD: stable convergence for aerial features)
        optimizer="SGD",
        lr0=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        lrf=0.01,
        patience=20,
        # Loss weights
        box=7.5,
        cls=0.7,
        dfl=1.5,
        # Augmentation (aerial-specific: no mixup/copy-paste)
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        degrees=45,
        translate=0.1,
        scale=0.3,  # reduced from 0.5 to protect small targets
        shear=0.0,
        perspective=0.0,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.0,
        # Training
        close_mosaic=10,
        amp=True,
        half=True,
        seed=0,
        deterministic=True,
        # Logging
        project="runs/aerial_train",
        name="yolo26s_v16",
        exist_ok=True,
        save_period=5,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("V16 (Best / Production) training complete!")
    print(f"mAP50:   {results.results_dict.get('mAP50', 'N/A')}")
    print(f"mAP50-95:{results.results_dict.get('mAP50-95', 'N/A')}")
    print("=" * 60)
