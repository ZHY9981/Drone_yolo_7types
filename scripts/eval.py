"""
YOLO26s Evaluation Script
==========================
Evaluate a trained model on the validation set with full metrics.

Usage:
    python scripts/eval.py --weights path/to/best.pt --data data/data.yaml

Output:
    - Per-class precision, recall, mAP50, mAP50-95
    - Confusion matrix (saved to runs/val/)
    - Detection visualizations (saved to runs/val/)
"""
import argparse
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26s evaluation")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained weights (best.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/data.yaml",
        help="Path to dataset config",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=800,
        help="Inference image size",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device (0, cpu, etc.)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=True,
        help="Use FP16 inference",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="eval",
        help="Experiment name for output directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print(f"Evaluating: {args.weights}")
    print(f"Dataset: {args.data}")
    print("=" * 60)

    model = YOLO(args.weights)

    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        half=args.half,
        plots=True,
        name=args.name,
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"mAP50:    {results.box.map50:.2%}")
    print(f"mAP50-95: {results.box.map:.2%}")
    print(f"Precision:{results.box.mp:.2%}")
    print(f"Recall:   {results.box.mr:.2%}")

    if hasattr(results, "ap_class_index") and hasattr(results, "class_map"):
        print("\nPer-class metrics:")
        for i, cls_id in enumerate(results.ap_class_index):
            cls_name = results.names.get(cls_id, f"class-{cls_id}")
            print(f"  {cls_name:15s}  AP50={results.box.ap50[i]:.2%}  AP={results.box.ap[i]:.2%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
