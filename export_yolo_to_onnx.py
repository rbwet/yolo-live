#!/usr/bin/env python3
"""
Export any YOLOv8 model to ONNX format.

Usage:
    # Export pretrained YOLOv8n (nano — fastest):
    python export_yolo_to_onnx.py --weights yolov8n.pt

    # Export custom trained model from runs/detect/train/weights/best.pt:
    python export_yolo_to_onnx.py --weights runs/detect/train/weights/best.pt

    # Export with dynamic axes (variable input size):
    python export_yolo_to_onnx.py --weights yolov8n.pt --dynamic

    # Specify output path:
    python export_yolo_to_onnx.py --weights my_model.pt --img 1280 -o custom.onnx

Requirements:
    pip install ultralytics
"""

import argparse
from pathlib import Path


def export(
    weights: str,
    img_size: int = 640,
    dynamic: bool = False,
    output: str | None = None,
    simplify: bool = True,
):
    """Export a YOLOv8 PyTorch model to ONNX.

    Args:
        weights: Path to .pt file or model name (e.g. "yolov8n.pt").
        img_size: Export image size (default 640).
        dynamic: Enable dynamic axes for variable input sizes.
        output: Custom output filename. Defaults to <original_name>.onnx.
        simplify: Run ONNX simplifier after export.
    """
    from ultralytics import YOLO

    print(f"Loading model: {weights}")
    model = YOLO(weights)

    onnx_path = model.export(
        format="onnx",
        imgsz=img_size,
        dynamic=dynamic,
        simplify=simplify,
    )

    # If user specified a custom output path, rename
    if output:
        Path(onnx_path).rename(output)
        onnx_path = output
        print(f"Renamed to: {output}")

    print(f"\nExport complete: {onnx_path}")
    print(f"Use with: python yolo_live.py --model {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv8 model to ONNX")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights or model name (yolov8n.pt)")
    parser.add_argument("--img", type=int, default=640, help="Image size for export (default: 640)")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input axes")
    parser.add_argument("-o", "--output", type=str, default=None, help="Custom output filename")
    parser.add_argument("--no-simplify", action="store_true", help="Skip ONNX simplifier")
    args = parser.parse_args()

    export(
        weights=args.weights,
        img_size=args.img,
        dynamic=args.dynamic,
        output=args.output,
        simplify=not args.no_simplify,
    )


if __name__ == "__main__":
    main()
