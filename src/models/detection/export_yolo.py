#!/usr/bin/env python3
"""Export a YOLO model to ONNX or TensorRT for production serving."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export Ultralytics YOLO weights to ONNX/TensorRT.")
    parser.add_argument("--weights", required=True, help="Path to .pt weights to export.")
    parser.add_argument(
        "--format",
        choices=("onnx", "engine"),
        default="onnx",
        help="Export target. Use 'engine' for TensorRT.",
    )
    parser.add_argument("--imgsz", type=int, default=960, help="Export image size.")
    parser.add_argument("--device", default=None, help="Export device, for example '0' or 'cpu'.")
    parser.add_argument("--half", action="store_true", help="Enable FP16 export when supported.")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic shapes for ONNX/TensorRT.")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph when supported.")
    parser.add_argument("--workspace", type=float, default=None, help="TensorRT workspace size in GiB.")
    parser.add_argument("--output-dir", default=None, help="Optional directory for the exported artifact.")
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path = Path(args.weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    export_kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "half": args.half,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
    }
    if args.device is not None:
        export_kwargs["device"] = args.device
    if args.workspace is not None:
        export_kwargs["workspace"] = args.workspace

    model = YOLO(str(weights_path))
    exported = Path(model.export(**export_kwargs)).resolve()

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / exported.name
        if exported != target:
            exported.replace(target)
            exported = target

    print(f"Exported {args.format} artifact: {exported}")


if __name__ == "__main__":
    main()
