import argparse
import os
import shutil
from pathlib import Path
from ultralytics import YOLO


def export_to_onnx(pt_path: str = "data/models/yolov8n.pt", output_path: str = "data/models/yolov8n.onnx", imgsz: int = 640):
    """Exports Ultralytics PyTorch model weights to ONNX format."""
    pt_file = Path(pt_path)
    if not pt_file.exists():
        raise FileNotFoundError(f"Model file not found: {pt_path}")

    print(f"Loading model from {pt_path}...")
    model = YOLO(str(pt_file))

    print(f"Exporting to ONNX (imgsz={imgsz}, opset=12, dynamic=False)...")
    exported_file = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)
    
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    if Path(exported_file) != target_path:
        shutil.copy(exported_file, str(target_path))
        print(f"Copied exported model to {target_path}")

    print(f"Successfully exported ONNX model to: {output_path}")
    return str(target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch YOLO model to ONNX for edge deployment.")
    parser.add_argument("--model", default="data/models/yolov8n.pt", help="Path to input .pt weights")
    parser.add_argument("--output", default="data/models/yolov8n.onnx", help="Path to output .onnx model")
    parser.add_argument("--imgsz", type=int, default=640, help="Input inference size")
    args = parser.parse_args()

    export_to_onnx(args.model, args.output, args.imgsz)
