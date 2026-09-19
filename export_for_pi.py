import json
from pathlib import Path

from ultralytics import YOLO


def main():
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "plant_best.pt"
    deploy_dir = project_dir / "pi_deploy"
    deploy_dir.mkdir(exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    # The trained classifier expects 224x224 crops, which matches the current runtime code.
    exported_path = model.export(format="onnx", imgsz=224, simplify=True, opset=12)

    exported_path = Path(exported_path)
    onnx_target = deploy_dir / "plant_best.onnx"
    if exported_path.resolve() != onnx_target.resolve():
        onnx_target.write_bytes(exported_path.read_bytes())

    labels_target = deploy_dir / "labels.json"
    labels_target.write_text(json.dumps(model.names, indent=2), encoding="utf-8")

    print(f"ONNX model saved to: {onnx_target}")
    print(f"Labels saved to: {labels_target}")


if __name__ == "__main__":
    main()
