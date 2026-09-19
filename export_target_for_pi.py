import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "plant_best_target.pt"
    deploy_dir = project_dir / "pi_deploy"
    deploy_dir.mkdir(exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    exported_path = Path(model.export(format="onnx", imgsz=224, simplify=True, opset=12))

    onnx_target = deploy_dir / "plant_best.onnx"
    shutil.copy2(exported_path, onnx_target)

    labels_target = deploy_dir / "labels.json"
    labels_target.write_text(json.dumps(model.names, indent=2), encoding="utf-8")

    print(f"Target ONNX model saved to: {onnx_target}")
    print(f"Target labels saved to: {labels_target}")


if __name__ == "__main__":
    main()
