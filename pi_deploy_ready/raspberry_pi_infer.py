import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    import depthai as dai
except ImportError:
    dai = None

try:
    import serial
except ImportError:
    serial = None


def parse_args():
    parser = argparse.ArgumentParser(description="Lightweight Raspberry Pi ONNX inference for CropBot.")
    parser.add_argument("--camera-type", default="oakd", choices=("oakd", "webcam"), help="Camera backend to use.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index when --camera-type webcam is used.")
    parser.add_argument("--onnx-model", default="plant_best.onnx", help="ONNX model path.")
    parser.add_argument("--labels", default="labels.json", help="JSON class labels path.")
    parser.add_argument("--image", help="Run inference on a single still image instead of the OAK-D camera.")
    parser.add_argument("--conf", type=float, default=0.98, help="Minimum confidence for plant detection.")
    parser.add_argument("--plant-margin", type=float, default=0.60, help="Required score gap between plant and the best non-plant class.")
    parser.add_argument("--min-plant-frames", type=int, default=6, help="Number of consecutive inference checks required before confirming detection.")
    parser.add_argument("--enable-spray", action="store_true", help="Send spray commands to Arduino after confirmed detection.")
    parser.add_argument("--arduino-port", default="/dev/ttyACM0", help="Arduino serial port path.")
    parser.add_argument("--arduino-baud", type=int, default=9600, help="Arduino serial baud rate.")
    parser.add_argument("--spray-cooldown", type=float, default=5.0, help="Minimum seconds between spray commands.")
    parser.add_argument("--frame-skip", type=int, default=3, help="Run inference every N frames.")
    parser.add_argument("--width", type=int, default=640, help="Requested OAK-D preview width.")
    parser.add_argument("--height", type=int, default=480, help="Requested OAK-D preview height.")
    parser.add_argument("--display", action="store_true", help="Show the live preview window.")
    parser.add_argument("--debug-scores", action="store_true", help="Print and display raw class scores for tuning.")
    parser.add_argument("--center-crop", action="store_true", help="Crop the center square before classification.")
    parser.add_argument(
        "--center-crop-ratio",
        type=float,
        default=1.0,
        help="Center crop size ratio. 1.0 keeps the full center square, 0.6 keeps a tighter center region.",
    )
    parser.add_argument("--reject-faces", action="store_true", help="Suppress plant detection when a human face is visible.")
    parser.add_argument("--save-debug-frame", action="store_true", help="Save the exact image used for inference as debug_inference_input.jpg.")
    return parser.parse_args()


class OakDCamera:
    def __init__(self, width=640, height=480):
        if dai is None:
            print("DepthAI is not installed. Install it with: pip install depthai")
            self.device = None
            self.queue = None
            self.opened = False
            return

        pipeline = dai.Pipeline()
        cam = self._create_color_camera(pipeline)
        cam.setPreviewSize(width, height)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout = self._create_xlink_out(pipeline)
        xout.setStreamName("video")
        cam.preview.link(xout.input)

        self.device = None
        self.queue = None
        self.opened = False

        try:
            self.device = dai.Device(pipeline)
            self.queue = self.device.getOutputQueue(name="video", maxSize=4, blocking=False)
            self.opened = True
        except Exception as exc:
            print("OAK-D camera not found. Check USB cable, external power, lsusb, and DepthAI permissions.")
            print(f"DepthAI details: {exc}")
            self.release()

    def isOpened(self):
        return self.opened and self.queue is not None

    def read(self):
        if not self.isOpened():
            return False, None
        packet = self.queue.get()
        if packet is None:
            return False, None
        return True, packet.getCvFrame()

    def release(self):
        self.queue = None
        if self.device is not None:
            self.device.close()
            self.device = None
        self.opened = False

    @staticmethod
    def _create_color_camera(pipeline):
        if hasattr(pipeline, "createColorCamera"):
            return pipeline.createColorCamera()

        node_module = getattr(dai, "node", None)
        if node_module is not None and hasattr(node_module, "ColorCamera"):
            return pipeline.create(node_module.ColorCamera)

        if hasattr(dai, "ColorCamera"):
            return pipeline.create(dai.ColorCamera)

        raise AttributeError("DepthAI ColorCamera API is not available in this installed version.")

    @staticmethod
    def _create_xlink_out(pipeline):
        if hasattr(pipeline, "createXLinkOut"):
            return pipeline.createXLinkOut()

        node_module = getattr(dai, "node", None)
        if node_module is not None and hasattr(node_module, "XLinkOut"):
            return pipeline.create(node_module.XLinkOut)

        if hasattr(dai, "XLinkOut"):
            return pipeline.create(dai.XLinkOut)

        raise AttributeError("DepthAI XLinkOut API is not available in this installed version.")


def open_camera(camera_type, camera_index, width, height):
    if camera_type == "oakd":
        return OakDCamera(width=width, height=height)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def load_labels(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(key): str(value) for key, value in raw.items()}


def connect_arduino(port, baud):
    if serial is None:
        print("Warning: pyserial is not installed. Spray control is disabled. Install it with: pip install pyserial")
        return None

    try:
        arduino = serial.Serial(port, baud, timeout=1)
        time.sleep(2.0)
        print(f"Arduino connected on {port}")
        return arduino
    except Exception as exc:
        print(f"Warning: could not connect to Arduino on {port}: {exc}")
        print("Continuing without pump control.")
        return None


def maybe_send_spray(arduino, enable_spray, cooldown, last_spray_time):
    if not enable_spray:
        return last_spray_time

    now = time.perf_counter()
    if now - last_spray_time < cooldown:
        return last_spray_time

    if arduino is None:
        print("Plant detected, but Arduino is not connected. Spray command was not sent.")
        return last_spray_time

    try:
        arduino.write(b"S")
        arduino.flush()
        print("Spray command sent to Arduino")
        return now
    except Exception as exc:
        print(f"Warning: failed to send spray command to Arduino: {exc}")
        return last_spray_time


def load_face_detector():
    if not hasattr(cv2, "CascadeClassifier"):
        print("Warning: this OpenCV build does not include CascadeClassifier. Face rejection is disabled.")
        return None

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        print(f"Warning: failed to load face detector from {cascade_path}")
        return None
    return detector


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(logits)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def crop_for_inference(frame, center_crop=False, crop_ratio=1.0):
    if not center_crop:
        return frame

    frame_height, frame_width = frame.shape[:2]
    side = min(frame_height, frame_width)
    side = max(32, int(side * min(max(crop_ratio, 0.1), 1.0)))
    y1 = (frame_height - side) // 2
    x1 = (frame_width - side) // 2
    return frame[y1 : y1 + side, x1 : x1 + side]


def prepare_input(frame, input_height, input_width, center_crop=False, crop_ratio=1.0):
    # For the demo, aim the OAK-D camera toward the plant/rose.
    cropped = crop_for_inference(frame, center_crop=center_crop, crop_ratio=crop_ratio)
    resized = cv2.resize(cropped, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return tensor, cropped, resized


def ensure_probabilities(output_array):
    row_sums = np.sum(output_array, axis=1, keepdims=True)
    looks_like_probabilities = (
        np.all(output_array >= 0.0)
        and np.all(output_array <= 1.0)
        and np.allclose(row_sums, 1.0, atol=1e-3)
    )
    if looks_like_probabilities:
        return output_array, False
    return softmax(output_array), True


def estimate_plant_boxes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([24, 30, 25]), np.array([100, 255, 255]))

    b_channel, g_channel, r_channel = cv2.split(frame.astype(np.float32))
    exg = 2.0 * g_channel - r_channel - b_channel
    exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, exg_mask = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = cv2.bitwise_and(green_mask, exg_mask)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(250, frame.shape[0] * frame.shape[1] * 0.002)

    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            continue
        boxes.append((x, y, w, h, area))

    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes[:3]


def expand_box(x, y, w, h, frame_width, frame_height, scale=0.35):
    pad_x = int(w * scale)
    pad_y = int(h * scale)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_width, x + w + pad_x)
    y2 = min(frame_height, y + h + pad_y)
    return x1, y1, x2, y2


def infer(session, input_name, frame, input_height, input_width, labels, center_crop=False, crop_ratio=1.0):
    tensor, cropped, resized = prepare_input(
        frame,
        input_height,
        input_width,
        center_crop=center_crop,
        crop_ratio=crop_ratio,
    )
    outputs = session.run(None, {input_name: tensor})
    scores = outputs[0]
    if scores.ndim > 2:
        scores = scores.reshape(scores.shape[0], -1)
    probabilities, softmax_applied = ensure_probabilities(scores)
    class_idx = int(np.argmax(probabilities[0]))
    class_name = labels.get(class_idx, str(class_idx))
    class_scores = {labels.get(idx, str(idx)): float(probabilities[0][idx]) for idx in range(probabilities.shape[1])}
    return class_name, class_scores, scores, softmax_applied, cropped, resized


def resolve_direction(frame_width, box):
    if box is None:
        return None

    x1, _, x2, _ = box
    center_x = (x1 + x2) / 2.0
    left_boundary = frame_width / 3.0
    right_boundary = 2.0 * frame_width / 3.0

    if center_x < left_boundary:
        return "Left"
    if center_x > right_boundary:
        return "Right"
    return "Center"


def draw_status(frame, confirmed_plant_detected, direction=None):
    text = f"Plant {direction}" if confirmed_plant_detected and direction else "Plant" if confirmed_plant_detected else "No plant"
    color = (0, 220, 0) if confirmed_plant_detected else (0, 180, 255)
    cv2.putText(frame, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def resolve_plant_score(labels, class_scores):
    plant_label = next((name for name in labels.values() if name.lower() == "plant"), None)
    if plant_label is None:
        plant_label = labels[min(labels)]
    plant_score = float(class_scores.get(plant_label, 0.0))
    best_non_plant_score = max(
        (float(score) for name, score in class_scores.items() if name.lower() != plant_label.lower()),
        default=0.0,
    )
    return plant_label, plant_score, best_non_plant_score


def maybe_save_debug_frame(resized_frame, enabled, output_dir):
    if enabled:
        output_path = output_dir / "debug_inference_input.jpg"
        saved = cv2.imwrite(str(output_path), resized_frame)
        if saved:
            print(f"Saved debug inference frame to: {output_path}")
        else:
            print(f"Warning: failed to save debug inference frame to: {output_path}")


def print_debug_scores(class_scores, predicted_class, raw_detected, confirmed_detected, streak, raw_output, softmax_applied):
    plant_score = class_scores.get("plant", 0.0)
    top_non_plant_name = max(
        (name for name in class_scores if name.lower() != "plant"),
        key=lambda name: class_scores[name],
        default="non_plant",
    )
    non_plant_score = class_scores.get(top_non_plant_name, 0.0)
    print(
        f"raw_output={np.array2string(raw_output, precision=4)} "
        f"softmax_applied={softmax_applied} "
        f"plant={plant_score:.4f} {top_non_plant_name}={non_plant_score:.4f} "
        f"predicted={predicted_class} raw={raw_detected} "
        f"confirmed={confirmed_detected} streak={streak}"
    )


def frame_has_face(frame, face_detector):
    if face_detector is None:
        return False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    return len(faces) > 0


def infer_best_candidate(
    session,
    input_name,
    frame,
    input_height,
    input_width,
    labels,
    center_crop=False,
    crop_ratio=1.0,
):
    frame_height, frame_width = frame.shape[:2]
    candidates = estimate_plant_boxes(frame)
    if not candidates:
        return None

    best_result = None
    best_plant_score = -1.0

    for x, y, w, h, area in candidates:
        x1, y1, x2, y2 = expand_box(x, y, w, h, frame_width, frame_height)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        predicted_class, class_scores, raw_output, softmax_applied, cropped, resized = infer(
            session,
            input_name,
            crop,
            input_height,
            input_width,
            labels,
            center_crop=center_crop,
            crop_ratio=crop_ratio,
        )
        _, plant_score, non_plant_score = resolve_plant_score(labels, class_scores)
        if plant_score > best_plant_score:
            best_plant_score = plant_score
            best_result = {
                "predicted_class": predicted_class,
                "class_scores": class_scores,
                "raw_output": raw_output,
                "softmax_applied": softmax_applied,
                "cropped": cropped,
                "resized": resized,
                "plant_score": plant_score,
                "non_plant_score": non_plant_score,
                "box": (x1, y1, x2, y2),
                "area": area,
            }

    return best_result


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    model_path = Path(args.onnx_model)
    labels_path = Path(args.labels)
    if not model_path.is_absolute():
        model_path = script_dir / model_path
    if not labels_path.is_absolute():
        labels_path = script_dir / labels_path
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = load_labels(labels_path)
    plant_label = next((name for name in labels.values() if name.lower() == "plant"), labels[min(labels)])
    face_detector = load_face_detector() if args.reject_faces else None

    arduino = connect_arduino(args.arduino_port, args.arduino_baud) if args.enable_spray else None
    last_spray_time = 0.0

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    input_height = int(input_shape[2]) if isinstance(input_shape[2], int) else 224
    input_width = int(input_shape[3]) if isinstance(input_shape[3], int) else 224
    output_meta = session.get_outputs()[0]
    print(f"ONNX input shape: {input_shape}")
    print(f"ONNX output shape: {output_meta.shape}")

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")

        candidate_result = infer_best_candidate(
            session,
            input_name,
            image,
            input_height,
            input_width,
            labels,
            center_crop=args.center_crop,
            crop_ratio=args.center_crop_ratio,
        )
        human_visible = frame_has_face(image, face_detector)
        if candidate_result is None:
            predicted_class = "no_plant"
            class_scores = {"plant": 0.0, "no_plant": 1.0}
            raw_output = np.array([[0.0, 1.0]], dtype=np.float32)
            softmax_applied = False
            plant_score = 0.0
            non_plant_score = 1.0
            resized = cv2.resize(image, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
            raw_plant_detected = False
        else:
            predicted_class = candidate_result["predicted_class"]
            class_scores = candidate_result["class_scores"]
            raw_output = candidate_result["raw_output"]
            softmax_applied = candidate_result["softmax_applied"]
            plant_score = candidate_result["plant_score"]
            non_plant_score = candidate_result["non_plant_score"]
            resized = candidate_result["resized"]
            raw_plant_detected = (
                plant_score >= args.conf
                and plant_score - non_plant_score >= args.plant_margin
                and not human_visible
            )
        confirmed_plant_detected = raw_plant_detected and max(args.min_plant_frames, 1) <= 1
        maybe_save_debug_frame(resized, args.save_debug_frame, script_dir)
        print(f"class_scores={class_scores}")
        print(f"plant_score={plant_score:.4f}")
        non_plant_name = max((name for name in class_scores if name.lower() != "plant"), key=lambda name: class_scores[name], default="non_plant")
        print(f"{non_plant_name}_score={class_scores.get(non_plant_name, 0.0):.4f}")
        if args.reject_faces:
            print(f"human_visible={human_visible}")
        print(f"predicted_class={predicted_class}")
        print(f"softmax_applied={softmax_applied}")
        print(f"final_decision={'Plant detected' if confirmed_plant_detected else 'No plant detected'}")
        if args.debug_scores:
            print_debug_scores(
                class_scores,
                predicted_class,
                raw_plant_detected,
                confirmed_plant_detected,
                1 if raw_plant_detected else 0,
                raw_output,
                softmax_applied,
            )
        return

    cap = open_camera(args.camera_type, args.camera_index, args.width, args.height)
    if not cap.isOpened():
        if args.camera_type == "oakd":
            raise RuntimeError("OAK-D camera not found. Check USB cable, external power, lsusb, and DepthAI permissions.")
        raise RuntimeError(f"Webcam not found on index {args.camera_index}.")

    frame_counter = 0
    last_predicted_class = "unknown"
    last_plant_score = 0.0
    last_non_plant_score = 0.0
    last_raw_plant_detected = False
    last_direction = None
    confirmed_plant_detected = False
    consecutive_plant_frames = 0
    class_scores = {"plant": 0.0, "no_plant": 1.0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        human_visible = frame_has_face(frame, face_detector)

        if frame_counter % max(args.frame_skip, 1) == 0:
            candidate_result = infer_best_candidate(
                session,
                input_name,
                frame,
                input_height,
                input_width,
                labels,
                center_crop=args.center_crop,
                crop_ratio=args.center_crop_ratio,
            )

            if candidate_result is None:
                class_scores = {"plant": 0.0, "no_plant": 1.0}
                raw_output = np.array([[0.0, 1.0]], dtype=np.float32)
                softmax_applied = False
                resized = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
                last_predicted_class = "no_plant"
                last_plant_score = 0.0
                last_non_plant_score = 1.0
                last_raw_plant_detected = False
                last_direction = None
            else:
                last_predicted_class = candidate_result["predicted_class"]
                class_scores = candidate_result["class_scores"]
                raw_output = candidate_result["raw_output"]
                softmax_applied = candidate_result["softmax_applied"]
                resized = candidate_result["resized"]
                last_plant_score = candidate_result["plant_score"]
                last_non_plant_score = candidate_result["non_plant_score"]
                last_direction = resolve_direction(frame.shape[1], candidate_result["box"])
                last_raw_plant_detected = (
                    last_plant_score >= args.conf
                    and last_plant_score - last_non_plant_score >= args.plant_margin
                    and not human_visible
                )

            maybe_save_debug_frame(resized, args.save_debug_frame, script_dir)

            if last_raw_plant_detected:
                consecutive_plant_frames += 1
            else:
                consecutive_plant_frames = 0

            confirmed_plant_detected = consecutive_plant_frames >= max(args.min_plant_frames, 1)

            if args.debug_scores:
                print_debug_scores(
                    class_scores,
                    last_predicted_class,
                    last_raw_plant_detected,
                    confirmed_plant_detected,
                    consecutive_plant_frames,
                    raw_output,
                    softmax_applied,
                )

            if confirmed_plant_detected:
                last_spray_time = maybe_send_spray(
                    arduino,
                    args.enable_spray,
                    args.spray_cooldown,
                    last_spray_time,
                )

        if args.display:
            draw_status(frame, confirmed_plant_detected, last_direction)
            cv2.imshow("CropBot Pi Deploy - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_counter += 1

    cap.release()
    if arduino is not None:
        arduino.close()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
