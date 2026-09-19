import argparse
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    import depthai as dai
except ImportError:
    dai = None

try:
    import serial
except ImportError:
    serial = None


def parse_args():
    parser = argparse.ArgumentParser(description="Live plant classification with an estimated plant box.")
    parser.add_argument("--model", default="plant_best.pt", help="Trained classifier path.")
    parser.add_argument("--detector", default="yolov8n.pt", help="General YOLO detector used to reject people.")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Legacy camera index argument. Ignored for OAK-D because DepthAI discovers the device automatically.",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index when --camera-type webcam is used.")
    parser.add_argument(
        "--camera-type",
        choices=("oakd", "webcam"),
        default="oakd",
        help="Camera backend to use. OAK-D uses DepthAI device discovery and does not need a port number or index.",
    )
    parser.add_argument("--conf", type=float, default=0.65, help="Minimum plant confidence to draw a box.")
    parser.add_argument("--imgsz", type=int, default=224, help="Classifier image size.")
    parser.add_argument("--detector-conf", type=float, default=0.35, help="Minimum confidence for the general detector.")
    parser.add_argument("--detector-imgsz", type=int, default=640, help="General detector image size.")
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--max-seconds", type=float, default=0, help="Stop after this many seconds. 0 runs until q.")
    parser.add_argument("--raw", action="store_true", help="Show raw camera only, without any model inference.")
    parser.add_argument("--debug", action="store_true", help="Show suppressed class/probability information.")
    parser.add_argument("--enable-spray", action="store_true", help="Send spray commands to Arduino after confirmed detection.")
    parser.add_argument("--arduino-port", default="/dev/ttyACM0", help="Arduino serial port path.")
    parser.add_argument("--arduino-baud", type=int, default=9600, help="Arduino serial baud rate.")
    parser.add_argument("--spray-cooldown", type=float, default=5.0, help="Minimum seconds between spray commands.")
    return parser.parse_args()


class OakDCamera:
    def __init__(self, width=640, height=480):
        if dai is None:
            print("DepthAI is not installed. Install it with: pip install depthai")
            self.width = width
            self.height = height
            self.pipeline = None
            self.device = None
            self.queue = None
            self.opened = False
            return

        self.width = width
        self.height = height
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.createColorCamera()
        cam.setPreviewSize(width, height)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

        xout = self.pipeline.createXLinkOut()
        xout.setStreamName("video")
        cam.preview.link(xout.input)

        self.device = None
        self.queue = None
        self.opened = False

        try:
            self.device = dai.Device(self.pipeline)
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
        frame_packet = self.queue.get()
        if frame_packet is None:
            return False, None
        frame = frame_packet.getCvFrame()
        return True, frame

    def release(self):
        self.queue = None
        if self.device is not None:
            self.device.close()
            self.device = None
        self.opened = False


def open_camera(camera_type, camera_index, width, height):
    if camera_type == "oakd":
        return OakDCamera(width=width, height=height)

    if platform.system() == "Windows":
        # DirectShow helps webcam enumeration on Windows, but is not valid on Raspberry Pi/Linux.
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


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


def maybe_send_spray(arduino, args, last_spray_time):
    if not args.enable_spray:
        return last_spray_time

    now = time.perf_counter()
    if now - last_spray_time < args.spray_cooldown:
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


def plant_probability(result):
    names = result.names
    plant_idx = next((idx for idx, name in names.items() if str(name).lower() == "plant"), None)
    if plant_idx is None:
        plant_idx = 0

    probs = result.probs.data.detach().cpu().numpy()
    return float(probs[int(plant_idx)])


def detections_by_name(result, wanted_names):
    matches = []
    wanted = {name.lower() for name in wanted_names}
    if result.boxes is None:
        return matches

    for box in result.boxes:
        cls_id = int(box.cls.item())
        name = str(result.names.get(cls_id, cls_id)).lower()
        if name in wanted:
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            matches.append((name, conf, x1, y1, x2, y2))

    return matches


def estimate_plant_boxes(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, np.array([25, 35, 35]), np.array([95, 255, 255]))

    b, g, r = cv2.split(frame.astype(np.float32))
    exg = 2 * g - r - b
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


def classify_crop(model, frame, x, y, w, h, imgsz):
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0
    result = model.predict(crop, imgsz=imgsz, verbose=False)[0]
    return plant_probability(result)


def draw_label(frame, text, x, y, color):
    cv2.putText(frame, text, (x, max(y - 10, 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def frame_is_black(frame):
    return float(frame.mean()) < 3.0 and int(frame.max()) < 40


def show_frame(window_name, frame, started_at, max_seconds):
    cv2.imshow(window_name, frame)
    should_quit = cv2.waitKey(1) & 0xFF == ord("q")
    timed_out = max_seconds > 0 and time.perf_counter() - started_at >= max_seconds
    return should_quit or timed_out


def main():
    args = parse_args()
    arduino = connect_arduino(args.arduino_port, args.arduino_baud) if args.enable_spray else None
    last_spray_time = 0.0
    model = None
    detector = None
    if not args.raw:
        model_path = Path(args.model)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Run train_plant_classifier.py first.")

        detector_path = Path(args.detector)
        if not detector_path.exists():
            raise FileNotFoundError(f"Detector not found: {detector_path}. Expected yolov8n.pt in this folder.")

        model = YOLO(str(model_path))
        detector = YOLO(str(detector_path))
    camera_index = args.camera_index if args.camera_type == "webcam" else args.camera
    cap = open_camera(args.camera_type, camera_index, args.width, args.height)
    if not cap.isOpened():
        if args.camera_type == "oakd":
            raise RuntimeError("OAK-D camera not found. Check USB cable, external power, lsusb, and DepthAI permissions.")
        raise RuntimeError(f"Could not open camera index {camera_index}. Try --camera-index 1 or check permissions.")

    started_at = time.perf_counter()
    prev_time = started_at
    window_name = "Plant camera - press q to quit"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        sprayed_this_frame = False

        if frame_is_black(frame):
            cv2.putText(
                frame,
                "Camera frame is black: check lens/privacy shutter/lighting/another app",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
            if show_frame(window_name, frame, started_at, args.max_seconds):
                break
            continue

        if args.raw:
            if show_frame(window_name, frame, started_at, args.max_seconds):
                break
            continue

        detector_result = detector.predict(
            frame,
            imgsz=args.detector_imgsz,
            conf=args.detector_conf,
            verbose=False,
        )[0]
        potted_plant_detections = detections_by_name(detector_result, {"potted plant"})

        if potted_plant_detections:
            color = (0, 220, 0)
            for _, conf, x1, y1, x2, y2 in potted_plant_detections:
                status = f"plant {conf * 100:.1f}%"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, status, x1, y1, color)

            last_spray_time = maybe_send_spray(arduino, args, last_spray_time)
            sprayed_this_frame = True

            if show_frame(window_name, frame, started_at, args.max_seconds):
                break
            continue

        now = time.perf_counter()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        color = (0, 220, 0)
        plant_found = False
        boxes = estimate_plant_boxes(frame)
        for x, y, w, h, _ in boxes:
            prob = classify_crop(model, frame, x, y, w, h, args.imgsz)
            if prob >= args.conf:
                plant_found = True
                status = f"plant {prob * 100:.1f}%"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                draw_label(frame, status, x, y, color)
                if not sprayed_this_frame:
                    last_spray_time = maybe_send_spray(arduino, args, last_spray_time)
                    sprayed_this_frame = True

        if args.debug and not plant_found:
            cv2.putText(
                frame,
                f"no plant | FPS {fps:.1f}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 180, 255),
                2,
                cv2.LINE_AA,
            )

        if show_frame(window_name, frame, started_at, args.max_seconds):
            break

    cap.release()
    if arduino is not None:
        arduino.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
