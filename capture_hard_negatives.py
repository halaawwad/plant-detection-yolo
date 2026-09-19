import argparse
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Capture webcam hard negatives for no_plant training.")
    parser.add_argument("--output-dir", default="hard_negatives", help="Folder to save negative frames.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index.")
    parser.add_argument("--count", type=int, default=24, help="How many frames to save.")
    parser.add_argument("--interval", type=float, default=0.4, help="Seconds between saved frames.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    return parser.parse_args()


def open_camera(camera_index, width, height):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_camera(args.camera_index, args.width, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {args.camera_index}")

    next_capture = time.perf_counter() + 2.0
    saved = 0
    window_name = "Hard negatives capture - stay in frame, press q to cancel"

    while saved < args.count:
        ok, frame = cap.read()
        if not ok:
            break

        remaining = max(args.count - saved, 0)
        countdown = max(next_capture - time.perf_counter(), 0.0)
        cv2.putText(frame, f"Capturing no_plant frames: {saved}/{args.count}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Next capture in {countdown:.1f}s", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Show face, shoulders, hands, clothes, empty room", (12, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if time.perf_counter() >= next_capture:
            output_path = output_dir / f"human_camera_{saved + 2:02d}.jpg"
            cv2.imwrite(str(output_path), frame)
            print(f"Saved {output_path}")
            saved += 1
            next_capture = time.perf_counter() + args.interval

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} hard negative frames to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
