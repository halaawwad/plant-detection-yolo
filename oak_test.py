import cv2

try:
    import depthai as dai
except ImportError:
    dai = None


class OakDCamera:
    def __init__(self, width=640, height=480):
        if dai is None:
            print("DepthAI is not installed. Install it with: pip install depthai")
            self.device = None
            self.queue = None
            self.opened = False
            return

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


def main():
    cap = OakDCamera()
    if not cap.isOpened():
        return

    window_name = "OAK-D test - press q to quit"
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame from the OAK-D camera.")
            break

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
