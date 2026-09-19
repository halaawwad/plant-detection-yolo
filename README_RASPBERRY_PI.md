# Raspberry Pi OAK-D Setup

This guide explains how to run the project on a Raspberry Pi using an OAK-D camera through DepthAI.

## A) Hardware Connection

- Connect the OAK-D USB data cable to a Raspberry Pi USB 3.0 port when available for the best bandwidth and stability.
- Connect external power to the OAK-D if your setup requires it.
- Power the Raspberry Pi with a proper power supply.
- If you later use serial control, connect the Arduino to the Raspberry Pi over USB.

Important notes:

- The OAK-D does not need a USB port number in the Python code.
- The OAK-D does not use a camera index like `0` or `1`.
- DepthAI automatically discovers the connected OAK-D device.
- USB 2.0 can still work in some setups, but USB 3.0 is recommended for better performance.
- Arduino spray control uses a serial port such as `/dev/ttyACM0` or `/dev/ttyUSB0`.

## B) Verify OAK-D Is Detected

Run:

```bash
lsusb
```

You should see something similar to:

```text
Intel Movidius MyriadX
```

## C) Activate the Python Environment

Run:

```bash
source oakenv/bin/activate
```

## D) Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

## E) Verify DepthAI Sees the OAK-D Camera

Run:

```bash
python3 -c 'import depthai as dai; print(dai.Device.getAllAvailableDevices())'
```

It should return a device list, not an empty list.

## F) Test the Camera Only

Run either:

```bash
python3 oak_test.py
```

or:

```bash
chmod +x test_oak_camera.sh
./test_oak_camera.sh
```

What to expect:

- A camera window should open.
- Press `q` to close it.

## G) Run the Full Model

Run either:

```bash
python3 live_plant_camera.py --camera-type oakd
```

Run with Arduino spray enabled:

```bash
python3 live_plant_camera.py --camera-type oakd --enable-spray --arduino-port /dev/ttyACM0 --spray-cooldown 5
```

or:

```bash
chmod +x run_oak_camera.sh
./run_oak_camera.sh
```

## H) Webcam Fallback

If you want to use a normal webcam instead of OAK-D, run:

```bash
python3 live_plant_camera.py --camera-type webcam --camera-index 0
```

On Raspberry Pi and Linux, webcam fallback uses normal OpenCV capture such as `cv2.VideoCapture(0)`.

## I) Troubleshooting

If `depthai` is not found:

```bash
pip install depthai
```

If `cv2.imshow` does not work:

- Make sure `opencv-python` is installed, not only `opencv-python-headless`.

If the OAK-D is not detected:

- Check the USB cable.
- Use a data cable, not a charge-only cable.
- Use a Raspberry Pi USB 3.0 port.
- Check external power for the OAK-D.
- Run `lsusb` again.
- It should show `Intel Movidius MyriadX`.
- You do not need to change the Python code for a different USB socket on the Raspberry Pi.

If you connect an Arduino over USB serial, first check which serial device name Linux assigned:

```bash
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null
```

Common results are:

- `/dev/ttyACM0`
- `/dev/ttyUSB0`

The Raspberry Pi sends `b"S"` to the Arduino after a confirmed plant detection when `--enable-spray` is used.

The Arduino receives `S` and can activate the pump relay on pin `9`.

The `--spray-cooldown` argument prevents the pump from spraying every frame.

If DepthAI shows permission warnings, add udev rules:

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="03e7", MODE="0666", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo reboot
```

## J) Files To Copy To Raspberry Pi

Copy these files and folders:

- `live_plant_camera.py`
- `oak_test.py`
- `plant_best.pt`
- `requirements.txt`
- `run_oak_camera.sh`
- `test_oak_camera.sh`
- `README_RASPBERRY_PI.md`
- `yolov8n.pt`
- any other project files required by the current code

## Quick Start

Run these commands on Raspberry Pi:

```bash
chmod +x run_oak_camera.sh
chmod +x test_oak_camera.sh
./test_oak_camera.sh
./run_oak_camera.sh
```

Direct Python commands:

```bash
python3 oak_test.py
python3 live_plant_camera.py --camera-type oakd
python3 live_plant_camera.py --camera-type oakd --enable-spray --arduino-port /dev/ttyACM0 --spray-cooldown 5
python3 live_plant_camera.py --camera-type webcam --camera-index 0
```
