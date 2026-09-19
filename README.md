\# 🌱 Plant Detection using YOLO



A computer vision project for real-time plant detection using \*\*YOLO\*\*, custom training data, and deployment support for \*\*Raspberry Pi\*\* and \*\*OAK cameras\*\*.



The project starts from pretrained YOLO weights and improves the model through additional training using custom-collected plant images, target-specific data, and hard-negative samples.



\---



\## 📌 Project Overview



This project was developed as the computer vision component of an intelligent agricultural robot.



The main goal is to detect plants from a live camera feed and provide the detection result to the robot control system.



\### System Flow



```text

Camera

&#x20;  ↓

Live Image

&#x20;  ↓

YOLO Model

&#x20;  ↓

Plant Detection

&#x20;  ↓

Detection Result

&#x20;  ↓

Raspberry Pi / Robot Logic

```



\---



\## 🧠 AI Model



The project uses a pretrained YOLO model as the starting point.



Instead of training a model completely from scratch, pretrained YOLO weights were used and then further trained using additional plant images and custom datasets.



This approach allows the model to benefit from previously learned visual features while adapting to the specific plants and environment required by the project.



\### Training Pipeline



```text

Pretrained YOLO Model

&#x20;       ↓

Additional Plant Images

&#x20;       ↓

Dataset Preparation

&#x20;       ↓

Target-Specific Dataset

&#x20;       ↓

Hard Negative Collection

&#x20;       ↓

Additional Training / Fine-Tuning

&#x20;       ↓

Best Model

&#x20;       ↓

PyTorch Model (.pt)

&#x20;       ↓

ONNX Export (.onnx)

&#x20;       ↓

Raspberry Pi / OAK Deployment

```



\---



\## 🎯 Training Improvements



Several steps were used to improve the model performance.



\### Custom Plant Data



Additional plant images were collected and added to the training dataset so the model could better recognize plants in the actual target environment.



\### Target-Specific Dataset



A dedicated target dataset was created to improve the model's performance on the specific type of plants and camera conditions used in the project.



\### Hard Negatives



Hard-negative images were collected using:



```text

capture\_hard\_negatives.py

```



These images contain scenes that should not be classified as the target plant.



Adding these samples helps reduce false-positive detections.



\---



\## 🤖 Trained Models



The repository contains the trained models in both \*\*PyTorch\*\* and \*\*ONNX\*\* formats.



```text

plant\_best.pt

plant\_best.onnx



plant\_best\_target.pt

plant\_best\_target.onnx

```



\### PyTorch Models



```text

plant\_best.pt

plant\_best\_target.pt

```



These models can be used with the YOLO / PyTorch environment for inference, testing, or additional training.



\### ONNX Models



```text

plant\_best.onnx

plant\_best\_target.onnx

```



ONNX provides a portable model format that can be used for deployment on different platforms and inference engines.



\---



\## 📂 Project Structure



```text

plant-detection-yolo/

│

├── build\_target\_dataset.py

├── capture\_hard\_negatives.py

├── export\_for\_pi.py

├── export\_target\_for\_pi.py

│

├── live\_plant\_camera.py

├── oak\_test.py

│

├── train\_plant\_classifier.py

├── train\_plant\_classifier\_gpu.bat

│

├── plant\_best.pt

├── plant\_best.onnx

├── plant\_best\_target.pt

├── plant\_best\_target.onnx

│

├── pi\_deploy\_ready/

│

├── run\_oak\_camera.sh

├── run\_plant\_camera.bat

├── test\_oak\_camera.sh

│

├── requirements.txt

├── README\_RASPBERRY\_PI.md

├── .gitignore

└── README.md

```



\---



\## 🛠 Main Scripts



\### `train\_plant\_classifier.py`



Main training script used to train and improve the plant model using the prepared dataset.



\---



\### `build\_target\_dataset.py`



Creates and prepares the target-specific dataset used during model training.



\---



\### `capture\_hard\_negatives.py`



Captures hard-negative images that are used to reduce false-positive detections.



\---



\### `live\_plant\_camera.py`



Runs the trained model on a live camera stream for real-time plant detection.



\---



\### `oak\_test.py`



Tests the OAK camera connection and verifies that the camera can capture frames correctly.



\---



\### `export\_for\_pi.py`



Exports the trained model into a format suitable for Raspberry Pi deployment.



\---



\### `export\_target\_for\_pi.py`



Exports the target-specific version of the trained model for deployment.



\---



\## 📷 Hardware



The vision system was designed to work with:



\- Raspberry Pi

\- OAK Camera

\- Agricultural robot platform



The camera captures the environment and sends image frames to the plant detection system.



The AI model analyzes the images and returns the plant detection result to the robot control logic.



\---



\## ⚙️ Installation



Clone the repository:



```bash

git clone https://github.com/halaawwad/plant-detection-yolo.git

cd plant-detection-yolo

```



Create a Python virtual environment:



```bash

python -m venv .venv

```



\### Windows



```cmd

.venv\\Scripts\\activate

```



\### Linux / Raspberry Pi



```bash

source .venv/bin/activate

```



Install the required dependencies:



```bash

pip install -r requirements.txt

```



\---



\## 📷 Test the OAK Camera



Run:



```bash

python oak\_test.py

```



On Raspberry Pi / Linux, the provided test script can also be used:



```bash

bash test\_oak\_camera.sh

```



\---



\## 🌱 Run Live Plant Detection



Run:



```bash

python live\_plant\_camera.py

```



On Windows:



```cmd

run\_plant\_camera.bat

```



For the OAK camera deployment environment:



```bash

bash run\_oak\_camera.sh

```



\---



\## 🍓 Raspberry Pi Deployment



The deployment-ready Raspberry Pi files are located inside:



```text

pi\_deploy\_ready/

```



Additional deployment information is available in:



```text

README\_RASPBERRY\_PI.md

```



\---



\## 📊 Dataset



The model was trained using multiple sources of plant images and additional custom-collected data.



The local development environment included:



```text

Dataset/

Dataset\_target\_specific/

flower\_photos/

hard\_negatives/

```



Large datasets are intentionally excluded from this GitHub repository.



Training outputs and temporary files are also excluded to keep the repository lightweight and focused on the source code and final trained models.



\---



\## 🚫 Files Excluded from GitHub



The following types of files are intentionally ignored:



```text

.venv/

\_\_pycache\_\_/

runs/

Dataset/

Dataset\_target\_specific/

flower\_photos/

hard\_negatives/

\*.zip

\*.tgz

```



Pretrained YOLO base weights are also excluded because they can be downloaded separately when needed.



\---



\## 🚀 Technologies



\- Python

\- YOLO

\- PyTorch

\- ONNX

\- OpenCV

\- Computer Vision

\- Deep Learning

\- Raspberry Pi

\- OAK Camera



\---



\## 🌾 Application



The plant detection model is intended to be integrated into an autonomous agricultural robot.



The complete system can use the detection result for tasks such as:



```text

Plant Detection

&#x20;      ↓

Target Confirmation

&#x20;      ↓

Robot Positioning

&#x20;      ↓

Agricultural Action

```



This allows computer vision to become part of the robot's automated decision-making process.



\---



\## 🔬 Project Goals



The main goals of this project are:



\- Detect plants in real time.

\- Improve detection using custom training data.

\- Reduce false detections using hard-negative samples.

\- Create a target-specific trained model.

\- Export the trained model for deployment.

\- Run the vision system on Raspberry Pi.

\- Integrate the AI model with an agricultural robot.



\---



\## 📄 Repository



GitHub:



```text

https://github.com/halaawwad/plant-detection-yolo

```



\---



\## 👩‍💻 Author



\*\*Hala Awwad\*\*



Computer Engineering Project



\---



\## 📌 Notes



The dataset used during development is not included in the repository because of its size.



The repository contains the source code, model export tools, deployment files, and final trained model weights required to demonstrate and deploy the plant detection system.



\---



\## 📄 License



This project is intended for academic, educational, and research purposes.

