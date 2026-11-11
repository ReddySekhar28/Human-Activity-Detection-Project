# Human Activity Detection using YOLO11-Pose and KNN

A computer vision and machine learning project that detects human body keypoints using YOLO11-Pose and classifies activity patterns into **Normal** and **Suspicious** categories using a K-Nearest Neighbors (KNN) classifier.

## Project Overview

This project combines deep learning and traditional machine learning:

1. YOLO11n-Pose detects people and estimates their body keypoints.
2. The system extracts 17 human body keypoints.
3. Keypoint coordinates are normalized relative to the detected person's bounding box.
4. A trained KNN classifier predicts whether the detected activity is Normal or Suspicious.
5. Webcam inference displays the prediction in real time and records the output video.

## System Architecture

```text
Webcam / Input Image
        |
        v
YOLO11n-Pose Model
        |
        v
Human Detection
        |
        v
17 Body Keypoints
        |
        v
Feature Extraction
(Normalized X, Y, Confidence)
        |
        v
Trained KNN Classifier
        |
        v
Normal / Suspicious
        |
        v
Display Prediction
```

## Technologies Used

- Python
- YOLO11n-Pose
- Ultralytics
- OpenCV
- PyTorch
- CUDA
- Scikit-learn
- K-Nearest Neighbors
- Pandas
- NumPy
- Matplotlib
- Joblib

## Project Structure

```text
Human_Activity_Detection_Project/
│
├── src/
│   ├── train_pose.py
│   ├── test_trained_pose.py
│   │
│   └── task_b/
│       ├── extract_keypoints.py
│       ├── train_knn.py
│       └── test_video.py
│
├── datasets/                  # Local dataset (not included)
│   └── activity/
│       ├── normal/
│       └── suspicious/
│
├── runs/                      # YOLO training outputs (not included)
├── outputs/                   # Generated results (not included)
├── weights/                   # Optional local model storage
│
├── bus.jpg
├── gpu_test.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ReddySekhar28/Human-Activity-Detection-Project.git
cd Human-Activity-Detection-Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install PyTorch

PyTorch installation depends on your operating system and hardware.

This project was tested with:

- PyTorch: 2.11.0+cu128
- Torchvision: 0.26.0+cu128
- CUDA: 12.8
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU

Install a compatible PyTorch build for your system using the official installation selector:

https://pytorch.org/get-started/locally/

## Dataset Preparation

The activity dataset is expected to have this structure:

```text
datasets/
└── activity/
    ├── normal/
    └── suspicious/
```

Place the corresponding images inside each class directory.

**Note:** The dataset is not included in this repository.

## Model Training and Execution

### Task A: YOLO11-Pose Training

The project uses the pretrained `yolo11n-pose.pt` model and the Ultralytics COCO-Pose configuration.

Place the pretrained model in the project root.

Run:

```bash
python src/train_pose.py
```

The selected experiment used:

| Parameter | Value |
|---|---|
| Model | YOLO11n-Pose |
| Dataset configuration | coco-pose.yaml |
| Epochs | 3 |
| Image size | 512 |
| Batch size | 4 |
| Dataset fraction | 10% |
| Device | CUDA GPU |
| Mixed precision | Enabled |

The selected trained checkpoint is expected at:

```text
runs/pose/outputs/coco_pose_fast/weights/best.pt
```

The pretrained and trained model weights are not included in this repository.

### Task B: Extract Human Keypoints

Prepare the activity dataset first, then run:

```bash
python src/task_b/extract_keypoints.py
```

This extracts 17 body keypoints per detected person and generates normalized coordinate and confidence features.

Generated file:

```text
outputs/task_b/keypoints.csv
```

### Train the KNN Classifier

Run:

```bash
python src/task_b/train_knn.py
```

The script:

- Loads the extracted keypoint dataset.
- Splits data into training and testing sets.
- Applies feature standardization.
- Compares KNN configurations with K values 3, 5, 7 and 9.
- Calculates accuracy, precision, recall and F1-score.
- Saves confusion matrices and the best classifier.

### Webcam Activity Detection

After preparing both trained models, run:

```bash
python src/task_b/test_video.py
```

The webcam inference script combines YOLO pose estimation and KNN classification.

Press `Q` to stop the webcam session.

The recorded video is saved to:

```text
outputs/task_b/live_test.mp4
```

## KNN Evaluation Results

The KNN classifier was evaluated using a stratified 75:25 train-test split with `random_state=42`.

| K | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 3 | 75.00% | 66.67% | 50.00% | 57.14% |
| 5 | 66.67% | 50.00% | 50.00% | 50.00% |
| **7** | **75.00%** | **60.00%** | **75.00%** | **66.67%** |
| 9 | 66.67% | 50.00% | 75.00% | 60.00% |

**Best tested configuration: K = 7**

K=7 achieved the highest F1-score among the four evaluated configurations.

These results represent the current dataset and test split. Further testing with larger and more diverse datasets is needed to assess generalization.

## Hardware and Environment

| Component | Configuration |
|---|---|
| Programming language | Python 3.14.3 |
| Deep learning framework | PyTorch 2.11.0 |
| Computer vision | OpenCV |
| Pose estimation | YOLO11n-Pose |
| Machine learning | Scikit-learn 1.9.1 |
| GPU | NVIDIA RTX 3050 6GB Laptop GPU |
| CUDA | 12.8 |

GPU verification:

```bash
python gpu_test.py
```

## Limitations and Future Improvements

- Expand the Normal and Suspicious activity datasets.
- Evaluate on larger, independent test datasets.
- Explore additional pose-based classifiers.
- Improve detection under poor lighting and partial occlusion.
- Support multiple-person activity classification.
- Evaluate real-world performance and false-positive rates.
- Improve temporal activity recognition using sequences of frames.

## Disclaimer

This is an academic computer vision project. The Normal and Suspicious labels represent model predictions based on the available training data. They do not establish intent, wrongdoing, or a real-world safety threat. The system should not be used as the sole basis for disciplinary, security, or other consequential decisions.

## GitHub Repository

[Human Activity Detection Project](https://github.com/ReddySekhar28/Human-Activity-Detection-Project)

## Author

**Gorrolla Reddy Sekhar**

GitHub: [ReddySekhar28](https://github.com/ReddySekhar28)