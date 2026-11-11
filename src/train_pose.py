
from ultralytics import YOLO
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Pretrained YOLO11n Pose model
MODEL_PATH = PROJECT_DIR / "yolo11n-pose.pt"

# Load pretrained weights
model = YOLO(str(MODEL_PATH))

print("Starting COCO-Pose fine-tuning...")
print("GPU: NVIDIA RTX 3050")
print("Dataset: COCO-Pose")
print("Training: 3 epochs, 512 image size, 10% dataset")

# Fine-tune model
results = model.train(
    data="coco-pose.yaml",
    epochs=3,
    imgsz=512,
    batch=4,
    device=0,
    workers=2,
    cache=False,
    amp=True,
    fraction=0.1,
    project=str(PROJECT_DIR / "outputs"),
    name="coco_pose_fast",
    exist_ok=True,
    plots=True,
    save=True
)

print("\nTraining completed!")

print(
    "Best model:",
    PROJECT_DIR / "runs" / "pose" / "outputs" /
    "coco_pose_fast" / "weights" / "best.pt"
)
