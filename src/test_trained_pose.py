
from ultralytics import YOLO
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

model_path = PROJECT_DIR / "outputs" / "coco8_pose_test" / "weights" / "best.pt"

model = YOLO(str(model_path))

print("Trained model loaded successfully!")

results = model.predict(
    source=str(PROJECT_DIR / "bus.jpg"),
    device=0,
    save=True,
    project=str(PROJECT_DIR / "outputs"),
    name="trained_pose_prediction"
)

for result in results:
    if result.keypoints is not None:
        print("Keypoint shape:", result.keypoints.data.shape)
        print("Detected people:", len(result.keypoints.data))
