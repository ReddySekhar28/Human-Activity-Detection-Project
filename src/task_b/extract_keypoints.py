
from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "datasets" / "activity"
MODEL_PATH = ROOT / "runs" / "pose" / "outputs" / "coco_pose_fast" / "weights" / "best.pt"
OUTPUT_DIR = ROOT / "outputs" / "task_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "keypoints.csv"

# Load trained YOLO pose model
model = YOLO(str(MODEL_PATH))

rows = []
image_extensions = {".jpg", ".jpeg", ".png"}

for label in ["normal", "suspicious"]:
    folder = DATASET / label

    for image_path in folder.iterdir():
        if image_path.suffix.lower() not in image_extensions:
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"Could not read: {image_path.name}")
            continue

        results = model.predict(
            source=image,
            imgsz=512,
            device=0,
            verbose=False
        )

        result = results[0]

        if result.keypoints is None or result.boxes is None:
            print(f"No person detected: {image_path.name}")
            continue

        keypoints = result.keypoints.data
        boxes = result.boxes.xyxy

        if len(keypoints) == 0:
            print(f"No person detected: {image_path.name}")
            continue

        # Select the person with the largest bounding box
        areas = (
            (boxes[:, 2] - boxes[:, 0]) *
            (boxes[:, 3] - boxes[:, 1])
        )

        person_index = int(areas.argmax())

        x1, y1, x2, y2 = boxes[person_index].tolist()
        person_keypoints = keypoints[person_index].cpu().numpy()

        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)

        features = []

        for x, y, confidence in person_keypoints:
            # Normalize coordinates relative to person bounding box
            x_norm = (x - x1) / width
            y_norm = (y - y1) / height

            features.extend([
                float(x_norm),
                float(y_norm),
                float(confidence)
            ])

        rows.append([
            label,
            image_path.name,
            *features
        ])

        print(f"Processed: {label}/{image_path.name}")

# Create CSV columns
columns = ["label", "source"]

for i in range(17):
    columns.extend([
        f"kp{i}_x",
        f"kp{i}_y",
        f"kp{i}_confidence"
    ])

with open(OUTPUT_CSV, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    writer.writerows(rows)

print("\nKeypoint extraction completed!")
print(f"Images processed: {len(rows)}")
print(f"CSV saved to: {OUTPUT_CSV}")
