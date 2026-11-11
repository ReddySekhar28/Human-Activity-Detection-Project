
from pathlib import Path
import cv2
import joblib
from ultralytics import YOLO

# Project paths
ROOT = Path(__file__).resolve().parents[2]

POSE_PATH = (
    ROOT / "runs" / "pose" / "outputs" /
    "coco_pose_fast" / "weights" / "best.pt"
)

KNN_PATH = ROOT / "outputs" / "task_b" / "best_knn_model.pkl"

OUTPUT_PATH = ROOT / "outputs" / "task_b" / "live_test.mp4"

# Create output directory
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Verify required files
if not POSE_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {POSE_PATH}")

if not KNN_PATH.exists():
    raise FileNotFoundError(f"KNN model not found: {KNN_PATH}")

# Load trained models
pose_model = YOLO(str(POSE_PATH))
knn_model = joblib.load(KNN_PATH)

print("YOLO Pose model loaded successfully!")
print("KNN classifier loaded successfully!")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera connection.")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 20.0

# Create video writer
writer = cv2.VideoWriter(
    str(OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create output video.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Unable to read webcam frame.")
            break

        results = pose_model.predict(
            frame,
            imgsz=512,
            device=0,
            verbose=False
        )

        result = results[0]
        display = frame.copy()

        if result.keypoints is not None and result.boxes is not None:

            keypoints = result.keypoints.data
            boxes = result.boxes.xyxy

            if len(keypoints) > 0:

                # Select person with the largest bounding box
                areas = (
                    (boxes[:, 2] - boxes[:, 0]) *
                    (boxes[:, 3] - boxes[:, 1])
                )

                index = int(areas.argmax())

                x1, y1, x2, y2 = boxes[index].tolist()
                points = keypoints[index].cpu().numpy()

                person_width = max(x2 - x1, 1)
                person_height = max(y2 - y1, 1)

                # Extract normalized keypoint features
                features = []

                for x, y, confidence in points:
                    features.extend([
                        float((x - x1) / person_width),
                        float((y - y1) / person_height),
                        float(confidence)
                    ])

                # KNN prediction
                probabilities = knn_model.predict_proba([features])[0]

                classes = knn_model.classes_
                best_index = probabilities.argmax()

                prediction = classes[best_index]
                confidence = float(probabilities[best_index])

                # Confidence threshold
                if confidence < 0.75:
                    text = f"UNCERTAIN ({confidence:.2f})"
                else:
                    text = f"{prediction.upper()} ({confidence:.2f})"

                # Draw bounding box
                cv2.rectangle(
                    display,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2
                )

                # Display prediction
                cv2.putText(
                    display,
                    text,
                    (int(x1), max(int(y1) - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        # Display and save webcam frame
        cv2.imshow("YOLO Pose + KNN Activity Detection", display)
        writer.write(display)

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

print("\nVideo testing completed.")
print("Saved video:", OUTPUT_PATH)
