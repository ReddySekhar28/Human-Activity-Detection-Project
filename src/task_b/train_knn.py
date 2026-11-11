
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Project paths
ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "outputs" / "task_b" / "keypoints.csv"
OUTPUT_DIR = ROOT / "outputs" / "task_b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load extracted keypoints
df = pd.read_csv(CSV_PATH)

print("Dataset loaded:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# Separate features and labels
X = df.drop(columns=["label", "source"])
y = df["label"]

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors="coerce")

if X.isnull().any().any():
    raise ValueError("Missing values found in keypoint features.")

if y.nunique() != 2:
    raise ValueError("Expected exactly two classes: normal and suspicious.")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Compare K values
k_values = [3, 5, 7, 9]
results = []
trained_models = {}

for k in k_values:
    print(f"\n{'=' * 40}")
    print(f"Training KNN with K = {k}")
    print("=" * 40)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions,
        pos_label="suspicious",
        zero_division=0
    )
    recall = recall_score(
        y_test, predictions,
        pos_label="suspicious",
        zero_division=0
    )
    f1 = f1_score(
        y_test, predictions,
        pos_label="suspicious",
        zero_division=0
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=["normal", "suspicious"],
        zero_division=0
    ))

    # Save confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["normal", "suspicious"]
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Suspicious"]
    )

    display.plot(cmap="Blues")
    plt.title(f"KNN Confusion Matrix (K={k})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_k{k}.png")
    plt.close()

    results.append({
        "K": k,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    trained_models[k] = model

# Compare all models
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "knn_results.csv", index=False)

print("\nKNN MODEL COMPARISON")
print(results_df.to_string(index=False))

# Select the model with the highest F1-score
best_k = int(results_df.loc[results_df["F1-score"].idxmax(), "K"])
best_model = trained_models[best_k]

joblib.dump(best_model, OUTPUT_DIR / "best_knn_model.pkl")

print(f"\nBest K value: {best_k}")
print("Best model saved: best_knn_model.pkl")
print("Results saved: knn_results.csv")
print("Confusion matrices saved.")
