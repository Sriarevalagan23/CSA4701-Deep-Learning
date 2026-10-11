"""Train and evaluate a fully connected neural network for grayscale digit classification."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
PLOTS = ROOT / "outputs" / "plots"
RESULTS = ROOT / "outputs" / "results"
for p in (TABLES, PLOTS, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

ds = load_digits()
X = ds.data.astype(np.float32) / 16.0
y = ds.target.astype(int)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# A fully connected neural network: 64 input features -> 64 hidden units -> 10 classes.
model = MLPClassifier(hidden_layer_sizes=(64,), activation="relu", solver="adam",
                      alpha=0.0001, learning_rate_init=0.001, max_iter=300,
                      random_state=42, early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=20)
model.fit(X_train, y_train)
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

pd.DataFrame(classification_report(y_test, pred, output_dict=True, zero_division=0)).T.to_csv(
    TABLES / "classification_report.csv")
pd.DataFrame(cm, index=[f"actual_{i}" for i in range(10)],
             columns=[f"pred_{i}" for i in range(10)]).to_csv(TABLES / "confusion_matrix.csv")
pd.DataFrame([
    ["Test accuracy", acc], ["Test error rate", 1-acc],
    ["Training samples", len(X_train)], ["Validation samples", len(X_val)],
    ["Test samples", len(X_test)], ["Iterations completed", model.n_iter_]
], columns=["metric", "value"]).to_csv(TABLES / "evaluation_summary.csv", index=False)
pd.DataFrame({"sample_index": np.arange(len(y_test)), "actual_label": y_test,
              "predicted_label": pred, "correct": pred == y_test}).to_csv(
    RESULTS / "test_predictions.csv", index=False)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap="Blues", colorbar=False)
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig(PLOTS / "confusion_matrix.png", dpi=160)
plt.close()

plt.figure(figsize=(7, 4))
plt.bar(["Correct", "Incorrect"], [(pred == y_test).sum(), (pred != y_test).sum()])
plt.ylabel("Number of test images")
plt.title("Test Prediction Outcomes")
plt.tight_layout()
plt.savefig(PLOTS / "test_outcomes.png", dpi=160)
plt.close()

print(f"Test accuracy: {acc:.4f}")
print(classification_report(y_test, pred, zero_division=0))
