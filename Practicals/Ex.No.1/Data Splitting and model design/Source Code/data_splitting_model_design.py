"""Create stratified train/validation/test splits and document the fully connected model."""
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"
PROC.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

ds = load_digits()
X = ds.data.astype("float32") / 16.0
y = ds.target.astype(int)
cols = [f"pixel_{i:02d}" for i in range(64)]

# 70% train, 15% validation, 15% test; stratify preserves class proportions
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

for name, xx, yy in [("train", X_train, y_train), ("validation", X_val, y_val), ("test", X_test, y_test)]:
    out = pd.DataFrame(xx, columns=cols)
    out.insert(0, "label", yy)
    out.to_csv(PROC / f"{name}_split.csv", index=False)

architecture = pd.DataFrame([
    ["Input", 64, "Each 8x8 image is flattened into 64 values"],
    ["Dense hidden layer", 64, "Fully connected; ReLU activation"],
    ["Output", 10, "One score for each digit class 0 through 9"],
], columns=["layer", "units", "description"])
architecture.to_csv(TABLES / "model_architecture.csv", index=False)

pd.DataFrame({"split": ["train", "validation", "test"],
              "samples": [len(y_train), len(y_val), len(y_test)]}).to_csv(
    TABLES / "split_summary.csv", index=False)
print("Saved stratified splits and model architecture table.")
