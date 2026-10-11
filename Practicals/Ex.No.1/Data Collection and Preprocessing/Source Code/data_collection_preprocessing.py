"""Load the sklearn handwritten digits dataset, export raw and normalized CSVs, and plot samples."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PLOTS = ROOT / "outputs" / "plots"
TABLES = ROOT / "outputs" / "tables"
for p in (RAW, PROC, PLOTS, TABLES):
    p.mkdir(parents=True, exist_ok=True)

ds = load_digits()
X = ds.data.astype(np.float32)  # shape: (samples, 64), each row is an 8x8 image flattened
y = ds.target.astype(int)
cols = [f"pixel_{i:02d}" for i in range(64)]

raw = pd.DataFrame(X, columns=cols)
raw.insert(0, "label", y)
raw.insert(0, "sample_id", np.arange(len(y)))
raw.to_csv(RAW / "digits_grayscale_raw.csv", index=False)

X_scaled = X / 16.0  # source pixel values are 0..16; normalized values become 0..1
processed = pd.DataFrame(X_scaled, columns=cols)
processed.insert(0, "label", y)
processed.insert(0, "sample_id", np.arange(len(y)))
processed.to_csv(PROC / "digits_grayscale_preprocessed.csv", index=False)

pd.DataFrame({
    "item": ["Total images", "Image height", "Image width", "Flattened pixel features",
             "Number of classes", "Raw pixel minimum", "Raw pixel maximum"],
    "value": [len(y), 8, 8, X.shape[1], len(np.unique(y)), float(X.min()), float(X.max())]
}).to_csv(TABLES / "dataset_summary.csv", index=False)
pd.DataFrame({"digit": np.arange(10), "count": [(y == i).sum() for i in range(10)]}).to_csv(
    TABLES / "class_distribution.csv", index=False)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(ds.images[i], cmap="gray", vmin=0, vmax=16)
    ax.set_title(f"Label: {y[i]}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(PLOTS / "sample_images.png", dpi=160)
plt.close()
print("Saved raw and normalized CSV datasets, summaries, and sample image plot.")
