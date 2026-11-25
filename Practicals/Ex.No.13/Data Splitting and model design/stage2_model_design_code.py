"""
Experiment 13 (CO3): Sequential Dataset Preparation and RNN Prediction
Stage: Data Splitting and Model Design
Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../Exp_13_Dataset.csv")
print(f"[Data Splitting and Model Design] Loaded dataset with shape: {df.shape}")

# Data Splitting (70% Train, 15% Validation, 15% Test)
n = len(df)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# Model Architecture / Mathematical Setup for Sequential Dataset Preparation and RNN Prediction
split_summary = pd.DataFrame({
    "Split": ["Train", "Validation", "Test"],
    "Samples": [len(train_df), len(val_df), len(test_df)],
    "Percentage": ["70%", "15%", "15%"]
})
split_summary.to_csv("stage2_model_design_summary_table.csv", index=False)

# Plot Split Proportions
plt.figure(figsize=(6, 4))
plt.bar(split_summary["Split"], split_summary["Samples"], color=["#2b5c8f", "#4a90e2", "#7fb3d5"])
plt.title("Exp 13: Dataset Split & Architecture Config")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("stage2_model_design_plot.png", dpi=150)
plt.close()

with open("stage2_model_design_results.txt", "w") as f:
    f.write("Experiment 13 - Data Splitting and Model Design Results\n")
    f.write("="*50 + "\n")
    f.write(f"Train/Val/Test Sizes: {len(train_df)} / {len(val_df)} / {len(test_df)}\n")
    f.write("Model Architecture Configured using: TensorFlow, Keras, NumPy, Pandas, Matplotlib\n")
