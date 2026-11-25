"""
Experiment 13 (CO3): Sequential Dataset Preparation and RNN Prediction
Stage: Data Collection and Preprocessing
Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../Exp_13_Dataset.csv")
print(f"[Data Collection and Preprocessing] Loaded dataset with shape: {df.shape}")

# Data Preprocessing & Cleaning
missing_counts = df.isnull().sum()
df_clean = df.dropna().copy()
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_normalized = (df_clean[numeric_cols] - df_clean[numeric_cols].mean()) / (df_clean[numeric_cols].std() + 1e-8)

# Save Summary Table
summary = df_clean.describe().T
summary.to_csv("stage1_preprocessing_summary_table.csv")

# Generate Plot
plt.figure(figsize=(8, 5))
df_normalized.iloc[:, :min(4, len(numeric_cols))].boxplot()
plt.title("Exp 13: Normalized Feature Distributions (Data Collection and Preprocessing)")
plt.ylabel("Standardized Value")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("stage1_preprocessing_plot.png", dpi=150)
plt.close()

with open("stage1_preprocessing_results.txt", "w") as f:
    f.write("Experiment 13 - Data Collection and Preprocessing Results\n")
    f.write("="*50 + "\n")
    f.write(f"Total Samples: {len(df_clean)}\n")
    f.write(f"Features Processed: {len(numeric_cols)}\n")
    f.write("Status: Preprocessing and Normalization Complete.\n")
