from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from neuroecho.model import CNNBiLSTMAutoencoder
from neuroecho.corruption import corrupt_batch
from neuroecho.metrics import regression_metrics

st.set_page_config(page_title="NeuroEcho", page_icon="🧠", layout="wide")
st.title("🧠 NeuroEcho")
st.caption("EEG signal reconstruction research prototype")

st.info("This prototype reconstructs artificially missing/corrupted EEG waveform segments. It does not decode thoughts or diagnose medical conditions.")

checkpoint = Path("artifacts/neuroecho_model.pt")
data_file = st.file_uploader("Upload prepared EEG windows (.npz)", type=["npz"])
if data_file is not None:
    loaded = np.load(data_file)
    windows = loaded["windows"].astype(np.float32)
elif Path("data/processed/windows.npz").exists():
    windows = np.load("data/processed/windows.npz")["windows"].astype(np.float32)
else:
    windows = None

if windows is None:
    st.warning("No prepared windows found. Follow the README to download and prepare the public EEG dataset.")
    st.stop()

idx = st.slider("Select EEG window", 0, len(windows)-1, 0)
clean = windows[idx:idx+1]
channel = st.selectbox("EEG channel", list(range(clean.shape[1])), format_func=lambda x: f"Channel {x+1}")
missing_fraction = st.slider("Missing segment proportion", 0.05, 0.50, 0.20, 0.05)
noise_std = st.slider("Added noise level", 0.0, 0.5, 0.15, 0.05)

if st.button("Run reconstruction", type="primary"):
    if not checkpoint.exists():
        st.error("Model checkpoint not found. Train the model first using scripts/train.py.")
        st.stop()
    corrupted, mask = corrupt_batch(clean, rng=np.random.default_rng(7),
                                    missing_fraction=missing_fraction, noise_std=noise_std)
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = CNNBiLSTMAutoencoder(channels=ckpt["channels"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        reconstructed = model(torch.from_numpy(corrupted)).numpy()

    ref = clean[0, channel]
    inp = corrupted[0, channel]
    pred = reconstructed[0, channel]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ref, name="Original reference", line=dict(width=2)))
    fig.add_trace(go.Scatter(y=inp, name="Corrupted input", line=dict(width=1)))
    fig.add_trace(go.Scatter(y=pred, name="Reconstructed output", line=dict(width=2)))
    fig.update_layout(title=f"EEG reconstruction — Channel {channel+1}",
                      xaxis_title="Sample", yaxis_title="Normalized amplitude",
                      height=480, legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    metrics = regression_metrics(ref, pred)
    c1, c2, c3 = st.columns(3)
    c1.metric("MSE", f"{metrics['MSE']:.4f}")
    c2.metric("MAE", f"{metrics['MAE']:.4f}")
    c3.metric("Pearson correlation", f"{metrics['Pearson correlation']:.4f}")
    out = pd.DataFrame({"original": ref, "corrupted": inp, "reconstructed": pred})
    st.download_button("Download reconstructed channel CSV",
                       out.to_csv(index=False).encode(),
                       file_name="neuroecho_reconstruction.csv", mime="text/csv")
