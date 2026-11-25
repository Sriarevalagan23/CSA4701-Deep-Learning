from pathlib import Path
import numpy as np
import mne
from scipy.signal import butter, sosfiltfilt

def load_edf(path, channels=8, target_sfreq=160):
    """Load EDF, retain EEG channels, filter, resample and return (channels, samples)."""
    raw = mne.io.read_raw_edf(str(path), preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, eog=False, stim=False, exclude=[])
    if len(raw.ch_names) == 0:
        raise ValueError("No EEG channels were found in this EDF file.")
    raw.filter(l_freq=1.0, h_freq=40.0, method="iir", verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")
    data = raw.get_data()
    data = data[:min(channels, data.shape[0])]
    # Robust channel-wise scaling reduces sensitivity to amplitude outliers.
    med = np.median(data, axis=1, keepdims=True)
    scale = np.median(np.abs(data - med), axis=1, keepdims=True) * 1.4826
    scale[scale < 1e-8] = 1.0
    data = (data - med) / scale
    return data.astype(np.float32), raw.info["sfreq"], raw.ch_names[:data.shape[0]]

def windows_from_signal(signal, window=256, stride=None):
    stride = stride or window
    chunks = []
    for start in range(0, signal.shape[1] - window + 1, stride):
        chunks.append(signal[:, start:start + window])
    if not chunks:
        raise ValueError(f"Recording is shorter than one {window}-sample window.")
    return np.stack(chunks).astype(np.float32)

def load_npz_windows(path):
    obj = np.load(path)
    x = obj["windows"].astype(np.float32)
    if x.ndim != 3:
        raise ValueError("Expected windows array shaped (windows, channels, samples).")
    return x
