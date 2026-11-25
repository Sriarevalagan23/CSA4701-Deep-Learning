import argparse
from pathlib import Path
import numpy as np
from neuroecho.data import load_edf, windows_from_signal

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default="data/raw")
p.add_argument("--output", default="data/processed")
p.add_argument("--channels", type=int, default=8)
p.add_argument("--window", type=int, default=256)
args = p.parse_args()

raw_dir, out_dir = Path(args.data_dir), Path(args.output)
out_dir.mkdir(parents=True, exist_ok=True)
edf_files = sorted(raw_dir.rglob("*.edf"))
if not edf_files:
    raise SystemExit("No EDF files found. Run scripts/download_eeg.py first.")
all_windows = []
for f in edf_files:
    print(f"Processing {f}")
    signal, sfreq, names = load_edf(f, channels=args.channels)
    all_windows.append(windows_from_signal(signal, window=args.window))
windows = np.concatenate(all_windows, axis=0)
np.savez_compressed(out_dir / "windows.npz", windows=windows)
print(f"Saved {len(windows)} windows, shape={windows.shape}, to {out_dir / 'windows.npz'}")
