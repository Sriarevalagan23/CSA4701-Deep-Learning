# NeuroEcho

Deep-learning research prototype for reconstructing artificially missing/corrupted EEG signal segments.

## Hardware profile
Designed to run on macOS, 8 GB RAM, CPU-only. Start with 8 EEG channels, 256-sample windows, small batches, and a small number of training epochs.

## Important scientific scope
This first implementation reconstructs EEG waveforms from synthetically corrupted versions of the same recordings. It does **not** infer thoughts, memories, or hidden mental content. Reconstruction quality is evaluated against the clean reference signal.

## 1. Setup (VS Code terminal)

```bash
cd NeuroEcho
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch installation fails, use the macOS instructions at https://pytorch.org/get-started/locally/ and select macOS + Pip.

## 2. Download a small public EEG subset

```bash
python scripts/download_eeg.py --subjects 1 --runs 1 2 --data-dir data/raw
```

The script uses MNE's EEGBCI/PhysioNet downloader. First download needs internet access.

## 3. Prepare training windows

```bash
python scripts/prepare_data.py --data-dir data/raw --output data/processed --channels 8 --window 256
```

## 4. Train

```bash
python scripts/train.py --data data/processed/windows.npz --epochs 5 --batch-size 8
```

Model checkpoint and training log are saved under `artifacts/`.

## 5. Launch dashboard

```bash
streamlit run app.py
```

Upload a prepared `.npz` file or use the sample test windows generated during preparation. Select a window and compare clean, corrupted, and reconstructed waveforms.

## Project structure

- `app.py`: Streamlit dashboard
- `neuroecho/data.py`: EDF loading and signal preprocessing
- `neuroecho/corruption.py`: reproducible synthetic missing/noisy segments
- `neuroecho/model.py`: compact CNN + BiLSTM denoising autoencoder
- `neuroecho/metrics.py`: reconstruction metrics
- `scripts/`: dataset download, preparation, training
- `artifacts/`: trained model outputs
- `data/raw/`: downloaded EDF files (not included in ZIP)
- `data/processed/`: prepared windows (not included in ZIP)

## Limitations
- EEG recordings and generated corruption masks are experimental inputs, not clinical measurements.
- The initial model is a baseline; compare against interpolation and filtering baselines before making research claims.
- Use participant-level train/validation/test splits for final experiments to avoid data leakage.
