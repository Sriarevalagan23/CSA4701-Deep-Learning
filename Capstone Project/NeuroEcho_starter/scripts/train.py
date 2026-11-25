import argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from neuroecho.model import CNNBiLSTMAutoencoder
from neuroecho.corruption import corrupt_batch

class EEGDataset(Dataset):
    def __init__(self, arr): self.arr = torch.from_numpy(arr)
    def __len__(self): return len(self.arr)
    def __getitem__(self, i): return self.arr[i]

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/processed/windows.npz")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--max-windows", type=int, default=1500)
args = p.parse_args()

torch.set_num_threads(2)
device = torch.device("cpu")
arr = np.load(args.data)["windows"].astype(np.float32)
if len(arr) > args.max_windows:
    arr = arr[:args.max_windows]
ds = EEGDataset(arr)
n_val = max(1, int(0.2 * len(ds)))
train_ds, val_ds = random_split(ds, [len(ds)-n_val, n_val],
                                generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size)
model = CNNBiLSTMAutoencoder(channels=arr.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.MSELoss()
Path("artifacts").mkdir(exist_ok=True)
history = []

for epoch in range(args.epochs):
    model.train()
    train_losses = []
    for clean in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        clean = clean.to(device)
        corrupted, mask = corrupt_batch(clean.numpy())
        corrupted = torch.from_numpy(corrupted).to(device)
        pred = model(corrupted)
        # Focus loss on missing regions; also retain a small whole-signal term.
        mask_t = torch.from_numpy(mask).to(device)
        missing = 1.0 - mask_t
        loss = (((pred-clean)**2 * missing).sum() / missing.sum().clamp_min(1)
                + 0.1 * loss_fn(pred, clean))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for clean in val_loader:
            clean = clean.to(device)
            corrupted, _ = corrupt_batch(clean.numpy(), rng=np.random.default_rng(100+epoch))
            pred = model(torch.from_numpy(corrupted).to(device))
            val_losses.append(loss_fn(pred, clean).item())
    row = {"epoch": epoch+1, "train_loss": float(np.mean(train_losses)),
           "val_mse": float(np.mean(val_losses))}
    history.append(row)
    print(row)

torch.save({"model_state": model.state_dict(), "channels": arr.shape[1],
            "window": arr.shape[2]}, "artifacts/neuroecho_model.pt")
with open("artifacts/training_history.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=history[0].keys())
    writer.writeheader(); writer.writerows(history)
print("Saved artifacts/neuroecho_model.pt")
