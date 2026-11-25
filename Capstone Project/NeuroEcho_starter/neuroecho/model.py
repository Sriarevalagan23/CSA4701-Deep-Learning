import torch
from torch import nn

class CNNBiLSTMAutoencoder(nn.Module):
    """Compact sequence autoencoder for short, multichannel EEG windows."""
    def __init__(self, channels=8, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(48), nn.GELU(),
        )
        self.temporal = nn.LSTM(
            input_size=48, hidden_size=hidden // 2,
            num_layers=1, batch_first=True, bidirectional=True
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden, 48, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(48, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, channels, kernel_size=7, padding=3),
        )

    def forward(self, x):
        z = self.encoder(x)          # (B, 48, T/2)
        z = z.transpose(1, 2)        # (B, T/2, 48)
        z, _ = self.temporal(z)      # (B, T/2, hidden)
        z = z.transpose(1, 2)
        y = self.decoder(z)
        return y[..., :x.shape[-1]]
