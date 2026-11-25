import numpy as np

def corrupt_batch(clean, rng=None, missing_fraction=0.20, noise_std=0.15):
    """Create synthetic gaps and additive noise. clean: (batch, channels, time)."""
    rng = rng or np.random.default_rng()
    x = clean.copy()
    mask = np.ones_like(x, dtype=np.float32)
    b, c, t = x.shape
    for i in range(b):
        for ch in range(c):
            # One contiguous missing segment per channel.
            gap = max(8, int(t * missing_fraction))
            start = int(rng.integers(0, max(1, t - gap + 1)))
            x[i, ch, start:start + gap] = 0.0
            mask[i, ch, start:start + gap] = 0.0
    x += rng.normal(0, noise_std, size=x.shape).astype(np.float32) * mask
    return x.astype(np.float32), mask
