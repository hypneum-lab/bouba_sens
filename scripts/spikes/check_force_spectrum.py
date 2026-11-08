"""Spectral check of force.pt channels (cardiac col 0, resp col 3) at 10 Hz."""

import sys

import numpy as np
import torch
from scipy.signal import welch

for p in sys.argv[1:]:
    f = torch.load(p).numpy()
    print(p, "shape", f.shape, "dtype", f.dtype, "duration_s", f.shape[0] / 10.0)
    nz = np.flatnonzero(np.abs(f[:, 0]) + np.abs(f[:, 3]) > 0)
    print(
        "  last nonzero frame",
        nz[-1] if nz.size else None,
        "zero-pad frames",
        f.shape[0] - 1 - (nz[-1] if nz.size else -1),
    )
    for name, c in (("cardiac", 0), ("resp", 3)):
        x = f[:, c] - f[:, c].mean()
        fr, pw = welch(x, fs=10.0, nperseg=1024)
        band = fr >= 0.05
        pk = fr[band][np.argmax(pw[band])]
        lo = pw[(fr >= 0.1) & (fr < 0.6)].sum()
        hi = pw[(fr >= 0.8) & (fr < 2.5)].sum()
        print(
            f"  {name:8s} peak={pk:.3f} Hz  P(0.1-0.6)/P(0.8-2.5)={lo / hi:.3f}  std={x.std():.3f}"
        )
