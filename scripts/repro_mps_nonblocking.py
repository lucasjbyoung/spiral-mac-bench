"""Minimal repro: on MPS a non_blocking copy from pageable host memory reads the
source after .to() returns, so reusing or freeing the source corrupts the copy.

    python repro_mps_nonblocking.py

Observed on Apple M5 Max, macOS 27.2, torch 2.11.0:
    non_blocking=False: 0/20 runs corrupted
    non_blocking=True: 20/20 runs corrupted
"""
import numpy as np
import torch

busy = torch.randn(4096, 4096, device='mps')
for nb in (False, True):
    bad_runs = 0
    for _ in range(20):
        for _ in range(4):  # keep the GPU queue busy so the copy is deferred
            busy = busy @ busy
            busy = busy / busy.norm()
        a = np.full(1_000_000, 7, dtype=np.int64)
        t = torch.from_numpy(a).to('mps', non_blocking=nb)
        a[:] = -1  # the host buffer is reused right after the call
        bad_runs += int((t.cpu() != 7).any())
    print(f'non_blocking={nb}: {bad_runs}/20 runs corrupted', flush=True)
