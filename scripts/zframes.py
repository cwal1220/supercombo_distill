"""Kept frames stored compressed: each [2,6,128,256] uint8 frame zstd-compressed (level 3) on its own in
<unit>/frames.zst, with the byte offsets in <unit>/frames_off.npy [M+1]. 1.7x smaller than frames.npy, 0.4 ms to
decompress a frame, random access by index (DataLoader workers, bank refresh).

load_frames(unit dir) gives frames.npy memory-mapped when it is there, else a ZFrames; both index like the array:
f[j], f[j, cam], f[js, cam] (js a list or array of kept-frame indices), len(f).

usage: .venv/bin/python scripts/zframes.py data/ep_label/<unit> ...   convert (verify, then delete frames.npy)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

SHAPE = (2, 6, 128, 256)


class ZFrames:
    def __init__(self, d: Path):
        self.path = Path(d) / "frames.zst"
        self.off = np.load(Path(d) / "frames_off.npy")
        self.shape = (len(self.off) - 1, *SHAPE)
        self._fd, self._local = None, None

    def __len__(self):
        return self.shape[0]

    def __getstate__(self):          # a fresh file descriptor and decompressors in each process
        return {**self.__dict__, "_fd": None, "_local": None}

    def frame(self, i: int) -> np.ndarray:
        if self._fd is None:
            import threading
            self._fd, self._local = os.open(self.path, os.O_RDONLY), threading.local()
        dec = getattr(self._local, "dec", None)
        if dec is None:                  # a decompressor per thread (one is not thread-safe); pread is
            import zstandard
            dec = self._local.dec = zstandard.ZstdDecompressor()
        i = int(i) % len(self)
        a, b = int(self.off[i]), int(self.off[i + 1])
        return np.frombuffer(dec.decompress(os.pread(self._fd, b - a, a)), np.uint8).reshape(SHAPE)

    def __getitem__(self, k):
        i, rest = (k[0], k[1:]) if isinstance(k, tuple) else (k, ())
        if isinstance(i, (int, np.integer)):
            return self.frame(i)[rest]
        idx = range(*i.indices(len(self))) if isinstance(i, slice) else np.asarray(i).ravel()
        return np.stack([self.frame(j)[rest] for j in idx])


def load_frames(d: Path):
    d = Path(d)
    return np.load(d / "frames.npy", mmap_mode="r") if (d / "frames.npy").exists() else ZFrames(d)


def has_frames(d: Path) -> bool:
    return (Path(d) / "frames.npy").exists() or (Path(d) / "frames.zst").exists()


def convert(d: Path, level: int = 3):
    import zstandard
    d = Path(d)
    src = np.load(d / "frames.npy", mmap_mode="r")
    comp = zstandard.ZstdCompressor(level=level)
    off = [0]
    with open(d / "frames.zst.tmp", "wb") as f:
        for i in range(len(src)):
            c = comp.compress(np.ascontiguousarray(src[i]).tobytes())
            f.write(c)
            off.append(off[-1] + len(c))
    np.save(d / "frames_off.npy", np.array(off, np.int64))
    (d / "frames.zst.tmp").rename(d / "frames.zst")
    z = ZFrames(d)
    for i in np.linspace(0, len(src) - 1, 25).astype(int):
        assert np.array_equal(z[i], src[i]), (d, i)
    raw = (d / "frames.npy").stat().st_size
    (d / "frames.npy").unlink()
    print(f"{d.name}: {len(src)} frames, {raw / 1e9:.2f} -> {off[-1] / 1e9:.2f} GB", flush=True)


if __name__ == "__main__":
    for u in sys.argv[1:]:
        if (Path(u) / "frames.npy").exists() and (Path(u) / "labels.npz").exists():
            convert(Path(u))
