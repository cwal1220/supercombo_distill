"""Factorize a checkpoint's two largest NPU FC layers into low-rank products (arch hist_rank, vis_rank), from its own
activations on training frames, so fine-tuning starts from (nearly) the same function.

On the AX620E these layers are bound by streaming their S8 weights: temporal.0 (6.6M weights, 0.58M cycles) and
vis_fc.0 (4.2M, 0.35M cycles) of R18q's 6.73M. Both factorizations are data-aware:
  - hist_rank r: P = the top-r principal directions of the hidden states (uncentred second moment), shared by the 24
    history tokens; temporal.0's columns of token t become W_t P^T (the current hidden and the rest stay).
  - vis_rank r: the top-r principal directions V of vis_fc.0's responses W x (Zhang et al. 2015, response-space low
    rank): vis_fc.0 = V^T W (no bias), vis_fc.1 = V with the old bias.
Prints the relative error of both layers' pre-activations on the collected frames.

usage: .venv/bin/python scripts/lowrank.py --ckpt ckpt/mac_r18q/ema.pt --out ckpt/lr/init.pt --hist-rank 128 --vis-rank 256
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import R18Backbone, Student, load_student  # noqa: E402
from train import EVAL_ROUTES  # noqa: E402
from zframes import load_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UNITS = ["2026-10-04--12-32-26-070__000", "2026-10-10--11-16-58-871__000", "2026-10-18--12-14-18-885__000",
         "2026-10-29--03-10-50-289__000"]


@torch.no_grad()
def collect(m: Student, units, frames_per_unit, every=3):
    """Closed loop over the first even frames of each unit (as eval_student runs it); every `every`-th frame keeps the
    inputs of vis_fc.0 and temporal.0 and every frame its hidden state."""
    got = {"vis": [], "tmp": []}
    keep = [False]
    def grab(name):
        def f(mod, i, o):           # a forward hook's return value would replace the output
            if keep[0]:
                got[name].append(i[0].clone())
        return f

    h1 = m.vis_fc[0].register_forward_hook(grab("vis"))
    h2 = m.temporal[0].register_forward_hook(grab("tmp"))
    hid = []
    for u in units:
        d = ROOT / "data/ep_label" / u
        assert not any(u.startswith(e) for e in EVAL_ROUTES), u
        lab = np.load(d / "labels.npz")
        frames = load_frames(d)
        pulse = lab["pulse"]
        n = min(len(lab["big"]), 200 + 2 * frames_per_unit)
        hist = np.zeros(((n + 1) // 2, 512), np.float32)
        for i, t in enumerate(range(4, n - n % 2, 2)):
            j = t // 2
            road = torch.from_numpy(np.concatenate([frames[j - 2, 0], frames[j, 0]])[None]).float()
            wide = torch.from_numpy(np.concatenate([frames[j - 2, 1], frames[j, 1]])[None]).float()
            des = np.zeros((100, 8), np.float32)
            lo = max(0, t - 99)
            des[100 - (t + 1 - lo):] = pulse[lo:t + 1]
            idx = np.arange(t - 96, t, 4) // 2
            fb = np.zeros((24, 512), np.float32)
            fb[idx >= 0] = hist[idx[idx >= 0]]
            keep[0] = t >= 100 and i % every == 0
            o = m(road, wide, torch.from_numpy(des.reshape(25, 4, 8).max(1)[None]), torch.from_numpy(fb[None]),
                  torch.tensor([[1.0, 0.0]]))
            hist[j] = o["out_hidden"][0].numpy()
        hid.append(torch.from_numpy(hist[50:]))
        print(f"{u}: {len(got['vis'])} samples", flush=True)
    h1.remove()
    h2.remove()
    return torch.cat(got["vis"]), torch.cat(got["tmp"]), torch.cat(hid)


def top_eig(c: torch.Tensor, r: int) -> torch.Tensor:
    """[r, n]: the top-r eigenvectors of a symmetric matrix, as rows."""
    w, v = torch.linalg.eigh(c.double())
    return v[:, w.argsort(descending=True)[:r]].T.float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hist-rank", type=int, default=128)
    ap.add_argument("--vis-rank", type=int, default=256)
    ap.add_argument("--frames", type=int, default=1500, help="even frames per unit")
    a = ap.parse_args()
    torch.set_num_threads(4)
    ck = torch.load(a.ckpt, map_location="cpu")
    arch = ck.get("arch", {})
    assert not ck.get("fused") and not arch.get("hist_rank") and not arch.get("vis_rank")
    m = load_student(ck)
    xv, xt, hid = collect(m, UNITS, a.frames)
    print(f"{len(xv)} samples, {len(hid)} hidden states", flush=True)
    st = {k: v.clone() for k, v in ck["model"].items()}
    T = m.policy_tokens - 1
    H = m.summarizer[-1].out_features
    yv = xv @ st["vis_fc.0.weight"].T
    ev, cv = torch.linalg.eigh((yv.T @ yv).double())
    eh, ch = torch.linalg.eigh((hid.T @ hid).double())
    for r in (32, 64, 96, 128, 192, 256, 384):   # residual energy share by rank
        print(f"rank {r:3d}: hidden residual {(eh[:-r].sum() / eh.sum()).sqrt().item():.4f}, "
              f"vis_fc response residual {(ev[:-r].sum() / ev.sum()).sqrt().item():.4f}", flush=True)

    if a.hist_rank:
        P = top_eig(hid.T @ hid / len(hid), a.hist_rank)                     # [r, 512]
        W = st.pop("temporal.0.weight")                                        # [512, T*H + rest]
        Wt = W[:, :T * H].reshape(W.shape[0], T, H)
        A = torch.einsum("oth,rh->otr", Wt, P).reshape(W.shape[0], T * a.hist_rank)
        st["hist_proj.weight"] = P
        st["temporal.0.weight"] = torch.cat([A, W[:, T * H:]], 1)
        y = xt @ W.T
        h = xt[:, :T * H].reshape(len(xt), T, H) @ P.T
        y2 = torch.cat([h.flatten(1), xt[:, T * H:]], 1) @ st["temporal.0.weight"].T
        e = ((hid @ P.T @ P - hid).norm() / hid.norm()).item()
        print(f"hist_rank {a.hist_rank}: hidden-state residual {e:.4f}, temporal.0 response error "
              f"{((y2 - y).norm() / y.norm()).item():.4f}", flush=True)

    if a.vis_rank:
        W, b = st.pop("vis_fc.0.weight"), st.pop("vis_fc.0.bias")              # [1024, 4096]
        y = xv @ W.T
        V = top_eig(y.T @ y / len(y), a.vis_rank).T                           # [1024, r]
        st["vis_fc.0.weight"] = V.T @ W
        st["vis_fc.1.weight"], st["vis_fc.1.bias"] = V, b
        y2 = xv @ st["vis_fc.0.weight"].T @ V.T
        print(f"vis_rank {a.vis_rank}: vis_fc.0 response error {((y2 - y).norm() / (y + b).norm()).item():.4f}", flush=True)

    arch = {**arch, "hist_rank": a.hist_rank, "vis_rank": a.vis_rank}
    s = Student(**{**arch, "pretrained": False})
    if ck.get("qat") and isinstance(s.backbone, R18Backbone):
        s.backbone.fuse_bn()
    s.load_state_dict(st)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": st, "arch": arch, "epoch": ck.get("epoch"), "fused": False, "qat": ck.get("qat"),
                "lowrank_from": str(a.ckpt)}, a.out)
    w = lambda k: st[k].numel()
    print(f"saved {a.out}: temporal.0 {ck['model']['temporal.0.weight'].numel() / 1e6:.2f}M -> "
          f"{(w('temporal.0.weight') + (w('hist_proj.weight') if a.hist_rank else 0)) / 1e6:.2f}M weights, vis_fc "
          f"{ck['model']['vis_fc.0.weight'].numel() / 1e6:.2f}M -> "
          f"{(w('vis_fc.0.weight') + (w('vis_fc.1.weight') if a.vis_rank else 0)) / 1e6:.2f}M", flush=True)


if __name__ == "__main__":
    main()
