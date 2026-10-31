"""Lane overlays on the road camera's model frame: a student (closed loop), the shipped board model (recorded), the
big teacher and LD (inner line edges, the lanes reference of eval_student.py), on frames picked by kind.

The student runs closed loop over the unit up to the last picked frame (its outputs are cached in
<out>/cache_<unit>_<ckpt run>.npz). Lanes are projected with the model frame's own intrinsics (medmodel: f 910,
centre 256, 47.6 on the 512x256 Y plane; the warp already applied calibration and camera mount), at the lane's own
z (LD: the mount height, flat road).

  kinds   straight  low lane error frames with |teacher plan y at 2 s| < 0.1 m (median of the student's error)
          curve     |teacher plan y at 2 s| > 1 m, desire none
          change    desire != none (lane change)
          worst     the student's largest lane error vs LD
  per frame the title gives the frame's lanes 1-2 error vs LD at 6-40 m (cm) for the student and the shipped model.

usage: .venv/bin/python scripts/viz_lanes.py --ckpt ckpt/mac_base_aug10/last.pt --units <unit> [<unit> ...]
           [--kinds straight,curve,change,worst] [--out outputs/lane_overlay]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import HEADS, POLICY_HEADS, Student  # noqa: E402
from train import BIG  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
T_IDXS = np.array([10.0 * (i / 32) ** 2 for i in range(33)])
LANE_X = (X_IDXS >= 6) & (X_IDXS <= 40)
F, CX, CY = 910.0, 256.0, 47.6
SCALE = 2   # 512x256 Y plane drawn at 1024x512
COLORS = {"LD": (60, 230, 60), "teacher": (255, 210, 0), "shipped": (40, 170, 255), "student": (255, 50, 50)}


@torch.no_grad()
def run_lanes(ckpt: Path, d: Path, lab, upto: int) -> dict:
    """Closed-loop student over the unit's even frames up to frame `upto`: lanes [N,4,33,2] (y, z) and lane_prob [N,4]."""
    ck = torch.load(ckpt, map_location="cpu")
    m = Student(**ck.get("arch", {})).eval()
    m.load_state_dict(ck["model"])
    frames = np.load(d / "frames.npy", mmap_mode="r")
    n = len(lab["big"])
    even = np.arange(4, min(n - n % 2, upto + 2), 2)
    hist = np.zeros(((n + 1) // 2, 512), np.float32)
    lanes, prob = [], []
    for t in even:
        j = t // 2
        road = torch.from_numpy(np.concatenate([frames[j - 2, 0], frames[j, 0]])[None]).float()
        wide = torch.from_numpy(np.concatenate([frames[j - 2, 1], frames[j, 1]])[None]).float()
        des = np.zeros((100, 8), np.float32)
        lo = max(0, t - 99)
        des[100 - (t + 1 - lo):] = lab["pulse"][lo:t + 1]
        idx = np.arange(t - 96, t, 4) // 2
        fb = np.zeros((24, 512), np.float32)
        fb[idx >= 0] = hist[idx[idx >= 0]]
        o = m(road, wide, torch.from_numpy(des.reshape(25, 4, 8).max(1)[None]), torch.from_numpy(fb[None]),
              torch.tensor([[1.0, 0.0]]))
        hist[j] = o["out_hidden"][0].numpy()
        y = m.outputs(o)
        lanes.append(y["lanes"][0, :264].numpy().reshape(4, 33, 2))
        prob.append(1 / (1 + np.exp(-y["lane_prob"][0, 1::2].numpy())))
    return {"even": even, "lanes": np.stack(lanes), "prob": np.stack(prob)}


def lane_err(lanes_y: np.ndarray, ld: np.ndarray) -> np.ndarray:
    """Per frame mean |lanes 1, 2 - LD| at 6-40 m where LD saw them (cm); NaN without LD."""
    ok = np.isfinite(ld[:, 1:3]) & LANE_X[None, None]
    e = np.where(ok, np.abs(lanes_y[:, 1:3] - np.nan_to_num(ld[:, 1:3])), 0).sum((1, 2))
    c = ok.sum((1, 2))
    return np.where(c > 5, e / np.maximum(c, 1) * 100, np.nan)


def rgb(frame6: np.ndarray) -> Image.Image:
    """YUV6 [6,128,256] -> RGB 512x256 (BT.601 full range)."""
    y = np.empty((256, 512), np.float32)
    y[0::2, 0::2], y[1::2, 0::2], y[0::2, 1::2], y[1::2, 1::2] = frame6[:4]   # the order that gives a smooth image
    u = np.repeat(np.repeat(frame6[4].astype(np.float32) - 128, 2, 0), 2, 1)
    v = np.repeat(np.repeat(frame6[5].astype(np.float32) - 128, 2, 0), 2, 1)
    out = np.stack([y + 1.402 * v, y - 0.344 * u - 0.714 * v, y + 1.772 * u], -1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def project(y: np.ndarray, z: np.ndarray, xmax: float = 100.0) -> list[tuple[float, float]]:
    pts = []
    for x, yy, zz in zip(X_IDXS, y, z):
        if x < 4 or x > xmax or not (np.isfinite(yy) and np.isfinite(zz)):
            continue
        pts.append((SCALE * (F * yy / x + CX), SCALE * (F * zz / x + CY)))
    return pts


def draw_lanes(dr: ImageDraw.ImageDraw, lanes_yz: np.ndarray, prob, color, width: int, dash: bool = False):
    for i in range(4):
        p = 1.0 if prob is None else float(prob[i])
        if p < 0.15:
            continue
        pts = project(lanes_yz[i, :, 0], lanes_yz[i, :, 1])
        w = max(1, round(width * (0.4 + 0.6 * p)))
        if dash:
            for a, b in zip(pts[::2], pts[1::2]):
                dr.line([a, b], fill=color, width=w)
        elif len(pts) > 1:
            dr.line(pts, fill=color, width=w, joint="curve")


def panel(d: Path, lab, t: int, s: dict, kind: str, font) -> Image.Image:
    frames = np.load(d / "frames.npy", mmap_mode="r")
    img = rgb(np.asarray(frames[t // 2, 0])).resize((512 * SCALE, 256 * SCALE), Image.BILINEAR)
    dr = ImageDraw.Draw(img)
    k = np.searchsorted(s["even"], t)
    a, b = BIG["lanes"]
    teacher = lab["big"][t, a:b][:264].reshape(4, 33, 2)
    h = float(lab["mount"][t][1])
    ld = np.stack([lab["ld_lanes"][t], np.full((4, 33), h)], -1)
    draw_lanes(dr, teacher, None, COLORS["teacher"], 3, dash=True)
    draw_lanes(dr, lab["rec_lanes"][t], lab["rec_lane_probabilities"][t], COLORS["shipped"], 5)
    draw_lanes(dr, s["lanes"][k], s["prob"][k], COLORS["student"], 5)
    draw_lanes(dr, ld, None, COLORS["LD"], 3, dash=True)
    es = lane_err(s["lanes"][k:k + 1, :, :, 0], lab["ld_lanes"][t:t + 1])[0]
    eb = lane_err(lab["rec_lanes"][t:t + 1, :, :, 0], lab["ld_lanes"][t:t + 1])[0]
    a, b = BIG["plan"]
    ty2 = np.interp(2.0, T_IDXS, lab["big"][t, a:b][:495].reshape(33, 15)[:, 1])
    title = (f"[{kind}] {d.name[:20]}  t={t / 20:.0f}s  v={lab['v_ego'][t] * 3.6:.0f} km/h  teacher y@2s {ty2:+.2f} m  "
             f"desire {int(lab['desire'][t])}   lane err vs LD: student {es:.0f} cm, shipped {eb:.0f} cm")
    dr.rectangle([0, 0, img.width, 30], fill=(0, 0, 0))
    dr.text((8, 5), title, fill=(255, 255, 255), font=font)
    return img


def legend(width: int, font) -> Image.Image:
    img = Image.new("RGB", (width, 34), (0, 0, 0))
    dr = ImageDraw.Draw(img)
    x = 10
    for name, label in [("student", "student (width = prob)"), ("shipped", "shipped"),
                        ("teacher", "big teacher (dashed)"), ("LD", "LD reference (dashed)")]:
        dr.rectangle([x, 10, x + 30, 22], fill=COLORS[name])
        dr.text((x + 38, 7), label, fill=(255, 255, 255), font=font)
        x += 60 + int(dr.textlength(label, font=font))
    return img


def pick(kind: str, lab, even, err_s, rng) -> int | None:
    a, b = BIG["plan"]
    big = lab["big"][even]
    ty2 = np.array([np.interp(2.0, T_IDXS, p[:, 1]) for p in big[:, a:b][:, :495].reshape(-1, 33, 15)])
    ok = (lab["v_ego"][even] > 5) & np.isfinite(err_s) & (np.arange(len(even)) > 100)
    des = lab["desire"][even]
    if kind == "straight":
        c = ok & (np.abs(ty2) < 0.1) & (des == 0)
        if not c.any():
            return None
        med = np.median(err_s[c])
        return int(even[np.flatnonzero(c)[np.argmin(np.abs(err_s[c] - med))]])
    if kind == "curve":
        c = ok & (np.abs(ty2) > 1.0) & (des == 0)
    elif kind == "change":
        c = ok & (des != 0)
    elif kind == "worst":
        c = ok
        return int(even[np.flatnonzero(c)[np.argmax(err_s[c])]]) if c.any() else None
    else:
        raise ValueError(kind)
    if not c.any():
        return None
    return int(even[rng.choice(np.flatnonzero(c))])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--units", nargs="+", required=True)
    ap.add_argument("--kinds", default="straight,curve,change,worst")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/lane_overlay")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    font = ImageFont.load_default(size=17)
    rng = np.random.default_rng(args.seed)
    for u in args.units:
        d = ROOT / "data/ep_label" / u
        lab = np.load(d / "labels.npz")
        cache = args.out / f"cache_{u}_{args.ckpt.parent.name}.npz"
        if cache.exists():
            s = dict(np.load(cache))
        else:
            s = run_lanes(args.ckpt, d, lab, len(lab["big"]))
            np.savez(cache, **s)
        err_s = lane_err(s["lanes"][..., 0], lab["ld_lanes"][s["even"]])
        err_b = lane_err(lab["rec_lanes"][s["even"]][..., 0], lab["ld_lanes"][s["even"]])
        ok = np.isfinite(err_s) & np.isfinite(err_b) & (lab["v_ego"][s["even"]] > 5)
        print(f"{u}: lane err vs LD, student {err_s[ok].mean():.1f} cm, shipped {err_b[ok].mean():.1f} cm "
              f"({ok.sum()} frames)")
        for kind in args.kinds.split(","):
            t = pick(kind, lab, s["even"], err_s, rng)
            if t is None:
                print(f"  {kind}: no frame")
                continue
            img = panel(d, lab, t, s, kind, font)
            sheet = Image.new("RGB", (img.width, img.height + 34))
            sheet.paste(img, (0, 0))
            sheet.paste(legend(img.width, font), (0, img.height))
            f = args.out / f"{u}_{kind}_{t}.png"
            sheet.save(f)
            print(f"  {kind}: frame {t} -> {f}")


if __name__ == "__main__":
    main()
