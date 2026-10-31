"""Side-by-side driving video: the shipped model (its outputs as the board recorded them) and a student as the NPU
runs it (S8 weights, eval_student --s8; = the board to 1e-4 since the fixed grid and --extremes builds), drawn on the
road camera's model frame (the 512x256 warped image both models see), plus a bird's-eye view of both with the big
teacher as reference and a lead-probability strip. 10 fps (the stored frames are the even ones, as the models run).

  camera panels   path (+-0.9 m at the plan's height), lane lines (width = probability), road edges, the lead
                  (chevron when probability > 0.5, outline 0.2-0.5) with its probability and distance
  bird's eye      x 0-120 m to the right, lateral +-8 m (right of the car down); both models' paths, lanes and
                  leads; the teacher's path and lead dashed yellow
  strip           lead probability over the clip: shipped, student, teacher; the cursor is the current frame

The student runs closed loop from 5 s before the clip (its history buffer holds its own hidden states of the last
4.8 s, which depend on the frames only, so this equals a run over the whole unit).

usage: .venv/bin/python scripts/viz_compare.py --ckpt ckpt/mac_lrc/ema.pt --unit 2026-10-05--16-28-44-410__000
           --start 65 --dur 25 [--name LRC] [--latency "11.2 ms"] [--title "line 1|line 2"]
           [--out outputs/viz_compare/<unit>_<start>s.mp4]"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_student as E  # noqa: E402
from student import load_student, quant_sim  # noqa: E402
from train import BIG  # noqa: E402
from viz_lanes import CX, CY, F, rgb  # noqa: E402
from zframes import load_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
SHIP, STUD, TEACH = (60, 170, 255), (255, 110, 50), (255, 215, 0)
S = 1.5                                    # camera panel scale (768x384)
W, HB, HCAM, HBEV, HSTRIP = 1536, 40, 384, 230, 110
XMAX = 120                                 # bird's-eye range (m)
NAME = "LRC"


@torch.no_grad()
def run_student(ckpt: Path, d: Path, lab, even: np.ndarray) -> dict:
    m = quant_sim(load_student(torch.load(ckpt, map_location="cpu", weights_only=False)).fuse())
    frames = load_frames(d)
    hist = np.zeros(((len(lab["big"]) + 1) // 2, 512), np.float32)
    out = {k: [] for k in ("plan", "lanes", "lane_prob", "road_edges", "lead", "lead_prob")}
    for t in range(max(4, int(even[0]) - 100), int(even[-1]) + 1, 2):
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
        if t < even[0]:
            continue
        y = {k: v[0].numpy() for k, v in m.outputs(o).items()}
        out["plan"].append(y["plan"][:495].reshape(33, 15)[:, :3])
        out["lanes"].append(y["lanes"][:264].reshape(4, 33, 2))
        out["lane_prob"].append(1 / (1 + np.exp(-y["lane_prob"][1::2])))
        out["road_edges"].append(y["road_edges"][:132].reshape(2, 33, 2))
        lead = y["lead"][:72].reshape(3, 6, 4)
        out["lead"].append(lead[0, 0, :2])
        out["lead_prob"].append(1 / (1 + np.exp(-y["lead_prob"][0])))
    return {k: np.stack(v) for k, v in out.items()}


def shipped(lab, even) -> dict:
    return {"plan": lab["rec_plan"][even], "lanes": lab["rec_lanes"][even], "lane_prob": lab["rec_lane_probabilities"][even],
            "road_edges": lab["rec_road_edges"][even],
            "lead": np.stack([lab["rec_lead_x"][even], lab["rec_lead_y"][even]], -1), "lead_prob": lab["rec_lead_probability"][even]}


def teacher(lab, even) -> dict:
    big = lab["big"][even]
    a, b = BIG["plan"]
    p = big[:, a:b][:, :495].reshape(-1, 33, 15)[:, :, :3]
    a, b = BIG["lead"]
    lead = big[:, a:a + 72].reshape(-1, 3, 6, 4)[:, 0, 0, :2]
    return {"plan": p, "lead": lead, "lead_prob": E.from_layout(big, BIG)["lead_prob"]}


def proj(x, y, z):
    return S * (F * y / x + CX), S * (F * z / x + CY)


def camera(frame6, q: dict, k: int, h: float, color, tq: dict) -> Image.Image:
    img = rgb(frame6).resize((int(512 * S), int(256 * S)), Image.BILINEAR).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    p = q["plan"][k]
    ok = (p[:, 0] > 3) & (p[:, 0] < 120)
    left = [proj(x, y - 0.9, z + h) for x, y, z in p[ok]]
    right = [proj(x, y + 0.9, z + h) for x, y, z in p[ok]]
    if len(left) > 1:
        dr.polygon(left + right[::-1], fill=(*color, 65), outline=(*color, 230))
    tp = tq["plan"][k]
    pts = [proj(x, y, z + h) for x, y, z in tp if 3 < x < 120]
    for a, b in zip(pts[::2], pts[1::2]):                          # teacher path, dashed
        dr.line([a, b], fill=(*TEACH, 200), width=2)
    xs = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
    sel = (xs > 4) & (xs < 100)
    for i in range(4):
        pr = float(q["lane_prob"][k][i])
        if pr < 0.15:
            continue
        pts = [proj(x, y, z) for x, (y, z) in zip(xs[sel], q["lanes"][k][i][sel])]
        dr.line(pts, fill=(255, 255, 255, int(255 * pr)), width=max(1, round(5 * pr)), joint="curve")
    for i in range(2):
        pts = [proj(x, y, z) for x, (y, z) in zip(xs[sel], q["road_edges"][k][i][sel])]
        dr.line(pts, fill=(255, 60, 60, 200), width=3, joint="curve")
    lx, ly = q["lead"][k]
    lp = float(q["lead_prob"][k])
    if lp > 0.2 and 2 < lx < 150:
        u, v = proj(lx, ly, h)
        sz = float(np.clip(12 * 30 / lx, 8, 16)) * S
        tri = [(u - sz, v + sz * 0.5), (u + sz, v + sz * 0.5), (u, v - sz * 0.4)]
        dr.polygon(tri, fill=(*color, 230) if lp > 0.5 else None, outline=(255, 255, 255, 255), width=2)
    img = Image.alpha_composite(img, ov).convert("RGB")
    return img


def bev(qs: list, k: int, tq: dict, font) -> Image.Image:
    img = Image.new("RGB", (W, HBEV), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    px, py, y0 = (W - 60) / XMAX, HBEV / 16, HBEV / 2          # XMAX m across, +-8 m lateral
    pt = lambda x, y: (40 + x * px, y0 + y * py)
    for x in range(0, XMAX + 1, 10):
        dr.line([pt(x, -8), pt(x, 8)], fill=(45, 45, 52))
        dr.text((pt(x, 8)[0] + 3 - (40 if x == XMAX else 0), HBEV - 18), f"{x} m", fill=(120, 120, 130), font=font)
    xs = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
    sel = xs <= XMAX
    for q, color in qs:
        for i in range(4):
            pr = float(q["lane_prob"][k][i])
            if pr >= 0.15:
                dr.line([pt(x, y) for x, y in zip(xs[sel], q["lanes"][k][i][sel, 0])], fill=tuple(int(c * (0.35 + 0.65 * pr)) for c in color), width=2)
        p = q["plan"][k]
        p = p[p[:, 0] <= XMAX]
        dr.line([pt(x, y) for x, y, _ in p], fill=color, width=5)
    tp = tq["plan"][k]
    tp = [pt(x, y) for x, y, _ in tp if x <= XMAX]
    for a, b in zip(tp[::2], tp[1::2]):
        dr.line([a, b], fill=TEACH, width=2)
    car = pt(0, 0)
    dr.rectangle([car[0] - 30, car[1] - 12, car[0], car[1] + 12], fill=(200, 200, 200))
    for q, color in [*qs, (tq, TEACH)]:
        lx, ly = q["lead"][k]
        lp = float(q["lead_prob"][k])
        if lp > 0.2 and 0 < lx < XMAX:
            u, v = pt(lx, ly)
            r = 9
            if q is tq:
                dr.ellipse([u - r - 3, v - r - 3, u + r + 3, v + r + 3], outline=TEACH, width=2)
            else:
                dr.rectangle([u - r, v - r, u + r, v + r], fill=color if lp > 0.5 else None, outline=color, width=2)
    dr.text((10, 6), "위에서 본 모습 (전방 →, 오른쪽 ↓)   실선: 경로   흐린 선: 차선   □ 앞차   노란 점선/○: teacher(참고)",
            fill=(200, 200, 205), font=font)
    return img


def strip(qs: list, tq: dict, k: int, font) -> Image.Image:
    img = Image.new("RGB", (W, HSTRIP), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    n = len(tq["lead_prob"])
    x0, x1, ya, yb = 150, W - 20, 12, HSTRIP - 14
    X = lambda i: x0 + (x1 - x0) * i / max(n - 1, 1)
    Y = lambda p: yb - (yb - ya) * p
    dr.line([(x0, Y(0.5)), (x1, Y(0.5))], fill=(70, 70, 80))
    dr.text((10, Y(1) - 4), "앞차 확률", fill=(200, 200, 205), font=font)
    dr.text((10, Y(0.5) - 10), "0.5", fill=(120, 120, 130), font=font)
    for i, (name, color) in enumerate([("출시", qs[0][1]), (NAME, qs[1][1]), ("teacher", TEACH)]):
        dr.text((10 + 45 * i, yb - 12), name, fill=color, font=font)
    for q, color, w in [(tq, TEACH, 1), *[(q, c, 2) for q, c in qs]]:
        dr.line([(X(i), Y(float(p))) for i, p in enumerate(q["lead_prob"])], fill=color, width=w)
    dr.line([(X(k), ya - 6), (X(k), yb + 6)], fill=(255, 230, 80), width=2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--unit", required=True)
    ap.add_argument("--start", type=float, required=True, help="seconds into the unit")
    ap.add_argument("--dur", type=float, default=25)
    ap.add_argument("--name", default="LRC")
    ap.add_argument("--latency", default="11.2 ms")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--title", help="a 2 s title card before the clip (lines split by |)")
    args = ap.parse_args()
    global NAME
    NAME = args.name
    d = ROOT / "data/ep_label" / args.unit
    lab = np.load(d / "labels.npz")
    t0 = int(args.start * 20) // 2 * 2
    even = np.arange(max(t0, 4), min(t0 + int(args.dur * 20), len(lab["big"]) - 1), 2)
    stud, ship, tq = run_student(args.ckpt, d, lab, even), shipped(lab, even), teacher(lab, even)
    out = args.out or ROOT / f"outputs/viz_compare/{args.unit}_{args.start:g}s.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    f18, f15 = ImageFont.truetype(FONT, 18), ImageFont.truetype(FONT, 15)
    H = HB + HCAM + HBEV + HSTRIP
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", "10", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                          stdin=subprocess.PIPE)
    frames = load_frames(d)
    if args.title:
        card = Image.new("RGB", (W, H), (12, 12, 16))
        dr = ImageDraw.Draw(card)
        for i, line in enumerate(args.title.split("|")):
            dr.text((80, H // 2 - 60 + 48 * i), line.strip(), fill=(235, 235, 240) if i == 0 else (170, 170, 180),
                    font=ImageFont.truetype(FONT, 34 if i == 0 else 24))
        for _ in range(20):
            ff.stdin.write(card.tobytes())
    h = float(lab["mount"][even[0]][1])
    for k, t in enumerate(even):
        canvas = Image.new("RGB", (W, H), (0, 0, 0))
        f6 = np.asarray(frames[t // 2, 0])
        for col, (q, color, title) in enumerate([(ship, SHIP, "출시 모델 (보드 기록, 14.0 ms)"),
                                                 (stud, STUD, f"{args.name} (NPU 연산, {args.latency})")]):
            canvas.paste(camera(f6, q, k, h, color, tq), (col * W // 2, HB))
            dr = ImageDraw.Draw(canvas)
            lp, (lx, _) = float(q["lead_prob"][k]), q["lead"][k]
            y2 = np.interp(2.0, E.T_IDXS, q["plan"][k][:, 1])
            dr.rectangle([col * W // 2, HB, (col + 1) * W // 2 - 1, HB + 30], fill=(0, 0, 0))
            dr.text((col * W // 2 + 10, HB + 5), f"{title}   앞차 {lp:.2f} · {lx:.0f} m   경로 y@2s {y2:+.2f} m",
                    fill=color, font=f18)
        dr = ImageDraw.Draw(canvas)
        tlp, (tlx, _) = float(tq["lead_prob"][k]), tq["lead"][k]
        ty2 = np.interp(2.0, E.T_IDXS, tq["plan"][k][:, 1])
        dr.text((10, 10), f"{args.unit}   t = {t / 20:5.1f} s   {lab['v_ego'][t] * 3.6:3.0f} km/h      "
                          f"teacher(참고): 앞차 {tlp:.2f} · {tlx:.0f} m, 경로 y@2s {ty2:+.2f} m", fill=(230, 230, 235), font=f18)
        canvas.paste(bev([(ship, SHIP), (stud, STUD)], k, tq, f15), (0, HB + HCAM))
        canvas.paste(strip([(ship, SHIP), (stud, STUD)], tq, k, f15), (0, HB + HCAM + HBEV))
        ff.stdin.write(canvas.tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"{out}: {len(even)} frames ({len(even) / 10:.1f} s)")


if __name__ == "__main__":
    main()
