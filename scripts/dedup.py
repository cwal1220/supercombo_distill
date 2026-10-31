"""Find places driven more than once in the edgepilot routes and drop the repeats driven in the same conditions.

The recordings carry no GPS, so places are matched by appearance:
  embed   one keyframe per second (H.264 GOP = 1 s) of every route: an ImageNet ResNet-50 (torchvision) place
          descriptor of the road above the hood (GeM-pooled layer3 and layer4 features, L2-normalised), the mean
          luma of that crop, and the local time of day (the route name is the board's local start time).
  match   for every pair of routes (and a route against itself more than 3 min apart), seconds whose
          descriptors are similar (cosine > --sim) and that line up in the same order for >= --run seconds
          (the matched second moves forward 0..3 s per second) are the same place.
  decide  two passes of a place are kept both when their conditions differ: another time-of-day class
          (day 07-17 h, twilight 05-07 / 17-19 h, night 19-05 h) or another luma class (dark < 45, dim < 90,
          bright). Otherwise one is kept: an eval route's pass first, then one already labelled, then the
          earliest route; eval routes keep every pass. A segment whose seconds are mostly dropped (>= --seg-drop)
          is not labelled (ep_label.py); the dropped seconds of the segments that stay are masked out of training
          (train.py, dedup.json "dropped_ns": [start, end) capture-time spans per route).

  verify  each matched run is kept only if its frames share a rigid scene (SIFT + RANSAC, see accept()):
          global descriptors alone pair look-alike night streets and tunnels.

usage: scripts/py scripts/dedup.py embed|match|thumbs|verify|decide [options]
Writes data/dedup/: emb_<route>.npz, thumb_<route>.npy, matches.npy, verify_scores.npy, dedup.json
(dropped seconds and segments).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ep"))
sys.path.insert(0, str(HERE))
import recording_reader as rr  # noqa: E402

ROOT = HERE.parent
DRIVES = ROOT / "data/edgepilot_drives/experiments"
OUT = ROOT / "data/dedup"
CROP = (60, 560)          # rows of the 720p frame: under the sky band, above the hood
EMB_W, EMB_H = 448, 192   # layer4 grid 14 x 6


def route_start(name: str) -> datetime:
    return datetime.strptime(name[:20], "%Y-%m-%d--%H-%M-%S")


def tod_class(hour: float) -> str:
    if 7 <= hour < 17:
        return "day"
    if 5 <= hour < 7 or 17 <= hour < 19:
        return "twilight"
    return "night"


def luma_class(y: float) -> str:
    return "dark" if y < 45 else ("dim" if y < 90 else "bright")


def keyframes(seg_dir: Path):
    import av
    info = rr.read_segment_index(seg_dir)
    key = info.frames[(info.frames["flags"] & 1) == 1]
    with av.open(str(rr.segment_video(seg_dir)), format="h264") as c:
        st = c.streams.video[0]
        st.codec_context.skip_frame = "NONKEY"
        for rec, f in zip(key, c.decode(st)):
            yield rec, f.to_ndarray(format="gray")


def embed_route(route: Path, model, device):
    import torch
    import torch.nn.functional as F
    dst = OUT / f"emb_{route.name}.npz"
    if dst.exists():
        return
    rows = {"t_ns": [], "seg": [], "luma": [], "emb": []}
    seg_dirs = sorted(d for d in (route / "segments").iterdir() if (d / "frames.bin").exists() and rr.segment_video(d))
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    for d in seg_dirs:
        batch, meta = [], []
        for rec, y in keyframes(d):
            crop = y[CROP[0]:CROP[1]]
            meta.append((int(rec["capture_timestamp_ns"]), int(d.name), float(crop.mean())))
            batch.append(crop)
        if not batch:
            continue
        with torch.no_grad():
            x = torch.from_numpy(np.stack(batch)).to(device).float().div(255).unsqueeze(1)
            x = F.interpolate(x, size=(EMB_H, EMB_W), mode="area").repeat(1, 3, 1, 1)
            x = (x - mean) / std
            m = model
            x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
            l3 = m.layer3(m.layer2(m.layer1(x)))
            l4 = m.layer4(l3)
            gem = lambda f: F.normalize(f.clamp(min=1e-6).pow(3).mean((2, 3)).pow(1 / 3), dim=-1)
            desc = F.normalize(torch.cat([gem(l3), gem(l4)], -1), dim=-1).half().cpu().numpy()
        for (t, s, l), e in zip(meta, desc):
            rows["t_ns"].append(t)
            rows["seg"].append(s)
            rows["luma"].append(l)
            rows["emb"].append(e)
    np.savez(dst, **{k: np.array(v) for k, v in rows.items()})
    print(f"{route.name}: {len(rows['t_ns'])} keyframes", flush=True)


def cmd_embed(args):
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    import torchvision
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2).cuda().eval()
    for r in sorted(DRIVES.iterdir()):
        if (r / "segments").exists():
            embed_route(r, model, "cuda")


THUMB_W, THUMB_H = 480, 270


def cmd_thumbs(args):
    """Every route's keyframes as 480x270 grey thumbnails (the verify stage's input), in emb order."""
    import cv2
    for r in sorted(DRIVES.iterdir()):
        e = OUT / f"emb_{r.name}.npz"
        dst = OUT / f"thumb_{r.name}.npy"
        if not e.exists() or dst.exists():
            continue
        t_ns = np.load(e)["t_ns"]
        out = np.lib.format.open_memmap(OUT / f"thumb_{r.name}.tmp.npy", mode="w+", dtype=np.uint8,
                                        shape=(len(t_ns), THUMB_H, THUMB_W))
        k = 0
        seg_dirs = sorted(d for d in (r / "segments").iterdir() if (d / "frames.bin").exists() and rr.segment_video(d))
        for d in seg_dirs:
            for rec, y in keyframes(d):
                assert int(rec["capture_timestamp_ns"]) == int(t_ns[k])
                out[k] = cv2.resize(y, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)
                k += 1
        out.flush()
        del out
        (OUT / f"thumb_{r.name}.tmp.npy").rename(dst)
        print(f"{r.name}: {k} thumbnails", flush=True)


def accept(scores: np.ndarray) -> bool:
    """A run is the same place when 2 of its 3 frame pairs have >= 15 RANSAC inliers or one has >= 30. Hand-checked
    contact sheets: different places score 0-11 (look-alike tunnels and night streets up to ~14); same places 27-80,
    but night and motion blur often leave one or two pairs of a true repeat at 12-20."""
    return bool((scores >= 15).sum() >= 2 or scores.max() >= 30)


def cmd_verify(args):
    """Score the matched runs by how much of a rigid scene their frame pairs share: SIFT + ratio test + RANSAC
    fundamental-matrix inliers for 3 pairs (15 %, 50 %, 85 % of the run); see accept() for the rule. Matches that
    stay at one pixel (windshield, hood) are dropped first."""
    import cv2
    m = np.load(OUT / "matches.npy", allow_pickle=True)
    thumbs = {}
    sift = cv2.SIFT_create(nfeatures=1500)
    cache = {}

    def feats(route, i):
        key = (route, i)
        if key not in cache:
            if route not in thumbs:
                thumbs[route] = np.load(OUT / f"thumb_{route}.npy", mmap_mode="r")
            img = np.ascontiguousarray(thumbs[route][i][:200])          # well above the hood
            cache[key] = sift.detectAndCompute(img, None)
        return cache[key]

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    keep, scores = [], []
    for n, (a, i0, i1, b, j0, j1, sim) in enumerate(m):
        res = []
        for f in (0.15, 0.5, 0.85):
            i = int(round(i0 + f * (i1 - i0)))
            j = int(round(j0 + f * (j1 - j0)))
            best = 0
            for dj in (-1, 0, 1):     # the run's second-to-second alignment is approximate
                jj = j + dj
                if jj < 0:
                    continue
                try:
                    (ka, da), (kb, db) = feats(a, i), feats(b, jj)
                except IndexError:
                    continue
                if da is None or db is None or len(ka) < 8 or len(kb) < 8:
                    continue
                pairs = matcher.knnMatch(da, db, k=2)
                good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
                if len(good) < 8:
                    continue
                pa = np.float32([ka[g.queryIdx].pt for g in good])
                pb = np.float32([kb[g.trainIdx].pt for g in good])
                moved = np.linalg.norm(pa - pb, axis=1) >= 3.0   # windshield and hood artefacts stay at one pixel
                pa, pb = pa[moved], pb[moved]
                if len(pa) < 8:
                    continue
                _, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
                best = max(best, int(mask.sum()) if mask is not None else 0)
            res.append(best)
        scores.append(res)
        if accept(np.array(res)):
            keep.append(m[n])
        if len(cache) > 20000:
            cache.clear()
        if n % 200 == 0:
            print(f"{n}/{len(m)} runs, {len(keep)} verified", flush=True)
    np.save(OUT / "matches_verified.npy", np.array(keep, dtype=object), allow_pickle=True)
    np.save(OUT / "verify_scores.npy", np.array(scores))
    print(f"{len(keep)} of {len(m)} runs verified ({sum(k[2] - k[1] + 1 for k in keep)} s)")


def runs_of(i_idx, j_idx, min_run):
    """Monotonic runs in a list of (i, best j) matches: j moves forward 0..3 per step of i (gaps of <= 2 s in i)."""
    out, start = [], 0
    for k in range(1, len(i_idx) + 1):
        if k == len(i_idx) or not (0 < i_idx[k] - i_idx[k - 1] <= 3 and 0 <= j_idx[k] - j_idx[k - 1] <= 4):
            if i_idx[k - 1] - i_idx[start] + 1 >= min_run:
                out.append((i_idx[start], i_idx[k - 1], j_idx[start], j_idx[k - 1]))
            start = k
    return out


def cmd_match(args):
    import torch
    embs = {p.stem[4:]: np.load(p) for p in sorted(OUT.glob("emb_*.npz"))}
    names = list(embs)
    e = {n: torch.from_numpy(embs[n]["emb"]).cuda().float() for n in names}
    matches = []   # (route a, i0, i1, route b, j0, j1, mean sim)
    for a in names:
        for b in names:
            if b < a:
                continue
            s = e[a] @ e[b].T
            if a == b:   # the same route: only passes more than 10 min apart
                ii = torch.arange(len(s), device="cuda")
                s[(ii[:, None] - ii[None, :]).abs() < 600] = -1
            best, j = s.max(1)
            ok = (best > args.sim).nonzero().flatten().cpu().numpy()
            jj = j[ok].cpu().numpy()
            for i0, i1, j0, j1 in runs_of(ok, jj, args.run):
                matches.append((a, int(i0), int(i1), b, int(j0), int(j1), float(best[i0:i1 + 1].mean())))
    np.save(OUT / "matches.npy", np.array(matches, dtype=object), allow_pickle=True)
    tot = sum(m[2] - m[1] + 1 for m in matches)
    print(f"{len(matches)} matched runs, {tot} s of route a matched elsewhere (sim > {args.sim}, run >= {args.run} s)")
    pairs = {}
    for m in matches:
        pairs[(m[0], m[3])] = pairs.get((m[0], m[3]), 0) + m[2] - m[1] + 1
    for (a, b), n in sorted(pairs.items(), key=lambda x: -x[1])[:40]:
        print(f"  {a} ~ {b}: {n} s")


def cmd_decide(args):
    from train import EVAL_ROUTES
    embs = {p.stem[4:]: np.load(p) for p in sorted(OUT.glob("emb_*.npz"))}
    allm = np.load(OUT / "matches.npy", allow_pickle=True)
    scores = np.load(OUT / "verify_scores.npy")
    assert len(scores) == len(allm), "verify_scores.npy is stale: rerun verify"
    matches = [x for x, s in zip(allm, scores) if accept(s)]
    print(f"{len(matches)} of {len(allm)} matched runs verified ({sum(x[2] - x[1] + 1 for x in matches)} s)")
    labelled = {d.name.split("__")[0] for d in (ROOT / "data/ep_label").iterdir() if (d / "labels.npz").exists()}

    def cond(r, i):
        t = route_start(r).hour + route_start(r).minute / 60 + (embs[r]["t_ns"][i] - embs[r]["t_ns"][0]) * 1e-9 / 3600
        return tod_class(t % 24), luma_class(float(embs[r]["luma"][i]))

    def priority(r):   # lower is kept first
        return (0 if any(r.startswith(e) for e in EVAL_ROUTES) else 1, 0 if r in labelled else 1, r)

    drop = {r: np.zeros(len(embs[r]["t_ns"]), bool) for r in embs}
    kept_diff = 0
    for a, i0, i1, b, j0, j1, sim in matches:
        n = i1 - i0 + 1
        for k in range(n):
            i = i0 + k
            j = j0 + int(round(k * (j1 - j0) / max(n - 1, 1)))
            if cond(a, i) != cond(b, j):
                kept_diff += 1
                continue
            loser = (a, i) if priority(a) > priority(b) or (a == b and i > j) else (b, j)
            if priority(loser[0])[0] == 0:   # eval routes are not trained on: both passes stay for evaluation
                continue
            drop[loser[0]][loser[1]] = True
    report, seg_drop = {}, []
    for r, d in drop.items():
        if not d.any():
            continue
        segs = embs[r]["seg"]
        for s in np.unique(segs):
            frac = d[segs == s].mean()
            if frac >= args.seg_drop:
                seg_drop.append([r, f"{s:03d}"])
        t_ns = embs[r]["t_ns"].astype(np.int64)
        end = np.append(t_ns[1:], t_ns[-1] + 1_000_000_000)    # a keyframe stands for the second up to the next one
        spans = []
        for k in np.flatnonzero(d):
            if spans and spans[-1][1] == int(t_ns[k]):
                spans[-1][1] = int(end[k])
            else:
                spans.append([int(t_ns[k]), int(end[k])])
        report[r] = {"dropped_s": int(d.sum()), "total_s": int(len(d)), "dropped_ns": spans}
    json.dump({"segments_not_labelled": seg_drop, "routes": report, "rule": vars(args)},
              open(OUT / "dedup.json", "w"), indent=1)
    tot = sum(len(v["t_ns"]) for v in embs.values())
    dropped = sum(v["dropped_s"] for v in report.values())
    print(f"{dropped} of {tot} s dropped ({100 * dropped / tot:.1f}%), {kept_diff} matched seconds kept for different "
          f"conditions, {len(seg_drop)} segments not labelled")
    for r, v in sorted(report.items(), key=lambda x: -x[1]["dropped_s"]):
        print(f"  {r}: {v['dropped_s']}/{v['total_s']} s dropped")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["embed", "match", "thumbs", "verify", "decide"])
    ap.add_argument("--sim", type=float, default=0.80)
    ap.add_argument("--run", type=int, default=8)
    ap.add_argument("--seg-drop", type=float, default=0.7)
    args = ap.parse_args()
    {"embed": cmd_embed, "match": cmd_match, "thumbs": cmd_thumbs, "verify": cmd_verify,
     "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    main()
