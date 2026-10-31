"""Distil the big driving model (and the LD lane model for lanes) into the student on edgepilot routes.

Data: ep_label.py's per-route labels.npz and frames.npy (warped views of the even frames), minus the spans
dedup.py dropped (places driven again in the same conditions). A sample is an even frame t: road and wide [12,128,256] = frames t-4 and t; desire [25,8] = the 100 last pulses max-pooled 4:1;
features_buffer [24,512] = the student's own hidden states at t-96, ..., t-4 (every 4th frame, oldest first) from
a bank that is recomputed with the current weights after every epoch (zeros before the first refresh, like the
runtime's empty history). This is the runtime's closed loop with stale weights.

Targets per head (raw openpilot outputs, the CPU heads' outputs):
  plan, lead, pose, road_transform, wide_from_device: the teacher's means (L1 divided by the teacher's std, log past 20)
      and log stds (L1);
  lanes, road_edges: y from LD (inner edges of the lines; LD near boundaries) where LD saw them, else the teacher;
      z and log stds from the teacher;
  lane_prob: existence logits (odd entries) from LD's valid probability, the others from the teacher;
  meta, lead_prob: soft BCE on the teacher's probabilities (lead_prob times |p - teacher p|^g with --lead-prob-focal g,
      Quality Focal Loss, Li et al. 2020: the frames the student gets wrong, e.g. far false leads, count most);
      desire_pred, desire_state: KL to the teacher.

--flip p: a share p of the samples is mirrored left-right (mirror_big(): images, labels, desire, the history from a
second bank of mirrored frames, traffic_convention [0, 1] since a mirrored right-hand-traffic road is a left-hand one).
--curve-sampling: samples drawn with weights 1 + [|teacher plan y at 2 s| > 0.3 m] + [> 1 m] + 2 [desire set],
halved when standing (v < 1 m/s, 15 % of the samples).
--qat: quantization-aware fine-tuning for the NPU: every NPU layer runs with its weights rounded to S8 per channel
as Pulsar2 does, with a straight-through gradient (student.quant_sim(qat=True)); RepConvs fold their branches on each
forward and round the folded weights, the BN statistics stay fixed. The bank is computed with the rounded weights too.
Checkpoints keep the float (unfused) weights; evaluate them with eval_student.py --s8.
--gap-aug p: a share p of the samples gets its first frame from another gap (0, 0.5, 1.5 or 2 times the normal 0.2 s),
i.e. the clip played at rate k, with the history zeroed (a drive start: the bank's states carry the true speed). They
learn what the rate fixes (gap_loss()): pose times k, the plan retimed (teacher at k t, velocities times k), the lead's
position; the spatial outputs (SPATIAL) as usual; nothing else. (v1 trained only the plan's t=0 vx and yaw rate, which
broke the plan's speed profile: long_acc 0.25 -> 1.12 after an epoch.)
Without it the student reads its speed off the scene (diag_gap.py: same frame twice still gives 63-72 % of the speed).
--hist-noise r, --hist-drop p: regularise the temporal input against shortcuts (comma's noise bottleneck on the vision
features, ChauffeurNet's past dropout): Gaussian noise of r times the per-dimension RMS of a unit 512-d state on the
history tokens and the current hidden as the policy sees them (the bank keeps the clean states), and the whole history
zeroed (a drive start) for a share p of the samples.
--ema d: also keep an exponential moving average of the weights (decay d per step), saved as <out>/ema.pt.

TensorBoard: <out>/tb (losses per head, grad norm, lr, samples/s every --tb-every steps); eval_student.py adds the
held-out metrics there. View with: tensorboard --logdir ckpt --bind_all

usage: scripts/py scripts/train.py --data data/ep_label --out ckpt/run1 [--epochs 20] [--batch 64]
"""
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from student import HEADS, POLICY_HEADS, Student, expand_state, load_student, plain_state, quant_sim  # noqa: E402
from lane_geom import slope_fix  # noqa: E402
from zframes import has_frames, load_frames  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)], np.float32)
# the big model's output slices (its metadata output_slices)
BIG = {"lanes": (0, 528), "lane_prob": (528, 536), "road_edges": (536, 800), "meta": (800, 855),
       "desire_pred": (855, 887), "pose": (887, 899), "wide_from_device": (899, 905), "road_transform": (905, 917),
       "plan": (917, 1907), "lead": (1907, 2051), "lead_prob": (2051, 2054), "desire_state": (2054, 2062)}
EVAL_ROUTES = ["2026-10-25--22-42-26-175", "2026-10-05--00-07-29-810", "2026-10-22--23-22-40-918",
               "2026-10-05--16-28-44-410", "2026-09-27--02-44-40-679"]
WEIGHTS = {"plan": 1.0, "lanes": 1.0, "road_edges": 0.5, "lead": 0.5, "lead_prob": 0.5, "lane_prob": 0.5,
           "meta": 0.3, "pose": 0.3, "road_transform": 0.3, "wide_from_device": 0.1, "desire_state": 0.3,
           "desire_pred": 0.1, "gap": 0.5}
MDN = ("plan", "lead", "pose", "road_transform", "wide_from_device", "lanes", "road_edges")
LD_SIGMA = torch.tensor(0.05 + 0.004 * X_IDXS)   # LD lane-point uncertainty (m) by distance
YUV6_FLIP = [2, 3, 0, 1, 4, 5, 8, 9, 6, 7, 10, 11]    # mirrored columns swap the Y phases (even col <-> odd col)
DESIRE_FLIP = [0, 2, 1, 4, 3, 6, 5, 7]                # none, turn L/R, lane change L/R, keep L/R
GAPS = (0, 1, 3, 4)            # --gap-aug: first frame j - g of kept frames (normal g = 2, i.e. t-4), ratio g / 2


def mirror_big(width: int = 2066):
    """(index, sign) so that big[index] * sign is the teacher output of the left-right mirrored scene: y, roll, yaw
    and their rates change sign; lanes, road edges, blinkers and desires swap sides; stds only move with their means."""
    idx, sign = np.arange(width), np.ones(width, np.float32)

    def part(name, shape, perm_axis=None, neg=(), neg_axis=-1):
        a, b = BIG[name]
        half = (b - a) // 2
        for off, mean in ((a, True), (a + half, False)):
            j = np.arange(half).reshape(shape)
            if perm_axis is not None:
                j = np.flip(j, perm_axis)
            idx[off:off + half] = off + j.reshape(-1)
            if mean and neg:
                sg = np.ones(shape, np.float32)
                sl = [slice(None)] * len(shape)
                sl[neg_axis] = list(neg)
                sg[tuple(sl)] = -1
                sign[off:off + half] = sg.reshape(-1)

    part("lanes", (4, 33, 2), 0, (0,))
    part("road_edges", (2, 33, 2), 0, (0,))
    part("plan", (33, 15), None, (1, 4, 7, 9, 11, 12, 14))
    part("lead", (3, 6, 4), None, (1,))
    part("pose", (6,), None, (1, 3, 5))
    part("road_transform", (6,), None, (1, 3, 5))
    part("wide_from_device", (3,), None, (0, 2))
    a, _ = BIG["lane_prob"]
    idx[a:a + 8] = a + np.flip(np.arange(8).reshape(4, 2), 0).reshape(-1)
    a, _ = BIG["meta"]
    for left in range(33, 55, 4):                    # LEFT_BLINKER 33::4 <-> RIGHT_BLINKER 34::4
        idx[a + left], idx[a + left + 1] = a + left + 1, a + left
    for k, n in (("desire_pred", 4), ("desire_state", 1)):
        a, _ = BIG[k]
        idx[a:a + 8 * n] = a + (np.arange(n)[:, None] * 8 + np.array(DESIRE_FLIP)[None]).reshape(-1)
    return idx, sign


MIRROR_IDX, MIRROR_SIGN = mirror_big()


def npz_mmap(path: Path, key: str) -> np.memmap:
    """Memory-map one array of an uncompressed .npz (np.savez stores its members as plain .npy files in the zip)."""
    with zipfile.ZipFile(path) as z:
        info = z.getinfo(key + ".npy")
    assert info.compress_type == zipfile.ZIP_STORED, f"{path}:{key} is compressed"
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        name_len, extra_len = struct.unpack("<HH", f.read(30)[26:30])    # the local file header
        f.seek(info.header_offset + 30 + name_len + extra_len)
        major, _ = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        offset = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=offset, order="F" if fortran else "C")


class Route:
    """One labelled unit. Labels and frames are memory-mapped on first use in each process, so a Route pickles to
    DataLoader workers without its data (macOS starts workers by spawn, not fork)."""
    ARRAYS = {"big": "big", "pulse": "pulse", "ld_lanes": "ld_lanes", "ld_prob": "ld_lane_prob",
              "ld_edges": "ld_edges", "v_ego": "v_ego", "desire": "desire", "mount": "mount"}

    def __init__(self, d: Path, dropped=()):
        self.dir, self.name = d, d.name
        self._mm = {}
        t_ns = np.asarray(npz_mmap(d / "labels.npz", "t_ns"), np.int64)
        self.n = len(t_ns)
        # samples: even frames with t-4 present, outside the spans dedup.py dropped (repeats in the same conditions);
        # the bank still covers every frame, so the history of the frames after a dropped span is intact
        self.samples = np.arange(4, self.n - self.n % 2, 2)
        t_ns = t_ns[self.samples]
        keep = np.ones(len(t_ns), bool)
        for a, b in dropped:
            keep &= ~((t_ns >= a) & (t_ns < b))
        self.samples = self.samples[keep]

    def __getattr__(self, k):   # big, pulse, ld_lanes, ld_prob, ld_edges, v_ego, desire, mount, frames
        if k not in Route.ARRAYS and k != "frames":
            raise AttributeError(k)
        if k not in self._mm:
            self._mm[k] = load_frames(self.dir) if k == "frames" else \
                npz_mmap(self.dir / "labels.npz", Route.ARRAYS[k])
        return self._mm[k]

    def __getstate__(self):
        return {**self.__dict__, "_mm": {}}


class Bank:
    """The student's hidden state per even frame of every route (fp16); with flip, also of the mirrored frames."""

    def __init__(self, routes, flip=False):
        self.h = {r.name: torch.zeros((r.n + 1) // 2, 512, dtype=torch.float16) for r in routes}
        self.hf = {r.name: torch.zeros((r.n + 1) // 2, 512, dtype=torch.float16) for r in routes} if flip else None

    def history(self, r: Route, t: int, flipped: bool = False) -> torch.Tensor:
        idx = np.arange(t - 96, t, 4) // 2      # frames t-96 .. t-4, every 4th
        out = torch.zeros(24, 512)
        ok = idx >= 0
        out[ok] = (self.hf if flipped else self.h)[r.name][idx[ok]].float()
        return out


class Samples(torch.utils.data.Dataset):
    def __init__(self, routes, bank, flip=0.0, gap=0.0, ld_fix=False):
        self.routes, self.bank, self.flip, self.gap, self.ld_fix = routes, bank, flip, gap, ld_fix
        self.index = [(i, t) for i, r in enumerate(routes) for t in r.samples]

    def curve_weights(self, night: float = 1.0) -> np.ndarray:
        """Per sample: 1 + [|teacher plan y at 2 s| > 0.3 m] + [> 1 m] + 2 [desire set], halved when standing
        (v < 1 m/s) (--curve-sampling); times `night` in units with mean road-camera luma < 150 (--night-weight).
        Despite the name these are mostly daytime units (checked 2026-10-31 against the route's hour, KST): the camera's
        auto exposure makes lit night streets bright (luma 165-171) and daytime tunnels / overcast dark (~125); 33 of
        the 35 units it picks are daytime, 141 of the 143 night units are not picked. So --night-weight w draws the
        daytime units w times more (most of the data is night: 1.33 M of 1.77 M frames)."""
        a = BIG["plan"][0]
        w = []
        for r in self.routes:
            rows = np.asarray(r.big[r.samples])        # teacher plan y at T_IDXS 14 and 15 (1.91 s, 2.20 s)
            y2 = np.abs(0.697 * rows[:, a + 14 * 15 + 1] + 0.303 * rows[:, a + 15 * 15 + 1])
            wr = (1.0 + (y2 > 0.3) + (y2 > 1.0) + 2.0 * (np.asarray(r.desire[r.samples]) != 0)) * \
                np.where(np.asarray(r.v_ego[r.samples]) < 1.0, 0.5, 1.0)
            if night != 1.0:
                js = np.linspace(0, len(r.frames) - 1, 8).astype(int)
                if np.mean([np.asarray(r.frames[j, 0])[:4].mean() for j in js]) < 150:
                    wr = wr * night
            w.append(wr)
        return np.concatenate(w)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        r = self.routes[i]
        g = 2
        if self.gap > 0 and torch.rand(()).item() < self.gap:
            g = GAPS[torch.randint(len(GAPS), ()).item()]
            g = g if t // 2 - g >= 0 else 2
        road = np.concatenate([r.frames[t // 2 - g, 0], r.frames[t // 2, 0]])
        wide = np.concatenate([r.frames[t // 2 - g, 1], r.frames[t // 2, 1]])
        des = np.zeros((100, 8), np.float32)
        lo = max(0, t - 99)
        des[100 - (t + 1 - lo):] = r.pulse[lo:t + 1]
        big, ld_lanes, ld_prob, ld_edges = np.array(r.big[t]), np.array(r.ld_lanes[t]), np.array(r.ld_prob[t]), np.array(r.ld_edges[t])
        if self.ld_fix:   # --ld-slope-fix: LD's lines on the teacher's road surface, not a flat road (lane_geom.py)
            off, h = np.asarray(r.mount[t], np.float64)
            for k_, n_ in (("lanes", 4), ("road_edges", 2)):
                a, _ = BIG[k_]
                z = big[a:a + n_ * 66].reshape(n_, 33, 2)[..., 1]
                if k_ == "lanes":
                    ld_lanes = slope_fix(ld_lanes, z, h, off)
                else:
                    ld_edges = slope_fix(ld_edges, z, h, off)
        fl = self.flip > 0 and torch.rand(()).item() < self.flip     # torch's RNG is seeded per worker
        if fl:
            road, wide = (np.ascontiguousarray(x[YUV6_FLIP, :, ::-1]) for x in (road, wide))
            des = des[:, DESIRE_FLIP]
            big = big[MIRROR_IDX] * MIRROR_SIGN
            ld_lanes, ld_prob, ld_edges = -ld_lanes[::-1].copy(), ld_prob[::-1].copy(), -ld_edges[::-1].copy()
        return {"road": torch.from_numpy(road), "wide": torch.from_numpy(wide),
                "desire": torch.from_numpy(des.reshape(25, 4, 8).max(1)),
                "feat": self.bank.history(r, t, fl) if g == 2 else torch.zeros(24, 512),
                **({"gap": torch.tensor(g / 2)} if self.gap > 0 else {}),
                "big": torch.from_numpy(big),
                "ld_lanes": torch.from_numpy(ld_lanes), "ld_prob": torch.from_numpy(ld_prob),
                "ld_edges": torch.from_numpy(ld_edges), "v": torch.tensor(float(r.v_ego[t])),
                "flip": torch.tensor(float(fl)), "route": i, "t": t}


def augment(x: torch.Tensor) -> torch.Tensor:
    """Photometric augmentation of a batch of YUV6 frame pairs [B, 12, H, W] (0..255; channels 0-3 Y, 4 U, 5 V of
    frame t-4, then the same of frame t). Per sample, the same for both frames: luma gain and offset, chroma
    saturation and shift (exposure, white balance), then per-pixel noise. 20 % of the samples stay as they are.
    No geometric change: the targets are tied to the calibrated view."""
    n, dev = x.shape[0], x.device
    u = lambda lo, hi, k: torch.empty(n, k, device=dev).uniform_(lo, hi)
    on = (torch.rand(n, 1, device=dev) < 0.8).float()
    gain, off, sat, shift = u(0.7, 1.3, 1), u(-25.0, 25.0, 1), u(0.7, 1.3, 1), u(-8.0, 8.0, 2)
    # one scale and bias per channel (identity for the samples left alone), so the whole thing is a few passes
    scale = 1 + on * (torch.cat([gain.expand(n, 4), sat.expand(n, 2)], 1) - 1)
    bias = on * torch.cat([off.expand(n, 4), 128 * (1 - sat) + shift], 1)
    shape = (n, 1, 6, 1, 1)
    f = torch.addcmul(bias.view(shape), x.view(n, 2, 6, *x.shape[2:]), scale.view(shape)).view_as(x)
    f += torch.randn_like(f) * (on * u(0.0, 4.0, 1)).view(n, 1, 1, 1)
    return f.clamp_(0, 255)


def batched_forward(model, b, device, aug=False, hist_noise=0.0, hist_drop=0.0):
    """The student on a batch: the backbone sees 2B images, the rest per sample. aug: augment() road and wide
    (each camera its own draw); hist_noise, hist_drop: --hist-noise, --hist-drop on the policy's input."""
    road = b["road"].to(device, non_blocking=True).float()
    wide = b["wide"].to(device, non_blocking=True).float()
    if aug:
        road, wide = augment(road), augment(wide)
    n = road.shape[0]
    x = torch.cat([road, wide], 0) * model.in_scale
    f = model.neck(model.backbone(x))
    feat = model.vis_fc(torch.cat([f[:n].flatten(1), f[n:].flatten(1)], 1))
    hidden = F.normalize(model.summarizer(feat).unsqueeze(1), dim=-1).squeeze(1)
    fl = b["flip"].to(device).view(n, 1) if "flip" in b else torch.zeros(n, 1, device=device)
    traffic = torch.cat([1 - fl, fl], 1)     # [1, 0] as recorded (right-hand traffic); [0, 1] for mirrored samples
    hist, h_in = b["feat"].to(device), hidden
    if hist_drop > 0:
        hist = hist * (torch.rand(n, 1, 1, device=device) >= hist_drop)
    if hist_noise > 0:
        sd = hist_noise / math.sqrt(hidden.shape[1])
        hist, h_in = hist + sd * torch.randn_like(hist), hidden + sd * torch.randn_like(hidden)
    hx = model.head_features(feat, h_in, hist, b["desire"].to(device), traffic)
    out = {k: model.cpu_heads[k](v) for k, v in hx.items()}
    return out, hidden


def capped(err, cap=20.0):
    """Past `cap` std an error grows only logarithmically (gradient cap/err). A hard clamp there gave no gradient at
    all: road_transform's first two means start beyond it and never moved (loss ~7)."""
    return torch.where(err > cap, cap + cap * torch.log(err.clamp(min=cap) / cap), err)


def mdn_loss(pred, tgt_mu, tgt_logstd, sigma=None, mask=None, cap=20.0):
    """L1 in units of the target std, capped()."""
    half = pred.shape[1] // 2
    mu, logstd = pred[:, :half], pred[:, half:]
    sig = sigma if sigma is not None else tgt_logstd.clamp(-7, 5).exp()
    err = capped((mu - tgt_mu).abs() / sig, cap)
    if mask is not None:
        err = err * mask
        lm = err.sum() / mask.sum().clamp(min=1)
    else:
        lm = err.mean()
    return lm + 0.1 * (logstd - tgt_logstd).abs().mean()


T_IDXS = torch.tensor([10.0 * (i / 32) ** 2 for i in range(33)])
PLAN_RATE_POW = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1], dtype=torch.float32)   # per plan column
SPATIAL = ("lanes", "road_edges", "lane_prob")     # outputs a frame gap does not change


def gap_loss(out, b, device):
    """--gap-aug samples (the clip played at rate k = gap ratio): the ego motion the gap implies, against the teacher
    retimed, L1 in units of the teacher's std at the original time, capped(): the pose means times k; the whole plan
    at time t is the teacher's at k t (positions and angles as they are, velocities and rates times k, accelerations
    times k^2; times past the 10 s horizon skipped); the current lead's x, y."""
    big = b["big"].to(device)
    k = b["gap"].to(device).float()
    n = big.shape[0]
    a, z = BIG["pose"]
    pose = (out["pose"][:, :6] - k[:, None] * big[:, a:a + 6]).abs() / big[:, a + 6:z].clamp(-7, 5).exp()
    a, z = BIG["plan"]
    mu, sig = big[:, a:a + 495].reshape(n, 33, 15), big[:, a + 495:z].reshape(n, 33, 15).clamp(-7, 5).exp()
    t = T_IDXS.to(device)
    kt = k[:, None] * t[None]
    hi = torch.searchsorted(t, kt.contiguous()).clamp(1, 32)
    w = ((kt - t[hi - 1]) / (t[hi] - t[hi - 1])).clamp(0, 1).unsqueeze(-1)
    gather = lambda i: torch.gather(mu, 1, i.unsqueeze(-1).expand(n, 33, 15))
    tgt = (gather(hi - 1) * (1 - w) + gather(hi) * w) * k[:, None, None] ** PLAN_RATE_POW.to(device)
    valid = (kt <= t[-1] + 1e-6).unsqueeze(-1).float()
    plan = capped((out["plan"][:, :495].reshape(n, 33, 15) - tgt).abs() / sig)
    plan = (plan * valid).sum() / (valid.sum() * 15).clamp(min=1)
    a, z = BIG["lead"]
    lead = (out["lead"][:, :72].reshape(n, 3, 6, 4)[:, 0, 0, :2] - big[:, a:a + 72].reshape(n, 3, 6, 4)[:, 0, 0, :2]).abs() / \
        big[:, a + 72:z].reshape(n, 3, 6, 4)[:, 0, 0, :2].clamp(-7, 5).exp()
    return (capped(pose).mean() + plan + capped(lead).mean()) / 3


def losses(out, b, device):
    """Total loss and its parts; --gap-aug samples (gap != 1) go to gap_loss() and the SPATIAL losses only."""
    if "gap" not in b:
        return head_losses(out, b, device)
    gap = b["gap"] != 1.0
    sub = lambda d, m: {k: v[m.to(v.device)] for k, v in d.items() if torch.is_tensor(v)}
    _, L = head_losses(sub(out, ~gap), sub(b, ~gap), device)
    if gap.any():
        _, La = head_losses(out, b, device)
        L.update({k: La[k] for k in SPATIAL})
        L["gap"] = gap_loss(sub(out, gap), sub(b, gap), device)
    else:
        L["gap"] = torch.zeros((), device=device)
    return sum(WEIGHTS[k] * v for k, v in L.items()), L


PLAN_LAT = [1, 4, 7, 11, 14]       # plan columns of the lateral motion: y, vy, ay, yaw, yaw rate
LOSS_OPTS = {"plan_lat": 1.0, "lead_focal": 0.0}      # --plan-lat-weight, --lead-prob-focal
PLAN_ABS_T = 21                    # plan times 0 ... 3.9 s (T_IDXS[20])


def head_losses(out, b, device):
    big = b["big"].to(device)
    tg = {k: big[:, a:z] for k, (a, z) in BIG.items()}
    L = {}
    for k in ("plan", "lead", "pose", "road_transform", "wide_from_device"):
        half = tg[k].shape[1] // 2
        w = None
        if k == "plan" and LOSS_OPTS["plan_lat"] != 1.0:   # a weighted mean: the lateral columns count more
            w = torch.ones(33, 15, device=device)
            w[:, PLAN_LAT] = LOSS_OPTS["plan_lat"]
            w = w.reshape(1, -1).expand(big.shape[0], -1)
        L[k] = mdn_loss(out[k], tg[k][:, :half], tg[k][:, half:], mask=w)
    n = big.shape[0]
    if WEIGHTS.get("plan_abs"):   # --plan-abs-weight: plan y up to 4 s in 10 cm units, not the teacher's std (wide in turns)
        dy = out["plan"][:, :495].reshape(n, 33, 15)[:, :PLAN_ABS_T, 1] - tg["plan"][:, :495].reshape(n, 33, 15)[:, :PLAN_ABS_T, 1]
        L["plan_abs"] = capped(dy.abs() / 0.1).mean()
    if WEIGHTS.get("lead_abs"):   # --lead-abs-weight: the current lead x, y (all 3 horizons) in 3 m / 0.3 m units, any lead_prob
        d = (out["lead"][:, :72].reshape(n, 3, 6, 4) - tg["lead"][:, :72].reshape(n, 3, 6, 4))[:, :, 0, :2].abs()
        L["lead_abs"] = capped(d / torch.tensor([3.0, 0.3], device=device)).mean()
    for k, ld, nl in (("lanes", b["ld_lanes"], 4), ("road_edges", b["ld_edges"], 2)):
        half = tg[k].shape[1] // 2
        mu_t = tg[k][:, :half].reshape(n, nl, 33, 2).clone()
        ls_t = tg[k][:, half:]
        ld = ld.to(device)
        have = torch.isfinite(ld)
        mu_t[..., 0] = torch.where(have, ld, mu_t[..., 0])     # LD y where LD saw the line, the teacher elsewhere
        sig = ls_t.reshape(n, nl, 33, 2).clamp(-7, 5).exp().clone()
        sig[..., 0] = torch.where(have, LD_SIGMA.to(device).expand_as(ld), sig[..., 0])
        L[k] = mdn_loss(out[k], mu_t.reshape(n, -1), ls_t, sigma=sig.reshape(n, -1))
    # lane existence: odd logits from LD, even ones from the teacher
    t_prob = torch.sigmoid(tg["lane_prob"]).clone()
    t_prob[:, 1::2] = b["ld_prob"].to(device)
    L["lane_prob"] = F.binary_cross_entropy_with_logits(out["lane_prob"], t_prob)
    for k in ("meta", "lead_prob"):
        p_t = torch.sigmoid(tg[k])
        bce = F.binary_cross_entropy_with_logits(out[k], p_t, reduction="none")
        if k == "lead_prob" and LOSS_OPTS["lead_focal"]:
            bce = bce * (torch.sigmoid(out[k]) - p_t).abs().pow(LOSS_OPTS["lead_focal"])
        L[k] = bce.mean()
    L["desire_state"] = F.kl_div(F.log_softmax(out["desire_state"], -1), F.softmax(tg["desire_state"], -1),
                                 reduction="batchmean")
    dp_s, dp_t = out["desire_pred"].reshape(n, 4, 8), tg["desire_pred"].reshape(n, 4, 8)
    L["desire_pred"] = F.kl_div(F.log_softmax(dp_s, -1), F.softmax(dp_t, -1), reduction="batchmean") / 4
    total = sum(WEIGHTS[k] * v for k, v in L.items())
    return total, L


@torch.no_grad()
def refresh_bank(model, routes, bank, device, batch=64, conv=None, flipped=False):
    """Recompute every frame's hidden state with the current weights (flipped: of the mirrored frames, into
    bank.hf). conv: the backbone+neck to use (e.g. a torch.compile'd one); every batch is padded to the same size
    so a compiled conv sees one shape."""
    from concurrent.futures import ThreadPoolExecutor
    model.eval()
    conv = conv or (lambda x: model.neck(model.backbone(x)))
    store = bank.hf if flipped else bank.h
    t0 = time.time()

    def load(r, s):
        """Batch s of route r: kept frames j and j-2, each read (decompressed) once, both cameras."""
        m = len(r.frames)
        j = np.arange(s, min(s + batch, m))
        n = len(j)
        j = np.concatenate([j, np.full(batch - n, j[-1])])    # pad the last batch
        jj = np.clip(j - 2, 0, None)              # frame t-4 is two kept frames back
        u, inv = np.unique(np.concatenate([jj, j]), return_inverse=True)
        fr = np.asarray(r.frames[u])
        a, b = fr[inv[:batch]], fr[inv[batch:]]
        return j, n, np.concatenate([a[:, 0], b[:, 0]], 1), np.concatenate([a[:, 1], b[:, 1]], 1)

    jobs = [(r, s) for r in routes for s in range(0, len(r.frames), batch)]
    with ThreadPoolExecutor(3) as ex:           # zstd frames decompress off the GIL while the GPU runs
        futs = [ex.submit(load, *jobs[i]) for i in range(min(3, len(jobs)))]
        for i, (r, _) in enumerate(jobs):
            j, n, road, wide = futs[i].result()
            if i + 3 < len(jobs):
                futs.append(ex.submit(load, *jobs[i + 3]))
            futs[i] = None
            road = torch.from_numpy(road).to(device).float()
            wide = torch.from_numpy(wide).to(device).float()
            if flipped:
                road, wide = road[:, YUV6_FLIP].flip(-1), wide[:, YUV6_FLIP].flip(-1)
            with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
                f = conv(torch.cat([road, wide], 0) * model.in_scale)
                feat = model.vis_fc(torch.cat([f[:batch].flatten(1), f[batch:].flatten(1)], 1))
                h = F.normalize(model.summarizer(feat).float().unsqueeze(1), dim=-1).squeeze(1)
            store[r.name][j[:n]] = h[:n].half().cpu()
    model.train()
    return time.time() - t0


def raise_open_files():
    """Every unit keeps 7 memory maps open (frames and 6 label arrays) in each DataLoader worker: lift the soft
    limit on open files (macOS defaults to 256)."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    for want in (65536, 10240, 4096):
        want = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if want <= soft:
            return
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            return
        except (ValueError, OSError):
            continue


def tb_writer(logdir: Path):
    """A TensorBoard SummaryWriter, or None when the tensorboard package is missing."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not installed: no TensorBoard logs", flush=True)
        return None
    return SummaryWriter(str(logdir), flush_secs=30)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data/ep_label")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4, help="AdamW weight decay")
    ap.add_argument("--aug", action="store_true", help="photometric augmentation of the training images (augment())")
    ap.add_argument("--flip", type=float, default=0.0, help="share of the samples mirrored left-right (mirror_big())")
    ap.add_argument("--ld-slope-fix", action="store_true",
                    help="LD lane and road-edge targets re-projected onto the teacher's road surface (lane_geom.slope_fix)")
    ap.add_argument("--curve-sampling", action="store_true",
                    help="draw samples weighted toward curves and desires (Samples.curve_weights())")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--arch", default="{}")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test)")
    ap.add_argument("--init", type=Path, help="start from this checkpoint's weights (a fresh schedule)")
    ap.add_argument("--device", default=default_device(), help="cuda, mps (Apple GPU) or cpu")
    ap.add_argument("--compile-refresh", action="store_true",
                    help="torch.compile the conv part of the bank refresh (2.5x faster for base on MPS with torch 2.14; "
                         "the Metal compiler failed on w48d)")
    ap.add_argument("--tb-every", type=int, default=20, help="TensorBoard scalars (<out>/tb) every this many steps")
    ap.add_argument("--qat", action="store_true", help="train with S8-rounded weights (student.quant_sim(qat=True))")
    ap.add_argument("--gap-aug", type=float, default=0.0, help="share of the samples with another frame gap, ego motion only (gap_loss())")
    ap.add_argument("--hist-noise", type=float, default=0.0, help="noise on the policy's state inputs, x per-dim RMS")
    ap.add_argument("--hist-drop", type=float, default=0.0, help="share of the samples with the history zeroed")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA decay of the weights per step (saved as ema.pt)")
    ap.add_argument("--refresh-every", type=int, default=1, help="recompute the history bank every N epochs")
    ap.add_argument("--epoch-samples", type=int, default=0, help="samples per epoch, drawn anew each epoch (default: all)")
    ap.add_argument("--plan-lat-weight", type=float, default=1.0,
                    help="weight of the plan's lateral columns (y, vy, ay, yaw, yaw rate) in its loss mean (lateral control)")
    ap.add_argument("--plan-abs-weight", type=float, default=0.0,
                    help="weight of an extra plan loss: |y - teacher y| up to 4 s in 10 cm units (turns count in cm, not in the teacher's std)")
    ap.add_argument("--lead-abs-weight", type=float, default=0.0,
                    help="weight of an extra lead loss: |x|, |y| of the current lead in 3 m / 0.3 m units (far leads count in m, "
                         "not in the teacher's std, 17-25 m past 60 m)")
    ap.add_argument("--lead-prob-weight", type=float, default=WEIGHTS["lead_prob"], help="weight of the lead_prob loss")
    ap.add_argument("--lead-weight", type=float, default=WEIGHTS["lead"], help="weight of the lead loss")
    ap.add_argument("--lead-prob-focal", type=float, default=0.0,
                    help="lead_prob BCE times |p - teacher p|^g (Quality Focal Loss): wrong frames (far false leads) count most")
    ap.add_argument("--night-weight", type=float, default=1.0,
                    help="with --curve-sampling: units with road-camera luma < 150 drawn this much more (mostly DAYTIME units, see curve_weights)")
    ap.add_argument("--init-partial", action="store_true",
                    help="--init loads all but the backbone and neck (e.g. a pretrained r18 backbone under trained heads)")
    ap.add_argument("--tags", default="", help="augmented replay units to train on too (<route>__<seg>__<tag>), comma "
                    "list; 'all' for every tag. Untagged units always")
    ap.add_argument("--dedup", type=Path, default=ROOT / "data/dedup/dedup.json",
                    help="dedup.py's decisions: the dropped spans are not trained on")
    args = ap.parse_args()
    LOSS_OPTS["plan_lat"] = args.plan_lat_weight
    LOSS_OPTS["lead_focal"] = args.lead_prob_focal
    WEIGHTS["lead"] = args.lead_weight
    if args.plan_abs_weight:
        WEIGHTS["plan_abs"] = args.plan_abs_weight
    WEIGHTS["lead_prob"] = args.lead_prob_weight
    if args.lead_abs_weight:
        WEIGHTS["lead_abs"] = args.lead_abs_weight
    args.out.mkdir(parents=True, exist_ok=True)
    raise_open_files()
    device = args.device
    dedup = json.load(open(args.dedup))["routes"] if args.dedup.exists() else {}
    print(f"dedup: {args.dedup if dedup else 'none'}", flush=True)
    routes = [Route(d, dedup.get(d.name.split("__")[0], {}).get("dropped_ns", ())) for d in sorted(args.data.iterdir())
              if (d / "labels.npz").exists() and has_frames(d)]
    tags = None if args.tags == "all" else set(filter(None, args.tags.split(",")))
    train_r = [r for r in routes if not any(r.name.startswith(e) for e in EVAL_ROUTES)   # units: route or route__chunk
               and (r.name.count("__") < 2 or tags is None or r.name.rsplit("__", 1)[1] in tags)]
    print(f"{len(train_r)} train routes, {sum(len(r.samples) for r in train_r)} samples, device {device}", flush=True)
    bank = Bank(train_r, flip=args.flip > 0)
    ds = Samples(train_r, bank, flip=args.flip, gap=args.gap_aug, ld_fix=args.ld_slope_fix)
    sampler = None
    if args.curve_sampling:
        t0 = time.time()
        w = ds.curve_weights(args.night_weight)
        sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(w).double(), args.epoch_samples or len(ds),
                                                         replacement=True)
    elif args.epoch_samples:
        sampler = torch.utils.data.RandomSampler(ds, num_samples=min(args.epoch_samples, len(ds)))
        print(f"curve sampling: samples weighted up {(w > 1).mean() * 100:.0f} % (drawn {(w * (w > 1)).sum() / w.sum() * 100:.0f} %), "
              f"standing {(w < 1).mean() * 100:.0f} % (drawn {(w * (w < 1)).sum() / w.sum() * 100:.0f} %) "
              f"({time.time() - t0:.0f} s)", flush=True)

    def refresh():
        dt = refresh_bank(model, train_r, bank, device, conv=refresh_conv)
        if args.flip > 0:
            dt += refresh_bank(model, train_r, bank, device, conv=refresh_conv, flipped=True)
        return dt
    model = Student(**json.loads(args.arch))
    if args.init:
        ck = torch.load(args.init, map_location="cpu")
        if ck.get("arch", {}) == json.loads(args.arch) and not args.init_partial:
            model = load_student(ck).train()
        else:   # a larger temporal input or temporal heads: the old weights, new input columns at zero
            assert not ck.get("fused"), "expand an unfused checkpoint"
            keep = ("backbone.", "neck.") if args.init_partial else ()
            grown = expand_state(model, ck["model"], ck.get("arch", {}), keep=keep)
            print(f"arch {ck.get('arch')} -> {args.arch}: expanded {grown}" +
                  (", backbone and neck kept (--init-partial)" if keep else ""), flush=True)
        print(f"weights from {args.init} (epoch {ck.get('epoch')}{', fused' if ck.get('fused') else ''})", flush=True)
    if args.qat:
        quant_sim(model, qat=True)
        print("QAT: S8 weights on the NPU layers", flush=True)
    fused = model.backbone.is_fused()
    model = model.to(device)
    refresh_conv = None
    if args.compile_refresh:
        refresh_conv = torch.compile(lambda x: model.neck(model.backbone(x)))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    ema = {k: v.detach().clone() for k, v in model.named_parameters()} if args.ema > 0 else None
    steps_per_epoch = (len(sampler) if sampler is not None else len(ds)) // args.batch
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total, 1), pct_start=0.05)
    step = 0
    log = open(args.out / "train.log", "a")
    if args.init:   # trained weights: the first epoch's history comes from them, not from zeros
        dt = refresh()
        print(f"bank from the initial weights in {dt:.0f} s", flush=True)
        log.write(f"bank from {args.init} in {dt:.0f} s\n")
    tb = tb_writer(args.out / "tb")
    win, wcnt, wt = {}, 0, time.time()      # TensorBoard window: means over the last --tb-every steps
    for ep in range(args.epochs):
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=sampler is None, sampler=sampler, num_workers=args.workers,
                                         drop_last=True, pin_memory=device == "cuda", persistent_workers=False, prefetch_factor=4)
        t0, agg, cnt = time.time(), {}, 0
        for b in dl:
            with torch.autocast(device, dtype=torch.bfloat16):
                out, _ = batched_forward(model, b, device, aug=args.aug, hist_noise=args.hist_noise, hist_drop=args.hist_drop)
            out = {k: v.float() for k, v in out.items()}
            loss, parts = losses(out, b, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if ema:
                with torch.no_grad():
                    for k, v in model.named_parameters():
                        ema[k].lerp_(v.detach(), 1 - args.ema)
            step += 1
            cnt += 1
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v.item()
            if tb:
                for k, v in [("total", loss.detach().item()), ("grad_norm", gnorm.item()), *((k, v.item()) for k, v in parts.items())]:
                    win[k] = win.get(k, 0.0) + v
                wcnt += 1
                if step % args.tb_every == 0:
                    for k, v in win.items():
                        tb.add_scalar("train/grad_norm" if k == "grad_norm" else f"loss/{k}", v / wcnt, step)
                    tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                    tb.add_scalar("train/samples_per_s", wcnt * args.batch / (time.time() - wt), step)
                    tb.add_scalar("train/epoch", ep + cnt / steps_per_epoch, step)
                    win, wcnt, wt = {}, 0, time.time()
            if step % 200 == 0:
                msg = (f"ep {ep} step {step}/{total} loss {loss.detach().item():.3f} " +
                       " ".join(f"{k} {v / cnt:.3f}" for k, v in agg.items()) +
                       f" | {cnt * args.batch / (time.time() - t0):.0f} samples/s lr {sched.get_last_lr()[0]:.2e}")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
            if args.max_steps and step >= args.max_steps:
                break
        torch.save({"model": plain_state(model), "arch": json.loads(args.arch), "epoch": ep, "fused": fused, "qat": args.qat}, args.out / "last.pt")
        if ema:
            st = plain_state(model)
            st.update({k.replace("parametrizations.weight.original", "weight"): v.clone() for k, v in ema.items()})
            torch.save({"model": st, "arch": json.loads(args.arch), "epoch": ep, "fused": fused, "qat": args.qat, "ema": args.ema},
                       args.out / "ema.pt")
        # the bank only feeds the next epoch: none after the last; --refresh-every N keeps it N epochs (stale weights)
        dt = refresh() if ep < args.epochs - 1 and (ep + 1) % args.refresh_every == 0 else 0.0
        msg = f"epoch {ep} done in {time.time() - t0:.0f} s, bank refreshed in {dt:.0f} s"
        print(msg, flush=True)
        log.write(msg + "\n")
        if tb:
            tb.add_scalar("epoch/minutes", (time.time() - t0) / 60, ep)
            tb.add_scalar("epoch/bank_refresh_s", dt, ep)
            tb.flush()
            wt = time.time()
        if args.max_steps and step >= args.max_steps:
            break


if __name__ == "__main__":
    main()
