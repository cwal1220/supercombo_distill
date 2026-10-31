"""The LD lane model (FVRFBGRLEDT0001819) as a lane teacher: inference, decoding, and projection to the road.

Network: BGR 640x320, raw 0..255 (the x/127.5-1 normalisation is folded into the first conv). Outputs, per the
svnet3 decoder (leddect0001819_defs.hpp, LEDDecoder::ConvertPred2LineModelImage / ParseLinePoints_):
  lane_line_cls_output [6 lines x 12]  valid(1, sigmoid) | shape(3) | double shape(4) | colour(4) (softmax)
  boundary_line_cls_output [4 x 10]    valid(1) | shape(9)
  lane_boundary_anchor_output [10 x 4] start/length per line and boundary: ext_start, length, start, end offsets
  lane_boundary_point_confidence [10 x 72]  per-point confidence; a point with confidence < 0 is skipped
  lane_delta_output[i] [2 x 72]        x (0..1 of the ROI width) and local width per prior y
  boundary_delta_output[i] [72]        x per prior y
Prior ys: 72 fixed values from 1.0 (ROI bottom) to 0.0 (top). A line is kept when its mean positive point
confidence >= 0 and its valid probability >= 0.5. Line order (led_pos): 0 ego left, 1 ego right, 2 right line of
the left lane, 3 left line of the right lane, 4 left line of the left lane, 5 right line of the right lane.
x is the centre of the painted line and the width output its half width (checked on MaixCAM2 frames: x +- half
lands on the two edges of the paint), so a line's inner edge, the one facing the lane it bounds, is x + half for
lines left of the car (0, 4) and x - half for lines right of it (1, 5).

Projection: each image point is cast onto the flat road in the openpilot calibrated frame (x forward, y right,
z down; camera at the origin, road at z = height) through the MaixCAM2 intrinsics and the recorded calibration
rpy, then shifted by the warp's camera mount (edgepilot's virtual camera `camera_offset_m` to the right), and
resampled at openpilot's X_IDXS.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

LD_ONNX = Path("/home/chan/Downloads/Models/FVRFBGRLEDT0001819/ONNX/FVRFBGRLEDT0001819.onnx")
LD_DEFS = Path("/home/chan/Downloads/Models/FVRFBGRLEDT0001819/svnet3/leddect0001819_defs.hpp")
if not LD_ONNX.exists():     # the Mac: the same network (same I/O) under ~/Documents/LD, prior ys from PRIOR_Y
    LD_ONNX = Path.home() / "Documents/LD/lane_detection.onnx"
# prior_y_00 .. prior_y_71 of leddect0001819_defs.hpp (the ax630 LD port's kPriorY), for when the file is not here
PRIOR_Y = [1.00000000, 0.98119122, 0.95924765, 0.94043887, 0.91849530, 0.89968652, 0.88087773, 0.86206895, 0.84326017,
           0.82445139, 0.80877745, 0.78996867, 0.77429467, 0.75862068, 0.74294674, 0.72727275, 0.71159875, 0.69592476,
           0.68025076, 0.66457683, 0.65203762, 0.63636363, 0.62068969, 0.60815048, 0.59561127, 0.57993734, 0.56739813,
           0.55485892, 0.54231977, 0.52978057, 0.51724142, 0.50470221, 0.49216300, 0.47962382, 0.47021943, 0.45768023,
           0.44514105, 0.43573666, 0.42319748, 0.41379309, 0.40125391, 0.39184952, 0.38557991, 0.37617555, 0.36677116,
           0.36050156, 0.35109717, 0.34169278, 0.33542320, 0.32601881, 0.31974921, 0.31034482, 0.30094042, 0.29467085,
           0.28526646, 0.27586207, 0.26959246, 0.26018807, 0.25078368, 0.24451411, 0.23510972, 0.22570533, 0.21943574,
           0.21003135, 0.20062695, 0.17241380, 0.14420062, 0.11598746, 0.08777429, 0.05956113, 0.03134796, 0.00000000]
NET_W, NET_H = 640, 320
N_LINES, N_BOUNDS, N_PTS = 6, 4, 72
X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])
VIEW_FROM_DEVICE = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])


def prior_ys() -> np.ndarray:
    if not LD_DEFS.exists():
        return np.array(PRIOR_Y, np.float32)
    text = LD_DEFS.read_text()
    vals = dict((int(i), float(v)) for i, v in re.findall(r"prior_y_(\d+)_\{([-0-9.]+)F\}", text))
    return np.array([vals[i] for i in range(N_PTS)], np.float32)


class LDModel:
    def __init__(self, path: Path = LD_ONNX):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.names = [o.name for o in self.sess.get_outputs()]
        self.ys = prior_ys()

    def run(self, bgr: np.ndarray) -> dict:
        """bgr: [320, 640, 3] uint8 -> raw outputs, flattened."""
        x = bgr.transpose(2, 0, 1)[None].astype(np.float32)
        return {k: v.reshape(-1) for k, v in zip(self.names, self.sess.run(None, {"input": x}))}

    def decode(self, out: dict) -> list[dict]:
        """Lines and boundaries as dicts: kind, pos, valid, conf, x and y (0..1 of the ROI) of the kept points."""
        res = []
        cls_l = out["line_feature_maps.lane_line_cls_output"].reshape(N_LINES, 12)
        cls_b = out["line_feature_maps.boundary_line_cls_output"].reshape(N_BOUNDS, 10)
        sl = out["line_feature_maps.lane_boundary_anchor_output"].reshape(N_LINES + N_BOUNDS, 4)
        conf = out["line_feature_maps.lane_boundary_point_confidence"].reshape(N_LINES + N_BOUNDS, N_PTS)
        for k in range(N_LINES + N_BOUNDS):
            line = k < N_LINES
            c = conf[k]
            pos_c = c[c > 0]
            lconf = float(pos_c.mean()) if len(pos_c) else -1.0
            valid = float(1 / (1 + np.exp(-(cls_l[k, 0] if line else cls_b[k - N_LINES, 0]))))
            if lconf < 0 or valid < 0.5:
                continue
            dw = out[f"line_feature_maps.lane_delta_output[{k}]"] if line else \
                out[f"line_feature_maps.boundary_delta_output[{k - N_LINES}]"]
            xs = dw[:N_PTS]
            # lines: half the painted width per point (the decoder's ics_local_widths = 2 w roi_w is the full width)
            half = dw[N_PTS:2 * N_PTS] if line else np.zeros(N_PTS, np.float32)
            ext_start, length = sl[k, 0], sl[k, 1]
            i0 = int(np.argmin(np.abs(self.ys - (1 - ext_start))))
            i1 = int(np.argmin(np.abs(self.ys - (1 - (ext_start + length)))))
            idx = np.arange(min(i0, i1), max(i0, i1) + 1)
            idx = idx[c[idx] >= 0]
            if len(idx) < 3:
                continue
            res.append(dict(kind="line" if line else "boundary", pos=k if line else k - N_LINES, valid=valid,
                            conf=lconf, x=xs[idx].astype(np.float64), half=half[idx].astype(np.float64),
                            y=self.ys[idx].astype(np.float64)))
        return res


def roi_input(bgr_full: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop (x, y, w, h) out of the full BGR frame and resize to 640x320 (area average when shrinking)."""
    import cv2
    x, y, w, h = roi
    return cv2.resize(bgr_full[y:y + h, x:x + w], (NET_W, NET_H), interpolation=cv2.INTER_AREA)


def rot_from_rpy(rpy) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def to_road(u: np.ndarray, v: np.ndarray, k: np.ndarray, rpy, height: float, offset: float = 0.0):
    """Image pixels -> (x forward, y right) on the road in the calibrated frame of the model's virtual camera."""
    rays = np.linalg.inv(k) @ np.stack([u, v, np.ones_like(u)])
    dev = VIEW_FROM_DEVICE.T @ rays
    cal = rot_from_rpy(rpy).T @ dev
    ok = cal[2] > 1e-4
    t = np.where(ok, height / np.where(ok, cal[2], 1.0), np.nan)
    return cal[0] * t, cal[1] * t - offset


def resample(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """y at X_IDXS within the observed x range (NaN outside)."""
    o = np.argsort(x)
    x, y = x[o], y[o]
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[keep], y[keep]
    out = np.full(len(X_IDXS), np.nan)
    if len(x) >= 3:
        inside = (X_IDXS >= x[0]) & (X_IDXS <= x[-1])
        out[inside] = np.interp(X_IDXS[inside], x, y)
    return out
