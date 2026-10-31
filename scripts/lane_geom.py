"""LD lines on hills. ep_label.py projects LD's image lines onto a flat road at the camera height (ld_teacher.to_road),
so where the road ahead rises or falls relative to the car's own plane LD's y is off: a rising road's points land
too far and the lane widens with distance (x 1.35 at 30 m with 0.6 m of rise, diag_lane_grade.py), a falling road's
narrows; students trained on LD y copy it. slope_fix() puts the same image points on the big teacher's road surface
instead: a flat point (X, y) at height h is the ray s (X, y + off, h) (off the camera offset, as to_road subtracts
it), and the road point on it is where the ray's height s h equals the teacher's height of that line z(s X)."""
import numpy as np

X_IDXS = np.array([192.0 * (i / 32) ** 2 for i in range(33)])


def slope_fix(ld: np.ndarray, z: np.ndarray, h, off) -> np.ndarray:
    """ld [..., L, 33]: y at X_IDXS on the flat road (NaN where LD saw nothing); z [..., L, 33]: the teacher's height
    of the same lines (below the camera, as h); h, off [...]: camera height and offset. Returns y at X_IDXS on the
    teacher's surface, NaN outside the re-projected range. A true point at x shows up in the flat projection at
    X = x h / z(x); that map is inverted along each line (its increasing part: beyond a crest the road is hidden)."""
    lead = ld.shape[:-1]
    ld2, z2 = ld.reshape(-1, 33), z.reshape(-1, 33)
    h2 = np.broadcast_to(np.asarray(h, np.float64)[..., None], lead).reshape(-1)
    o2 = np.broadcast_to(np.asarray(off, np.float64)[..., None], lead).reshape(-1)
    out = np.full(ld2.shape, np.nan, np.float32)
    xs = np.linspace(2.0, 192.0, 400)
    for i in range(len(ld2)):
        seen = np.isfinite(ld2[i])
        if seen.sum() < 3 or not np.isfinite(z2[i]).all():
            continue
        zz = np.interp(xs, X_IDXS, z2[i])
        ok = zz > 0.05
        xf = np.where(ok, xs * h2[i] / np.where(ok, zz, 1.0), np.inf)
        inc = np.r_[True, np.diff(xf) > 0] & ok
        inc = np.cumprod(inc).astype(bool)                       # up to the first fold (a crest) or a bad height
        if inc.sum() < 3:
            continue
        X = X_IDXS[seen]
        inside = (X >= xf[inc][0]) & (X <= xf[inc][-1])
        if inside.sum() < 3:
            continue
        xt = np.interp(X[inside], xf[inc], xs[inc])
        s = xt / X[inside]
        yt = s * (ld2[i][seen][inside] + o2[i]) - o2[i]
        keep = (X_IDXS >= min(xt[0], X[inside][0])) & (X_IDXS <= xt[-1])   # the nearest point moves < 0.1 m
        out[i, keep] = np.interp(X_IDXS[keep], xt, yt)
    return out.reshape(ld.shape)
