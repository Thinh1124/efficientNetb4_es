"""Parametric, region-restricted artifact operators.

Every operator takes the source crop (BGR uint8), a soft alpha mask in [0, 1] for the chosen
region and a strength in [0, 1], and returns the modified image. The final image is
alpha * modified + (1 - alpha) * source, so pixels outside the region are untouched and the
manipulation mask is known exactly.
"""
from typing import Callable, Dict, Optional

import cv2
import numpy as np


def soft_alpha(region: np.ndarray, rng: np.random.Generator, size: int) -> np.ndarray:
    """Randomly deformed, feathered version of a binary region mask (SBI-style)."""
    m = region.astype(np.uint8)
    k = int(rng.integers(1, max(2, size // 40))) * 2 + 1
    m = cv2.dilate(m, np.ones((k, k), np.uint8)) if rng.random() < 0.5 else cv2.erode(m, np.ones((3, 3), np.uint8))
    if m.sum() == 0:
        m = region.astype(np.uint8)
    # elastic jitter of the mask outline
    grid = max(4, size // 32)
    dx = cv2.resize(rng.normal(0, size / 60, (grid, grid)).astype(np.float32), (size, size))
    dy = cv2.resize(rng.normal(0, size / 60, (grid, grid)).astype(np.float32), (size, size))
    xs, ys = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
    m = cv2.remap(m.astype(np.float32), xs + dx, ys + dy, cv2.INTER_LINEAR)
    blur = int(rng.integers(2, max(3, size // 16))) * 2 + 1
    alpha = cv2.GaussianBlur(m, (blur, blur), 0)
    peak = float(alpha.max())
    return np.clip(alpha / peak, 0, 1) if peak > 0 else alpha


def _blend_boundary(img, alpha, s, rng, **_):
    """SBI-like: slightly transformed + colour-jittered copy of the same face blended in."""
    h, w = img.shape[:2]
    scale = 1 + rng.uniform(-0.04, 0.04) * (0.5 + s)
    tx, ty = rng.uniform(-0.02, 0.02, 2) * w * (0.5 + s)
    matrix = np.float32([[scale, 0, (1 - scale) * w / 2 + tx], [0, scale, (1 - scale) * h / 2 + ty]])
    moved = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    return _color_mismatch(moved, alpha, 0.4 * s, rng)


def _color_mismatch(img, alpha, s, rng, **_):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    shift = rng.uniform(-1, 1, 3) * np.array([25, 15, 15]) * (0.2 + s)
    gain = 1 + rng.uniform(-0.15, 0.15) * s
    lab[..., 0] = lab[..., 0] * gain
    lab += shift
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _blur_resample(img, alpha, s, rng, **_):
    h, w = img.shape[:2]
    f = 1.0 - 0.75 * s  # 1.0 -> 0.25
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR if rng.random() < 0.5 else cv2.INTER_CUBIC)
    k = int(1 + 4 * s) * 2 + 1
    return cv2.GaussianBlur(up, (k, k), 0) if s > 0.5 else up


def _over_smooth(img, alpha, s, rng, **_):
    d = int(5 + 10 * s)
    sigma = 30 + 120 * s
    out = img
    for _ in range(1 + int(2 * s)):
        out = cv2.bilateralFilter(out, d, sigma, sigma)
    return out


def _warp_geometry(img, alpha, s, rng, **_):
    h, w = img.shape[:2]
    ys, xs = np.nonzero(alpha > 0.5)
    if len(xs) == 0:
        return img
    cx, cy = xs.mean(), ys.mean()
    radius = max(xs.std(), ys.std(), 4.0) * 2.0
    amp = (0.02 + 0.06 * s) * w
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction) + 1e-6
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fall = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * radius ** 2)).astype(np.float32)
    if rng.random() < 0.5:  # bulge / pinch instead of shift
        k = (rng.choice([-1, 1]) * (0.1 + 0.3 * s)) * fall
        mx, my = gx + k * (gx - cx), gy + k * (gy - cy)
    else:
        mx, my = gx + amp * direction[0] * fall, gy + amp * direction[1] * fall
    return cv2.remap(img, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _texture_inconsistency(img, alpha, s, rng, donor=None, **_):
    """Paste the region from another face crop (donor), roughly colour-matched."""
    if donor is None:
        return _over_smooth(img, alpha, s, rng)
    donor = cv2.resize(donor, (img.shape[1], img.shape[0]))
    m = alpha > 0.5
    if m.any():
        mean_src = img[m].mean(axis=0)
        mean_don = donor[m].mean(axis=0)
        donor = np.clip(donor.astype(np.float32) + (mean_src - mean_don) * (1 - 0.6 * s), 0, 255).astype(np.uint8)
    return donor


def _authentic_degrade(img, alpha, s, rng, **_):
    """Label 0: whole-image degradations a real video can have (compression, blur, noise)."""
    out = img
    if rng.random() < 0.7:
        q = int(90 - 65 * s)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if rng.random() < 0.4:
        k = int(1 + 2 * s) * 2 + 1
        out = cv2.GaussianBlur(out, (k, k), 0)
    if rng.random() < 0.3:
        out = np.clip(out + rng.normal(0, 2 + 6 * s, out.shape), 0, 255).astype(np.uint8)
    return out


OPS: Dict[str, Callable] = {
    "blend_boundary": _blend_boundary,
    "color_mismatch": _color_mismatch,
    "blur_resample": _blur_resample,
    "over_smooth": _over_smooth,
    "warp_geometry": _warp_geometry,
    "texture_inconsistency": _texture_inconsistency,
    "authentic_degrade": _authentic_degrade,
    # authentic_seam reuses texture_inconsistency with a donor from the SAME real video (label stays 0)
    "authentic_seam": _texture_inconsistency,
}


def apply(img: np.ndarray, alpha: np.ndarray, artifact: str, strength: float, rng: np.random.Generator,
          donor: Optional[np.ndarray] = None) -> np.ndarray:
    modified = OPS[artifact](img, alpha, strength, rng, donor=donor)
    if artifact == "authentic_degrade":
        return modified
    a = alpha[..., None].astype(np.float32)
    return np.clip(a * modified.astype(np.float32) + (1 - a) * img.astype(np.float32), 0, 255).astype(np.uint8)
