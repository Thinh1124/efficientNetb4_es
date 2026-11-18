"""Face-region geometry shared by pseudo-masks (P1), synthesis (P2) and error reports.

Regions are built from the 5 SCRFD/InsightFace keypoints expressed in crop coordinates,
in the order: left eye, right eye, nose tip, left mouth corner, right mouth corner.
"""
from typing import Dict, Sequence

import cv2
import numpy as np

REGIONS = ("eyes", "nose", "mouth", "skin", "boundary")


def parse_kps(value) -> np.ndarray:
    """Parse '"x1 y1 x2 y2 ..."' (10 numbers) or an array into a (5, 2) float array."""
    if isinstance(value, str):
        numbers = [float(v) for v in value.split()]
    else:
        numbers = [float(v) for v in np.asarray(value).reshape(-1)]
    if len(numbers) != 10:
        raise ValueError(f"Expected 10 keypoint numbers, got {len(numbers)}")
    return np.asarray(numbers, dtype=np.float32).reshape(5, 2)


def format_kps(kps: np.ndarray) -> str:
    return " ".join(f"{v:.2f}" for v in np.asarray(kps, dtype=np.float32).reshape(-1))


def _ellipse(size: int, center, axes, angle_deg: float) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    center_i = (int(round(center[0])), int(round(center[1])))
    axes_i = (max(1, int(round(axes[0]))), max(1, int(round(axes[1]))))
    cv2.ellipse(mask, center_i, axes_i, angle_deg, 0, 360, 1, thickness=-1)
    return mask


def _kernel(radius: float) -> np.ndarray:
    k = max(3, int(round(radius)) * 2 + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def face_geometry(kps: np.ndarray) -> Dict[str, object]:
    kps = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    left_eye, right_eye, nose, mouth_l, mouth_r = kps
    eye_vec = right_eye - left_eye
    d = float(np.linalg.norm(eye_vec)) or 1.0
    angle = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (mouth_l + mouth_r) / 2.0
    vertical = float(np.linalg.norm(mouth_center - eye_center)) or d
    return {
        "kps": kps,
        "d": d,
        "angle": angle,
        "eye_center": eye_center,
        "mouth_center": mouth_center,
        "vertical": vertical,
    }


def face_hull(kps: np.ndarray, size: int) -> np.ndarray:
    """Approximate face ellipse (forehead to chin, cheek to cheek)."""
    g = face_geometry(kps)
    d = g["d"]
    center = (g["eye_center"] + g["mouth_center"]) / 2.0
    half_h = g["vertical"] / 2.0 + 0.9 * d
    half_w = 1.1 * d
    return _ellipse(size, center, (half_w, half_h), g["angle"])


def region_masks(kps: np.ndarray, size: int) -> Dict[str, np.ndarray]:
    """Return binary uint8 masks (0/1) of shape (size, size) for each region plus 'face'."""
    g = face_geometry(kps)
    kps = g["kps"]
    d, angle = g["d"], g["angle"]
    left_eye, right_eye, nose, mouth_l, mouth_r = kps

    face = face_hull(kps, size)
    eyes = np.maximum(
        _ellipse(size, left_eye, (0.38 * d, 0.22 * d), angle),
        _ellipse(size, right_eye, (0.38 * d, 0.22 * d), angle),
    )
    nose_center = nose - 0.2 * (nose - g["eye_center"])
    nose_m = _ellipse(size, nose_center, (0.26 * d, 0.36 * d), angle)
    mouth_half = float(np.linalg.norm(mouth_r - mouth_l)) / 2.0
    mouth = _ellipse(size, g["mouth_center"], (mouth_half + 0.15 * d, 0.26 * d), angle)

    ring = 0.15 * d
    outer = cv2.dilate(face, _kernel(ring))
    inner = cv2.erode(face, _kernel(ring))
    boundary = (outer > 0) & (inner == 0)

    inner_parts = (eyes > 0) | (nose_m > 0) | (mouth > 0)
    skin = (inner > 0) & ~inner_parts

    return {
        "eyes": (eyes > 0).astype(np.uint8),
        "nose": ((nose_m > 0) & ~(eyes > 0)).astype(np.uint8),
        "mouth": ((mouth > 0) & ~(nose_m > 0)).astype(np.uint8),
        "skin": skin.astype(np.uint8),
        "boundary": boundary.astype(np.uint8),
        "face": face,
    }


def region_ratios(mask: np.ndarray, regions: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Fraction of each region covered by a binary manipulation mask."""
    m = mask > 0
    out = {}
    for name in REGIONS:
        r = regions[name] > 0
        area = int(r.sum())
        out[name] = float((m & r).sum()) / area if area else 0.0
    return out


def dominant_region(ratios: Dict[str, float], min_ratio: float = 0.05) -> str:
    name = max(REGIONS, key=lambda r: ratios.get(r, 0.0))
    return name if ratios.get(name, 0.0) >= min_ratio else "none"


def crop_square(image: np.ndarray, x1: float, y1: float, side: float, size: int) -> np.ndarray:
    """Crop a square (x1, y1, side) from image and resize to size x size; pads outside with black."""
    s = size / float(side)
    matrix = np.array([[s, 0.0, -x1 * s], [0.0, s, -y1 * s]], dtype=np.float32)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def square_box(bbox: Sequence[float], margin: float):
    """Square crop box (x1, y1, side) centred on a detector bbox and enlarged by margin."""
    bx1, by1, bx2, by2 = [float(v) for v in bbox]
    cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    side = max(bx2 - bx1, by2 - by1) * margin
    return cx - side / 2.0, cy - side / 2.0, side
