"""P1.1 — Pseudo manipulation masks for Celeb-DF fakes from |fake - source real| differences.

For each fake frame in meta.csv the source real video (src_video) is decoded at the same
frame index and cropped with the *fake* crop box, so both crops are pixel-aligned if the
naming assumption holds. Alignment is verified per frame with the mean absolute difference
on a ring *outside* the face (border_mad): a face swap only changes the face, so the
background must match up to compression noise.

Outputs
-------
<masks-root>/<rel_path>   binary mask PNG (0/255), same size as the crop
<out-csv>                 rel_path, mask_rel, aligned, border_mad, mask_area, ratio_<region>..., dominant_region

Example
-------
python scripts/make_pseudo_masks.py --celebdf-root /data/Celeb-DF-v2 --meta data/celebdf/meta.csv \
    --masks-root data/celebdf/masks --out-csv data/celebdf/masks.csv
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_celebdf import FAKE_RE, read_frames  # noqa: E402
from synth.regions import REGIONS, crop_square, dominant_region, parse_kps, region_masks, region_ratios  # noqa: E402


def diff_mask(fake: np.ndarray, real: np.ndarray, face: np.ndarray, min_thr: float):
    f = cv2.GaussianBlur(fake, (5, 5), 0).astype(np.float32)
    r = cv2.GaussianBlur(real, (5, 5), 0).astype(np.float32)
    diff = np.abs(f - r).max(axis=2)

    size = fake.shape[0]
    face_zone = cv2.dilate(face, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 8 | 1, size // 8 | 1)))
    ring = cv2.dilate(face_zone, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 6 | 1, size // 6 | 1)))
    outside = (ring == 0)
    border_mad = float(diff[outside].mean()) if outside.any() else float("nan")

    diff8 = np.clip(diff, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(diff8[face_zone > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) \
        if (face_zone > 0).any() else (min_thr, None)
    thr = max(float(otsu), min_thr)
    mask = ((diff8 > thr) & (face_zone > 0)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size // 32 | 1, size // 32 | 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask, border_mad


def source_video(fake_video_id: str, naming: str):
    m = FAKE_RE.match(fake_video_id.split("/")[-1])
    if not m:
        return None
    a, b, k = m.groups()
    return f"Celeb-real/id{a if naming == 'target' else b}_{k}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--celebdf-root", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--masks-root", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--size", type=int, default=224, help="crop size used in prepare_celebdf.py extract")
    p.add_argument("--src-naming", choices=["target", "source"], default="target",
                   help="target: id{A}_id{B}_{k} -> Celeb-real/id{A}_{k}; source: -> Celeb-real/id{B}_{k}")
    p.add_argument("--min-thr", type=float, default=12.0, help="minimum diff threshold (0-255)")
    p.add_argument("--border-mad-thr", type=float, default=6.0, help="max background diff to call a pair aligned")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--limit-videos", type=int, default=0, help="debug: only first N fake videos")
    args = p.parse_args()

    root, masks_root = Path(args.celebdf_root), Path(args.masks_root)
    meta = pd.read_csv(args.meta)
    fakes = meta[meta["label"] == 1]
    videos = sorted(fakes["video_id"].unique())
    if args.limit_videos:
        videos = videos[: args.limit_videos]
    videos = [v for i, v in enumerate(videos) if i % args.num_shards == args.shard]

    out_csv = Path(args.out_csv)
    if args.num_shards > 1:
        out_csv = out_csv.with_name(f"{out_csv.stem}_shard{args.shard:03d}{out_csv.suffix}")
    done = set()
    if out_csv.exists():
        done = set(pd.read_csv(out_csv)["rel_path"])

    def video_file(video_id: str):
        for ext in (".mp4", ".avi", ".mov", ".mkv"):
            path = root / f"{video_id}{ext}"
            if path.exists():
                return path
        return None

    try:
        from tqdm import tqdm
        videos_iter = tqdm(videos)
    except ImportError:
        videos_iter = videos

    header = not out_csv.exists()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stats = {"aligned": 0, "total": 0, "missing_src": 0}
    with out_csv.open("a", encoding="utf-8", newline="") as sink:
        for vid in videos_iter:
            rows = fakes[(fakes["video_id"] == vid) & ~fakes["rel_path"].isin(done)]
            if rows.empty:
                continue
            fake_path, src_id = video_file(vid), source_video(vid, args.src_naming)
            src_path = video_file(src_id) if isinstance(src_id, str) else None
            records = []
            if fake_path is None or src_path is None:
                stats["missing_src"] += len(rows)
                for _, row in rows.iterrows():
                    records.append({"rel_path": row["rel_path"], "mask_rel": "", "aligned": False, "border_mad": np.nan,
                                    "mask_area": np.nan, **{f"ratio_{r}": np.nan for r in REGIONS},
                                    "dominant_region": "none"})
            else:
                wanted = rows["frame_idx"].astype(int).tolist()
                fake_frames, real_frames = read_frames(fake_path, wanted), read_frames(src_path, wanted)
                for _, row in rows.iterrows():
                    idx = int(row["frame_idx"])
                    if idx not in fake_frames or idx not in real_frames:
                        continue
                    kps = parse_kps(row["kps"])
                    crop_size = args.size
                    fake = crop_square(fake_frames[idx], row["x1"], row["y1"], row["side"], crop_size)
                    real = crop_square(real_frames[idx], row["x1"], row["y1"], row["side"], crop_size)
                    regions = region_masks(kps, crop_size)
                    mask, border_mad = diff_mask(fake, real, regions["face"], args.min_thr)
                    aligned = bool(np.isfinite(border_mad) and border_mad <= args.border_mad_thr)
                    ratios = region_ratios(mask, regions)
                    mask_rel = row["rel_path"]
                    out_path = masks_root / mask_rel
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), mask * 255)
                    stats["total"] += 1
                    stats["aligned"] += int(aligned)
                    records.append({"rel_path": row["rel_path"], "mask_rel": mask_rel, "aligned": aligned,
                                    "border_mad": round(border_mad, 3), "mask_area": round(float(mask.mean()), 4),
                                    **{f"ratio_{r}": round(v, 4) for r, v in ratios.items()},
                                    "dominant_region": dominant_region(ratios) if aligned else "none"})
            if records:
                pd.DataFrame.from_records(records).to_csv(sink, header=header, index=False)
                header = False
                sink.flush()

    total = max(1, stats["total"])
    print(f"Frames processed: {stats['total']}, aligned: {stats['aligned']} ({100 * stats['aligned'] / total:.1f}%), "
          f"missing source video: {stats['missing_src']}")
    if stats["total"] and stats["aligned"] / total < 0.5:
        print("[WARN] Fewer than half of the pairs align. The fake->source naming assumption in "
              "prepare_celebdf.py (id{A}_id{B}_{k} -> Celeb-real/id{A}_{k}) is probably wrong; rerun with --src-naming source.")



if __name__ == "__main__":
    main()
