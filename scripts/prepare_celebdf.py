"""P0 — Celeb-DF v2 preparation: splits, frame extraction + face crops, manifest.

Subcommands
-----------
split    Build the official split (518 test videos from List_of_testing_videos.txt)
         and an identity-disjoint split (S-id). Writes one JSON file.
extract  Decode sampled frames, detect the face (SCRFD via InsightFace), save square
         crops and per-frame metadata (crop box + 5 keypoints). Supports sharding so
         several processes/GPUs can run in parallel. Resumable.
merge    Merge the shard files into one manifest CSV with split columns.

Example
-------
python scripts/prepare_celebdf.py split   --celebdf-root /data/Celeb-DF-v2 --out data/celebdf/splits.json
python scripts/prepare_celebdf.py extract --celebdf-root /data/Celeb-DF-v2 --splits data/celebdf/splits.json \
       --frames-root data/celebdf/frames --shard 0 --num-shards 1
python scripts/prepare_celebdf.py merge   --splits data/celebdf/splits.json --frames-root data/celebdf/frames \
       --out data/celebdf/meta.csv
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.regions import crop_square, format_kps, square_box  # noqa: E402

REAL_FOLDERS = ("Celeb-real", "YouTube-real")
FAKE_FOLDER = "Celeb-synthesis"
FAKE_RE = re.compile(r"^id(\d+)_id(\d+)_(\d+)$")
CELEB_REAL_RE = re.compile(r"^id(\d+)_(\d+)$")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


# --------------------------------------------------------------------------- split
def list_videos(root: Path) -> Dict[str, Dict]:
    videos: Dict[str, Dict] = {}
    for folder in REAL_FOLDERS + (FAKE_FOLDER,):
        folder_dir = root / folder
        if not folder_dir.is_dir():
            print(f"[warn] missing folder: {folder_dir}")
            continue
        for path in sorted(folder_dir.iterdir()):
            if path.suffix.lower() not in VIDEO_EXTS:
                continue
            vid = f"{folder}/{path.stem}"
            info = {"file": f"{folder}/{path.name}", "folder": folder, "label": 1 if folder == FAKE_FOLDER else 0,
                    "target_id": None, "source_id": None, "src_video": None}
            if folder == FAKE_FOLDER:
                m = FAKE_RE.match(path.stem)
                if m:
                    a, b, k = m.groups()
                    info["target_id"], info["source_id"] = int(a), int(b)
                    # ASSUMPTION: id{A}_id{B}_{k} = real video id{A}_{k} with the face of id{B}.
                    # make_pseudo_masks.py measures how many pairs actually align.
                    info["src_video"] = f"Celeb-real/id{a}_{k}"
            elif folder == "Celeb-real":
                m = CELEB_REAL_RE.match(path.stem)
                if m:
                    info["target_id"] = int(m.group(1))
            videos[vid] = info
    return videos


def read_official_test(root: Path) -> set:
    list_file = root / "List_of_testing_videos.txt"
    if not list_file.exists():
        raise FileNotFoundError(f"Missing {list_file}")
    test = set()
    for line in list_file.read_text().splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        rel = parts[-1].replace("\\", "/")
        folder, name = rel.split("/")[-2:]
        test.add(f"{folder}/{Path(name).stem}")
    return test


def _assign_val(videos: Dict[str, Dict], key: str, val_ratio: float, rng: random.Random) -> None:
    for label in (0, 1):
        pool = sorted(v for v, i in videos.items() if i[key] == "train" and i["label"] == label)
        rng.shuffle(pool)
        for vid in pool[: int(round(len(pool) * val_ratio))]:
            videos[vid][key] = "val"


def cmd_split(args) -> None:
    root = Path(args.celebdf_root)
    videos = list_videos(root)
    if not videos:
        raise SystemExit("No videos found. Check --celebdf-root.")
    test = read_official_test(root)
    missing = sorted(test - set(videos))
    if missing:
        print(f"[warn] {len(missing)} test videos from the list were not found, e.g. {missing[:3]}")

    rng = random.Random(args.seed)
    for vid, info in videos.items():
        info["official"] = "test" if vid in test else "train"
    _assign_val(videos, "official", args.val_ratio, rng)

    identities = sorted({i["target_id"] for i in videos.values() if i["target_id"] is not None}
                        | {i["source_id"] for i in videos.values() if i["source_id"] is not None})
    heldout = set(random.Random(args.seed + 1).sample(identities, min(args.sid_heldout, len(identities))))
    yt_rng = random.Random(args.seed + 2)
    for vid, info in sorted(videos.items()):
        ids = {i for i in (info["target_id"], info["source_id"]) if i is not None}
        if not ids:  # YouTube-real: no identity label
            info["sid"] = "test" if yt_rng.random() < args.yt_test_ratio else "train"
        elif ids <= heldout:
            info["sid"] = "test"
        elif ids.isdisjoint(heldout):
            info["sid"] = "train"
        else:
            info["sid"] = "test" if args.sid_mode == "loose" else "drop"
    _assign_val(videos, "sid", args.val_ratio, rng)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"celebdf_root": str(root), "seed": args.seed, "val_ratio": args.val_ratio,
                        "sid_heldout_ids": sorted(heldout), "sid_mode": args.sid_mode,
                        "assumption_fake_naming": "id{A}_id{B}_{k} -> source real Celeb-real/id{A}_{k}"},
               "videos": videos}
    out.write_text(json.dumps(payload, indent=1))
    for key in ("official", "sid"):
        counts = Counter((i[key], "fake" if i["label"] else "real") for i in videos.values())
        print(f"[{key}] " + ", ".join(f"{s}/{l}={n}" for (s, l), n in sorted(counts.items())))
    print(f"Wrote {out}")


# ------------------------------------------------------------------------- extract
class Detector:
    """Face detector returning (bbox[4], kps[5,2], score) for the main face or None."""

    def __init__(self, kind: str, ctx_id: int, det_size: int) -> None:
        self.kind = kind
        if kind == "insightface":
            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:
                raise SystemExit("pip install insightface onnxruntime-gpu  (see requirements-llm.txt)") from exc
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=providers)
            self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        elif kind != "center":
            raise ValueError(f"Unknown detector {kind}")

    def __call__(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        if self.kind == "center":  # debug/testing only: fixed central box + canonical keypoints
            side = min(h, w) * 0.5
            x1, y1 = (w - side) / 2, (h - side) / 2
            canon = np.array([[0.35, 0.40], [0.65, 0.40], [0.50, 0.55], [0.38, 0.72], [0.62, 0.72]], np.float32)
            return np.array([x1, y1, x1 + side, y1 + side], np.float32), canon * side + [x1, y1], 1.0
        faces = self.app.get(frame_bgr)
        if not faces:
            return None
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) * float(f.det_score))
        return np.asarray(best.bbox, np.float32), np.asarray(best.kps, np.float32), float(best.det_score)


def sample_indices(n_frames: int, k: int) -> List[int]:
    if n_frames <= 0:
        return []
    return sorted(set(np.linspace(0, n_frames - 1, num=min(k, n_frames)).round().astype(int).tolist()))


def read_frames(path: Path, wanted: List[int]) -> Dict[int, np.ndarray]:
    """Sequential decode (exact frame indices, unlike CAP_PROP_POS_FRAMES seeking)."""
    cap = cv2.VideoCapture(str(path))
    out: Dict[int, np.ndarray] = {}
    wanted_set, last = set(wanted), max(wanted) if wanted else -1
    idx = 0
    while idx <= last:
        ok = cap.grab()
        if not ok:
            break
        if idx in wanted_set:
            ok, frame = cap.retrieve()
            if ok:
                out[idx] = frame
        idx += 1
    cap.release()
    return out


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def cmd_extract(args) -> None:
    root = Path(args.celebdf_root)
    splits = json.loads(Path(args.splits).read_text())["videos"]
    frames_root = Path(args.frames_root)
    shard_file = frames_root / "_shards" / f"meta_shard{args.shard:03d}.jsonl"
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if shard_file.exists():
        for line in shard_file.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["video_id"])

    vids = sorted(splits)
    if args.limit_per_folder:
        per_folder: Dict[str, int] = Counter()
        kept = []
        for vid in vids:
            folder = splits[vid]["folder"]
            if per_folder[folder] < args.limit_per_folder:
                kept.append(vid)
                per_folder[folder] += 1
        vids = kept
    vids = [v for i, v in enumerate(vids) if i % args.num_shards == args.shard and v not in done]
    print(f"Shard {args.shard}/{args.num_shards}: {len(vids)} videos to process ({len(done)} already done)")

    detector = Detector(args.detector, args.ctx_id, args.det_size)
    try:
        from tqdm import tqdm
        iterator = tqdm(vids)
    except ImportError:
        iterator = vids

    with shard_file.open("a", encoding="utf-8") as sink:
        for vid in iterator:
            info = splits[vid]
            video_path = root / info["file"]
            n_frames = video_frame_count(video_path)
            k = args.frames_fake if info["label"] == 1 else args.frames_real
            frames = read_frames(video_path, sample_indices(n_frames, k))
            rows = []
            for idx, frame in sorted(frames.items()):
                det = detector(frame)
                if det is None or det[2] < args.min_score:
                    continue
                bbox, kps, score = det
                x1, y1, side = square_box(bbox, args.margin)
                crop = crop_square(frame, x1, y1, side, args.size)
                kps_crop = (kps - [x1, y1]) * (args.size / side)
                rel = f"{vid}/{idx:05d}.png"
                out_path = frames_root / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_path), crop)
                rows.append({"rel_path": rel, "frame_idx": idx, "x1": round(float(x1), 2), "y1": round(float(y1), 2),
                             "side": round(float(side), 2), "det_score": round(score, 4), "kps": format_kps(kps_crop)})
            sink.write(json.dumps({"video_id": vid, "n_frames": n_frames, "rows": rows}) + "\n")
            sink.flush()


# --------------------------------------------------------------------------- merge
def cmd_merge(args) -> None:
    import pandas as pd

    splits = json.loads(Path(args.splits).read_text())["videos"]
    shard_dir = Path(args.frames_root) / "_shards"
    records, empty = [], []
    for shard in sorted(shard_dir.glob("meta_shard*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            info = splits[item["video_id"]]
            if not item["rows"]:
                empty.append(item["video_id"])
            for row in item["rows"]:
                records.append({**row, "video_id": item["video_id"], "folder": info["folder"], "label": info["label"],
                                "n_frames": item["n_frames"], "target_id": info["target_id"],
                                "source_id": info["source_id"], "src_video": info["src_video"],
                                "official": info["official"], "sid": info["sid"]})
    df = pd.DataFrame.from_records(records).drop_duplicates("rel_path")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {len(df)} frames from {df['video_id'].nunique()} videos")
    if empty:
        print(f"[warn] {len(empty)} videos with no detected face, e.g. {empty[:3]}")
    for key in ("official", "sid"):
        print(df.groupby([key, "label"]).agg(frames=("rel_path", "size"), videos=("video_id", "nunique")))
    missing = sorted(set(splits) - set(df["video_id"]))
    if missing:
        print(f"[info] {len(missing)} videos not (yet) extracted, e.g. {missing[:3]}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split")
    s.add_argument("--celebdf-root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--val-ratio", type=float, default=0.1)
    s.add_argument("--sid-heldout", type=int, default=12, help="number of identities reserved for S-id test")
    s.add_argument("--sid-mode", choices=["strict", "loose"], default="strict",
                   help="strict: fakes mixing seen+held-out identities are dropped; loose: put in test")
    s.add_argument("--yt-test-ratio", type=float, default=0.1)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_split)

    e = sub.add_parser("extract")
    e.add_argument("--celebdf-root", required=True)
    e.add_argument("--splits", required=True)
    e.add_argument("--frames-root", required=True)
    e.add_argument("--frames-real", type=int, default=64)
    e.add_argument("--frames-fake", type=int, default=32)
    e.add_argument("--size", type=int, default=224)
    e.add_argument("--margin", type=float, default=1.3)
    e.add_argument("--min-score", type=float, default=0.5)
    e.add_argument("--detector", choices=["insightface", "center"], default="insightface")
    e.add_argument("--det-size", type=int, default=640)
    e.add_argument("--ctx-id", type=int, default=0, help="GPU id for InsightFace, -1 for CPU")
    e.add_argument("--shard", type=int, default=0)
    e.add_argument("--num-shards", type=int, default=1)
    e.add_argument("--limit-per-folder", type=int, default=0, help="debug: only N videos per folder")
    e.set_defaults(func=cmd_extract)

    m = sub.add_parser("merge")
    m.add_argument("--splits", required=True)
    m.add_argument("--frames-root", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(func=cmd_merge)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
