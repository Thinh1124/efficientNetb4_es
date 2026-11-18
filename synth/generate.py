"""P2 — Generate structured synthetic data from Celeb-DF REAL frames.

Each image is produced by a recipe (region, artifact, strength) so its manipulation mask and
labels are exact. Modes:

  --plan plan.json        counts per cell from llm_planner.py (D2 = LLM, D3 = rule)
  --random N              N images spread uniformly over all cells (baseline D1)
  --grid --per-cell K     K images per cell per strength in --strengths (diagnostic probe set)

Source frames are taken ONLY from `--split` of `--protocol` (train for training data, val for
the probe set). The script refuses to use test frames.

Example
-------
python -m synth.generate --meta data/celebdf/meta.csv --frames-root data/celebdf/frames \
    --protocol official --split train --random 30000 --out-dir data/celebdf/synth/D1 --round 0
python -m synth.generate ... --split val --grid --per-cell 40 --out-dir data/celebdf/synth/probe_val
"""
import argparse
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.artifacts import apply, soft_alpha  # noqa: E402
from synth.recipes import FAKE_ARTIFACTS, REAL_OPS, REAL_REGION, REGIONS, all_cells, load_plan  # noqa: E402
from synth.regions import parse_kps, region_masks  # noqa: E402

_STATE: Dict = {}


def _init_worker(frames_root: str, out_dir: str, reals: pd.DataFrame, min_visible: float) -> None:
    _STATE.update(frames_root=Path(frames_root), out_dir=Path(out_dir), reals=reals, min_visible=min_visible,
                  by_video={v: g.index.to_numpy() for v, g in reals.groupby("video_id")})


def _aligned_donor(src_kps: np.ndarray, donor_row, size: int) -> np.ndarray:
    donor = cv2.imread(str(_STATE["frames_root"] / donor_row["rel_path"]))
    matrix, _ = cv2.estimateAffinePartial2D(parse_kps(donor_row["kps"]), src_kps)
    if matrix is None:
        return cv2.resize(donor, (size, size))
    return cv2.warpAffine(donor, matrix, (size, size), borderMode=cv2.BORDER_REFLECT)


def _make_one(job: Dict) -> Optional[Dict]:
    rng = np.random.default_rng(job["seed"])
    reals: pd.DataFrame = _STATE["reals"]
    src = reals.loc[job["src_index"]]
    img = cv2.imread(str(_STATE["frames_root"] / src["rel_path"]))
    if img is None:
        return None
    size = img.shape[0]
    kps = parse_kps(src["kps"])
    regions = region_masks(kps, size)
    artifact = job["artifact"]
    region = job["region"] if job["region"] != REAL_REGION else str(rng.choice(REGIONS))

    donor = None
    if artifact == "texture_inconsistency":  # different video (different identity most of the time)
        other = reals[reals["video_id"] != src["video_id"]]
        donor = _aligned_donor(kps, other.iloc[int(rng.integers(len(other)))], size) if len(other) else None
    elif artifact == "authentic_seam":  # same video, another frame -> stays real
        idx = _STATE["by_video"][src["video_id"]]
        idx = idx[idx != job["src_index"]]
        if len(idx) == 0:
            return None
        donor = _aligned_donor(kps, reals.loc[int(rng.choice(idx))], size)

    strength = job["strength"]
    for attempt in range(3):
        alpha = soft_alpha(regions[region], rng, size)
        out = apply(img, alpha, artifact, strength, rng, donor=donor)
        zone = alpha > 0.5 if artifact != "authentic_degrade" else np.ones(alpha.shape, bool)
        visible = float(np.abs(out.astype(np.float32) - img.astype(np.float32))[zone].mean()) if zone.any() else 0.0
        if artifact in REAL_OPS or visible >= _STATE["min_visible"]:
            break
        strength = min(1.0, strength + 0.25)
    else:
        return None

    label = 0 if artifact in REAL_OPS else 1
    name = f"{job['job_id']:08d}.png"
    rel = f"images/{'real' if label == 0 else 'fake'}/{artifact}/{job['region']}/{name}"
    mask_rel = f"masks/{artifact}/{job['region']}/{name}"
    for path, data in ((rel, out), (mask_rel, (alpha * 255).astype(np.uint8))):
        full = _STATE["out_dir"] / path
        full.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full), data)
    return {"rel_path": rel, "label": label, "region": job["region"] if label else REAL_REGION,
            "op_region": region, "artifact": artifact, "strength": round(strength, 3), "visible": round(visible, 2),
            "mask_rel": mask_rel if label else "", "src_rel_path": src["rel_path"], "src_video_id": src["video_id"],
            "video_id": f"synth::{src['video_id']}", "round": job["round"], "kps": src["kps"]}


def build_jobs(args, reals: pd.DataFrame, start_id: int) -> List[Dict]:
    rng = np.random.default_rng(args.seed)
    cells: List[Dict] = []
    if args.plan:
        for it in load_plan(args.plan):
            for _ in range(it.count):
                cells.append({"region": it.region, "artifact": it.artifact,
                              "strength": float(rng.uniform(*it.strength))})
    elif args.random:
        all_c = all_cells()
        for i in range(args.random):
            region, artifact = all_c[i % len(all_c)]
            cells.append({"region": region, "artifact": artifact, "strength": float(rng.uniform(0.2, 0.9))})
    elif args.grid:
        for region, artifact in all_cells():
            for s in args.strengths:
                cells += [{"region": region, "artifact": artifact, "strength": float(s)}] * args.per_cell
    else:
        raise SystemExit("Choose one of --plan, --random, --grid")

    # sample source frames video-balanced: pick a video, then a frame
    videos = reals["video_id"].unique()
    by_video = {v: g.index.to_numpy() for v, g in reals.groupby("video_id")}
    jobs = []
    for i, cell in enumerate(cells):
        vid = videos[rng.integers(len(videos))]
        jobs.append({**cell, "src_index": int(rng.choice(by_video[vid])), "seed": int(args.seed * 1_000_003 + i),
                     "job_id": start_id + i, "round": args.round})
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", required=True)
    p.add_argument("--frames-root", required=True)
    p.add_argument("--protocol", default="official")
    p.add_argument("--split", default="train", choices=["train", "val"])
    p.add_argument("--out-dir", required=True)
    p.add_argument("--plan")
    p.add_argument("--random", type=int, default=0)
    p.add_argument("--grid", action="store_true")
    p.add_argument("--per-cell", type=int, default=40)
    p.add_argument("--strengths", type=float, nargs="+", default=[0.3, 0.6, 0.9])
    p.add_argument("--round", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-visible", type=float, default=3.0, help="min mean |diff| inside the region (0-255)")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    args = p.parse_args()

    meta = pd.read_csv(args.meta)
    reals = meta[(meta[args.protocol] == args.split) & (meta["label"] == 0)].copy()
    if reals.empty:
        raise SystemExit("No real frames for this protocol/split")
    assert (reals[args.protocol] != "test").all(), "refusing to synthesise from test frames"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_csv = out_dir / "synth_meta.csv"
    start_id = 0
    if meta_csv.exists():
        prev = pd.read_csv(meta_csv)
        if (prev["round"] == args.round).any():
            raise SystemExit(f"Round {args.round} already exists in {meta_csv}; use a new --round")
        start_id = int(prev["rel_path"].str.extract(r"(\d+)\.png$")[0].astype(int).max()) + 1

    jobs = build_jobs(args, reals, start_id)
    print(f"Generating {len(jobs)} images from {reals['video_id'].nunique()} real videos "
          f"({args.protocol}/{args.split}) with {args.workers} workers")
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(args.frames_root, str(out_dir), reals, args.min_visible)) as pool:
        results = [r for r in pool.imap_unordered(_make_one, jobs, chunksize=32) if r]

    if not results:
        raise SystemExit("No image passed the visibility check; lower --min-visible or raise strengths")
    df = pd.DataFrame(results).sort_values("rel_path")
    asked = pd.DataFrame(jobs).groupby(["artifact", "region"]).size().rename("asked")
    kept = df.groupby(["artifact", "region"]).size().rename("kept")
    drop = pd.concat([asked, kept], axis=1).fillna(0)
    drop["drop_rate"] = (1 - drop["kept"] / drop["asked"]).round(2)
    bad = drop[drop["drop_rate"] > 0.3]
    if not bad.empty:
        print("[warn] cells with >30% dropped (artifact not visible at this strength / no donor):")
        print(bad.to_string())
    df.to_csv(meta_csv, mode="a", header=not meta_csv.exists(), index=False)
    print(f"Kept {len(df)}/{len(jobs)} (dropped {len(jobs) - len(df)} not visible / no donor). Wrote {meta_csv}")
    print(df.groupby(["artifact", "region"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
