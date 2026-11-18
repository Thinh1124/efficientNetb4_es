"""P1.3 — Quality control of MLLM annotations and per-finding confidence q.

Subcommands
-----------
sample-review  Export N frames (stratified by artifact) to a CSV + side-by-side images for two
               human reviewers. Reviewers fill region_ok / artifact_ok with 1 or 0 per finding.
calibrate      Read the filled reviewer CSVs -> per-artifact precision (calib) + Cohen's kappa.
score          q = confidence * agree_mask * calib[artifact]; writes annotations_qc.jsonl with
               frame-level region_vec[5] and artifact_vec[7] (max q), plus a summary.

agree_mask: 1 if the pseudo-mask covers >= --min-ratio of the claimed region, --disagree-weight
otherwise. A non-"authentic" finding on a REAL frame is counted as an MLLM false positive.

Example
-------
python scripts/qc_annotations.py sample-review --ann data/celebdf/annotations/train_mllm.jsonl \
    --frames-root data/celebdf/frames --masks-root data/celebdf/masks --n 200 --out-dir data/celebdf/review
python scripts/qc_annotations.py calibrate --reviews data/celebdf/review/reviewer_A.csv data/celebdf/review/reviewer_B.csv \
    --out data/celebdf/annotations/calib.json
python scripts/qc_annotations.py score --ann data/celebdf/annotations/train_mllm.jsonl --masks-csv data/celebdf/masks.csv \
    --calib data/celebdf/annotations/calib.json --out data/celebdf/annotations/train_qc.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.annotate_mllm import overlay  # noqa: E402
from synth.recipes import ARTIFACTS, REGIONS  # noqa: E402


def load_ann(path: str) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return pd.DataFrame(rows)


def explode(ann: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for _, r in ann.iterrows():
        for j, f in enumerate(r["findings"]):
            recs.append({"rel_path": r["rel_path"], "label": r["label"], "finding_idx": j, **f})
    return pd.DataFrame(recs)


def cmd_sample_review(args) -> None:
    ann = load_ann(args.ann)
    findings = explode(ann[ann["parse_ok"]])
    rng = np.random.default_rng(args.seed)
    per_art = max(1, args.n // max(1, findings["artifact"].nunique()))
    picked_paths = []
    for _, g in findings.groupby("artifact"):
        paths = g["rel_path"].unique()
        picked_paths += list(rng.choice(paths, size=min(per_art, len(paths)), replace=False))
    picked = findings[findings["rel_path"].isin(set(picked_paths[: args.n]))].copy()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    for path in picked["rel_path"].unique():
        image = cv2.imread(str(Path(args.frames_root) / path))
        panels = [image]
        mask_path = Path(args.masks_root) / path if args.masks_root else None
        if mask_path is not None and mask_path.exists():
            panels.append(overlay(image, cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)))
        big = cv2.resize(np.concatenate(panels, axis=1), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(img_dir / (path.replace("/", "__"))), big)
    picked["image"] = "images/" + picked["rel_path"].str.replace("/", "__", regex=False)
    picked["region_ok"] = ""
    picked["artifact_ok"] = ""
    cols = ["image", "rel_path", "label", "finding_idx", "region", "artifact", "severity", "confidence",
            "region_ok", "artifact_ok"]
    for reviewer in ("A", "B"):
        picked[cols].to_csv(out_dir / f"reviewer_{reviewer}.csv", index=False)
    print(f"{picked['rel_path'].nunique()} frames / {len(picked)} findings -> {out_dir} "
          f"(fill region_ok, artifact_ok with 1/0 in reviewer_A.csv and reviewer_B.csv independently)")


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_calibrate(args) -> None:
    reviews = [pd.read_csv(p) for p in args.reviews]
    for r in reviews:
        if r[["region_ok", "artifact_ok"]].isna().any().any():
            raise SystemExit("Some region_ok/artifact_ok cells are empty")
    key = ["rel_path", "finding_idx"]
    merged = reviews[0]
    for i, r in enumerate(reviews[1:], start=1):
        merged = merged.merge(r[key + ["region_ok", "artifact_ok"]], on=key, suffixes=("", f"_{i}"))
    ok_cols = [c for c in merged.columns if c.startswith("artifact_ok")]
    reg_cols = [c for c in merged.columns if c.startswith("region_ok")]
    merged["artifact_ok_mean"] = merged[ok_cols].mean(axis=1)
    merged["region_ok_mean"] = merged[reg_cols].mean(axis=1)
    calib = {a: round(float(g["artifact_ok_mean"].mean()), 3) for a, g in merged.groupby("artifact")}
    located = merged[merged["artifact"] != "authentic"]  # "authentic" findings have no region to judge
    out = {"artifact_precision": calib,
           "region_precision": round(float(located["region_ok_mean"].mean()), 3) if len(located) else None,
           "n_findings": int(len(merged))}
    if len(ok_cols) >= 2:
        out["kappa_artifact"] = round(kappa(merged[ok_cols[0]].values.astype(int), merged[ok_cols[1]].values.astype(int)), 3)
        if len(located):
            out["kappa_region"] = round(kappa(located[reg_cols[0]].values.astype(int),
                                              located[reg_cols[1]].values.astype(int)), 3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    if out["region_precision"] is not None and out["region_precision"] < 0.6:
        print("[GO/NO-GO] region precision < 0.6: use pseudo-mask regions + synthetic labels instead of MLLM regions.")


def cmd_score(args) -> None:
    ann = load_ann(args.ann)
    masks = pd.read_csv(args.masks_csv).set_index("rel_path") if args.masks_csv else None
    calib = json.loads(Path(args.calib).read_text())["artifact_precision"] if args.calib else {}
    stats = {"frames": 0, "parse_fail": 0, "findings": 0, "fake_findings": 0, "mask_disagree": 0,
             "real_false_positive": 0}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        for _, r in ann.iterrows():
            stats["frames"] += 1
            if not r["parse_ok"]:
                stats["parse_fail"] += 1
                continue
            ratios = None
            if masks is not None and r["rel_path"] in masks.index and bool(masks.loc[r["rel_path"], "aligned"]):
                ratios = {g: float(masks.loc[r["rel_path"], f"ratio_{g}"]) for g in REGIONS}
            region_vec, artifact_vec, scored = np.zeros(len(REGIONS)), np.zeros(len(ARTIFACTS)), []
            for f in r["findings"]:
                stats["findings"] += 1
                q = float(f["confidence"]) * float(calib.get(f["artifact"], 1.0))
                if f["artifact"] != "authentic":
                    if r["label"] == 0:
                        stats["real_false_positive"] += 1
                        q *= args.disagree_weight
                    else:
                        stats["fake_findings"] += 1
                    if r["label"] == 1 and ratios is not None and ratios.get(f["region"], 0.0) < args.min_ratio:
                        stats["mask_disagree"] += 1
                        q *= args.disagree_weight
                    if f["region"] in REGIONS:
                        region_vec[REGIONS.index(f["region"])] = max(region_vec[REGIONS.index(f["region"])], q)
                artifact_vec[ARTIFACTS.index(f["artifact"])] = max(artifact_vec[ARTIFACTS.index(f["artifact"])], q)
                scored.append({**f, "q": round(q, 4)})
            sink.write(json.dumps({"rel_path": r["rel_path"], "label": int(r["label"]), "findings": scored,
                                   "region_vec": region_vec.round(4).tolist(),
                                   "artifact_vec": artifact_vec.round(4).tolist(),
                                   "regions": list(REGIONS), "artifacts": list(ARTIFACTS)}) + "\n")
    print(json.dumps(stats, indent=1))
    if stats["fake_findings"]:
        print(f"mask disagreement rate (non-authentic findings on fake frames): "
              f"{stats['mask_disagree'] / stats['fake_findings']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample-review")
    s.add_argument("--ann", required=True)
    s.add_argument("--frames-root", required=True)
    s.add_argument("--masks-root")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", required=True)
    s.set_defaults(func=cmd_sample_review)
    c = sub.add_parser("calibrate")
    c.add_argument("--reviews", nargs="+", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=cmd_calibrate)
    sc = sub.add_parser("score")
    sc.add_argument("--ann", required=True)
    sc.add_argument("--masks-csv")
    sc.add_argument("--calib")
    sc.add_argument("--min-ratio", type=float, default=0.05)
    sc.add_argument("--disagree-weight", type=float, default=0.2)
    sc.add_argument("--out", required=True)
    sc.set_defaults(func=cmd_score)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
