"""P2.3 — Per-cell error report that the planner reads.

Inputs (from scripts/predict_csv.py with the current checkpoint):
  --probe-preds   predictions on the probe grid (synth.generate --grid on VAL real frames)
  --val-preds     predictions on Celeb-DF val frames
  --masks-csv     optional, to break down val false negatives by dominant manipulated region

Per fake cell (region, artifact): AUC of probe fakes vs clean real val frames, TPR@0.5, and
TPR per strength. Per real op: FPR@0.5. Output: report.json + report.md (compact table for the LLM).

Example
-------
python -m synth.error_report --probe-preds results/round1/probe.csv --val-preds results/round1/val.csv \
    --masks-csv data/celebdf/masks.csv --out results/round1/report
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from synth.recipes import FAKE_ARTIFACTS, REAL_OPS, REGIONS


def build_report(probe: pd.DataFrame, val: pd.DataFrame, masks: pd.DataFrame = None, thr: float = 0.5) -> dict:
    real_probs = val.loc[val["label"] == 0, "prob"].to_numpy()
    cells = []
    for region in REGIONS:
        for artifact in FAKE_ARTIFACTS:
            g = probe[(probe["region"] == region) & (probe["artifact"] == artifact)]
            if g.empty or len(real_probs) == 0:
                continue
            y = np.r_[np.ones(len(g)), np.zeros(len(real_probs))]
            s = np.r_[g["prob"].to_numpy(), real_probs]
            by_strength = {f"{k:.1f}": round(float((v["prob"] >= thr).mean()), 3)
                           for k, v in g.groupby(g["strength"].round(1))}
            cells.append({"region": region, "artifact": artifact, "n": int(len(g)),
                          "auc": round(float(roc_auc_score(y, s)), 4),
                          "tpr": round(float((g["prob"] >= thr).mean()), 3), "tpr_by_strength": by_strength})
    real_ops = []
    for op in REAL_OPS:
        g = probe[probe["artifact"] == op]
        if not g.empty:
            real_ops.append({"op": op, "n": int(len(g)), "fpr": round(float((g["prob"] >= thr).mean()), 3),
                             "mean_prob": round(float(g["prob"].mean()), 3)})

    y, s = val["label"].to_numpy().astype(int), val["prob"].to_numpy()
    val_summary = {"n": int(len(val)), "auc": round(float(roc_auc_score(y, s)), 4) if len(set(y)) == 2 else None,
                   "acc": round(float(((s >= thr) == y).mean()), 4),
                   "fpr_real": round(float((s[y == 0] >= thr).mean()), 4) if (y == 0).any() else None,
                   "fnr_fake": round(float((s[y == 1] < thr).mean()), 4) if (y == 1).any() else None}
    if masks is not None:
        fakes = val[val["label"] == 1].merge(masks[["rel_path", "dominant_region"]], on="rel_path", how="left")
        fakes["dominant_region"] = fakes["dominant_region"].fillna("none")
        val_summary["fnr_by_dominant_region"] = {
            k: {"n": int(len(g)), "fnr": round(float((g["prob"] < thr).mean()), 3)}
            for k, g in fakes.groupby("dominant_region")}
    return {"threshold": thr, "val": val_summary, "cells": sorted(cells, key=lambda c: c["auc"]), "real_ops": real_ops}


def to_markdown(report: dict) -> str:
    lines = ["## Celeb-DF validation", json.dumps(report["val"]), "",
             "## Synthetic probe cells (fake), sorted by AUC ascending (worst first)",
             "| region | artifact | AUC | TPR@0.5 | TPR by strength |", "|---|---|---|---|---|"]
    for c in report["cells"]:
        lines.append(f"| {c['region']} | {c['artifact']} | {c['auc']:.3f} | {c['tpr']:.2f} | "
                     f"{' '.join(f'{k}:{v:.2f}' for k, v in c['tpr_by_strength'].items())} |")
    lines += ["", "## Real-side probes (label 0): false positive rate",
              "| op | FPR@0.5 | mean prob |", "|---|---|---|"]
    lines += [f"| {r['op']} | {r['fpr']:.2f} | {r['mean_prob']:.2f} |" for r in report["real_ops"]]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-preds", required=True)
    p.add_argument("--val-preds", required=True)
    p.add_argument("--masks-csv")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", required=True, help="output prefix, writes <out>.json and <out>.md")
    args = p.parse_args()

    report = build_report(pd.read_csv(args.probe_preds), pd.read_csv(args.val_preds),
                          pd.read_csv(args.masks_csv) if args.masks_csv else None, args.threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    out.with_suffix(".md").write_text(to_markdown(report))
    print(to_markdown(report))


if __name__ == "__main__":
    main()
