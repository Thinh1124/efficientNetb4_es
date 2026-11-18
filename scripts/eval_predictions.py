"""Frame- and video-level metrics with bootstrap CIs (resampling videos) for prediction CSVs.

Input: one or more CSVs from predict_csv.py (columns rel_path, video_id, label, prob).
Video score = mean frame probability.

Example
-------
python scripts/eval_predictions.py results/B0_seed*/test_official.csv --out results/summary_official.csv
"""
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def eer(y: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2)


def metrics(y: np.ndarray, s: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"auc": float("nan"), "ap": float("nan"), "eer": float("nan"), "acc": float(((s >= thr) == y).mean())}
    return {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
            "eer": eer(y, s), "acc": float(((s >= thr) == y).mean())}


def evaluate(df: pd.DataFrame, n_boot: int, seed: int) -> Dict[str, float]:
    df = df.copy()
    df["label"] = df["label"].astype(int)
    video = df.groupby("video_id").agg(label=("label", "first"), prob=("prob", "mean")).reset_index()
    out = {f"frame_{k}": v for k, v in metrics(df["label"].values, df["prob"].values).items()}
    out.update({f"video_{k}": v for k, v in metrics(video["label"].values, video["prob"].values).items()})
    out["n_frames"], out["n_videos"] = len(df), len(video)

    if n_boot > 0 and video["label"].nunique() == 2:
        rng = np.random.default_rng(seed)
        groups = {vid: (g["label"].values, g["prob"].values) for vid, g in df.groupby("video_id")}
        vprob = video.set_index("video_id")
        vids_by_label = [video.loc[video["label"] == c, "video_id"].values for c in (0, 1)]
        v_auc, f_auc = [], []
        for _ in range(n_boot):
            pick = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in vids_by_label])
            vs = vprob.loc[pick]
            v_auc.append(roc_auc_score(vs["label"], vs["prob"]))
            f_auc.append(roc_auc_score(np.concatenate([groups[v][0] for v in pick]),
                                       np.concatenate([groups[v][1] for v in pick])))
        out["video_auc_lo"], out["video_auc_hi"] = np.percentile(v_auc, [2.5, 97.5]).tolist()
        out["frame_auc_lo"], out["frame_auc_hi"] = np.percentile(f_auc, [2.5, 97.5]).tolist()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csvs", nargs="+")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    args = p.parse_args()

    rows = []
    for path in args.csvs:
        res = evaluate(pd.read_csv(path), args.n_boot, args.seed)
        res["file"] = path
        rows.append(res)
        ci = (f" [{res['video_auc_lo']:.4f}, {res['video_auc_hi']:.4f}]" if "video_auc_lo" in res else "")
        print(f"{path}: frame AUC {res['frame_auc']:.4f} | video AUC {res['video_auc']:.4f}{ci} | "
              f"ACC {res['frame_acc']:.4f} | AP {res['frame_ap']:.4f} | EER {res['frame_eer']:.4f}")
    table = pd.DataFrame(rows)
    if len(rows) > 1:
        num = table.select_dtypes("number")
        print("\nmean ± std over files:")
        for col in ("frame_auc", "video_auc", "frame_acc", "frame_ap", "frame_eer"):
            print(f"  {col}: {num[col].mean():.4f} ± {num[col].std(ddof=1):.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
