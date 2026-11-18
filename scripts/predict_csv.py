"""Run a trained checkpoint on a manifest split or a synthetic set and write per-frame predictions.

Output CSV columns: rel_path, video_id, label, prob  (+ the synth columns when --synth is used).

Examples
--------
# Celeb-DF test (official protocol)
python scripts/predict_csv.py --checkpoint checkpoints/celebdf_B0_seed0/best.pth \
    --manifest data/celebdf/meta.csv --frames-root data/celebdf/frames --protocol official --split test \
    --out results/B0_seed0/test_official.csv

# Robustness (P-robust): JPEG quality 30
python scripts/predict_csv.py ... --perturb jpeg:30 --out results/B0_seed0/test_jpeg30.csv

# Synthetic probe grid (P2 error report)
python scripts/predict_csv.py --checkpoint ... --synth data/celebdf/synth/probe_val/synth_meta.csv \
    --out results/round1/probe.csv
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import get_eval_transform  # noqa: E402
from models import build_model  # noqa: E402


def perturb(image_rgb: np.ndarray, spec: str) -> np.ndarray:
    """spec: jpeg:Q | blur:K | resize:F | none."""
    if not spec or spec == "none":
        return image_rgb
    kind, _, value = spec.partition(":")
    if kind == "jpeg":
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(value)])
        return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "blur":
        k = int(value) | 1
        return cv2.GaussianBlur(image_rgb, (k, k), 0)
    if kind == "resize":
        h, w = image_rgb.shape[:2]
        f = float(value)
        small = cv2.resize(image_rgb, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    raise ValueError(f"Unknown perturbation: {spec}")


class ListDataset(Dataset):
    def __init__(self, root: Path, frame: pd.DataFrame, transform, spec: str) -> None:
        self.root, self.frame, self.transform, self.spec = root, frame.reset_index(drop=True), transform, spec

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        image = cv2.imread(str(self.root / row["rel_path"]))
        if image is None:
            raise RuntimeError(f"Cannot read {self.root / row['rel_path']}")
        image = perturb(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.spec)
        return self.transform(image=image)["image"], i


def load_model(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = build_model(
        backbone=str(cfg.get("backbone", "redesigned_favit")),
        pretrained=False,
        dropout=float(cfg.get("dropout", 0.4)),
        image_size=int(cfg.get("image_size", 224)),
        freq_in_channels=int(cfg.get("freq_in_channels", 3)),
        freq_dim=int(cfg.get("freq_dim", 128)),
        use_freq=bool(cfg.get("use_freq", True)),
        favit_checkpoint=cfg.get("favit_checkpoint"),
        swin_checkpoint=cfg.get("swin_checkpoint"),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.to(device).eval(), cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest")
    p.add_argument("--frames-root")
    p.add_argument("--protocol", default="official")
    p.add_argument("--split", default="test")
    p.add_argument("--synth", help="synth_meta.csv instead of a manifest split")
    p.add_argument("--synth-root", help="defaults to the folder of --synth")
    p.add_argument("--perturb", default="none", help="jpeg:Q | blur:K | resize:F")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, cfg = load_model(args.checkpoint, device)

    if args.synth:
        frame = pd.read_csv(args.synth)
        root = Path(args.synth_root) if args.synth_root else Path(args.synth).parent
        if "video_id" not in frame.columns:
            frame["video_id"] = frame["rel_path"]
    else:
        if not (args.manifest and args.frames_root):
            raise SystemExit("--manifest and --frames-root are required without --synth")
        frame = pd.read_csv(args.manifest)
        frame = frame[frame[args.protocol] == args.split]
        root = Path(args.frames_root)

    dataset = ListDataset(root, frame, get_eval_transform(int(cfg.get("image_size", 224))), args.perturb)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    probs = np.zeros(len(dataset), dtype=np.float32)
    with torch.no_grad():
        for images, idx in loader:
            out = model(images.to(device, non_blocking=True))
            logits = out[0] if isinstance(out, tuple) else out
            probs[idx.numpy()] = torch.sigmoid(logits.view(-1)).float().cpu().numpy()

    result = dataset.frame.copy()
    result["prob"] = probs
    keep = [c for c in result.columns if c in {"rel_path", "video_id", "label", "prob", "region", "artifact",
                                                 "strength", "op_label", "round", "src_rel_path"}]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result[keep].to_csv(out, index=False)
    print(f"Wrote {out} ({len(result)} rows, perturb={args.perturb})")


if __name__ == "__main__":
    main()
