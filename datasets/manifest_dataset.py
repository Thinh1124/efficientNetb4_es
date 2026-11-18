"""Datasets driven by CSV manifests (Celeb-DF meta.csv and synthetic synth_meta.csv).

A manifest row points at an image by a POSIX-style path relative to a root folder, so the
same CSV works on Windows and Linux. The split is chosen per protocol column
("official" or "sid"), which avoids duplicating images for each protocol.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


def _read_rgb(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class ManifestFrameDataset(Dataset):
    """Frames listed in a manifest CSV. Returns (image, label, rel_path) like the folder datasets."""

    def __init__(
        self,
        manifest: str,
        root: str,
        split: str,
        protocol: str = "official",
        transform=None,
        real_percent: float = 100.0,
        fake_percent: float = 100.0,
        seed: int = 42,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        df = pd.read_csv(manifest)
        if protocol not in df.columns:
            raise KeyError(f"Protocol column '{protocol}' not in {manifest}")
        df = df[df[protocol] == split]
        df = self._subsample_videos(df, label=0, percent=real_percent, seed=seed)
        df = self._subsample_videos(df, label=1, percent=fake_percent, seed=seed)
        if df.empty:
            raise RuntimeError(f"No rows for protocol={protocol} split={split} in {manifest}")
        self.frame = df.reset_index(drop=True)
        # Same layout as DeepfakeFrameDataset.samples so pos_weight counting keeps working.
        self.samples: List[Tuple[str, float]] = list(zip(self.frame["rel_path"], self.frame["label"].astype(float)))

    @staticmethod
    def _subsample_videos(df: pd.DataFrame, label: int, percent: float, seed: int) -> pd.DataFrame:
        """Low-data protocol (P-low): keep a percentage of *videos* (not frames) of one class."""
        if percent >= 100.0:
            return df
        videos = sorted(df.loc[df["label"] == label, "video_id"].unique())
        random.Random(seed).shuffle(videos)
        keep = set(videos[: max(1, int(len(videos) * percent / 100.0))])
        return df[(df["label"] != label) | df["video_id"].isin(keep)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image = _read_rgb(self.root / rel_path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.float32), rel_path


class SynthFrameDataset(Dataset):
    """Images produced by synth/generate.py, listed in synth_meta.csv."""

    def __init__(self, manifest: str, root: Optional[str] = None, transform=None,
                 rounds: Optional[Sequence[int]] = None) -> None:
        self.root = Path(root) if root else Path(manifest).parent
        self.transform = transform
        df = pd.read_csv(manifest)
        if rounds is not None:
            df = df[df["round"].isin(list(rounds))]
        if df.empty:
            raise RuntimeError(f"No synthetic rows in {manifest}")
        self.frame = df.reset_index(drop=True)
        self.samples: List[Tuple[str, float]] = list(zip(self.frame["rel_path"], self.frame["label"].astype(float)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image = _read_rgb(self.root / rel_path)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image, torch.tensor(label, dtype=torch.float32), rel_path


def build_manifest_datasets(config: Dict, train_transform, eval_transform):
    """Train/val datasets from config keys: manifest, frames_root, protocol, synth (list)."""
    from torch.utils.data import ConcatDataset

    protocol = str(config.get("protocol", "official"))
    common = dict(manifest=config["manifest"], root=config["frames_root"], protocol=protocol)
    train_parts = [ManifestFrameDataset(
        split="train", transform=train_transform,
        real_percent=float(config.get("train_real_percent", 100)),
        fake_percent=float(config.get("train_fake_percent", 100)),
        seed=int(config.get("seed", 42)), **common)]
    for entry in config.get("synth") or []:
        train_parts.append(SynthFrameDataset(entry["manifest"], entry.get("root"), train_transform,
                                             entry.get("rounds")))
    val = ManifestFrameDataset(split="val", transform=eval_transform, **common)
    train = train_parts[0] if len(train_parts) == 1 else ConcatDataset(train_parts)
    return train, val
