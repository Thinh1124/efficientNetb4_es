"""P1.2 — Structured artifact annotation of Celeb-DF *train* frames with a vision-language model.

Fake frames are shown together with a pseudo-mask overlay (hint of where the face was
changed); real frames are shown alone. The model must answer with a fixed JSON schema:
region in {eyes, nose, mouth, skin, boundary}, artifact in ARTIFACTS, severity 0-3,
confidence 0-1. Output is appended to a JSONL file and the run is resumable.

Backends
--------
transformers  Qwen2.5-VL loaded in-process (default model Qwen/Qwen2.5-VL-7B-Instruct, optional 4-bit).
openai        Any OpenAI-compatible chat endpoint with image input, e.g. `vllm serve Qwen/Qwen2.5-VL-7B-Instruct`.

Examples
--------
# preview prompts and overlays, no model
python scripts/annotate_mllm.py --meta data/celebdf/meta.csv --frames-root data/celebdf/frames \
    --masks-csv data/celebdf/masks.csv --masks-root data/celebdf/masks \
    --out data/celebdf/annotations/train_mllm.jsonl --n-fake 20 --n-real 5 --dry-run

# real run on GPU
python scripts/annotate_mllm.py ... --n-fake 8000 --n-real 2000 --backend transformers --batch-size 8
"""
import argparse
import base64
import json
import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synth.recipes import ARTIFACTS, REGIONS  # noqa: E402

PROMPT_FAKE = """You are a face-forensics expert. Image 1 is a face crop from a video. Image 2 is the same crop where
the red area marks pixels that differ from the original video (a hint of where the face may have been changed;
it can be imperfect).

Inspect Image 1 and list visible manipulation artifacts. Only report what you can actually see.
Allowed regions: {regions}
Allowed artifacts: {artifacts}
severity: 0 (none) to 3 (obvious). confidence: 0.0 to 1.0, how sure you are that the artifact is really visible.
If you see nothing, return one finding with artifact "authentic".

Answer with JSON only, no other text:
{{"findings": [{{"region": "...", "artifact": "...", "severity": 0, "confidence": 0.0}}]}}"""

PROMPT_REAL = """You are a face-forensics expert. The image is a face crop from a video.

Inspect it and list visible manipulation artifacts, if any. Only report what you can actually see.
Allowed regions: {regions}
Allowed artifacts: {artifacts}
severity: 0 (none) to 3 (obvious). confidence: 0.0 to 1.0.
If you see nothing suspicious, return one finding with artifact "authentic".

Answer with JSON only, no other text:
{{"findings": [{{"region": "...", "artifact": "...", "severity": 0, "confidence": 0.0}}]}}"""


def overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    m = mask > 127
    red = np.zeros_like(out)
    red[..., 2] = 255
    out[m] = (0.55 * out[m] + 0.45 * red[m]).astype(np.uint8)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 0, 255), 1)
    return out


def parse_findings(text: str) -> Optional[List[Dict]]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    raw = data.get("findings") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return None
    findings = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        artifact = str(item.get("artifact", "")).strip().lower()
        region = str(item.get("region", "")).strip().lower()
        if artifact not in ARTIFACTS:
            continue
        if artifact != "authentic" and region not in REGIONS:
            continue
        try:
            severity = int(round(float(item.get("severity", 0))))
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        findings.append({"region": region if region in REGIONS else "none", "artifact": artifact,
                         "severity": min(3, max(0, severity)), "confidence": min(1.0, max(0.0, confidence))})
    return findings


# ------------------------------------------------------------------------ backends
def resolve_dtype(name: str):
    """bf16 on Ampere+ (A100, L4), fp16 on older GPUs such as the Colab T4."""
    import torch

    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32" or not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class TransformersBackend:
    def __init__(self, model_name: str, load_4bit: bool, max_pixels: int, dtype: str = "auto") -> None:
        import torch
        from transformers import AutoProcessor

        dtype = resolve_dtype(dtype)

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        except ImportError as exc:
            raise SystemExit("transformers>=4.49 is required for Qwen2.5-VL") from exc
        kwargs = {"torch_dtype": dtype, "device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        self.model = ModelCls.from_pretrained(model_name, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=128 * 28 * 28, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "left"
        self.torch = torch

    def __call__(self, batch: List[Dict], max_new_tokens: int) -> List[str]:
        from PIL import Image

        texts, images = [], []
        for item in batch:
            content = [{"type": "image"} for _ in item["images"]] + [{"type": "text", "text": item["prompt"]}]
            texts.append(self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                            tokenize=False, add_generation_prompt=True))
            images.extend(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in item["images"])
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)


class OpenAIBackend:
    def __init__(self, api_base: str, model_name: str) -> None:
        self.url = api_base.rstrip("/") + "/chat/completions"
        self.model = model_name
        self.key = os.environ.get("OPENAI_API_KEY", "EMPTY")

    def __call__(self, batch: List[Dict], max_new_tokens: int) -> List[str]:
        outputs = []
        for item in batch:
            content = []
            for im in item["images"]:
                ok, buf = cv2.imencode(".png", im)
                content.append({"type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf).decode()}})
            content.append({"type": "text", "text": item["prompt"]})
            body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": content}],
                               "max_tokens": max_new_tokens, "temperature": 0}).encode()
            req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json",
                                                                       "Authorization": f"Bearer {self.key}"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                outputs.append(json.loads(resp.read())["choices"][0]["message"]["content"])
        return outputs


# ---------------------------------------------------------------------------- main
def select_frames(meta: pd.DataFrame, masks: Optional[pd.DataFrame], protocol: str,
                  n_fake: int, n_real: int, seed: int) -> pd.DataFrame:
    train = meta[meta[protocol] == "train"]
    fakes = train[train["label"] == 1]
    if masks is not None:
        good = masks[masks["aligned"].astype(bool)][["rel_path", "mask_rel"]]
        fakes = fakes.merge(good, on="rel_path", how="inner")
    reals = train[train["label"] == 0].assign(mask_rel="")
    # spread the budget over videos: at most ceil(n / n_videos) frames per video
    def spread(df, n):
        if df.empty or n <= 0:
            return df.iloc[0:0]
        per_video = max(1, int(np.ceil(n / df["video_id"].nunique())))
        df = df.groupby("video_id", group_keys=False).apply(
            lambda g: g.sample(min(len(g), per_video), random_state=seed))
        return df.sample(min(n, len(df)), random_state=seed)
    return pd.concat([spread(fakes, n_fake), spread(reals, n_real)], ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--meta", required=True)
    p.add_argument("--frames-root", required=True)
    p.add_argument("--masks-csv")
    p.add_argument("--masks-root")
    p.add_argument("--protocol", default="official", help="frames are taken from this protocol's train split")
    p.add_argument("--out", required=True)
    p.add_argument("--n-fake", type=int, default=8000)
    p.add_argument("--n-real", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["transformers", "openai"], default="transformers")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--api-base", default="http://localhost:8000/v1")
    p.add_argument("--load-4bit", action="store_true")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                   help="auto = bf16 if the GPU supports it, else fp16 (Colab T4)")
    p.add_argument("--max-pixels", type=int, default=448 * 448)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--dry-run", action="store_true", help="write prompts + overlay images, do not call a model")
    args = p.parse_args()

    meta = pd.read_csv(args.meta)
    masks = pd.read_csv(args.masks_csv) if args.masks_csv else None
    todo = select_frames(meta, masks, args.protocol, args.n_fake, args.n_real, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["rel_path"] for l in out.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = todo[~todo["rel_path"].isin(done)].reset_index(drop=True)
    print(f"Frames to annotate: {len(todo)} (already done: {len(done)})")

    fmt = dict(regions=", ".join(REGIONS), artifacts=", ".join(ARTIFACTS))
    backend = None
    if not args.dry_run:
        backend = (TransformersBackend(args.model, args.load_4bit, args.max_pixels, args.dtype) if args.backend == "transformers"
                   else OpenAIBackend(args.api_base, args.model))
    preview_dir = out.parent / "dry_run_preview"

    with out.open("a", encoding="utf-8") as sink:
        for start in range(0, len(todo), args.batch_size):
            rows = todo.iloc[start:start + args.batch_size]
            batch = []
            for _, row in rows.iterrows():
                image = cv2.imread(str(Path(args.frames_root) / row["rel_path"]))
                hint = bool(row["label"] == 1 and isinstance(row.get("mask_rel"), str) and row["mask_rel"]
                            and args.masks_root)
                images = [image]
                if hint:
                    mask = cv2.imread(str(Path(args.masks_root) / row["mask_rel"]), cv2.IMREAD_GRAYSCALE)
                    images.append(overlay(image, mask))
                prompt = (PROMPT_FAKE if hint else PROMPT_REAL).format(**fmt)
                batch.append({"row": row, "images": images, "prompt": prompt, "hint": hint})

            if args.dry_run:
                preview_dir.mkdir(parents=True, exist_ok=True)
                for item in batch:
                    stem = item["row"]["rel_path"].replace("/", "__").rsplit(".", 1)[0]
                    cv2.imwrite(str(preview_dir / f"{stem}.png"), np.concatenate(item["images"], axis=1))
                    (preview_dir / f"{stem}.txt").write_text(item["prompt"], encoding="utf-8")
                continue

            texts = backend(batch, args.max_new_tokens)
            for item, text in zip(batch, texts):
                findings = parse_findings(text)
                sink.write(json.dumps({"rel_path": item["row"]["rel_path"], "video_id": item["row"]["video_id"],
                                       "label": int(item["row"]["label"]), "with_mask_hint": item["hint"],
                                       "parse_ok": findings is not None, "findings": findings or [],
                                       "raw": text, "model": args.model}) + "\n")
            sink.flush()
            print(f"  {min(start + args.batch_size, len(todo))}/{len(todo)}", flush=True)

    if args.dry_run:
        print(f"Dry run: previews in {preview_dir}")


if __name__ == "__main__":
    main()
