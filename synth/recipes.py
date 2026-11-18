"""Shared vocabulary (regions / artifacts) and the plan schema used by the planner and generator."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from synth.regions import REGIONS

FAKE_ARTIFACTS = (
    "blend_boundary",
    "color_mismatch",
    "blur_resample",
    "over_smooth",
    "warp_geometry",
    "texture_inconsistency",
)
# Annotation vocabulary (P1) = fake artifacts + "authentic"
ARTIFACTS = FAKE_ARTIFACTS + ("authentic",)
# Label-0 densification ops: degraded real and "seam real" (paste from another frame of the same video)
REAL_OPS = ("authentic_degrade", "authentic_seam")
REAL_REGION = "any"

__all__ = ["REGIONS", "FAKE_ARTIFACTS", "ARTIFACTS", "REAL_OPS", "REAL_REGION", "PlanItem", "validate_plan",
           "load_plan", "save_plan", "all_cells"]


@dataclass
class PlanItem:
    region: str
    artifact: str
    strength: Tuple[float, float]
    count: int
    reason: str = ""

    @property
    def label(self) -> int:
        return 0 if self.artifact in REAL_OPS else 1


def all_cells() -> List[Tuple[str, str]]:
    cells = [(r, a) for r in REGIONS for a in FAKE_ARTIFACTS]
    return cells + [(REAL_REGION, op) for op in REAL_OPS]


def validate_plan(raw: Dict, budget: int, max_per_cell: int) -> Tuple[List[PlanItem], List[str]]:
    """Validate and repair an LLM plan. Returns (items, warnings). Never raises on bad items."""
    warnings: List[str] = []
    items: Dict[Tuple[str, str], PlanItem] = {}
    entries = raw.get("plan", []) if isinstance(raw, dict) else []
    if not isinstance(entries, list):
        entries = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            warnings.append(f"item {i}: not an object")
            continue
        artifact = str(e.get("artifact", "")).strip().lower()
        region = str(e.get("region", "")).strip().lower()
        if artifact in REAL_OPS:
            region = REAL_REGION
        elif artifact not in FAKE_ARTIFACTS or region not in REGIONS:
            warnings.append(f"item {i}: invalid cell ({region}, {artifact})")
            continue
        try:
            lo, hi = [float(v) for v in (e.get("strength") or [0.3, 0.7])][:2]
            count = int(e.get("count", 0))
        except (TypeError, ValueError):
            warnings.append(f"item {i}: bad strength/count")
            continue
        lo, hi = sorted((min(1.0, max(0.05, lo)), min(1.0, max(0.05, hi))))
        count = max(0, min(count, max_per_cell))
        if count == 0:
            continue
        key = (region, artifact)
        if key in items:
            items[key].count = min(max_per_cell, items[key].count + count)
        else:
            items[key] = PlanItem(region, artifact, (lo, hi), count, str(e.get("reason", ""))[:300])
    plan = list(items.values())
    total = sum(it.count for it in plan)
    if total > budget and total > 0:
        scale = budget / total
        for it in plan:
            it.count = max(1, int(it.count * scale))
        warnings.append(f"total {total} > budget {budget}: scaled down")
    return plan, warnings


def save_plan(path: str, plan: List[PlanItem], extra: Dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({**extra, "plan": [asdict(it) for it in plan]}, indent=1))


def load_plan(path: str) -> List[PlanItem]:
    raw = json.loads(Path(path).read_text())
    return [PlanItem(e["region"], e["artifact"], tuple(e["strength"]), int(e["count"]), e.get("reason", ""))
            for e in raw["plan"]]
