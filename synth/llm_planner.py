"""P2.3 — Decide what to synthesise next round from the error report.

Modes (same budget, so D1/D2/D3 are comparable):
  llm     D2: an instruction LLM reads report.md and returns a JSON plan (OpenAI-compatible endpoint,
          e.g. `vllm serve Qwen/Qwen2.5-7B-Instruct`, or any hosted API via --api-base + OPENAI_API_KEY).
  rule    D3: budget split in proportion to (1 - AUC) of the worst cells + FPR of real ops.
  random  D1: budget spread evenly over all cells.

The plan is validated (allowed cells, strength range, per-cell cap, total budget) before use,
and every prompt/answer is appended to a log for the report.

Example
-------
python -m synth.llm_planner --report results/round1/report.json --mode llm --budget 10000 \
    --round 2 --out plans/round2_llm.json --log runs/planner_log.jsonl
"""
import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from synth.error_report import to_markdown
from synth.recipes import REAL_OPS, REGIONS, FAKE_ARTIFACTS, all_cells, save_plan, validate_plan

SYSTEM = "You plan synthetic training data for a face-forgery detector. You answer with JSON only."

USER = """The detector is a ViT trained on Celeb-DF. Below is its error report on the validation set and on a
synthetic probe grid (each fake probe has one artifact in one face region).

{report}

Decide which synthetic images to generate for the next training round to fix the weaknesses, while
not over-fitting to easy cells and keeping false positives on real videos low.
Rules:
- total count must be <= {budget}; at most {max_per_cell} per cell.
- regions: {regions}; fake artifacts: {artifacts}.
- real-side ops (label 0, region "any"): {real_ops}. Use them if false positives on real probes are high.
- strength is [low, high] within [0.05, 1.0]; lower strength = subtler artifact.
- give a short reason per item.
Return exactly:
{{"plan": [{{"region": "...", "artifact": "...", "strength": [0.2, 0.5], "count": 1000, "reason": "..."}}]}}"""


def call_llm(api_base: str, model: str, prompt: str, temperature: float) -> str:
    body = json.dumps({"model": model, "temperature": temperature, "max_tokens": 1500,
                       "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
    req = urllib.request.Request(api_base.rstrip("/") + "/chat/completions", data=body.encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"]


def rule_plan(report: dict, budget: int, top_k: int) -> dict:
    cells = sorted(report["cells"], key=lambda c: c["auc"])[:top_k]
    weights = [(c, max(1e-3, 1 - c["auc"])) for c in cells]
    weights += [({"region": "any", "artifact": r["op"]}, r["fpr"]) for r in report["real_ops"] if r["fpr"] > 0.05]
    total = sum(w for _, w in weights) or 1.0
    plan = []
    for c, w in weights:
        weak = c.get("tpr_by_strength", {})
        low = min((float(k) for k, v in weak.items() if v < 0.8), default=0.3)
        plan.append({"region": c["region"], "artifact": c["artifact"], "strength": [max(0.05, low - 0.15), low + 0.2],
                     "count": int(budget * w / total), "reason": "rule: proportional to 1-AUC / FPR"})
    return {"plan": plan}


def random_plan(budget: int) -> dict:
    cells = all_cells()
    per = budget // len(cells)
    return {"plan": [{"region": r, "artifact": a, "strength": [0.2, 0.9], "count": per, "reason": "uniform"}
                     for r, a in cells]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", help="report.json from synth.error_report (not needed for --mode random)")
    p.add_argument("--mode", choices=["llm", "rule", "random"], required=True)
    p.add_argument("--budget", type=int, default=10000)
    p.add_argument("--max-per-cell", type=int, default=3000)
    p.add_argument("--top-k", type=int, default=8, help="rule mode: number of worst cells")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--log", default="runs/planner_log.jsonl")
    p.add_argument("--api-base", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--retries", type=int, default=2)
    args = p.parse_args()

    report = json.loads(Path(args.report).read_text()) if args.report else None
    prompt, answer = None, None
    if args.mode == "random":
        raw = random_plan(args.budget)
    elif args.mode == "rule":
        raw = rule_plan(report, args.budget, args.top_k)
    else:
        prompt = USER.format(report=to_markdown(report), budget=args.budget, max_per_cell=args.max_per_cell,
                             regions=", ".join(REGIONS), artifacts=", ".join(FAKE_ARTIFACTS),
                             real_ops=", ".join(REAL_OPS))
        raw = {}
        for attempt in range(args.retries + 1):
            answer = call_llm(args.api_base, args.model, prompt, args.temperature)
            match = re.search(r"\{.*\}", answer, flags=re.S)
            try:
                raw = json.loads(match.group(0)) if match else {}
            except json.JSONDecodeError:
                raw = {}
            if raw.get("plan"):
                break
            print(f"[warn] attempt {attempt}: unparsable plan, retrying")
        if not raw.get("plan"):
            raise SystemExit("LLM did not return a usable plan; see the log")

    plan, warnings = validate_plan(raw, args.budget, args.max_per_cell)
    for w in warnings:
        print(f"[plan] {w}")
    save_plan(args.out, plan, {"round": args.round, "mode": args.mode, "budget": args.budget})
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.log).open("a", encoding="utf-8") as log:
        log.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "round": args.round, "mode": args.mode,
                              "model": args.model if args.mode == "llm" else None, "prompt": prompt,
                              "answer": answer, "raw_plan": raw, "warnings": warnings,
                              "final_total": sum(it.count for it in plan)}) + "\n")
    print(f"Plan with {len(plan)} cells, {sum(it.count for it in plan)} images -> {args.out}")
    for it in plan:
        print(f"  {it.region:9s} {it.artifact:22s} {it.strength} x{it.count}  {it.reason}")


if __name__ == "__main__":
    main()
