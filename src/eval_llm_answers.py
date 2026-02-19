import json
import time
import argparse
import re
from pathlib import Path
from typing import Dict, Any, List

from rag_engine import answer_denial

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "eval" / "golden_set.jsonl"
OUT_DIR = ROOT / "eval" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_list(x):
    return x if isinstance(x, list) else []


def _norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _keyword_variants(kw: str) -> List[str]:
    """
    Expand golden keywords into acceptable variants/synonyms.
    This makes scoring fair when the model uses equivalent wording.
    """
    k = _norm(kw)

    # Default: include itself
    variants = {k}

    # Hand-tuned synonym expansions (extend anytime you add new golden keywords)
    synonym_map = {
        "prior authorization": [
            "prior auth", "preauth", "pre-authorization", "pa", "utn", "unique tracking number",
            "authorization number", "pa number",
        ],
        "documentation": [
            "supporting documentation", "clinical documentation", "medical documentation",
            "records", "medical records", "documentation submitted",
        ],
        "claim submission": [
            "submit claim", "claim submitted", "submission", "resubmit", "resubmission",
            "file claim", "claim filing",
        ],
        "duplicate claim": [
            "duplicate", "already processed", "previously processed", "already paid",
            "duplicate submission", "duplicate billing",
        ],
        "adjustment": [
            "adjust", "adjusted claim", "claim adjustment", "adjustment request",
            "corrected claim", "replacement claim",
        ],
        "resubmission": [
            "resubmit", "resubmitted", "submit again", "re-submit", "resubmission",
            "correct and resubmit",
        ],
        "documentation request": [
            "additional documentation request", "adr", "records requested",
            "request for documentation", "medical records request",
        ],
        "medical records": [
            "records", "clinical notes", "chart notes", "documentation", "medical documentation",
        ],
        "timeframe": [
            "within the required timeframe", "within timeframe", "deadline", "timely",
            "within the required time", "time limit",
        ],
    }

    if k in synonym_map:
        for v in synonym_map[k]:
            variants.add(_norm(v))

    # A few generic morphological expansions
    if k.endswith("ion"):
        variants.add(k[:-3])  # crude stem
    if " " in k:
        variants.add(k.replace(" ", ""))  # rare but harmless

    return sorted(variants)


def score_case(output: Dict[str, Any], expected_keywords: List[str]) -> Dict[str, Any]:
    text_blob = " ".join([
        str(output.get("denial_summary", "")),
        " ".join(_safe_list(output.get("likely_missing_items", []))),
        " ".join(_safe_list(output.get("recommended_actions", []))),
        str(output.get("appeal_guidance", "")),
    ])
    text_blob = _norm(text_blob)

    hits = 0
    missing = []

    for kw in expected_keywords:
        variants = _keyword_variants(kw)
        if any(v in text_blob for v in variants):
            hits += 1
        else:
            missing.append(kw)

    coverage = hits / max(1, len(expected_keywords))

    json_ok = (
        isinstance(output.get("denial_summary", ""), str)
        and isinstance(output.get("likely_missing_items", []), list)
        and isinstance(output.get("recommended_actions", []), list)
        and isinstance(output.get("appeal_guidance", ""), str)
    )

    return {
        "keyword_hits": hits,
        "keyword_total": len(expected_keywords),
        "keyword_coverage": coverage,
        "missing_keywords": missing,
        "json_ok": json_ok,
        "num_citations": len(_safe_list(output.get("citations", []))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_cases", type=int, default=5)
    args = parser.parse_args()

    cases = []
    with GOLD.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    cases = cases[: args.max_cases]

    results = []
    t_all0 = time.perf_counter()

    for i, c in enumerate(cases, 1):
        case_id = c.get("case_id", f"case_{i}")
        denial_text = c["denial_text"]
        question = c.get("question", "")
        expected_keywords = c.get("expected_policy_keywords", []) or c.get("expected_keywords", [])

        t0 = time.perf_counter()
        out = answer_denial(denial_text=denial_text, question=question)
        t1 = time.perf_counter()

        scores = score_case(out, expected_keywords)

        results.append({
            "case_id": case_id,
            "latency_sec": round(t1 - t0, 3),
            **scores,
        })

        (OUT_DIR / f"{case_id}.json").write_text(
            json.dumps(out, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

        print(
            f"[{i}/{len(cases)}] {case_id} | latency={t1-t0:.2f}s | "
            f"coverage={scores['keyword_coverage']:.2f} | json_ok={scores['json_ok']}"
        )

    t_all1 = time.perf_counter()

    avg_latency = sum(r["latency_sec"] for r in results) / max(1, len(results))
    json_ok_rate = sum(1 for r in results if r["json_ok"]) / max(1, len(results))
    avg_coverage = sum(r["keyword_coverage"] for r in results) / max(1, len(results))

    summary = {
        "num_cases": len(results),
        "avg_latency_sec": round(avg_latency, 3),
        "json_ok_rate": round(json_ok_rate, 3),
        "avg_keyword_coverage": round(avg_coverage, 3),
        "total_runtime_sec": round(t_all1 - t_all0, 3),
        "per_case": results,
    }

    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== Phase 6B Summary ===")
    print(json.dumps(
        {k: summary[k] for k in ["num_cases", "avg_latency_sec", "json_ok_rate", "avg_keyword_coverage", "total_runtime_sec"]},
        indent=2
    ))


if __name__ == "__main__":
    main()
