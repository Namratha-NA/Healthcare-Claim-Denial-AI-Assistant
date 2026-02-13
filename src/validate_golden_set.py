import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "eval" / "golden_set.jsonl"
TAXONOMY = ROOT / "eval" / "denial_taxonomy.json"

REQUIRED_TOP_LEVEL = {"case_id", "denial_text", "denial_category", "codes", "expected_policy_keywords", "notes"}
REQUIRED_CODES_KEYS = {"icd10", "hcpcs", "cpt"}


def main():
    if not GOLDEN.exists():
        raise FileNotFoundError(f"Missing: {GOLDEN}")
    if not TAXONOMY.exists():
        raise FileNotFoundError(f"Missing: {TAXONOMY}")

    taxonomy = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    valid_categories = set(taxonomy.keys())

    errors = []
    n = 0

    with GOLDEN.open("r", encoding="utf-8-sig") as f:

        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON ({e})")
                continue

            missing = REQUIRED_TOP_LEVEL - set(obj.keys())
            if missing:
                errors.append(f"Line {i}: missing fields {sorted(missing)}")

            cat = obj.get("denial_category")
            if cat not in valid_categories:
                errors.append(f"Line {i}: invalid denial_category '{cat}' (not in taxonomy)")

            codes = obj.get("codes", {})
            if not isinstance(codes, dict):
                errors.append(f"Line {i}: codes must be an object/dict")
            else:
                missing_codes = REQUIRED_CODES_KEYS - set(codes.keys())
                if missing_codes:
                    errors.append(f"Line {i}: codes missing keys {sorted(missing_codes)}")

            kw = obj.get("expected_policy_keywords")
            if not isinstance(kw, list) or len(kw) < 2:
                errors.append(f"Line {i}: expected_policy_keywords should be a list with at least 2 items")

            denial_text = obj.get("denial_text", "")
            if not isinstance(denial_text, str) or len(denial_text) < 30:
                errors.append(f"Line {i}: denial_text too short (<30 chars)")

    print(f"Validated {n} cases from {GOLDEN.name}")
    if errors:
        print("\n❌ Errors found:")
        for e in errors:
            print("-", e)
        raise SystemExit(1)

    print("✅ Golden set looks good.")


if __name__ == "__main__":
    main()
