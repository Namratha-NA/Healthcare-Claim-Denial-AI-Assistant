from __future__ import annotations

from rag_engine import answer_denial


def print_answer(result: dict):
    print("\n==============================")
    print("Claim Denial Assistant Output")
    print("==============================")

    print("\n Denial Summary")
    print(result.get("denial_summary", ""))

    missing = result.get("likely_missing_items", [])
    if missing:
        print("\n Likely Missing Items")
        for i, item in enumerate(missing, start=1):
            print(f"{i}. {item}")

    print("\n Recommended Actions")
    for i, a in enumerate(result.get("recommended_actions", []), start=1):
        print(f"{i}. {a}")

    print("\n Evidence (Top 2)")
    for i, s in enumerate(result.get("evidence_snippets", [])[:2], start=1):
        print(f"\n[{i}] {s}")

    print("\n Citations (Top-k)")
    for c in result.get("citations", []):
        print(f"- {c['filename']} | page {c['page_num']} | {c['chunk_id']}")


def main():
    print("\nHealthcare Claim Denial AI Assistant (CLI)")
    print("Type 'exit' to quit.\n")

    denial_text = input("Paste denied claim text:\n> ").strip()
    if not denial_text:
        print("No denial text provided. Exiting.")
        return

    while True:
        q = input("\nAsk a question (or press Enter for default analysis):\n> ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = answer_denial(denial_text, question=q if q else None, k=8)
        print_answer(result)


if __name__ == "__main__":
    main()
