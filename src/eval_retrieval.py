from __future__ import annotations

import json
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "processed" / "faiss_index"
GOLDEN = ROOT / "eval" / "golden_set.jsonl"

K = 5


def main():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": K})

    total = 0
    hits = 0

    with GOLDEN.open("r", encoding="utf-8-sig") as f:
        for line in tqdm(f, desc="Evaluating retrieval"):
            case = json.loads(line)
            q = case["denial_text"]
            expected = [kw.lower() for kw in case.get("expected_policy_keywords", [])]

            docs = retriever.invoke(q)
            blob = "\n".join([d.page_content.lower() for d in docs])

            ok = any(kw in blob for kw in expected) if expected else False
            total += 1
            hits += int(ok)

            if total <= 3 and docs:
                print("\n--- Sample ---")
                print("Case:", case["case_id"])
                print("Expected keywords:", expected)
                print("Top doc meta:", docs[0].metadata)

    print(f"\nHit@{K}: {hits}/{total} = {hits/total:.2%}")


if __name__ == "__main__":
    main()
