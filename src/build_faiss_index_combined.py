from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

POLICY_CHUNKS = ROOT / "data" / "processed" / "chunks" / "policy_chunks_v3.jsonl"
PA_CHUNKS = ROOT / "data" / "processed" / "chunks" / "pa_chunks.jsonl"

INDEX_DIR = ROOT / "data" / "processed" / "faiss_index_v2"


def load_jsonl_as_docs(path: Path, corpus_tag: str) -> list[Document]:
    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj.get("text", ""),
                    metadata={
                        "chunk_id": obj.get("chunk_id", ""),
                        "doc_id": obj.get("doc_id", ""),
                        "filename": obj.get("filename", ""),
                        "page_num": obj.get("page_num", -1),
                        "source": obj.get("source", ""),
                        "corpus": corpus_tag,
                    },
                )
            )
    return docs


def main():
    if not POLICY_CHUNKS.exists():
        raise FileNotFoundError(f"Missing: {POLICY_CHUNKS}")
    if not PA_CHUNKS.exists():
        raise FileNotFoundError(f"Missing: {PA_CHUNKS}")

    print("Loading policy chunks...")
    policy_docs = load_jsonl_as_docs(POLICY_CHUNKS, corpus_tag="cms_claims_manual")

    print("Loading prior-authorization chunks...")
    pa_docs = load_jsonl_as_docs(PA_CHUNKS, corpus_tag="cms_prior_auth")

    docs = policy_docs + pa_docs
    print(f"Total docs to embed: {len(docs)} (policies={len(policy_docs)}, pa={len(pa_docs)})")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Building FAISS index...")
    vs = FAISS.from_documents(tqdm(docs, desc="Embedding"), embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))

    print(f"✅ Saved combined FAISS index to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
