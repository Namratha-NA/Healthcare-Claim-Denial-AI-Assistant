from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data" / "processed" / "chunks" / "policy_chunks_v3.jsonl"
INDEX_DIR = ROOT / "data" / "processed" / "faiss_index"


def main():
    if not CHUNKS.exists():
        raise FileNotFoundError(f"Missing: {CHUNKS}")

    docs = []
    with CHUNKS.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading chunks"):
            obj = json.loads(line)
            docs.append(
                Document(
                    page_content=obj["text"],
                    metadata={
                        "chunk_id": obj["chunk_id"],
                        "doc_id": obj["doc_id"],
                        "filename": obj["filename"],
                        "page_num": obj["page_num"],
                        "source": obj.get("source", ""),
                    },
                )
            )

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"Embedding {len(docs)} chunks and building FAISS index...")
    vs = FAISS.from_documents(docs, embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    print(f"Saved FAISS index to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
