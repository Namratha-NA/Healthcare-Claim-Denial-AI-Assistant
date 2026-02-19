from __future__ import annotations

from llm_client import generate_llm_structured
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def detect_category(denial_text: str, question: str = "") -> str:
    t = f"{denial_text} {question}".lower()

    # Prior Auth
    if any(x in t for x in ["prior authorization", "preauth", "pa number", "utn", "authorization number"]):
        return "prior_auth"

    # Duplicate claim
    if any(x in t for x in ["duplicate", "already been processed", "previously processed", "already paid"]):
        return "duplicate"

    # Documentation request / ADR / records missing in timeframe
    if any(x in t for x in ["additional documentation", "documentation request", "medical records", "records were not received", "within the required timeframe", "timeframe"]):
        return "documentation_request"

    # Medical necessity
    if any(x in t for x in ["medical necessity", "not medically necessary", "does not meet criteria"]):
        return "medical_necessity"

    # Missing info
    if any(x in t for x in ["missing", "incomplete", "invalid", "information is needed", "documentation not included"]):
        return "missing_info"

    # Coding
    if any(x in t for x in ["modifier", "cpt", "icd", "coding", "diagnosis code"]):
        return "coding"

    # Timely filing
    if any(x in t for x in ["timely filing", "filed late", "past filing limit"]):
        return "timely_filing"

    # Eligibility
    if any(x in t for x in ["not eligible", "eligibility", "coverage terminated"]):
        return "eligibility"

    return "general"

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "processed" / "faiss_index_v2"

DEFAULT_K = 4
_RETRIEVER_CACHE = {}


@dataclass
class Citation:
    filename: str
    page_num: int
    chunk_id: str


def _clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def load_retriever(k: int = DEFAULT_K):
    # Cache by k so we don't reload the embedding model every question
    if k in _RETRIEVER_CACHE:
        return _RETRIEVER_CACHE[k]

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vs = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vs.as_retriever(search_kwargs={"k": k})
    _RETRIEVER_CACHE[k] = retriever
    return retriever


def retrieve_documents(query: str, k: int = DEFAULT_K):
    retriever = load_retriever(k=k)
    # Newer LangChain uses invoke()
    return retriever.invoke(query)


def must_contain_filter(docs, required_terms: List[str]):
    if not required_terms:
        return docs

    filtered = []
    for d in docs:
        t = (d.page_content or "").lower()
        if any(term.lower() in t for term in required_terms):
            filtered.append(d)

    return filtered


def answer_denial(
    denial_text: str,
    question: Optional[str] = None,
    k: int = DEFAULT_K,
):
    query = denial_text + " " + (question or "")
    docs = retrieve_documents(query, k=k)

    category = detect_category(denial_text, question or "")

    required_terms: List[str] = []
    if category == "prior_auth":
        required_terms = ["prior authorization", "utn", "unique tracking number", "pa number", "preauth"]
    elif "npi" in (denial_text + " " + (question or "")).lower():
        required_terms = ["billing provider", "provider", "npi", "provider identifier", "claim"]

    filtered = must_contain_filter(docs, required_terms)
    if filtered:
        docs = filtered

    # Build citations FROM FINAL docs
    citations = []
    for d in docs[:5]:
        m = d.metadata or {}
        citations.append(
            {
                "filename": m.get("filename", ""),
                "page_num": int(m.get("page_num", -1)),
                "chunk_id": m.get("chunk_id", ""),
            }
        )

    # Evidence snippets FROM FINAL docs
    evidence_snippets = []
    for d in docs[:2]:
        evidence_snippets = [(d.page_content or "")[:600] for d in docs[:3]]


    llm_response = generate_llm_structured(
        denial_text=denial_text,
        question=question or "",
        evidence_snippets=evidence_snippets,
    )

    llm_response["citations"] = citations
    return llm_response





if __name__ == "__main__":
    sample_denial = (
        "Claim denied: Prior authorization required for the billed service. "
        "No PA number found on claim."
    )

    out = answer_denial(
        sample_denial,
        question="Why was this denied and what should I fix?",
        k=5,
    )

    print(json.dumps(out, indent=2, ensure_ascii=False))
