from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "processed" / "faiss_index_v2"

DEFAULT_K = 8
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

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    retriever = vs.as_retriever(search_kwargs={"k": k})
    _RETRIEVER_CACHE[k] = retriever
    return retriever


def build_response_llm_free(
    denial_text: str,
    question: Optional[str],
    docs,
) -> Dict[str, Any]:

    citations: List[Citation] = []
    evidence_snippets: List[str] = []

    # Build citations
    for d in docs[:5]:
        m = d.metadata or {}
        citations.append(
            Citation(
                filename=m.get("filename", ""),
                page_num=int(m.get("page_num", -1)),
                chunk_id=m.get("chunk_id", ""),
            )
        )

    # Keep top 2 evidence snippets (cleaned + shortened)
    for d in docs[:2]:
        snippet = _clean_ws(d.page_content)
        evidence_snippets.append(snippet[:800])

    dt = denial_text.lower()

    # Extract likely missing issues from denial text
    missing_items = []

    if "prior authorization" in dt or "preauth" in dt or "pa number" in dt:
        missing_items.append("Prior Authorization / UTN (Unique Tracking Number) missing on claim")

    if "npi" in dt:
        missing_items.append("Provider NPI issue")

    if "modifier" in dt:
        missing_items.append("Modifier issue")

    if "timely filing" in dt or "late" in dt:
        missing_items.append("Timely filing exceeded")

    if "medical necessity" in dt:
        missing_items.append("Medical necessity documentation missing or not met")

    # Create denial summary
    if "prior authorization" in dt or "pa number" in dt or "preauth" in dt:
        denial_summary = (
            "This denial indicates the service requires prior authorization and the claim "
            "is missing the required authorization identifier (e.g., UTN / authorization number), "
            "so the claim is not payable as submitted."
        )
    else:
        denial_summary = (
            "This denial indicates a billing or documentation requirement was not met "
            "based on CMS policy."
        )

    # Question-aware recommended actions
    q = (question or "").lower()

    if "medicare advantage" in q:
        checklist = [
            "This CMS prior authorization process applies to Medicare Fee-for-Service (FFS), not Medicare Advantage (MA).",
            "Confirm the patient’s coverage type for the date of service (FFS vs MA).",
            "If Medicare Advantage, follow the plan-specific authorization and appeal process.",
        ]

    elif "add" in q or "include" in q or "need to" in q:
        checklist = [
            "Add the required UTN/authorization number to the claim in the appropriate claim field.",
            "Ensure the UTN matches the approved service and date of service.",
            "Verify all provider identifiers (NPI) and beneficiary information are correct.",
            "Resubmit the corrected claim within timely filing limits.",
        ]

    elif "non-affirm" in q:
        checklist = [
            "If the PA request was non-affirmed, the associated claim may be denied.",
            "Review the non-affirmation reason and identify missing documentation or unmet criteria.",
            "Submit a new PA request or appeal with the missing documentation before resubmitting the claim.",
        ]

    else:
        checklist = [
            "Confirm the service requires prior authorization under CMS OPD/DMEPOS rules.",
            "Locate the UTN (Unique Tracking Number) from the prior authorization decision.",
            "Add the UTN to the claim and verify all claim identifiers are correct.",
            "Attach supporting documentation as required by CMS guidance.",
            "Resubmit the corrected claim.",
        ]

    return {
        "denial_text": denial_text,
        "question": question or "default denial analysis",
        "denial_summary": denial_summary,
        "likely_missing_items": missing_items,
        "recommended_actions": checklist,
        "evidence_snippets": evidence_snippets,
        "citations": [c.__dict__ for c in citations],
    }

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
) -> Dict[str, Any]:
    # Retrieve more than k, then filter/rerank down to top-k
    retrieve_k = max(25, k)

    retriever = load_retriever(k=retrieve_k)

    # Build query using BOTH denial text + question
    if question:
        query = f"{question}\n\nContext: {denial_text}"
    else:
        query = denial_text.strip()


    # Keyword boosting signals
    boost_terms: List[str] = []
    dt = denial_text.lower()

    is_pa = ("prior authorization" in dt) or ("preauth" in dt) or ("pre-authorization" in dt) or ("pa number" in dt)
    if is_pa:
        boost_terms += ["prior authorization", "preauthorization", "pre-certification", "authorization number", "referral"]
    if "timely filing" in dt or "late" in dt:
        boost_terms += ["timely filing", "filing limit"]
    if "duplicate" in dt:
        boost_terms += ["duplicate claim", "adjustment", "resubmission"]
    if "modifier" in dt or "bundl" in dt:
        boost_terms += ["modifier", "bundling", "NCCI"]
    if "medical necessity" in dt or "not medically necessary" in dt:
        boost_terms += ["medical necessity", "coverage criteria", "documentation requirements"]
    if "coordination of benefits" in dt or "secondary" in dt or "eob" in dt:
        boost_terms += ["coordination of benefits", "primary payer", "EOB"]

    if boost_terms:
        query = query + "\nKey terms: " + ", ".join(boost_terms)
        
    if question:
        # add strong keywords from the question into boost terms
        q_words = re.findall(r"[a-zA-Z]{4,}", question.lower())
        boost_terms += list(dict.fromkeys(q_words))[:10]

    # Retrieve candidate docs
    docs = retriever.invoke(query)

    # MUST-CONTAIN FILTER (apply BEFORE rerank)
    required_terms: List[str] = []
    if is_pa:
        required_terms = ["authorization", "prior authorization", "preauthorization", "pre-certification", "referral"]

    if required_terms:
        filtered = []
        for d in docs:
            t = (d.page_content or "").lower()
            if any(term.lower() in t for term in required_terms):
                filtered.append(d)
        # Use filtered if we found any; otherwise fall back
        if filtered:
            docs = filtered

    # Rerank by keyword overlap (hybrid)
    denial_keywords = re.findall(r"[a-zA-Z]{4,}", dt)
    denial_keywords = list(dict.fromkeys(denial_keywords))[:25]

    def score_doc(text: str) -> int:
        t = text.lower()
        score = 0
        for term in boost_terms:
            if term.lower() in t:
                score += 5
        for w in denial_keywords:
            if w in t:
                score += 1
        return score

    scored = [(score_doc(d.page_content), d) for d in docs]
    scored.sort(key=lambda x: x[0], reverse=True)

    reranked = [d for s, d in scored if s > 0][:k]
    if len(reranked) < k:
        reranked = [d for _, d in scored][:k]

    # Build grounded response
    return build_response_llm_free(denial_text=denial_text, question=question, docs=reranked)


if __name__ == "__main__":
    sample_denial = "Claim denied: Prior authorization required for the billed service. No PA number found on claim."
    out = answer_denial(sample_denial, question="Why was this denied and what should I fix?", k=5)
    print(json.dumps(out, indent=2, ensure_ascii=False))
