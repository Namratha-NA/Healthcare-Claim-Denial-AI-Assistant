from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import re

def detect_category(denial_text: str, question: str = "") -> str:
    t = f"{denial_text} {question}".lower()

    if any(x in t for x in ["prior authorization", "preauth", "pa number", "utn", "authorization number"]):
        return "prior_auth"

    if any(x in t for x in ["medical necessity", "not medically necessary", "does not meet criteria"]):
        return "medical_necessity"

    if any(x in t for x in ["missing", "incomplete", "invalid", "information is needed", "documentation not included"]):
        return "missing_info"

    if any(x in t for x in ["modifier", "cpt", "icd", "coding", "diagnosis code"]):
        return "coding"

    if any(x in t for x in ["timely filing", "filed late", "past filing limit"]):
        return "timely_filing"

    if any(x in t for x in ["duplicate", "previously processed", "already paid"]):
        return "duplicate"

    if any(x in t for x in ["not eligible", "eligibility", "coverage terminated"]):
        return "eligibility"

    return "general"

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

    for d in (docs or [])[:5]:
        m = d.metadata or {}
        citations.append(
            Citation(
                filename=m.get("filename", ""),
                page_num=int(m.get("page_num", -1)) if m.get("page_num", -1) is not None else -1,
                chunk_id=m.get("chunk_id", ""),
            )
        )

    # ----------------------------
    # Evidence snippets (top 2)
    # ----------------------------
    for d in (docs or [])[:2]:
        snippet = _clean_ws(d.page_content or "")
        evidence_snippets.append(snippet[:800])

    dt = (denial_text or "").lower()
        # ----------------------------
    # Detect question intent 
    # ----------------------------
    q = (question or "").lower().strip()

    def _has_any(text: str, phrases: List[str]) -> bool:
        return any(p in text for p in phrases)

    if not q:
        intent = "general"
    elif _has_any(q, [
        "criteria", "requirement", "requirements", "standard", "standards",
        "policy", "rule", "rules", "what does cms use", "how does cms decide",
        "how is medical necessity determined", "coverage criteria"
    ]):
        intent = "criteria"
    elif _has_any(q, [
        "appeal", "appealable", "reconsideration", "redetermination",
        "reopen", "reopening", "escalate", "dispute", "grievance"
    ]):
        intent = "appeal"
    elif _has_any(q, [
        "documentation", "document", "attach", "include", "records", "notes",
        "what documentation", "what should i include", "what do i include",
        "what do i need to include"
    ]):
        intent = "documentation"
    elif _has_any(q, [
        "resubmit", "resubmission", "corrected", "correct", "fix", "update",
        "how to fix", "what should i do", "next steps", "how do i resolve",
        "what do i need to add", "what do i add", "add to the claim"
    ]):
        intent = "resubmit"
    elif _has_any(q, [
        "what does this mean", "what does it mean", "meaning", "explain",
        "why was this denied", "why denied", "root cause"
    ]):
        intent = "explain"
    elif _has_any(q, [
        "what happens if", "if i", "in that case", "scenario", "edge case"
    ]):
        intent = "scenario"
    else:
        intent = "general"


    # ----------------------------
    # Default fallback checklist
    # (prevents UnboundLocalError)
    # ----------------------------
    checklist = [
        "Review the denial reason and identify the missing requirement using the cited CMS policy excerpts.",
        "Correct claim fields/codes and attach required documentation.",
        "Resubmit the corrected claim within timely filing limits or file an appeal if the claim was compliant.",
    ]

    # ----------------------------
    # Detect category from denial + question
    # ----------------------------
    combined = f"{dt} {q}"

    if any(x in combined for x in ["prior authorization", "preauth", "pa number", "utn", "authorization number"]):
        category = "prior_auth"
    elif any(x in combined for x in ["npi", "not enrolled", "enroll", "enrollment"]):
        category = "provider_enrollment"
    elif any(x in combined for x in ["medical necessity", "not medically necessary", "does not meet criteria"]):
        category = "medical_necessity"
    elif any(x in combined for x in ["timely filing", "filed late", "past filing limit", "late"]):
        category = "timely_filing"
    elif any(x in combined for x in ["duplicate", "previously processed", "already paid"]):
        category = "duplicate"
    elif any(x in combined for x in ["eligibility", "not eligible", "coverage terminated"]):
        category = "eligibility"
    elif any(x in combined for x in ["modifier", "cpt", "icd", "coding", "diagnosis code"]):
        category = "coding"
    elif any(x in combined for x in ["missing", "incomplete", "invalid", "documentation not included", "information is needed"]):
        category = "missing_info"
    else:
        category = "general"

    # ----------------------------
    # Extract likely missing issues (simple tags)
    # ----------------------------
    missing_items: List[str] = []

    if category == "prior_auth":
        missing_items.append("Prior Authorization / UTN (Unique Tracking Number) missing on claim")

    if category == "provider_enrollment":
        missing_items.append("Provider NPI missing/invalid or provider not enrolled")

    if category == "coding":
        if "modifier" in combined:
            missing_items.append("Modifier issue")
        if "diagnosis" in combined or "icd" in combined:
            missing_items.append("Diagnosis/ICD code issue")
        if "cpt" in combined or "hcpcs" in combined:
            missing_items.append("Procedure/CPT/HCPCS code issue")

    if category == "timely_filing":
        missing_items.append("Timely filing exceeded")

    if category == "medical_necessity":
        missing_items.append("Medical necessity documentation missing or not met")

    if category == "duplicate":
        missing_items.append("Duplicate claim/service previously processed")

    if category == "eligibility":
        missing_items.append("Eligibility/coverage issue for date of service")

    if category == "missing_info":
        missing_items.append("Missing/incomplete claim information or documentation")

    # ----------------------------
    # Create denial summary (category-aware)
    # ----------------------------
    if category == "prior_auth":
        denial_summary = (
            "This denial indicates the service requires prior authorization and the claim is missing the required "
            "authorization identifier (e.g., UTN / authorization number), so the claim is not payable as submitted."
        )
    elif category == "provider_enrollment":
        denial_summary = (
            "This denial indicates the billing provider identifier (NPI) is missing/invalid or the provider is not "
            "enrolled/eligible to bill Medicare, so the claim cannot be processed as submitted."
        )
    elif category == "medical_necessity":
        denial_summary = (
            "This denial indicates the service was considered not medically necessary based on the documentation submitted, "
            "or the documentation did not demonstrate that the service meets coverage/medical-necessity requirements."
        )
    else:
        denial_summary = (
            "This denial indicates a billing or documentation requirement was not met based on CMS policy."
        )

    # ----------------------------
    # Detect question intent
    # ----------------------------
    if any(x in q for x in ["criteria", "rule", "standard", "definition", "what does cms use", "how does cms decide"]):
        intent = "criteria"
    elif any(x in q for x in ["appeal", "appealable", "reconsideration", "redetermination", "grievance", "reopen"]):
        intent = "appeal"
    elif any(x in q for x in ["documentation", "document", "include", "attach", "records", "notes", "proof"]):
        intent = "documentation"
    elif any(x in q for x in ["resubmit", "corrected", "fix", "update", "add to claim"]):
        intent = "resubmit"
    elif any(x in q for x in ["what does this mean", "what is this", "explain", "meaning"]):
        intent = "explain"
    else:
        intent = "general"

    # ----------------------------
    # Question override (Medicare Advantage)
    # Applies regardless of category
    # ----------------------------
    if "medicare advantage" in q:
        checklist = [
            "Confirm whether the patient has Medicare Fee-for-Service (FFS) or Medicare Advantage (MA) coverage for the date of service.",
            "CMS FFS guidance may not apply to MA plans; MA plans follow plan-specific authorization, billing, and appeals rules.",
            "If the patient is MA, check the plan’s prior authorization requirements and resubmission/appeal process.",
        ]

    # ----------------------------
    # Category + intent-aware actions
    # ----------------------------
    elif category == "prior_auth":
        if "non-affirm" in q:
            checklist = [
                "If the prior authorization request was non-affirmed, the associated claim may be denied even if a UTN exists.",
                "Review the non-affirmation reason and identify missing documentation or unmet coverage criteria.",
                "Submit a new PA request (or appeal/re-review) with the missing documentation before resubmitting the claim.",
            ]
        elif any(x in q for x in ["add", "include", "need to"]):
            checklist = [
                "Add the required UTN/authorization number to the claim in the appropriate claim field.",
                "Ensure the UTN matches the approved service and date of service.",
                "Attach/retain the PA decision and supporting documentation per the cited CMS guidance.",
                "Resubmit the corrected claim within timely filing limits.",
            ]
        elif intent == "scenario":
            checklist = [
                "Identify which scenario applies (no PA submitted vs PA submitted but non-affirmed vs PA approved but UTN missing).",
                "If no PA was submitted, submit PA first (if required) and then resubmit the claim with UTN.",
                "If PA was non-affirmed, correct documentation/criteria and re-request or appeal before resubmitting.",
                "If PA was approved but UTN missing, add UTN and resubmit.",
            ]
        else:
            checklist = [
                "Confirm the service requires prior authorization under CMS OPD/DMEPOS rules for the date of service.",
                "Locate the UTN (Unique Tracking Number) from the prior authorization decision.",
                "Add the UTN to the claim and verify all claim identifiers are correct.",
                "Attach/retain supporting documentation as required by CMS guidance.",
                "Resubmit the corrected claim.",
            ]

    elif category == "provider_enrollment":
        if intent == "explain":
            checklist = [
                "This typically means the claim is missing the billing provider NPI, the NPI is invalid, or the provider is not enrolled/active with Medicare for billing.",
                "Check that the billing provider and rendering provider NPIs are placed in the correct claim fields.",
                "Verify Medicare enrollment status (active vs inactive/excluded) before resubmitting.",
            ]
        else:
            checklist = [
                "Confirm the billing provider NPI is present and valid on the claim (correct digits, no typos).",
                "Verify the billing/rendering provider is enrolled in Medicare for the service type and location (and is not excluded).",
                "Ensure the claim uses the correct billing provider vs rendering provider NPI in the required fields.",
                "Correct the NPI/enrollment issue and resubmit; if enrollment is pending, wait for approval before resubmitting.",
            ]

    elif category == "medical_necessity":
        if intent == "criteria":
            checklist = [
                "Use the cited CMS policy sections to identify the specific coverage/medical necessity criteria that apply (diagnosis/indications, frequency limits, prerequisites).",
                "Confirm the documentation explicitly links the patient’s condition and symptoms to the medical need for the billed service.",
                "Verify codes support the criteria (ICD-10 reflects the qualifying condition; CPT/HCPCS matches what was provided).",
                "Create a quick mapping: policy criteria → where it appears in the clinical note (criteria-to-evidence table).",
            ]
        elif intent == "appeal":
            checklist = [
                "Yes—medical necessity denials are commonly appealable (follow the appropriate Medicare appeal level for your claim type).",
                "Include additional supporting documentation: progress notes, imaging/labs, prior treatment attempts and outcomes, and a physician letter of medical necessity.",
                "Reference the cited CMS policy sections and explicitly explain how your documentation satisfies the criteria.",
                "If appropriate, request a peer-to-peer review or include guideline support to strengthen the rationale.",
            ]
        elif intent == "documentation":
            checklist = [
                "Include clinical documentation supporting medical necessity: diagnosis, symptoms, severity, functional impact, and provider assessment.",
                "Attach objective evidence where applicable (test results, imaging, labs, screening results).",
                "Document prior conservative treatments attempted and outcomes (if required by policy).",
                "Ensure the provider’s order/referral and plan of care align with the billed service and dates of service.",
            ]
        elif intent == "resubmit":
            checklist = [
                "Update attachments with documentation that directly supports the medical necessity criteria in the cited policy.",
                "Confirm CPT/HCPCS, ICD-10, modifiers, and units match what is documented in the medical record.",
                "Resubmit within timely filing limits; if resubmission isn’t allowed, submit an appeal with the corrected documentation instead.",
            ]
        elif intent == "scenario":
            checklist = [
                "If documentation exists but wasn’t submitted, include it on appeal/reopening with policy citations.",
                "If criteria were not met, document conservative treatment/prerequisites (if required) and appeal with updated evidence.",
                "If coding mismatch caused the denial, correct ICD-10/CPT linkage and resubmit/appeal as allowed.",
            ]
        else:
            checklist = [
                "Submit supporting medical necessity documentation (clinical notes, diagnosis, symptoms, prior conservative treatment, and test results) tied to the billed service.",
                "Ensure documentation clearly supports why the service is reasonable and necessary for the patient’s condition.",
                "Verify CPT/HCPCS and ICD-10 codes align with the documented diagnosis and service provided.",
                "If already submitted and denied, file an appeal with additional documentation referencing the cited CMS policy sections.",
            ]

    elif category == "coding":
        checklist = [
            "Validate CPT/HCPCS and ICD-10 codes match the clinical documentation and the service performed.",
            "Check required modifiers, units, place of service, and diagnosis-to-procedure linkage.",
            "Correct coding/claim fields and resubmit; if coding was correct, appeal with documentation and policy citations.",
        ]

    elif category == "missing_info":
        checklist = [
            "Identify which required fields or attachments are missing (patient/provider identifiers, diagnosis, modifiers, supporting documentation).",
            "Correct the claim fields and include missing documentation per the cited CMS requirements.",
            "Resubmit the corrected claim within timely filing limits.",
        ]

    elif category == "timely_filing":
        checklist = [
            "Confirm Medicare timely filing rules and whether the claim missed the filing deadline.",
            "If an exception applies, gather proof (payer delay, retro eligibility, EOBs/correspondence) and submit with a reopening/appeal.",
            "Submit a reopening/appeal request referencing the cited CMS guidance and include supporting evidence.",
        ]

    elif category == "duplicate":
        checklist = [
            "Check whether the service line was previously processed (paid/denied) and whether an adjustment/correction is needed.",
            "If this is a corrected claim, ensure the correct claim frequency/type-of-bill indicators are used.",
            "If the prior processing was incorrect, submit a reopening/adjustment rather than a duplicate claim.",
        ]

    elif category == "eligibility":
        checklist = [
            "Verify beneficiary eligibility and coverage dates for the date of service.",
            "If Medicare is secondary, include primary payer EOB/COB information.",
            "Correct eligibility/coverage information and resubmit or bill the appropriate payer.",
        ]

    # else: keep the default fallback checklist

    return {
        "denial_text": denial_text,
        "question": question or "default denial analysis",
        "denial_summary": denial_summary,
        "likely_missing_items": missing_items,
        "recommended_actions": checklist,
        "evidence_snippets": evidence_snippets,
        "citations": [c.__dict__ for c in citations],
        "category": category,
        "intent": intent,
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
