**Healthcare Claim Denial AI Assistant**

An LLM-powered Retrieval-Augmented Generation (RAG) system that analyzes Medicare claim denials, retrieves relevant CMS policy evidence, and generates structured remediation guidance with citations.
Built as an end-to-end AI system using FAISS, LangGraph, Ollama (local LLM), and Streamlit.

**Key Functions**

- Analyzes Medicare claim denial text
- Retrieves relevant CMS policy documents
- Generates structured JSON responses:
  - Denial summary
  - Likely missing items
  - Recommended actions
  - Appeal guidance
- Provides CMS citations for transparency
- Includes a structured evaluation framework 
- Runs fully locally 

**Architecture Overview**

Denied Claim Text
        ↓
Intent Detection
        ↓
FAISS Retrieval (CMS Manuals)
        ↓
Evidence Filtering
        ↓
Ollama LLM (Structured JSON Output)
        ↓
LangGraph Workflow
        ↓
Streamlit UI

**Tech Stack**

- Python
- LangChain
- LangGraph
- FAISS
- Sentence Transformers
- Ollama (LLaMA 3)
- Streamlit

**Evaluation**

Tested on 20 curated Medicare denial cases.

- JSON validity: 100%
- Avg keyword coverage: ~0.70–0.80
- Fully structured outputs
- Citation-supported responses

**Project Structure**

src/
 ├── rag_engine.py
 ├── llm_client.py
 ├── langgraph_workflow.py
 ├── ui_app.py
 ├── eval_llm_answers.py

eval/
 ├── golden_set.jsonl
 ├── outputs/

**Key Highlights**

- Retrieval-Augmented Generation with CMS policy grounding
- Intent-aware filtering to reduce hallucinations
- Structured JSON enforcement
- Evaluation-driven LLM development
- Fully local LLM deployment

 **Author**

Namratha Nagathihalli Anantha
