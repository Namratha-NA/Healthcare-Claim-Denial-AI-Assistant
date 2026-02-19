# Healthcare Claim Denial AI Assistant

An LLM-powered Retrieval-Augmented Generation (RAG) system that analyzes Medicare claim denials, retrieves relevant CMS policy evidence, and generates structured remediation guidance with citations.
Built as an end-to-end AI system using FAISS, LangGraph, Ollama (local LLM), and Streamlit.

### Business Problem

Healthcare providers lose significant revenue due to Medicare claim denials caused by:

- Missing prior authorizations
- Medical necessity issues
- Coding errors
- Documentation gaps
- Duplicate claims
- Provider enrollment problems

Denial resolution is:
- Manual
- Time-consuming
- Policy-heavy
- Error-prone

Staff must search hundreds of CMS manual pages to determine why a claim was denied and how to fix it.

### Solution Overview

The Healthcare Claim Denial AI Assistant uses Retrieval-Augmented Generation (RAG) to ground LLM outputs in CMS policy documents:

- Analyze Medicare denial text
- Retrieve relevant CMS manuals using FAISS
- Filter evidence based on intent detection
- Generate structured JSON outputs:
  - Denial summary
  - Likely missing items
  - Recommended actions
  - Appeal guidance
- Provide CMS citations for transparency
- Run fully locally (no external API dependency)

### Architecture

- Denied Claim Text
        ↓
- Intent Detection
        ↓
- FAISS Retrieval (CMS Manuals)
        ↓
- Evidence Filtering
        ↓
- Ollama LLM (Structured JSON Output)
        ↓
- LangGraph Workflow
        ↓
- Streamlit UI

### Demo

Below is the working Streamlit interface powered by LangGraph + Ollama (local LLM). Watch the full working demo here:
https://www.loom.com/share/38a7d7a3c0834ca7aaa98b9cacd1bc90

### Tech Stack

- Python
- LangChain
- LangGraph
- FAISS
- Sentence Transformers
- Ollama (LLaMA 3)
- Streamlit

### Evaluation Framework

The system includes a structured evaluation pipeline using curated Medicare denial cases.

- 20 curated denial cases
- Expected policy keywords per case
- Latency measurement
- JSON structure validation
- Citation verification

### Results

- JSON validity: **100%**
- Average keyword coverage: **~0.70–0.80**
- Structured outputs across all cases
- Citation-backed responses
- Average latency: ~70–85 seconds (local LLM)

### Project Structure

src/
 ├── rag_engine.py
 ├── llm_client.py
 ├── langgraph_workflow.py
 ├── ui_app.py
 ├── eval_llm_answers.py

eval/
 ├── golden_set.jsonl
 ├── outputs/

### Business Impact

- Faster denial turnaround
- Reduced revenue leakage
- Improved claim resubmission accuracy
- Scalable denial management automation

### Future Improvements

- Implement real-time streaming responses
- Add analytics dashboard (denial trends & metrics)
- Expand evaluation dataset beyond 20 cases

 ### Author

Namratha Nagathihalli Anantha




