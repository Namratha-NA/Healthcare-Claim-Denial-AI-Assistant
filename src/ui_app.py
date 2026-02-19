import streamlit as st
from langgraph_workflow import build_graph

st.set_page_config(page_title="Healthcare Claim Denial AI", layout="wide")

st.title("🏥 Healthcare Claim Denial AI Assistant")
st.markdown("Analyze Medicare claim denials using CMS policy evidence and an LLM-powered RAG system.")

@st.cache_resource
def get_graph():
    return build_graph()

denial_text = st.text_area("Paste Denied Claim Text:", height=150)
question = st.text_input("Ask a Question (optional):")

if st.button("Analyze Claim"):

    if not denial_text.strip():
        st.warning("Please enter denial text.")
        st.stop()

    graph = get_graph()

    with st.spinner("Analyzing claim using AI..."):
        try:
            result = graph.invoke({"denial_text": denial_text, "question": question})
            output = result.get("result", {})
        except Exception as e:
            st.error(f"Error during analysis: {e}")
            st.stop()

    st.divider()

    st.subheader("Denial Summary")
    st.write(output.get("denial_summary", "No summary generated."))

    missing_items = output.get("likely_missing_items", [])
    if missing_items:
        st.subheader("Likely Missing Items")
        for item in missing_items:
            st.write(f"- {item}")

    actions = output.get("recommended_actions", [])
    if actions:
        st.subheader("Recommended Actions")
        for i, action in enumerate(actions, 1):
            st.write(f"{i}. {action}")

    appeal = output.get("appeal_guidance", "")
    if appeal:
        st.subheader("Appeal Guidance")
        st.write(appeal)

    citations = output.get("citations", [])
    if citations:
        st.subheader("Citations")
        for c in citations:
            st.write(f"- {c.get('filename','')} | page {c.get('page_num','')}")
