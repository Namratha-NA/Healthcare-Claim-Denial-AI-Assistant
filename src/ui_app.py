import streamlit as st
from langgraph_workflow import build_graph

st.set_page_config(page_title="Healthcare Claim Denial AI", layout="wide")

st.title(" Healthcare Claim Denial AI Assistant")

denial_text = st.text_area(
    "Paste Denied Claim Text:",
    height=150,
)

question = st.text_input(
    "Ask a Question (optional):",
)

if st.button("Analyze Claim"):

    if not denial_text.strip():
        st.warning("Please enter denial text.")
    else:
        graph = build_graph()

        result = graph.invoke({
            "denial_text": denial_text,
            "question": question,
        })

        output = result["result"]

        st.subheader(" Denial Summary")
        st.write(output.get("denial_summary", ""))

        st.subheader(" Likely Missing Items")
        for item in output.get("likely_missing_items", []):
            st.write(f"- {item}")

        st.subheader(" Recommended Actions")
        for action in output.get("recommended_actions", []):
            st.write(f"- {action}")

        st.subheader(" Citations")
        for citation in output.get("citations", []):
            st.write(
                f"- {citation['filename']} | page {citation['page_num']}"
            )
