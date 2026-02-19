from typing import TypedDict
from langgraph.graph import StateGraph, END
from rag_engine import answer_denial

class DenialState(TypedDict):
    denial_text: str
    question: str
    result: dict

def run_rag(state: DenialState) -> DenialState:
    result = answer_denial(
        denial_text=state["denial_text"],
        question=state.get("question", ""),
    )
    return {**state, "result": result}

def build_graph():
    graph = StateGraph(DenialState)
    graph.add_node("rag_node", run_rag)
    graph.set_entry_point("rag_node")
    graph.add_edge("rag_node", END)
    return graph.compile()

