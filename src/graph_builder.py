from langgraph.graph import StateGraph, END
from src.graph_nodes import (
    MarketingState,
    create_search_node,
    create_researcher_node,
    create_strategist_node,
    create_copywriter_node,
    create_editor_node
)

def build_graph(call_qwen):
    """Build and compile the LangGraph workflow."""
    workflow = StateGraph(MarketingState)

    # Create node instances
    search_node = create_search_node()
    researcher_node = create_researcher_node(call_qwen)
    strategist_node = create_strategist_node(call_qwen)
    copywriter_node = create_copywriter_node(call_qwen)
    editor_node = create_editor_node(call_qwen)

    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("copywriter", copywriter_node)
    workflow.add_node("editor", editor_node)

    # Set entry point and edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "researcher")
    workflow.add_edge("researcher", "strategist")
    workflow.add_edge("strategist", "copywriter")
    workflow.add_edge("copywriter", "editor")
    workflow.add_edge("editor", END)

    return workflow.compile()