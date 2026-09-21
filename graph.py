from langgraph.graph import StateGraph, END
from state import ForgeState
from nodes import (
    orchestrator_node, route_after_orchestration, parse_docs_node,
    grade_extraction_node, route_after_grading, generate_tool_node,
    test_via_a2a_node, route_after_test, package_server_node,
)
from cli import get_docs_input

def build_graph():
    workflow = StateGraph(ForgeState)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("parse_docs", parse_docs_node)
    workflow.add_node("grade_extraction", grade_extraction_node)
    workflow.add_node("generate_tool", generate_tool_node)
    workflow.add_node("test_via_a2a", test_via_a2a_node)
    workflow.add_node("package_server", package_server_node)

    workflow.set_entry_point("orchestrator")
    workflow.add_conditional_edges("orchestrator", route_after_orchestration,
        {"simple_path": "parse_docs", "complex_path": "parse_docs"})
    workflow.add_edge("parse_docs", "grade_extraction")
    workflow.add_conditional_edges("grade_extraction", route_after_grading,
        {"retry": "parse_docs", "continue": "generate_tool"})
    workflow.add_edge("generate_tool", "test_via_a2a")
    workflow.add_conditional_edges("test_via_a2a", route_after_test,
        {"retry": "generate_tool", "done": "package_server"})
    workflow.add_edge("package_server", END)
    return workflow.compile()

if __name__ == "__main__":
    raw_docs = get_docs_input()
    graph = build_graph()
    result = graph.invoke({
        "raw_docs": raw_docs, "orchestration_plan": None, "endpoints": None,
        "generated_code": None, "retry_count": 0, "missing_fields": None,
        "codegen_attempt": 0, "test_passed": None, "test_output": None, "output_path": None,
    })
    print("\n=== FINAL ===")
    print(f"Plan: {result['orchestration_plan']}")
    print(f"Output saved to: {result['output_path']}")