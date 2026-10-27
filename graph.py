r"""
GRAPH = the wiring. Which node runs after which, and where the loops are.

                              orchestrator
                             /            \
                  (OpenAPI spec)          (prose docs)
                       |                       |
                  parse_docs              index_docs
                       |                       |
                       |              discover_endpoints
                       |                       |
                       |                   retrieve <-----------+
                       |                       |                |
                       |               grade_retrieval -> rewrite_query     AGENTIC RAG LOOP
                       |                       |    \___ next endpoint ___/
                       +----------+------------+
                                  v
                          grade_extraction
                                  |
                            generate_tool <--(retry, with errors)--+
                                  |                                |
                           security_review ------------------------+        SELF-CORRECTING LOOP
                                  |            (reads the code,    |        (either agent can send
                                  |             never runs it)     |         the code back)
                            test_via_a2a --------------------------+
                                  |            (runs the code against mocked HTTP)
                           package_server -> END
                    (if the review never passes: report only, nothing packaged or run)
"""

from langgraph.graph import StateGraph, END
from state import ForgeState
from nodes import (
    orchestrator_node, route_by_input, parse_docs_node,
    grade_extraction_node, route_after_grading, generate_tool_node,
    security_review_node, route_after_review,
    test_via_a2a_node, route_after_test, package_server_node,
)
from rag_nodes import (
    index_docs_node, discover_endpoints_node, route_after_discovery,
    retrieve_node, grade_retrieval_node, route_after_retrieval_grade, rewrite_query_node,
)
from cli import get_docs_input

# Each node visit counts as a step. The RAG loop visits several nodes per
# endpoint, so the default limit of 25 would be hit on any real API.
RUN_CONFIG = {"recursion_limit": 5000}


def build_graph():
    workflow = StateGraph(ForgeState)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("parse_docs", parse_docs_node)
    workflow.add_node("index_docs", index_docs_node)
    workflow.add_node("discover_endpoints", discover_endpoints_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_retrieval", grade_retrieval_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("grade_extraction", grade_extraction_node)
    workflow.add_node("generate_tool", generate_tool_node)
    workflow.add_node("security_review", security_review_node)
    workflow.add_node("test_via_a2a", test_via_a2a_node)
    workflow.add_node("package_server", package_server_node)

    workflow.set_entry_point("orchestrator")
    workflow.add_conditional_edges("orchestrator", route_by_input,
        {"spec": "parse_docs", "prose": "index_docs"})

    # OpenAPI path
    workflow.add_edge("parse_docs", "grade_extraction")

    # Prose path: the agentic RAG loop
    workflow.add_edge("index_docs", "discover_endpoints")
    workflow.add_conditional_edges("discover_endpoints", route_after_discovery,
        {"retrieve": "retrieve", "none_found": "grade_extraction"})
    workflow.add_edge("retrieve", "grade_retrieval")
    workflow.add_conditional_edges("grade_retrieval", route_after_retrieval_grade,
        {"rewrite": "rewrite_query", "next_endpoint": "retrieve", "done": "grade_extraction"})
    workflow.add_edge("rewrite_query", "retrieve")

    # Shared: generate, review and validate over A2A, self-correct, package.
    # The review comes first because it never runs the code; the validator does.
    workflow.add_conditional_edges("grade_extraction", route_after_grading,
        {"continue": "generate_tool", "abort": END})
    workflow.add_edge("generate_tool", "security_review")
    workflow.add_conditional_edges("security_review", route_after_review,
        {"validate": "test_via_a2a", "retry": "generate_tool", "rejected": "package_server"})
    workflow.add_conditional_edges("test_via_a2a", route_after_test,
        {"retry": "generate_tool", "done": "package_server"})
    workflow.add_edge("package_server", END)
    return workflow.compile()


def initial_state(raw_docs: str, source_url: str = None, selection: dict = None, extra_auth: list = None) -> dict:
    return {
        "raw_docs": raw_docs, "source_url": source_url,
        "selection": selection or {}, "extra_auth": extra_auth or [],
        "api_meta": None, "selection_note": None,
        "orchestration_plan": None, "max_code_attempts": 2,
        "endpoints": None, "missing_fields": None,
        "rag_candidates": None, "rag_cursor": 0, "rag_round": 0, "rag_query": None,
        "rag_next_query": None, "rag_context": None, "rag_trace": [],
        "code_batches": None, "generated_code": None, "codegen_attempt": 0,
        "security_passed": None, "security_output": None, "security_warnings": [], "fix_errors": None,
        "test_passed": None, "test_output": None, "test_errors": None,
        "attempt_history": [], "output_path": None, "mcp_client_check": None,
    }


def run(raw_docs: str, source_url: str = None, selection: dict = None, extra_auth: list = None) -> dict:
    return build_graph().invoke(initial_state(raw_docs, source_url, selection, extra_auth), RUN_CONFIG)


if __name__ == "__main__":
    raw_docs, source_url, options = get_docs_input()
    result = run(raw_docs, source_url, options["selection"], options["extra_auth"])
    print("\n=== FINAL ===")
    print(f"Plan: {result['orchestration_plan']}")
    if result["output_path"]:
        print(f"Output saved to: {result['output_path']}")
    elif result.get("security_passed") is False:
        print("Nothing was packaged: the security review rejected the code "
              "(see validation_report.json in the API's folder under generated/)")
    else:
        print("Nothing was packaged.")
