"""
STATE = the shared notebook every step of the graph reads from and writes to.

Think of it like a clipboard passed along an assembly line. Each node looks
at what's already written on it, does its one job, writes its result back,
and hands it to the next node.

In LangGraph this clipboard is a dictionary with a fixed shape, defined
below with TypedDict.
"""

from typing import TypedDict, Optional, List


class ForgeState(TypedDict):
    # --- input ---
    raw_docs: str                       # the API documentation text we start with
    source_url: Optional[str]           # where the docs came from (used to resolve relative server URLs)

    # --- orchestrator ---
    orchestration_plan: Optional[str]   # "simple" or "complex"
    max_code_attempts: int              # retry budget for codegen, set by the orchestrator

    # --- extraction ---
    endpoints: Optional[List[dict]]     # the endpoint definitions code is generated from
    missing_fields: Optional[List[str]] # endpoints dropped for lacking a required field

    # --- agentic RAG loop (prose docs only) ---
    rag_candidates: Optional[List[dict]]  # endpoints discovered in the docs: [{"method", "path"}]
    rag_cursor: int                       # which candidate is being worked on
    rag_round: int                        # retrieval attempts so far for that candidate
    rag_query: Optional[str]              # the search query for the current round
    rag_next_query: Optional[str]         # the better query the grader proposed
    rag_context: Optional[List[str]]      # chunks retrieved for the current round
    rag_trace: List[dict]                 # record of every round: query, sufficient or not, what was missing

    # --- generate / validate loop ---
    code_batches: Optional[List[str]]   # generated code, one string per batch of endpoints
    generated_code: Optional[str]       # all batches joined together
    codegen_attempt: int
    test_passed: Optional[bool]
    test_output: Optional[str]          # human-readable summary from the validator
    test_errors: Optional[List[dict]]   # structured errors from the validator (fed back to the LLM)
    attempt_history: List[dict]         # one record per attempt: what failed, what was regenerated

    # --- output ---
    output_path: Optional[str]
