"""
STATE = the shared notebook every step of the graph reads from and writes to.

Think of it like a clipboard that gets passed from person to person on an
assembly line. Each person (node) looks at what's already written on it,
does their one job, writes their result back, and hands it to the next
person. Nobody needs to know what happened before their turn — they just
trust the clipboard has what they need.

In LangGraph, this clipboard is just a dictionary with a fixed shape,
defined below using TypedDict (this just means: "this dictionary will
always have exactly these keys, with these types").
"""

from typing import TypedDict, Optional, List


class ForgeState(TypedDict):
    raw_docs: str                    # the API documentation text we start with
    orchestration_plan: Optional[str] 
    endpoints: Optional[List[dict]]    # filled in by Step 1 (parse_docs_node)
    generated_code: Optional[str]    # filled in by Step 2 (generate_tool_node)
    retry_count: int
    missing_fields: Optional[List[str]]
    codegen_attempt: int
    test_passed: Optional[bool]
    test_output: Optional[str]
    output_path: Optional[str]