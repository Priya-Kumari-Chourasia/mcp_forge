"""
NODES = the individual workers on the assembly line.

A "node" in LangGraph is a plain Python function with one rule: it takes
the state (the clipboard) in, and returns the updated state out. LangGraph
handles calling them in the right order.

This file holds the orchestrator, the OpenAPI path, and the codegen loop:

    generate_tool -> validator agent (over A2A) -> regenerate ONLY the
    failing batches, with the validator's errors and the previous code

The prose-docs path (agentic RAG) is in rag_nodes.py. Both loops follow the
same idea: when a check fails, the retry is TOLD WHAT WENT WRONG instead of
just being asked again.
"""

import ast
import json
import os
import re

from state import ForgeState
from llm import call_llm
from spec_parser import is_openapi_spec, extract_endpoints_from_openapi
from a2a_client import validate_via_a2a, ValidatorUnreachable

REQUIRED_FIELDS = ["method", "path", "description"]
VALIDATOR_AGENT_URL = os.getenv("VALIDATOR_AGENT_URL", "http://localhost:8000")
BATCH_SIZE = 15  # endpoints per LLM call - keeps each request well within context limits

# The orchestrator's decision changes how hard the pipeline tries:
# a complex API gets a bigger retry budget for code generation.
CODE_ATTEMPTS_BY_PLAN = {"simple": 2, "complex": 4}


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

def orchestrator_node(state: ForgeState) -> ForgeState:
    """Reads the docs and decides: is this a simple or a complex API?
    The answer sets the retry budget for the codegen loop."""
    spec = is_openapi_spec(state["raw_docs"])
    if spec is not None:
        # Structured spec: count endpoints directly, no LLM needed.
        endpoints = extract_endpoints_from_openapi(spec, state.get("source_url"))
        plan = "complex" if len(endpoints) > 10 else "simple"
        detail = f"{len(endpoints)} endpoint(s), no LLM needed"
    else:
        # Prose docs: a sample is enough to judge complexity, and keeps the request small.
        sample = state["raw_docs"][:3000]
        prompt = f"""Analyze these API docs and decide: is this a SIMPLE or COMPLEX API?

SIMPLE means:
- Few endpoints (1-10)
- Clear, straightforward documentation
- No auth OR simple API key auth

COMPLEX means:
- Many endpoints (10+)
- Unclear or inconsistent docs
- OAuth, custom auth, or multiple auth schemes
- Pagination, rate limiting, or other edge cases

Respond with ONLY one word: simple OR complex

API docs (sample):
{sample}"""
        plan = call_llm(prompt).strip().lower()
        if plan not in CODE_ATTEMPTS_BY_PLAN:
            print(f"[WARNING] Unexpected plan: {plan}, defaulting to complex")
            plan = "complex"
        detail = "decided by LLM"

    state["orchestration_plan"] = plan
    state["max_code_attempts"] = CODE_ATTEMPTS_BY_PLAN[plan]
    print(f"ORCHESTRATOR -> plan: {plan} ({detail}); codegen retry budget: {state['max_code_attempts']} attempt(s)")
    return state


# ----------------------------------------------------------------------
# Extraction: structured spec -> parsed in code; prose -> agentic RAG (rag_nodes.py)
# ----------------------------------------------------------------------

def route_by_input(state: ForgeState) -> str:
    """A structured spec needs no retrieval and no LLM. Prose docs go through agentic RAG."""
    if is_openapi_spec(state["raw_docs"]) is not None:
        print("  ROUTING: OpenAPI spec -> deterministic parser")
        return "spec"
    print("  ROUTING: prose docs -> agentic RAG")
    return "prose"


def parse_docs_node(state: ForgeState) -> ForgeState:
    spec = is_openapi_spec(state["raw_docs"])
    state["endpoints"] = extract_endpoints_from_openapi(spec, state.get("source_url"))
    print(f"STEP 1 (parse_docs) -> OpenAPI spec, {len(state['endpoints'])} endpoint(s), no LLM needed")
    return state


def grade_extraction_node(state: ForgeState) -> ForgeState:
    """Final gate before code generation: drop endpoints that lack a required field."""
    endpoints = state["endpoints"] or []
    usable, issues = [], []
    for i, ep in enumerate(endpoints):
        missing = [f for f in REQUIRED_FIELDS if not (isinstance(ep, dict) and ep.get(f))]
        if missing:
            issues.append(f"endpoint {i}: missing {missing}")
        else:
            usable.append(ep)
    state["endpoints"] = usable
    state["missing_fields"] = issues
    print(f"STEP 1b (grade_extraction) -> {len(usable)} usable endpoint(s), dropped: {issues}")
    return state


def route_after_grading(state: ForgeState) -> str:
    if not state["endpoints"]:
        print("  DECISION: no usable endpoints were extracted, stopping")
        return "abort"
    return "continue"


# ----------------------------------------------------------------------
# The codegen loop: generate code, validate it, regenerate what failed
# ----------------------------------------------------------------------

CODEGEN_INSTRUCTIONS = """Write Python functions (MCP tools) - exactly ONE function per endpoint below.

Rules:
- Start with: import os, requests
- Use snake_case function names built from the method and path
  (e.g. GET /pet/findByStatus -> get_pet_find_by_status). Names must be unique.
- Each function must be completely SELF-CONTAINED. Do NOT define shared helper
  functions - this code is combined with output from other independent batches.
- The URL is the endpoint's base_url followed by its path. If base_url is empty,
  use os.environ["API_BASE_URL"] instead.
- Path parameters and required query parameters are required function arguments.
  Optional query parameters are arguments that default to None, and are only sent
  when they are not None.
- If the endpoint has a "request_body", add an argument `body: dict` and send it
  with json=body (or data=body when content_type is not JSON)List the body's
  field names in the docstring..
- Add type hints to every argument and a one-line docstring taken from the description.
- Call requests.<method>(..., timeout=30), then return response.json() if the response
  is JSON, otherwise response.text.

Return ONLY valid Python code. No markdown fences, no explanations.
"""


def build_codegen_prompt(batch, previous_code=None, errors=None):
    """First attempt: instructions + endpoints.
    Retry: the same, PLUS the code that failed and the validator's errors.
    That extra context is what makes the loop self-correcting - without it,
    the model would be asked the identical question and (at temperature 0)
    would give the identical wrong answer."""
    prompt = f"{CODEGEN_INSTRUCTIONS}\nEndpoints:\n{json.dumps(batch, indent=2)}\n"
    if previous_code and errors:
        error_lines = "\n".join(f"- {e}" for e in errors)
        prompt += f"""
Your previous code for these endpoints FAILED validation.

Validator errors:
{error_lines}

Previous code:
{previous_code}

Fix these problems and return the complete corrected code for ALL endpoints above.
"""
    return prompt


def generate_one_batch(batch, label, previous_code=None, errors=None, max_syntax_retries=2):
    """Generates code for one batch. A syntax error is the cheapest failure to
    catch, so it is retried right here (with the error message) instead of
    costing a full trip to the validator."""
    code = None
    for _ in range(max_syntax_retries):
        code = call_llm(build_codegen_prompt(batch, previous_code, errors))
        try:
            ast.parse(code)
            return code
        except SyntaxError as e:
            print(f"  {label} -> SyntaxError, retrying this batch with the error")
            previous_code, errors = code, [f"SyntaxError: {e}"]
    print(f"  {label} -> still has a syntax error, passing it on to the validator")
    return code


def top_level_function_names(code: str):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None          # None means "this batch does not even parse"
    return [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def errors_by_batch(errors, code_batches):
    """Works out WHICH batches each validator error belongs to, so only those
    get regenerated. Returns {batch_index: [error text, ...]}.

    - an error naming a function  -> the batch(es) that define that function
    - an error naming an endpoint -> the batch that endpoint was in
    - anything else (e.g. a syntax error) -> batches that don't parse, or all of them
    """
    names_per_batch = [top_level_function_names(code) for code in code_batches]
    result = {}

    for err in errors:
        targets = set()
        if err.get("function"):
            targets = {i for i, names in enumerate(names_per_batch) if names and err["function"] in names}
        elif err.get("endpoint_index") is not None:
            targets = {err["endpoint_index"] // BATCH_SIZE}
        if not targets:
            targets = {i for i, names in enumerate(names_per_batch) if names is None}
        if not targets:
            targets = set(range(len(code_batches)))
        for i in targets:
            result.setdefault(i, []).append(err["error"])
    return result


def generate_tool_node(state: ForgeState) -> ForgeState:
    attempt = state.get("codegen_attempt", 0) + 1
    endpoints = state["endpoints"]
    batches = [endpoints[i:i + BATCH_SIZE] for i in range(0, len(endpoints), BATCH_SIZE)]
    code_batches = state.get("code_batches")

    if attempt == 1 or not code_batches:
        print(f"STEP 2 (generate_tool) attempt 1 -> {len(endpoints)} endpoint(s) in {len(batches)} batch(es)")
        code_batches = []
        for i, batch in enumerate(batches):
            code_batches.append(generate_one_batch(batch, f"batch {i + 1}/{len(batches)}"))
            print(f"  batch {i + 1}/{len(batches)} done ({len(batch)} endpoint(s))")
        regenerated = list(range(len(batches)))
    else:
        # Retry: regenerate only the batches the validator complained about,
        # and give each one its own errors plus the code that produced them.
        failing = errors_by_batch(state.get("test_errors") or [], code_batches)
        print(f"STEP 2 (generate_tool) attempt {attempt} -> regenerating {len(failing)} of "
              f"{len(batches)} batch(es) using the validator's errors")
        for i, batch_errors in sorted(failing.items()):
            code_batches[i] = generate_one_batch(
                batches[i], f"batch {i + 1}/{len(batches)}",
                previous_code=code_batches[i], errors=batch_errors,
            )
            print(f"  batch {i + 1}/{len(batches)} regenerated ({len(batch_errors)} error(s) fed back)")
        regenerated = sorted(failing)

    state["code_batches"] = code_batches
    state["generated_code"] = "\n\n".join(code_batches)
    state["codegen_attempt"] = attempt
    state["attempt_history"] = state.get("attempt_history", []) + [
        {"attempt": attempt, "batches": [i + 1 for i in regenerated]}
    ]
    return state


def test_via_a2a_node(state: ForgeState) -> ForgeState:
    """Hands the code, and the endpoints it must implement, to the validator
    agent over the A2A protocol (see a2a_client.py)."""
    print(f"STEP 3 (validate via A2A) attempt {state['codegen_attempt']}")
    try:
        result = validate_via_a2a(VALIDATOR_AGENT_URL, state["generated_code"], state["endpoints"])
        passed, output, errors = result["passed"], result["output"], result["errors"]
    except ValidatorUnreachable:
        print("ERROR: Could not reach the validator agent.")
        print("  Start it first, in another terminal: python validator_agent.py")
        passed, output = False, "validator unreachable"
        errors = [{"function": None, "endpoint_index": None, "error": output}]

    state["test_passed"] = passed
    state["test_output"] = output
    state["test_errors"] = errors

    # Record the outcome on this attempt's history entry.
    state["attempt_history"][-1].update({"passed": passed, "errors": [e["error"] for e in errors]})

    print(f"  verdict: passed={passed}")
    if passed:
        print(f"  {output}")
    for e in errors:
        print(f"  - {e['error']}")
    return state


def route_after_test(state: ForgeState) -> str:
    if state["test_output"] == "validator unreachable":
        print("  DECISION: validator is not running, finishing without retrying")
        return "done"
    if not state["test_passed"] and state["codegen_attempt"] < state["max_code_attempts"]:
        print("  DECISION: validation failed, sending errors back to generate_tool")
        return "retry"
    print(f"  DECISION: {'passed' if state['test_passed'] else 'out of attempts'}, finishing")
    return "done"


# ----------------------------------------------------------------------
# Packaging: wrap the validated functions as a runnable MCP server
# ----------------------------------------------------------------------

def package_as_mcp_server(generated_code: str, server_name: str):
    """Wraps raw function definitions as a runnable MCP server: adds the
    MCPServer import/instance, a @server.tool() decorator above each function
    (found via ast, not string guessing), and a __main__ block that starts it."""
    tree = ast.parse(generated_code)
    lines = generated_code.split("\n")
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    tool_info = [{"name": n.name, "description": ast.get_docstring(n) or "No description"} for n in functions]

    # Insert decorators bottom-to-top so earlier line numbers don't shift.
    for lineno in sorted((n.lineno for n in functions), reverse=True):
        lines.insert(lineno - 1, "@server.tool()")

    header = (f'"""\nAuto-generated MCP server: {server_name}\nGenerated by MCP-Forge.\n"""\n'
              f'from mcp.server.mcpserver import MCPServer\n\nserver = MCPServer("{server_name}")\n\n')
    footer = '\n\nif __name__ == "__main__":\n    server.run()\n'
    return header + "\n".join(lines) + footer, tool_info


def package_server_node(state: ForgeState) -> ForgeState:
    first = state["endpoints"][0] if state.get("endpoints") else {}
    safe_name = re.sub(r'[^a-zA-Z0-9]+', '-', first.get("base_url") or "api").strip('-').lower() or "generated-api"
    output_dir = os.path.join("generated", safe_name)
    os.makedirs(output_dir, exist_ok=True)

    # The record of both loops: every codegen attempt and every retrieval round.
    with open(os.path.join(output_dir, "validation_report.json"), "w", encoding="utf-8") as f:
        json.dump({
            "passed": bool(state.get("test_passed")),
            "attempts": state.get("attempt_history", []),      # the codegen loop
            "retrieval": state.get("rag_trace", []),           # the agentic RAG loop (prose docs only)
        }, f, indent=2)

    try:
        server_code, tool_info = package_as_mcp_server(state["generated_code"], safe_name)
    except SyntaxError as e:
        raw_path = os.path.join(output_dir, "raw_output_INVALID.py")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(f"# WARNING: syntax error, could not be packaged as a real MCP server.\n# Error: {e}\n\n")
            f.write(state["generated_code"])
        print(f"\nERROR: Generated code has a syntax error and could not be packaged: {e}")
        print(f"  Raw output saved for inspection: {raw_path}")
        state["output_path"] = raw_path
        return state

    server_path = os.path.join(output_dir, "server.py")
    with open(server_path, "w", encoding="utf-8") as f:
        f.write(server_code)
    with open(os.path.join(output_dir, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write("mcp>=2\nrequests\n")
    readme = f"# {safe_name} - MCP Server\n\nAuto-generated by MCP-Forge.\n\n## Tools\n\n"
    for t in tool_info:
        readme += f"- **{t['name']}**: {t['description']}\n"
    readme += (f'\n## Add to Claude Desktop\n\n```json\n{{\n  "mcpServers": {{\n    "{safe_name}": {{\n'
               f'      "command": "python",\n      "args": ["{os.path.abspath(server_path)}"]\n    }}\n  }}\n}}\n```\n')
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    status = "PASSED" if state.get("test_passed") else "FAILED (needs manual review)"
    print(f"\nPackaged MCP server -> {output_dir}/")
    print(f"  {len(tool_info)} tool(s): {[t['name'] for t in tool_info]}")
    print(f"  Status: {status}")
    print("  Attempts:")
    for record in state.get("attempt_history", []):
        outcome = "passed" if record.get("passed") else f"failed with {len(record.get('errors', []))} error(s)"
        print(f"    attempt {record['attempt']}: generated batch(es) {record['batches']} -> {outcome}")
    state["output_path"] = server_path
    return state
