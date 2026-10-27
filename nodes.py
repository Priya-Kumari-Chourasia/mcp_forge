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

The LLM writes only the tool functions. The fixed parts of the server (base
URL, authentication, error handling) come from server_template.py, so they
are the same, tested code in every generated server.
"""

import ast
import fnmatch
import json
import os
from collections import Counter

from state import ForgeState
from llm import call_llm
from spec_parser import is_openapi_spec, extract_endpoints_from_openapi, get_api_meta
from a2a_client import check_via_a2a, AgentUnreachable
from security_rules import strip_invisible
import server_template

REQUIRED_FIELDS = ["method", "path", "description"]
VALIDATOR_AGENT_URL = os.getenv("VALIDATOR_AGENT_URL", "http://localhost:8000")
SECURITY_AGENT_URL = os.getenv("SECURITY_AGENT_URL", "http://localhost:8001")
# Endpoints per LLM call - keeps each request within the LLM's limits. For an API with
# very long docs on Groq's free tier, a smaller batch (MCP_FORGE_BATCH_SIZE=8) avoids
# requests that are too large for one minute's token allowance.
BATCH_SIZE = max(1, int(os.getenv("MCP_FORGE_BATCH_SIZE", "15")))

# A server with hundreds of tools overwhelms the AI using it (and costs a lot to
# generate), so by default only this many endpoints are kept. Choose which ones
# with --tags / --paths, or raise the cap with --max-tools.
DEFAULT_MAX_TOOLS = 40

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
    state["api_meta"] = get_api_meta(spec, state.get("source_url"))     # title, base URL, auth schemes
    print(f"STEP 1 (parse_docs) -> OpenAPI spec, {len(state['endpoints'])} endpoint(s), no LLM needed")
    return state


def _select(endpoints: list, selection: dict):
    """Keeps the endpoints the user asked for. Tags and path patterns narrow the
    set (an endpoint must match both when both are given); then the cap applies.
    Returns (kept endpoints, a note saying what was left out, or None)."""
    tags = {t.lower() for t in selection.get("tags") or []}
    patterns = selection.get("paths") or []
    max_tools = selection.get("max_tools") or DEFAULT_MAX_TOOLS

    chosen = [
        ep for ep in endpoints
        if (not tags or tags & {t.lower() for t in ep.get("tags") or []})
        and (not patterns or any(fnmatch.fnmatchcase(ep["path"], p) for p in patterns))
    ]
    notes = []
    if len(chosen) < len(endpoints):
        notes.append(f"{len(chosen)} of {len(endpoints)} endpoint(s) match the chosen tags/paths")
    if len(chosen) > max_tools:
        notes.append(f"kept the first {max_tools} of {len(chosen)}; choose others with --tags or --paths, "
                     f"or raise --max-tools")
        chosen = chosen[:max_tools]
    return chosen, ("; ".join(notes) or None)


def grade_extraction_node(state: ForgeState) -> ForgeState:
    """Final gate before code generation, shared by both paths:
    drop endpoints that lack a required field, keep the ones the user chose,
    settle the API's base URL and credentials, and give every endpoint the
    tool name the generated function must have."""
    endpoints = state["endpoints"] or []
    usable, issues = [], []
    for i, ep in enumerate(endpoints):
        missing = [f for f in REQUIRED_FIELDS if not (isinstance(ep, dict) and ep.get(f))]
        if missing:
            issues.append(f"endpoint {i}: missing {missing}")
        else:
            usable.append(ep)

    usable, note = _select(usable, state.get("selection") or {})

    # One base URL for the whole server. A spec states it; for prose docs, use the
    # one most endpoints agree on.
    meta = dict(state.get("api_meta") or {"title": "", "base_url": "", "auth": []})
    if not meta.get("base_url"):
        known = Counter(ep["base_url"] for ep in usable if ep.get("base_url"))
        meta["base_url"] = known.most_common(1)[0][0] if known else ""
    for ep in usable:
        ep["base_url"] = meta["base_url"]

    # Credentials the docs did not declare but the user said the API needs (--auth).
    declared = {(a["type"], a.get("name")) for a in meta.get("auth") or []}
    meta["auth"] = list(meta.get("auth") or []) + [
        a for a in state.get("extra_auth") or [] if (a["type"], a.get("name")) not in declared]

    server_template.assign_tool_names(usable)

    state["endpoints"] = usable
    state["missing_fields"] = issues
    state["selection_note"] = note
    state["api_meta"] = meta
    print(f"STEP 1b (grade_extraction) -> {len(usable)} usable endpoint(s), dropped: {issues}")
    if note:
        print(f"  selection: {note}")
    return state


def route_after_grading(state: ForgeState) -> str:
    if not state["endpoints"]:
        print("  DECISION: no usable endpoints were extracted, stopping")
        return "abort"
    return "continue"


# ----------------------------------------------------------------------
# The codegen loop: generate code, validate it, regenerate what failed
# ----------------------------------------------------------------------

# The rulebook sent to the LLM every time it writes tool code.
#
# The LLM writes ONLY the tool functions. Base URL, authentication and error
# handling are in the header from server_template.py, so each tool is a short
# call to _request() and there is far less for the LLM to get wrong.
#
# Return type rule (added after testing a generated server in Claude Desktop):
# the MCP SDK checks each tool's return value against its return annotation.
# Petstore's find-by-status returns a JSON ARRAY, so a tool annotated
# `-> Union[dict, str]` was rejected with a bare "Error executing tool".
# `Union[dict, list, str]` covers JSON objects, JSON arrays and plain text.
CODEGEN_INSTRUCTIONS = """Write the tool functions of an MCP server - exactly ONE function per endpoint below.

Your code goes directly under a header that is already written. The header imports
os, Annotated, Any, Literal, Optional, Union and pydantic's Field, and defines:

    _request(method, path, *, path_params=None, params=None, json=None, data=None, headers=None)

_request fills in the path's placeholders (encoding each value), adds the API's base URL
and authentication, leaves out params whose value is None, raises a clear error when the
API returns an error, and returns the response.

Rules (a security review checks them before the code is run):
- Do NOT write imports, constants, helper functions or decorators. Write only the tool functions.
- Name each function exactly as the endpoint's "name".
- Each function makes exactly one _request(...) call and returns what it returns. The method
  and the path are plain string literals copied exactly from the endpoint, placeholders
  included: _request("GET", "/pet/{petId}", path_params={"petId": pet_id}).
  Never fill in the path yourself (no f-strings): _request does it, safely. Never write
  full URLs, never use requests, httpx or os, and never read environment variables.
- Use only the function's own arguments, variables it assigns itself, _request and the
  type names above.
- One argument per path, query and header parameter, named in snake_case. Send each one
  under its documented name: path parameters in path_params, query parameters in params,
  header parameters in headers. Example: params={"status": status} if the documented name
  is status.
- Path parameters and required query parameters have no default. Every other parameter
  is Optional and defaults to None.
- Type every argument as Annotated[<type>, Field(description="...")], using the
  parameter's description (or a short one of your own). integer -> int, number -> float,
  boolean -> bool, array -> list[<item type>], anything else -> str. If the parameter
  lists "enum" values, the type is Literal[...] of those values (list[Literal[...]] for
  an array). An optional parameter's type is Optional[...] inside the Annotated.
- If the endpoint has a "request_body", add an argument named body of type dict - required
  if the body is required, otherwise Optional[dict] = None - whose Field description lists
  the body's fields (mark the required ones). Send it with json=body, or data=body when
  the content_type is not JSON. If the request_body's "type" is "array", the body is a
  list of such objects: use list[dict] instead of dict.
- Never add arguments for API keys, tokens or passwords: the header handles them.
- Return type: -> Union[dict, list, str]
- Docstring: one line, taken from the endpoint's description.

Example, for the endpoint
{"name": "find_pets_by_status", "method": "GET", "path": "/pet/findByStatus",
 "parameters": [{"name": "status", "in": "query", "required": true, "type": "string",
                 "description": "Status to filter by", "enum": ["available", "pending", "sold"]}]}

def find_pets_by_status(
    status: Annotated[Literal["available", "pending", "sold"], Field(description="Status to filter by")],
) -> Union[dict, list, str]:
    \"\"\"Finds pets by status.\"\"\"
    return _request("GET", "/pet/findByStatus", params={"status": status})

and for the endpoint
{"name": "get_pet_by_id", "method": "GET", "path": "/pet/{petId}",
 "parameters": [{"name": "petId", "in": "path", "required": true, "type": "integer",
                 "description": "ID of pet to return"}]}

def get_pet_by_id(
    pet_id: Annotated[int, Field(description="ID of pet to return")],
) -> Union[dict, list, str]:
    \"\"\"Find pet by ID.\"\"\"
    return _request("GET", "/pet/{petId}", path_params={"petId": pet_id})

Return ONLY valid Python code. No markdown fences, no explanations.
"""


MAX_DESCRIPTION = 200      # characters; long descriptions would push a batch past the LLM's token limit


def _short(text: str) -> str:
    """The first sentence of a description, at most MAX_DESCRIPTION characters.
    Some specs (GitHub's) have paragraph-long parameter descriptions; the first
    sentence is what a tool's argument description needs.
    Invisible characters are removed first: they are text an LLM reads but a person
    reviewing the docs cannot see (see security_rules.py)."""
    text = " ".join(strip_invisible(text).split())
    first = text.split(". ")[0]
    first = first if first.endswith(".") or len(first) == len(text) else first + "."
    return first if len(first) <= MAX_DESCRIPTION else first[:MAX_DESCRIPTION - 3].rstrip() + "..."


def codegen_view(endpoint: dict) -> dict:
    """What the LLM is shown for one endpoint. The base URL is left out on purpose:
    _request adds it, so the LLM has no reason to write a URL."""
    view = {key: endpoint[key] for key in ("name", "method", "path") if key in endpoint}
    view["description"] = _short(endpoint.get("description"))
    view["parameters"] = [
        {**p, "description": _short(p["description"])} if p.get("description") else p
        for p in endpoint.get("parameters") or []
    ]
    if endpoint.get("request_body"):
        body = dict(endpoint["request_body"])
        if body.get("field_details"):
            body["field_details"] = [{**f, "description": _short(f["description"])} if f.get("description") else f
                                     for f in body["field_details"]]
        view["request_body"] = body
    return view


def build_codegen_prompt(batch, previous_code=None, errors=None):
    """First attempt: instructions + endpoints.
    Retry: the same, PLUS the code that failed and the errors it got (from the
    security reviewer or the validator, whichever rejected it).
    That extra context is what makes the loop self-correcting - without it,
    the model would be asked the identical question and (at temperature 0)
    would give the identical wrong answer."""
    view = [codegen_view(ep) for ep in batch]
    prompt = f"{CODEGEN_INSTRUCTIONS}\nEndpoints:\n{json.dumps(view, indent=2)}\n"
    if previous_code and errors:
        error_lines = "\n".join(f"- {e}" for e in errors)
        prompt += f"""
Your previous code for these endpoints FAILED validation.

Errors (from the security review or the validator):
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
        # Retry: regenerate only the batches the reviewer or validator complained about,
        # and give each one its own errors plus the code that produced them.
        failing = errors_by_batch(state.get("fix_errors") or [], code_batches)
        print(f"STEP 2 (generate_tool) attempt {attempt} -> regenerating {len(failing)} of "
              f"{len(batches)} batch(es) using the errors they got")
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


def security_review_node(state: ForgeState) -> ForgeState:
    """Hands the generated tools to the security reviewer agent over A2A, BEFORE
    the validator runs them (the validator executes the code; the reviewer only
    reads it). Only the tools are sent: the header above them is the fixed,
    hand-written template."""
    print(f"STEP 3 (security review via A2A) attempt {state['codegen_attempt']}")
    try:
        result = check_via_a2a(SECURITY_AGENT_URL, state["generated_code"], state["endpoints"])
    except AgentUnreachable:
        # Fail closed: without a review, the code must not be run.
        print("ERROR: Could not reach the security agent. Start it first: python security_agent.py")
        result = {"passed": False, "output": "security agent unreachable", "warnings": [],
                  "errors": [{"function": None, "endpoint_index": None, "rule": None,
                              "error": "security agent unreachable"}]}

    state["security_passed"] = result["passed"]
    state["security_output"] = result["output"]
    state["security_warnings"] = result["warnings"]
    if not result["passed"]:
        state["fix_errors"] = result["errors"]
        state["attempt_history"][-1].update({"stage": "security review", "passed": False,
                                             "errors": [e["error"] for e in result["errors"]]})

    print(f"  verdict: passed={result['passed']}")
    for e in result["errors"]:
        print(f"  - {e['error']}")
    for w in result["warnings"]:
        print(f"  warning: {w['warning']}")
    return state


def route_after_review(state: ForgeState) -> str:
    if state["security_passed"]:
        return "validate"
    if state["security_output"] != "security agent unreachable" and state["codegen_attempt"] < state["max_code_attempts"]:
        print("  DECISION: security review failed, sending its findings back to generate_tool")
        return "retry"
    print("  DECISION: security review failed, nothing will be run or packaged")
    return "rejected"


def code_for_validation(state: ForgeState) -> str:
    """The tools exactly as they will run in the packaged server: under the same header."""
    return server_template.render_header(state.get("api_meta") or {}) + "\n\n" + state["generated_code"]


def test_via_a2a_node(state: ForgeState) -> ForgeState:
    """Hands the code, and the endpoints it must implement, to the validator
    agent over the A2A protocol (see a2a_client.py)."""
    print(f"STEP 4 (validate via A2A) attempt {state['codegen_attempt']}")
    try:
        result = check_via_a2a(VALIDATOR_AGENT_URL, code_for_validation(state), state["endpoints"])
        passed, output, errors = result["passed"], result["output"], result["errors"]
    except AgentUnreachable:
        print("ERROR: Could not reach the validator agent.")
        print("  Start it first, in another terminal: python validator_agent.py")
        passed, output = False, "validator unreachable"
        errors = [{"function": None, "endpoint_index": None, "error": output}]

    state["test_passed"] = passed
    state["test_output"] = output
    state["test_errors"] = errors
    state["fix_errors"] = errors

    # Record the outcome on this attempt's history entry.
    state["attempt_history"][-1].update({"stage": "validation", "passed": passed,
                                         "errors": [e["error"] for e in errors]})

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
# Packaging: a runnable MCP server, then a check with a real MCP client
# ----------------------------------------------------------------------

def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def package_server_node(state: ForgeState) -> ForgeState:
    meta = state.get("api_meta") or {}
    output_dir = os.path.join("generated", server_template.folder_name(meta))
    os.makedirs(output_dir, exist_ok=True)

    # The record of both loops: every codegen attempt and every retrieval round.
    warnings = [w["warning"] for w in state.get("security_warnings") or []]
    report = {
        "passed": bool(state.get("security_passed") and state.get("test_passed")),
        "security_review": {"passed": bool(state.get("security_passed")), "summary": state.get("security_output"),
                            "warnings": warnings},
        "attempts": state.get("attempt_history", []),      # the codegen loop
        "retrieval": state.get("rag_trace", []),           # the agentic RAG loop (prose docs only)
        "selection": state.get("selection_note"),
    }

    if not state.get("security_passed"):
        # Fail closed: code the security review rejected is never packaged as a server and
        # never started. It is kept as .txt for inspection, so it cannot be run by accident.
        # Files from an earlier run of the same API are removed: they would not match this report.
        for stale in ("server.py", "claude_desktop_config.json", "README.md", "requirements.txt"):
            if os.path.exists(os.path.join(output_dir, stale)):
                os.remove(os.path.join(output_dir, stale))
        _write(os.path.join(output_dir, "rejected_tools.txt"), state.get("generated_code") or "")
        report_path = os.path.join(output_dir, "validation_report.json")
        _write(report_path, json.dumps(report, indent=2))
        print(f"\nREJECTED by the security review: nothing was packaged or run.")
        print(f"  {state.get('security_output')}")
        print(f"  Report: {report_path}")
        state["output_path"] = None
        return state

    try:
        server_code, tool_info = server_template.package_server(state["generated_code"], state["endpoints"], meta)
    except SyntaxError as e:
        raw_path = os.path.join(output_dir, "raw_output_INVALID.py")
        _write(raw_path, f"# WARNING: syntax error, could not be packaged as a real MCP server.\n# Error: {e}\n\n"
                         + state["generated_code"])
        _write(os.path.join(output_dir, "validation_report.json"), json.dumps(report, indent=2))
        print(f"\nERROR: Generated code has a syntax error and could not be packaged: {e}")
        print(f"  Raw output saved for inspection: {raw_path}")
        state["output_path"] = raw_path
        return state

    server_path = os.path.join(output_dir, "server.py")
    _write(server_path, server_code)
    _write(os.path.join(output_dir, "requirements.txt"), server_template.SERVER_REQUIREMENTS)
    _write(os.path.join(output_dir, "README.md"), server_template.readme(meta, tool_info, server_path, warnings))
    _write(os.path.join(output_dir, "claude_desktop_config.json"),
           json.dumps(server_template.claude_desktop_config(meta, server_path), indent=2))

    # Final check on the actual file: start it and connect an MCP client, as Claude Desktop does.
    client_check = server_template.check_with_mcp_client(server_path, [t["name"] for t in tool_info])
    report["mcp_client_check"] = client_check
    _write(os.path.join(output_dir, "validation_report.json"), json.dumps(report, indent=2))

    status = "PASSED" if state.get("test_passed") else "FAILED (needs manual review)"
    print(f"\nPackaged MCP server -> {output_dir}/")
    print(f"  {len(tool_info)} tool(s): {[t['name'] for t in tool_info]}")
    print(f"  Validation: {status}")
    if warnings:
        print(f"  Security review: {len(warnings)} warning(s) - see README.md")
    if client_check["passed"]:
        print(f"  MCP client check: started server.py over stdio and listed {client_check['tools_listed']} tool(s)")
    else:
        print(f"  MCP client check FAILED: {client_check['error'] or 'missing ' + str(client_check['missing'])}")
    print("  Attempts:")
    for record in state.get("attempt_history", []):
        outcome = ("passed" if record.get("passed") else
                   f"failed {record.get('stage', 'validation')} with {len(record.get('errors', []))} error(s)")
        print(f"    attempt {record['attempt']}: generated batch(es) {record['batches']} -> {outcome}")
    print(f"  Claude Desktop config: {os.path.join(output_dir, 'claude_desktop_config.json')}")
    state["output_path"] = server_path
    state["mcp_client_check"] = client_check
    return state
