"""
NODES = the individual workers on the assembly line.

A "node" in LangGraph is just a plain Python function with one rule:
it takes the state (the clipboard) in, and returns the state (updated)
out. That's it. LangGraph handles calling them in the right order and
passing the clipboard between them — you just write normal Python
functions.

We have two nodes right now:
  1. parse_docs_node    -> reads raw docs, extracts structured info
  2. generate_tool_node -> reads that structured info, writes the code
"""

import json
import requests
from state import ForgeState
from llm import call_llm
from spec_parser import is_openapi_spec, extract_endpoints_from_openapi
import os, re
import ast  # add this import at the top, alongside your existing imports

REQUIRED_FIELDS = ["method", "path", "description"]
TEST_RUNNER_URL = "http://localhost:8000/tasks/send"
MAX_CODE_ATTEMPTS = 2


def orchestrator_node(state: ForgeState) -> ForgeState:
    """
    READ the raw docs and DECIDE: is this a simple or complex API?

    If it's a structured OpenAPI spec, count endpoints directly -
    no LLM needed, and no risk of the request being too large for
    the API to even accept.
    """
    spec = is_openapi_spec(state["raw_docs"])
    if spec is not None:
        endpoints = extract_endpoints_from_openapi(spec)
        plan = "complex" if len(endpoints) > 10 else "simple"
        state["orchestration_plan"] = plan
        print(f"ORCHESTRATOR -> plan: {plan} ({len(endpoints)} endpoint(s), no LLM needed)")
        return state

    # Prose docs: only send a reasonable-sized SAMPLE, not the whole thing.
    # Deciding simple-vs-complex doesn't require reading every word, and
    # this protects against ever hitting the API's request-size limit
    # again on a very long prose document.
    sample = state["raw_docs"][:3000]
    prompt = f"""Analyze these API docs and decide: is this a SIMPLE or COMPLEX API?

SIMPLE means:
- Few endpoints (1-10)
- Clear, straightforward documentation
- No auth OR simple API key auth
- Consistent response formats
- No rate limiting complexity

COMPLEX means:
- Many endpoints (10+)
- Unclear or inconsistent docs
- OAuth, custom auth, or multiple auth schemes
- Variable response formats
- Rate limiting, pagination, or edge cases

Respond with ONLY one word: simple OR complex

API docs (sample):
{sample}"""

    response = call_llm(prompt)
    plan = response.strip().lower()
    if plan not in ["simple", "complex"]:
        print(f"[WARNING] Unexpected plan: {plan}, defaulting to complex")
        plan = "complex"

    state["orchestration_plan"] = plan
    print(f"ORCHESTRATOR -> plan: {plan}")
    return state


def route_after_orchestration(state: ForgeState) -> str:
    """
    After the orchestrator decides, route to the right path:
    - "simple" → simple_path (fast, fewer retries expected)
    - "complex" → complex_path (thorough, more validation)
    """
    plan = state["orchestration_plan"]
    print(f"  ROUTING: {plan} path")
    
    if plan == "simple":
        return "simple_path"
    else:  # complex or anything weird
        return "complex_path"
    
    
def parse_docs_node(state: ForgeState) -> ForgeState:
    attempt = state.get("retry_count", 0) + 1

    spec = is_openapi_spec(state["raw_docs"])
    if spec is not None:
        endpoints = extract_endpoints_from_openapi(spec)
        state["endpoints"] = endpoints
        state["retry_count"] = attempt
        print(f"STEP 1 (parse_docs) -> OpenAPI spec detected, {len(endpoints)} endpoint(s), no LLM needed")
        return state

    sample = state["raw_docs"][:6000]
    prompt = f"""Attempt {attempt}: Extract the endpoint info as a JSON ARRAY.
Each item MUST have method, path, description at the top level.

API docs (sample):
{sample}"""
    response = call_llm(prompt)
    parsed = json.loads(response)
    if isinstance(parsed, dict):
        parsed = [parsed]
    state["endpoints"] = parsed
    state["retry_count"] = attempt
    print(f"STEP 1 (parse_docs) attempt {attempt} -> {len(parsed)} endpoint(s) found (via LLM)")
    return state


def grade_extraction_node(state: ForgeState) -> ForgeState:
    endpoints = state["endpoints"]
    all_missing = []
    for i, ep in enumerate(endpoints):
        missing = [f for f in REQUIRED_FIELDS if not ep.get(f)]
        if missing:
            all_missing.append(f"endpoint {i} ({ep.get('path', '?')}): missing {missing}")
    state["missing_fields"] = all_missing
    print(f"STEP 1b (grade) -> issues: {all_missing}")
    return state

def route_after_grading(state: ForgeState) -> str:
    if not state["missing_fields"]:
        print("  DECISION: extraction complete, moving to generate_tool")
        return "continue"
    if state["retry_count"] >= 2:
        print(f"  DECISION: giving up after {state['retry_count']} attempts, missing: {state['missing_fields']}")
        return "continue"
    print("  DECISION: incomplete, retrying parse_docs")
    return "retry"

BATCH_SIZE = 15  # endpoints per LLM call - keeps each request well within context limits

def generate_one_batch(batch, batch_num, total_batches, max_batch_retries=2):
    """Generate code for ONE batch, and if it comes back with a syntax
    error, retry just this batch immediately - instead of letting one
    bad batch force a full regeneration of all 22."""
    response = None
    for local_attempt in range(1, max_batch_retries + 1):
        prompt = f"""Write Python functions (MCP tools) — ONE function per endpoint below.

IMPORTANT: Use snake_case for every function name (e.g., get_boards_by_id_board,
NOT getBoardsByIdBoard). This must be consistent across all functions, since
this file will be combined with output from other independent batches that
don't see each other's naming choices.

IMPORTANT: Each function must be completely SELF-CONTAINED. Do NOT define
any shared/helper functions (like a private _request or _build_query
helper) that other functions might rely on — every function must build
its own request and call requests.<method>(...) directly inside itself.
This code will be combined with output from other independent batches,
so nothing can depend on a helper defined elsewhere.

Return ONLY valid Python code. No markdown fences, no explanations.

Endpoints:
{json.dumps(batch, indent=2)}"""
        response = call_llm(prompt)
        try:
            ast.parse(response)
            return response
        except SyntaxError:
            print(f"  batch {batch_num}/{total_batches} attempt {local_attempt} -> SyntaxError, retrying just this batch")
    print(f"  batch {batch_num}/{total_batches} -> still broken after {max_batch_retries} local attempts, using it anyway")
    return response


def generate_tool_node(state: ForgeState) -> ForgeState:
    attempt = state.get("codegen_attempt", 0) + 1
    endpoints = state["endpoints"]

    batches = [endpoints[i:i + BATCH_SIZE] for i in range(0, len(endpoints), BATCH_SIZE)]
    print(f"STEP 2 (generate_tool) attempt {attempt} -> {len(endpoints)} endpoint(s) in {len(batches)} batch(es)")

    all_code_parts = []
    for i, batch in enumerate(batches):
        code = generate_one_batch(batch, i + 1, len(batches))
        all_code_parts.append(code)
        print(f"  batch {i+1}/{len(batches)} done ({len(batch)} endpoint(s))")

    state["generated_code"] = "\n\n".join(all_code_parts)
    state["codegen_attempt"] = attempt
    print(f"STEP 2 (generate_tool) attempt {attempt} -> code generated across {len(batches)} batch(es)")
    return state
def test_via_a2a_node(state: ForgeState) -> ForgeState:
    task = {"id": f"test-{state['codegen_attempt']}", "message": {"parts": [{"type": "text", "text": state["generated_code"]}]}}
    try:
        response = requests.post(TEST_RUNNER_URL, json=task, timeout=10)
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.ConnectionError:
        print("ERROR: Could not reach the test-runner service.")
        print("  Start it first, in another terminal: python test_runner_service.py")
        state["test_passed"] = False
        state["test_output"] = "test-runner unreachable"
        return state
    except requests.exceptions.Timeout:
        print("ERROR: Test-runner took too long to respond.")
        state["test_passed"] = False
        state["test_output"] = "test-runner timed out"
        return state

    passed = result["status"]["state"] == "completed"
    output = result["artifacts"][0]["parts"][0]["text"]
    state["test_passed"] = passed
    state["test_output"] = output
    print(f"STEP 3 (test via A2A) -> passed={passed}, output={output}")
    return state
"""
def save_output_node(state: ForgeState) -> ForgeState:
    first = state["endpoints"][0] if state.get("endpoints") else {}
    safe_name = re.sub(r'[^a-zA-Z0-9]+', '-', first.get("base_url", "api")).strip('-').lower() or "generated-api"
    output_dir = os.path.join("generated", safe_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "mcp_server.py")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(state["generated_code"])
    status = "PASSED" if state.get("test_passed") else "FAILED (needs manual review)"
    print(f"\nSaved to: {output_path} | Status: {status}")
    state["output_path"] = output_path
    return state
"""
def route_after_test(state: ForgeState) -> str:
    if not state["test_passed"] and state["codegen_attempt"] < MAX_CODE_ATTEMPTS:
        print("  DECISION: test failed, retrying generate_tool")
        return "retry"
    print(f"  DECISION: {'passed' if state['test_passed'] else 'giving up'}, finishing")
    return "done"



def package_as_mcp_server(generated_code: str, server_name: str):
    """Take raw function definitions and wrap them as a real, runnable
    MCP server: adds the MCPServer import/instance, a @server.tool()
    decorator above each function (found via ast, not string guessing),
    and a __main__ block that actually starts it."""
    tree = ast.parse(generated_code)
    lines = generated_code.split("\n")

    tool_info = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node) or "No description"
            tool_info.append({"name": node.name, "description": docstring})

    # Insert decorators bottom-to-top so earlier line numbers don't shift
    func_start_lines = sorted(
        (n.lineno for n in tree.body if isinstance(n, ast.FunctionDef)), reverse=True
    )
    for lineno in func_start_lines:
        lines.insert(lineno - 1, "@server.tool()")

    body = "\n".join(lines)
    header = f'"""\nAuto-generated MCP server: {server_name}\nGenerated by MCP-Forge.\n"""\nfrom mcp.server.mcpserver import MCPServer\n\nserver = MCPServer("{server_name}")\n\n'
    footer = '\n\nif __name__ == "__main__":\n    server.run()\n'
    return header + body + footer, tool_info


def package_server_node(state: ForgeState) -> ForgeState:
    first = state["endpoints"][0] if state.get("endpoints") else {}
    safe_name = re.sub(r'[^a-zA-Z0-9]+', '-', first.get("base_url", "api")).strip('-').lower() or "generated-api"
    output_dir = os.path.join("generated", safe_name)
    os.makedirs(output_dir, exist_ok=True)

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
        f.write("mcp\nrequests\n")
    readme = f"# {safe_name} - MCP Server\n\nAuto-generated by MCP-Forge.\n\n## Tools\n\n"
    for t in tool_info:
        readme += f"- **{t['name']}**: {t['description']}\n"
    readme += f'\n## Add to Claude Desktop\n\n```json\n{{\n  "mcpServers": {{\n    "{safe_name}": {{\n      "command": "python",\n      "args": ["{os.path.abspath(server_path)}"]\n    }}\n  }}\n}}\n```\n'
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)

    status = "PASSED" if state.get("test_passed") else "FAILED (needs manual review)"
    print(f"\nPackaged MCP server -> {output_dir}/")
    print(f"  {len(tool_info)} tool(s): {[t['name'] for t in tool_info]}")
    print(f"  Status: {status}")
    state["output_path"] = server_path
    return state