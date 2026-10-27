"""
VALIDATOR AGENT - a separate agent the pipeline talks to over the A2A protocol.

It runs as its own service: FastAPI + uvicorn, with the A2A protocol handled
by the official a2a-sdk (see a2a_server.py, shared with the security agent):

  GET  /.well-known/agent-card.json   the Agent Card: who this agent is, what
                                      skill it offers, and where/how to call it
  POST /                              JSON-RPC 2.0 endpoint (send the header
                                      "A2A-Version: 1.0"; the SDK client does)
         SendMessage  -> validates the code in the message, returns a Task
         GetTask      -> returns a previously completed Task by id
  GET  /docs                          FastAPI's page documenting the endpoints

A caller needs to know nothing about this agent in advance except its URL:
it fetches the Agent Card (discovery), then sends a Message. The Message has
two parts - the generated code (text) and the endpoints it must implement
(data). The reply is a Task whose artifacts hold the verdict:
{"passed": bool, "errors": [...]}.

This agent RUNS the code it is sent, so the pipeline sends code here only
after the security reviewer (security_agent.py) has passed it.

Two layers of checks:
  1. STATIC      - is it valid Python, with functions, and no duplicate names?
  2. BEHAVIOURAL - called through a real MCP server, does each tool accept its
                   arguments, send the documented method, path, required query
                   parameters and body, and return a result the MCP client
                   accepts? Done in a child process with HTTP mocked (see
                   behaviour_check.py), so no real API is ever touched.

Run it:  python validator_agent.py      (listens on http://127.0.0.1:8000)
"""

import ast
import json
import os
import subprocess
import sys
import tempfile

import a2a_server
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill

BEHAVIOUR_CHECK_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behaviour_check.py")
BEHAVIOUR_TIMEOUT_SECONDS = 30


# ----------------------------------------------------------------------
# A2A layer
# ----------------------------------------------------------------------

def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="MCP-Forge Validator",
        description="Validates generated MCP tool code: static checks, then a behavioural check "
                    "that calls every tool through an MCP server against mocked HTTP.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(url=base_url, protocol_binding="JSONRPC", protocol_version="1.0"),
        ],
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/x-python", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        skills=[AgentSkill(
            id="validate_mcp_tools",
            name="Validate MCP tools",
            description="Send Python tool functions as a text part and the endpoints they must implement "
                        "as a data part ({\"endpoints\": [...]}). Returns a pass/fail verdict with one "
                        "structured error per problem.",
            tags=["validation", "mcp", "code"],
        )],
    )


class ValidatorExecutor(a2a_server.CheckExecutor):
    """The agent's own logic, in the shape the A2A SDK expects (see a2a_server.CheckExecutor)."""
    artifact_prefix = "validation"
    log_name = "validator"

    def check(self, code, endpoints):
        return validate_code(code, endpoints)


def create_app(public_url: str):
    """The validator as a FastAPI app. public_url is the address callers use to
    reach it; it goes into the Agent Card, so it must be the real one (for
    example http://validator:8000/ inside Docker, not localhost)."""
    return a2a_server.create_app(build_agent_card(public_url), ValidatorExecutor(), title="MCP-Forge Validator",
                                 description="An A2A agent that validates generated MCP tool code.")


def serve_in_thread(host: str = "127.0.0.1", port: int = 0):
    """Starts the validator on a background thread; returns (base_url, server)."""
    return a2a_server.serve_in_thread(create_app, host, port)


# ----------------------------------------------------------------------
# The validation skill itself
# ----------------------------------------------------------------------

def _error(text, function=None, endpoint_index=None):
    return {"function": function, "endpoint_index": endpoint_index, "error": text}


def _failed(errors):
    return {"passed": False, "errors": errors, "output": "; ".join(e["error"] for e in errors)}


def validate_code(code: str, endpoints: list = None) -> dict:
    """Returns {"passed": bool, "output": summary text, "errors": [structured errors]}."""
    # ---------- layer 1: static checks ----------
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return _failed([_error(f"SyntaxError: {e}")])

    # The tools are the public top-level functions. Names starting with _ are
    # helpers, such as _request() from the server template, not tools.
    function_names = [n.name for n in tree.body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")]
    if not function_names:
        return _failed([_error("No tool functions were defined in the generated code.")])

    # Batches are generated by independent LLM calls, so two batches can pick
    # the same function name. Python would silently keep only the last one.
    duplicates = sorted({name for name in function_names if function_names.count(name) > 1})
    if duplicates:
        return _failed([
            _error(f"'{name}' is defined more than once; every function needs a unique name", function=name)
            for name in duplicates
        ])

    # ---------- layer 2: behavioural checks ----------
    if endpoints:
        errors = run_behaviour_check(code, endpoints)
        if errors:
            return _failed(errors)
        return {"passed": True, "errors": [],
                "output": f"{len(function_names)} tool(s) passed: each was called through an MCP server and sent "
                          f"the documented method, path, required parameters and body: {function_names}"}

    return {"passed": True, "errors": [],
            "output": f"{len(function_names)} function(s) defined (static checks only): {function_names}"}


def _environment_without_secrets() -> dict:
    """The child process gets the environment MINUS anything that looks like a
    credential, so generated code cannot read the API key."""
    hidden = ("KEY", "TOKEN", "SECRET", "PASSWORD")
    return {name: value for name, value in os.environ.items()
            if not any(word in name.upper() for word in hidden)}


def run_behaviour_check(code: str, endpoints: list) -> list:
    """Runs behaviour_check.py on the code in a CHILD PROCESS with a timeout.

    A child process means LLM-written code never runs inside this server,
    and a tool that hangs gets killed instead of freezing the validator.
    (This is process isolation, not a security sandbox.)
    """
    with tempfile.TemporaryDirectory() as tmp:
        code_path = os.path.join(tmp, "generated_tools.py")
        endpoints_path = os.path.join(tmp, "endpoints.json")
        result_path = os.path.join(tmp, "result.json")
        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)
        with open(endpoints_path, "w", encoding="utf-8") as f:
            json.dump(endpoints, f)

        try:
            proc = subprocess.run(
                [sys.executable, BEHAVIOUR_CHECK_SCRIPT, code_path, endpoints_path, result_path],
                cwd=tmp, capture_output=True, text=True, timeout=BEHAVIOUR_TIMEOUT_SECONDS,
                env=_environment_without_secrets(),
            )
        except subprocess.TimeoutExpired:
            return [_error(f"behaviour check timed out after {BEHAVIOUR_TIMEOUT_SECONDS}s "
                           f"(a tool is probably blocking or looping)")]

        if not os.path.exists(result_path):
            return [_error(f"behaviour check crashed: {proc.stderr.strip()[-500:]}")]
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)["errors"]


if __name__ == "__main__":
    # VALIDATOR_HOST=0.0.0.0 lets other machines (or containers) reach it; the default
    # only accepts connections from this machine. VALIDATOR_PUBLIC_URL is the address
    # written into the Agent Card, for when callers reach it under another name.
    port = int(os.getenv("VALIDATOR_PORT", "8000"))
    a2a_server.run_standalone(create_app, "Validator agent", os.getenv("VALIDATOR_HOST", "127.0.0.1"), port,
                              os.getenv("VALIDATOR_PUBLIC_URL", f"http://localhost:{port}/"))
