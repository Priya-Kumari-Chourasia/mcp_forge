"""
VALIDATOR AGENT - a separate agent the pipeline talks to over the A2A protocol.

It runs in its own process and implements the A2A (Agent2Agent) JSON-RPC
binding, protocol version 1.0, on top of Flask:

  GET  /.well-known/agent-card.json   the Agent Card: who this agent is, what
                                      skill it offers, and where/how to call it
  POST /                              JSON-RPC 2.0 endpoint
         SendMessage  -> validates the code in the message, returns a Task
         GetTask      -> returns a previously completed Task by id

A caller needs to know nothing about this agent in advance except its URL:
it fetches the Agent Card (discovery), then sends a Message. The Message has
two parts - the generated code (text) and the endpoints it must implement
(data). The reply is a Task whose artifacts hold the verdict.

Note on task state: a Task is COMPLETED whenever validation RAN, even if the
code failed it - "the code is wrong" is the agent's answer, not a failure of
the agent. The verdict lives in the artifact: {"passed": bool, "errors": [...]}.
TASK_STATE_FAILED is reserved for the validator itself breaking.

Not implemented: streaming, push notifications, task cancellation, auth.

Two layers of checks:
  1. STATIC      - is it valid Python, with functions, and no duplicate names?
  2. BEHAVIOURAL - does each tool, when called, send the right HTTP method to
                   the right URL? Done in a child process with HTTP mocked
                   (see behaviour_check.py), so no real API is ever touched.
"""

import ast
import json
import os
import subprocess
import sys
import tempfile
import uuid

from flask import Flask, request, jsonify
from google.protobuf import json_format
from google.protobuf.timestamp_pb2 import Timestamp

from a2a.helpers.proto_helpers import new_data_artifact, new_text_artifact
from a2a.types.a2a_pb2 import (
    AgentCapabilities, AgentCard, AgentInterface, AgentSkill,
    SendMessageRequest, SendMessageResponse, Task, TaskState, TaskStatus,
)

app = Flask(__name__)

BEHAVIOUR_CHECK_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behaviour_check.py")
BEHAVIOUR_TIMEOUT_SECONDS = 30

TASKS = {}      # task id -> Task, so GetTask can return finished tasks


# ----------------------------------------------------------------------
# A2A layer
# ----------------------------------------------------------------------

def build_agent_card(base_url: str) -> AgentCard:
    return AgentCard(
        name="MCP-Forge Validator",
        description="Validates generated MCP tool code: static checks, then a behavioural check "
                    "that every tool sends the documented HTTP method and URL.",
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


@app.get("/.well-known/agent-card.json")
def agent_card():
    return jsonify(json_format.MessageToDict(build_agent_card(request.host_url)))


def _rpc_result(rpc_id, message):
    return jsonify({"jsonrpc": "2.0", "id": rpc_id, "result": json_format.MessageToDict(message)})


def _rpc_error(rpc_id, code, text):
    return jsonify({"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": text}})


@app.post("/")
def json_rpc():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
        return _rpc_error(None, -32600, "Invalid JSON-RPC request")
    rpc_id, method, params = body.get("id"), body.get("method"), body.get("params") or {}

    if method == "SendMessage":
        try:
            send_request = json_format.ParseDict(params, SendMessageRequest())
        except json_format.ParseError as e:
            return _rpc_error(rpc_id, -32602, f"Invalid params: {e}")
        return _rpc_result(rpc_id, SendMessageResponse(task=handle_message(send_request.message)))

    if method == "GetTask":
        task = TASKS.get(params.get("id"))
        if task is None:
            return _rpc_error(rpc_id, -32001, "Task not found")
        return _rpc_result(rpc_id, task)

    return _rpc_error(rpc_id, -32601, f"Method not found: {method}")


def handle_message(message) -> Task:
    """Runs the validation skill on an incoming A2A Message and wraps the result in a Task."""
    code, endpoints = "", []
    for part in message.parts:
        if part.HasField("text"):
            code = part.text
        elif part.HasField("data"):
            endpoints = json_format.MessageToDict(part.data).get("endpoints", [])

    task_id = str(uuid.uuid4())
    print(f"[validator] task {task_id}: {len(endpoints)} endpoint(s) to check")
    now = Timestamp()
    now.GetCurrentTime()
    try:
        result = validate_code(code, endpoints)
        state = TaskState.TASK_STATE_COMPLETED
        artifacts = [
            new_text_artifact("validation-summary", result["output"]),
            new_data_artifact("validation-verdict", {"passed": result["passed"], "errors": result["errors"]}),
        ]
        print(f"[validator] task {task_id} -> passed={result['passed']}")
    except Exception as e:                      # the validator itself broke
        state = TaskState.TASK_STATE_FAILED
        artifacts = [new_text_artifact("validator-error", f"{type(e).__name__}: {e}")]

    task = Task(
        id=task_id,
        context_id=message.context_id or str(uuid.uuid4()),
        status=TaskStatus(state=state, timestamp=now),
        artifacts=artifacts,
        history=[message],
    )
    TASKS[task_id] = task
    return task


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

    function_names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not function_names:
        return _failed([_error("No functions were defined in the generated code.")])

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
                        "output": f"{len(function_names)} tool(s) each called the correct method and URL "
                    f"and registered on an MCP server: {function_names}"}

    return {"passed": True, "errors": [],
            "output": f"{len(function_names)} function(s) defined (static checks only): {function_names}"}


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
            )
        except subprocess.TimeoutExpired:
            return [_error(f"behaviour check timed out after {BEHAVIOUR_TIMEOUT_SECONDS}s "
                           f"(a tool is probably blocking or looping)")]

        if not os.path.exists(result_path):
            return [_error(f"behaviour check crashed: {proc.stderr.strip()[-500:]}")]
        with open(result_path, encoding="utf-8") as f:
            return json.load(f)["errors"]


if __name__ == "__main__":
    print("Validator agent listening on http://localhost:8000")
    print("  Agent Card: http://localhost:8000/.well-known/agent-card.json")
    app.run(port=8000)
