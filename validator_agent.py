"""
VALIDATOR AGENT - a separate agent the pipeline talks to over the A2A protocol.

It runs as its own service: FastAPI + uvicorn, with the A2A protocol handled
by the official a2a-sdk (JSON-RPC binding, protocol version 1.0):

  GET  /.well-known/agent-card.json   the Agent Card: who this agent is, what
                                      skill it offers, and where/how to call it
  POST /                              JSON-RPC 2.0 endpoint (send the header
                                      "A2A-Version: 1.0"; the SDK client does)
         SendMessage  -> validates the code in the message, returns a Task
         GetTask      -> returns a previously completed Task by id
  GET  /docs                          FastAPI's page documenting the endpoints

Why the SDK's server and not hand-written JSON-RPC: the protocol (request
parsing, method routing, error codes, task lifecycle, version check) is the
SDK's job, kept in step with the A2A spec by its maintainers. This file only
holds the agent's own logic - the ValidatorExecutor below.

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
  2. BEHAVIOURAL - called through a real MCP server, does each tool accept its
                   arguments, send the documented method, path, required query
                   parameters and body, and return a result the MCP client
                   accepts? Done in a child process with HTTP mocked (see
                   behaviour_check.py), so no real API is ever touched.

Run it:  python validator_agent.py      (listens on http://127.0.0.1:8000)
"""

import ast
import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import OrderedDict

import uvicorn
from fastapi import FastAPI

from a2a.helpers.proto_helpers import (
    get_data_parts, get_text_parts, new_data_part, new_task_from_user_message, new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import add_a2a_routes_to_fastapi, create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskStore, TaskUpdater
from a2a.types.a2a_pb2 import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.errors import UnsupportedOperationError

BEHAVIOUR_CHECK_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behaviour_check.py")
BEHAVIOUR_TIMEOUT_SECONDS = 30

# Finished tasks are kept so GetTask can return them, but not forever: a long-running
# validator would otherwise grow without limit. Beyond this many, the oldest are forgotten.
MAX_STORED_TASKS = 500


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


class BoundedTaskStore(TaskStore):
    """The SDK's in-memory task store, plus a limit: it remembers the newest
    MAX_STORED_TASKS tasks and forgets older ones. (The SDK also has a
    database-backed store; a validator's verdicts are only read straight after
    validation, so durable storage would add a database for no benefit.)"""

    def __init__(self, max_tasks: int = MAX_STORED_TASKS):
        self.max_tasks = max_tasks
        self._store = InMemoryTaskStore()
        self._order = OrderedDict()          # task id -> the call context it was saved with, oldest first

    async def save(self, task, context):
        await self._store.save(task, context)
        self._order[task.id] = context
        self._order.move_to_end(task.id)
        while len(self._order) > self.max_tasks:
            old_id, old_context = self._order.popitem(last=False)
            await self._store.delete(old_id, old_context)

    async def get(self, task_id, context):
        return await self._store.get(task_id, context)

    async def list(self, params, context):
        return await self._store.list(params, context)

    async def delete(self, task_id, context):
        self._order.pop(task_id, None)
        await self._store.delete(task_id, context)


class ValidatorExecutor(AgentExecutor):
    """The agent's own logic, in the shape the A2A SDK expects: read the message,
    validate, publish the verdict as artifacts, and close the task."""

    async def execute(self, context: RequestContext, event_queue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        parts = context.message.parts
        code = "\n".join(get_text_parts(parts))
        endpoints = next((d.get("endpoints", []) for d in get_data_parts(parts) if isinstance(d, dict)), [])
        print(f"[validator] task {task.id}: {len(endpoints)} endpoint(s) to check")

        try:
            # validate_code waits up to 30 s for a child process. Run it on a worker
            # thread so the server keeps answering other requests meanwhile.
            result = await asyncio.to_thread(validate_code, code, endpoints)
        except Exception as e:                      # the validator itself broke
            await updater.add_artifact([new_text_part(f"{type(e).__name__}: {e}")], name="validator-error")
            await updater.failed()
            return

        await updater.add_artifact([new_text_part(result["output"])], name="validation-summary")
        await updater.add_artifact([new_data_part({"passed": result["passed"], "errors": result["errors"]})],
                                   name="validation-verdict")
        await updater.complete()
        print(f"[validator] task {task.id} -> passed={result['passed']}")

    async def cancel(self, context: RequestContext, event_queue) -> None:
        # A validation takes seconds and has nothing to clean up, so cancelling is not offered.
        raise UnsupportedOperationError()


def create_app(public_url: str) -> FastAPI:
    """The validator as a FastAPI app. public_url is the address callers use to
    reach it; it goes into the Agent Card, so it must be the real one (for
    example http://validator:8000/ inside Docker, not localhost)."""
    card = build_agent_card(public_url)
    handler = DefaultRequestHandler(agent_executor=ValidatorExecutor(), task_store=BoundedTaskStore(),
                                    agent_card=card)
    app = FastAPI(title="MCP-Forge Validator", version="1.0.0",
                  description="An A2A agent that validates generated MCP tool code.")
    add_a2a_routes_to_fastapi(app, agent_card_routes=create_agent_card_routes(card),
                              jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"))
    return app


def serve_in_thread(host: str = "127.0.0.1", port: int = 0):
    """Starts the validator in a background thread of THIS process and returns
    (base_url, server). The web UI uses it to run as one command, and the tests
    use it with port=0 (any free port). Binding the socket first means the port
    is known before the Agent Card is built. Stop it with server.should_exit = True."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    base_url = f"http://{host}:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(create_app(base_url + "/"), log_level="warning"))
    threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True).start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("the validator agent did not start within 15 seconds")
        time.sleep(0.05)
    return base_url, server


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
    host = os.getenv("VALIDATOR_HOST", "127.0.0.1")
    port = int(os.getenv("VALIDATOR_PORT", "8000"))
    public_url = os.getenv("VALIDATOR_PUBLIC_URL", f"http://localhost:{port}/")
    print(f"Validator agent listening on http://{host}:{port}")
    print(f"  Agent Card: {public_url.rstrip('/')}/.well-known/agent-card.json")
    print(f"  API docs:   {public_url.rstrip('/')}/docs")
    uvicorn.run(create_app(public_url), host=host, port=port, log_level="warning")
