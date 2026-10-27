"""
A2A SERVER - the plumbing both agents share (validator_agent.py, security_agent.py).

Each agent is a FastAPI app on which the official a2a-sdk server handles the
A2A protocol (JSON-RPC binding, protocol version 1.0): request parsing, method
routing, error codes, the task lifecycle and the A2A-Version check. An agent
file supplies only two things: its Agent Card and its check function.

Both agents do the same kind of job: take code (a text part) and the endpoints
it implements (a data part), run a check, and answer with a verdict. So one
executor class, CheckExecutor, serves both.

  GET  /.well-known/agent-card.json   the Agent Card
  POST /                              JSON-RPC: SendMessage, GetTask
  GET  /docs                          FastAPI's page describing these endpoints

Note on task state: a Task is COMPLETED whenever the check RAN, even if the
code failed it - "the code is wrong" is the agent's answer, not a failure of
the agent. TASK_STATE_FAILED is reserved for the agent itself breaking.
"""

import asyncio
import socket
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
from a2a.utils.errors import UnsupportedOperationError

# Finished tasks are kept so GetTask can return them, but not forever: a long-running
# agent would otherwise grow without limit. Beyond this many, the oldest are forgotten.
MAX_STORED_TASKS = 500


class BoundedTaskStore(TaskStore):
    """The SDK's in-memory task store, plus a limit: it remembers the newest
    MAX_STORED_TASKS tasks and forgets older ones. (The SDK also has a
    database-backed store; verdicts are only read straight after the check,
    so durable storage would add a database for no benefit.)"""

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


class CheckExecutor(AgentExecutor):
    """Reads the message (code as text, {"endpoints": [...]} as data), runs
    `check(code, endpoints)`, and publishes two artifacts:
      <prefix>-summary   the human-readable result (text)
      <prefix>-verdict   everything else the check returned (data), e.g.
                         {"passed": bool, "errors": [...]}
    Subclasses set `check` and `artifact_prefix`."""

    artifact_prefix = "check"
    log_name = "agent"

    def check(self, code: str, endpoints: list) -> dict:
        raise NotImplementedError

    async def execute(self, context: RequestContext, event_queue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        parts = context.message.parts
        code = "\n".join(get_text_parts(parts))
        endpoints = next((d.get("endpoints", []) for d in get_data_parts(parts) if isinstance(d, dict)), [])
        print(f"[{self.log_name}] task {task.id}: {len(endpoints)} endpoint(s) to check")

        try:
            # A check can take seconds (the validator waits for a child process).
            # Running it on a worker thread keeps the server answering other requests.
            result = await asyncio.to_thread(self.check, code, endpoints)
        except Exception as e:                      # the agent itself broke
            await updater.add_artifact([new_text_part(f"{type(e).__name__}: {e}")], name=f"{self.artifact_prefix}-error")
            await updater.failed()
            return

        verdict = {key: value for key, value in result.items() if key != "output"}
        await updater.add_artifact([new_text_part(result["output"])], name=f"{self.artifact_prefix}-summary")
        await updater.add_artifact([new_data_part(verdict)], name=f"{self.artifact_prefix}-verdict")
        await updater.complete()
        print(f"[{self.log_name}] task {task.id} -> passed={result['passed']}")

    async def cancel(self, context: RequestContext, event_queue) -> None:
        # A check takes seconds and has nothing to clean up, so cancelling is not offered.
        raise UnsupportedOperationError()


def create_app(card, executor: AgentExecutor, title: str, description: str) -> FastAPI:
    """An agent as a FastAPI app, with the A2A routes mounted by the SDK."""
    handler = DefaultRequestHandler(agent_executor=executor, task_store=BoundedTaskStore(), agent_card=card)
    app = FastAPI(title=title, version=card.version, description=description)
    add_a2a_routes_to_fastapi(app, agent_card_routes=create_agent_card_routes(card),
                              jsonrpc_routes=create_jsonrpc_routes(handler, rpc_url="/"))
    return app


def serve_in_thread(make_app, host: str = "127.0.0.1", port: int = 0):
    """Starts an agent in a background thread of THIS process and returns
    (base_url, server). make_app(public_url) builds the FastAPI app. The web UI
    uses this to run as one command, and the tests use it with port=0 (any free
    port). Binding the socket first means the port is known before the Agent
    Card is built. Stop it with server.should_exit = True."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    base_url = f"http://{host}:{sock.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(make_app(base_url + "/"), log_level="warning"))
    threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True).start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError(f"the agent at {base_url} did not start within 15 seconds")
        time.sleep(0.05)
    return base_url, server


def run_standalone(make_app, name: str, host: str, port: int, public_url: str):
    """`python <agent>.py`: serve the agent in this process until stopped."""
    print(f"{name} listening on http://{host}:{port}")
    print(f"  Agent Card: {public_url.rstrip('/')}/.well-known/agent-card.json")
    print(f"  API docs:   {public_url.rstrip('/')}/docs")
    uvicorn.run(make_app(public_url), host=host, port=port, log_level="warning")
