"""
A2A CLIENT - how the pipeline talks to its two agents: the security reviewer
and the validator.

Uses the official a2a-sdk client. Two steps, exactly as the protocol intends:

  1. DISCOVERY - fetch the agent's Agent Card from /.well-known/agent-card.json.
     The card says what the agent can do and which URL and protocol binding
     to use. Nothing about an agent is hard-coded here except its address.
  2. SendMessage - send a Message (code as a text part, endpoints as a data
     part) and read the verdict out of the Task's artifacts.

Both agents answer in the same shape, so one function calls either.
"""

import asyncio

import httpx
from google.protobuf import json_format

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import new_data_part, new_message, new_text_part
from a2a.types.a2a_pb2 import Role, SendMessageRequest, TaskState


class AgentUnreachable(Exception):
    pass


def _finding(entry: dict, key: str) -> dict:
    """One error or warning as the pipeline uses it. JSON numbers come back as
    floats; endpoint_index is used as a list index later."""
    index = entry.get("endpoint_index")
    return {"function": entry.get("function"), "endpoint_index": None if index is None else int(index),
            "rule": entry.get("rule"), key: entry[key]}


async def _check(agent_url: str, code: str, endpoints: list) -> dict:
    async with httpx.AsyncClient(timeout=120) as http:
        # 1. Discovery
        try:
            card = await A2ACardResolver(http, agent_url).get_agent_card()
        except Exception as e:
            raise AgentUnreachable(str(e)) from e
        skills = [skill.id for skill in card.skills]
        print(f"  A2A: discovered agent '{card.name}' (skills: {skills})")

        # 2. SendMessage
        client = ClientFactory(ClientConfig(streaming=False, httpx_client=http)).create(card)
        message = new_message(
            [new_text_part(code, media_type="text/x-python"), new_data_part({"endpoints": endpoints})],
            role=Role.ROLE_USER,
        )
        task = None
        async for event in client.send_message(SendMessageRequest(message=message)):
            if event.HasField("task"):
                task = event.task

    if task is None:
        raise RuntimeError(f"the agent '{card.name}' did not return a task")
    state = TaskState.Name(task.status.state)
    print(f"  A2A: task {task.id} -> {state}")

    summary, verdict = "", None
    for artifact in task.artifacts:
        for part in artifact.parts:
            if part.HasField("text"):
                summary = part.text
            elif part.HasField("data"):
                verdict = json_format.MessageToDict(part.data)

    if task.status.state != TaskState.TASK_STATE_COMPLETED or verdict is None:
        # The agent itself failed - that is different from "the code failed the check".
        text = f"{card.name} error ({state}): {summary}"
        return {"passed": False, "output": text, "warnings": [],
                "errors": [{"function": None, "endpoint_index": None, "rule": None, "error": text}]}

    return {"passed": bool(verdict.get("passed")), "output": summary,
            "errors": [_finding(e, "error") for e in verdict.get("errors", [])],
            "warnings": [_finding(w, "warning") for w in verdict.get("warnings", [])]}


def check_via_a2a(agent_url: str, code: str, endpoints: list) -> dict:
    """Asks the agent at agent_url to check the code. Returns
    {"passed", "output", "errors", "warnings"}. Synchronous, so a normal
    (non-async) LangGraph node can call it."""
    return asyncio.run(_check(agent_url, code, endpoints))
