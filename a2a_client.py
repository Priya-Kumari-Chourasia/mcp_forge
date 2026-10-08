"""
A2A CLIENT - how the pipeline talks to the validator agent.

Uses the official a2a-sdk client. Two steps, exactly as the protocol intends:

  1. DISCOVERY - fetch the agent's Agent Card from /.well-known/agent-card.json.
     The card says what the agent can do and which URL and protocol binding
     to use. Nothing about the validator is hard-coded here except its address.
  2. SendMessage - send a Message (code as a text part, endpoints as a data
     part) and read the verdict out of the Task's artifacts.
"""

import asyncio

import httpx
from google.protobuf import json_format

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers.proto_helpers import new_data_part, new_message, new_text_part
from a2a.types.a2a_pb2 import Role, SendMessageRequest, TaskState


class ValidatorUnreachable(Exception):
    pass


async def _validate(agent_url: str, code: str, endpoints: list) -> dict:
    async with httpx.AsyncClient(timeout=120) as http:
        # 1. Discovery
        try:
            card = await A2ACardResolver(http, agent_url).get_agent_card()
        except Exception as e:
            raise ValidatorUnreachable(str(e)) from e
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
        raise RuntimeError("the validator agent did not return a task")
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
        # The validator itself failed - that is different from "the code failed validation".
        text = f"validator agent error ({state}): {summary}"
        return {"passed": False, "output": text,
                "errors": [{"function": None, "endpoint_index": None, "error": text}]}

    errors = [
        {"function": e.get("function"),
         # JSON numbers come back as floats; endpoint_index is used as a list index later.
         "endpoint_index": None if e.get("endpoint_index") is None else int(e["endpoint_index"]),
         "error": e["error"]}
        for e in verdict.get("errors", [])
    ]
    return {"passed": bool(verdict.get("passed")), "output": summary, "errors": errors}


def validate_via_a2a(agent_url: str, code: str, endpoints: list) -> dict:
    """Synchronous wrapper so a normal (non-async) LangGraph node can call it."""
    return asyncio.run(_validate(agent_url, code, endpoints))
