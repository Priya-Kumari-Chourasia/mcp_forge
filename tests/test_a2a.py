"""The validator must behave as an A2A agent: discoverable by its Agent Card,
callable with SendMessage, by the official a2a-sdk client."""
import asyncio
import threading
import time
import uuid

import requests

from a2a_client import validate_via_a2a
from server_template import render_header

GOOD = render_header({"title": "", "base_url": "https://catfact.ninja", "auth": []}) + '''

def get_fact() -> Union[dict, list, str]:
    """Returns a random cat fact."""
    return _request("GET", "/fact")
'''
ENDPOINTS = [{"method": "GET", "path": "/fact", "base_url": "https://catfact.ninja", "description": "fact"}]


def rpc(url, method, params, version="1.0"):
    """A raw JSON-RPC call. A2A 1.0 requires the A2A-Version header (the SDK client sends it)."""
    headers = {"A2A-Version": version} if version else {}
    return requests.post(url, headers=headers,
                         json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params}).json()


def send_message(url, code):
    return rpc(url, "SendMessage", {"message": {
        "messageId": str(uuid.uuid4()), "role": "ROLE_USER",
        "parts": [{"text": code}, {"data": {"endpoints": ENDPOINTS}}],
    }})


def test_agent_card_is_served_at_the_well_known_path(validator_agent_base_url):
    card = requests.get(f"{validator_agent_base_url}/.well-known/agent-card.json").json()
    assert card["name"] == "MCP-Forge Validator"
    assert card["skills"][0]["id"] == "validate_mcp_tools"
    interface = card["supportedInterfaces"][0]
    assert interface["protocolBinding"] == "JSONRPC" and interface["protocolVersion"] == "1.0"
    assert interface["url"].startswith(validator_agent_base_url)   # the card tells callers where to send messages


def test_official_sdk_client_can_discover_and_call_the_agent(validator_agent_base_url):
    result = validate_via_a2a(validator_agent_base_url, GOOD, ENDPOINTS)
    assert result == {"passed": True, "errors": [],
                      "output": "1 tool(s) passed: each was called through an MCP server and sent the documented "
                                "method, path, required parameters and body: ['get_fact']"}


def test_send_message_returns_a_completed_task_with_the_verdict_in_artifacts(validator_agent_base_url):
    task = send_message(validator_agent_base_url, GOOD)["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert [a["name"] for a in task["artifacts"]] == ["validation-summary", "validation-verdict"]
    assert task["artifacts"][1]["parts"][0]["data"] == {"passed": True, "errors": []}


def test_code_that_fails_validation_is_still_a_completed_task(validator_agent_base_url):
    """'The code is wrong' is the agent's answer, not a failure of the agent."""
    task = send_message(validator_agent_base_url, GOOD.replace('_request("GET"', '_request("POST"'))["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    verdict = task["artifacts"][1]["parts"][0]["data"]
    assert verdict["passed"] is False
    assert verdict["errors"][0]["function"] == "get_fact"


def test_get_task_returns_a_finished_task(validator_agent_base_url):
    task_id = send_message(validator_agent_base_url, GOOD)["result"]["task"]["id"]
    assert rpc(validator_agent_base_url, "GetTask", {"id": task_id})["result"]["id"] == task_id
    assert rpc(validator_agent_base_url, "GetTask", {"id": "nope"})["error"]["code"] == -32001


def test_unknown_method_is_a_json_rpc_error(validator_agent_base_url):
    assert rpc(validator_agent_base_url, "tasks/send", {})["error"]["code"] == -32601


def test_a_request_without_the_protocol_version_is_refused(validator_agent_base_url):
    """No A2A-Version header means protocol 0.3, which this 1.0 agent does not speak."""
    error = rpc(validator_agent_base_url, "GetTask", {"id": "x"}, version=None)["error"]
    assert error["code"] == -32009 and "not supported" in error["message"]


def test_the_agent_keeps_answering_while_a_validation_runs(validator_agent_base_url):
    """Validation waits seconds for a child process; it runs on a worker thread,
    so the agent can still answer other requests meanwhile."""
    slow = "import time\ntime.sleep(3)\n" + GOOD
    worker = threading.Thread(target=send_message, args=(validator_agent_base_url, slow))
    worker.start()
    time.sleep(0.5)                                    # the slow validation is now running
    started = time.time()
    card = requests.get(f"{validator_agent_base_url}/.well-known/agent-card.json", timeout=5).json()
    assert card["name"] == "MCP-Forge Validator" and time.time() - started < 1.5
    worker.join()


def test_old_tasks_are_forgotten_beyond_the_limit():
    from a2a.server.context import ServerCallContext
    from a2a.types.a2a_pb2 import Task
    from validator_agent import BoundedTaskStore

    async def scenario():
        store, context = BoundedTaskStore(max_tasks=2), ServerCallContext()
        for task_id in ("t1", "t2", "t3"):
            await store.save(Task(id=task_id, context_id="c"), context)
        return [await store.get(task_id, context) is not None for task_id in ("t1", "t2", "t3")]

    assert asyncio.run(scenario()) == [False, True, True]


def test_the_agent_documents_its_own_endpoints(validator_agent_base_url):
    """FastAPI generates an OpenAPI description, browsable at /docs."""
    schema = requests.get(f"{validator_agent_base_url}/openapi.json").json()
    assert {"/", "/.well-known/agent-card.json"} <= set(schema["paths"])
    assert requests.get(f"{validator_agent_base_url}/docs").status_code == 200
