"""The validator must behave as an A2A agent: discoverable by its Agent Card,
callable with SendMessage, by the official a2a-sdk client."""
import requests

from a2a_client import validate_via_a2a

GOOD = '''import requests

def get_fact() -> dict:
    """Returns a random cat fact."""
    return requests.get("https://catfact.ninja/fact", timeout=30).json()
'''
ENDPOINTS = [{"method": "GET", "path": "/fact", "base_url": "https://catfact.ninja", "description": "fact"}]


def rpc(url, method, params):
    return requests.post(url, json={"jsonrpc": "2.0", "id": "1", "method": method, "params": params}).json()


def send_message(url, code):
    return rpc(url, "SendMessage", {"message": {
        "messageId": "m1", "role": "ROLE_USER",
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
                      "output": "1 tool(s) each called the correct method and URL "
                                "and registered on an MCP server: ['get_fact']"}


def test_send_message_returns_a_completed_task_with_the_verdict_in_artifacts(validator_agent_base_url):
    task = send_message(validator_agent_base_url, GOOD)["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert [a["name"] for a in task["artifacts"]] == ["validation-summary", "validation-verdict"]
    assert task["artifacts"][1]["parts"][0]["data"] == {"passed": True, "errors": []}


def test_code_that_fails_validation_is_still_a_completed_task(validator_agent_base_url):
    """'The code is wrong' is the agent's answer, not a failure of the agent."""
    task = send_message(validator_agent_base_url, GOOD.replace("https://catfact.ninja", ""))["result"]["task"]
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
