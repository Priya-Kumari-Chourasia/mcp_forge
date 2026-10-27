"""The "Try it" panel: an LLM chats with a generated server over MCP, as Claude Desktop does.
The LLM is scripted so the test is deterministic; the server and its MCP connection are real."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from langchain_core.messages import AIMessage, ToolMessage

import server_template as T
import try_it

TOOLS = '''
def find_pets_by_status(
    status: Annotated[Literal["available", "pending", "sold"], Field(description="Status to filter by")],
) -> Union[dict, list, str]:
    """Finds pets by status."""
    return _request("GET", "/pet/findByStatus", params={"status": status})


def get_pet_by_id(pet_id: Annotated[int, Field(description="ID of the pet")]) -> Union[dict, list, str]:
    """Find pet by ID."""
    return _request("GET", f"/pet/{pet_id}")


def delete_pet(pet_id: Annotated[int, Field(description="ID of the pet")]) -> Union[dict, list, str]:
    """Deletes a pet."""
    return _request("DELETE", f"/pet/{pet_id}")
'''
ENDPOINTS = [{"name": "find_pets_by_status", "method": "GET"}, {"name": "get_pet_by_id", "method": "GET"},
             {"name": "delete_pet", "method": "DELETE"}]


class ScriptedModel:
    """Stands in for ChatGroq: returns the given replies in order and records what it was shown."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.offered = None
        self.seen = []

    def bind_tools(self, tools):
        self.offered = [t["function"]["name"] for t in tools]
        return self

    async def ainvoke(self, messages):
        self.seen.append(list(messages))
        return self.replies.pop(0)


@pytest.fixture
def server_path(tmp_path, monkeypatch):
    """A packaged Petstore server pointed at a tiny fake API on this machine."""
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/v3/pet/findByStatus"):
                code, body = 200, [{"id": 4, "name": "Dog 1", "status": "available"}]
            else:
                code, body = 404, {"message": "Pet not found"}
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    api = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=api.serve_forever, daemon=True).start()
    meta = {"title": "Swagger Petstore", "base_url": f"http://127.0.0.1:{api.server_port}/v3", "auth": []}
    code, _ = T.package_server(TOOLS, ENDPOINTS, meta)
    path = tmp_path / "server.py"
    path.write_text(code, encoding="utf-8")
    yield str(path)
    api.shutdown()


def call(name, args, call_id="c1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_the_model_calls_a_tool_over_mcp_and_answers_from_its_result(server_path):
    model = ScriptedModel([call("find_pets_by_status", {"status": "available"}),
                           AIMessage(content="The first available pet is Dog 1.")])
    conversation = [try_it.system_message("Swagger Petstore")]

    outcome = try_it.ask(server_path, conversation, "Find available pets", model)

    assert outcome["answer"] == "The first available pet is Dog 1."
    assert outcome["trace"] == [{"tool": "find_pets_by_status", "args": {"status": "available"}, "error": False,
                                 "result": '[{"id": 4, "name": "Dog 1", "status": "available"}]'}]
    # The real API data went back to the model as a tool result.
    tool_message = model.seen[1][-1]
    assert isinstance(tool_message, ToolMessage) and "Dog 1" in tool_message.content


def test_an_api_error_reaches_the_model_with_its_status_code(server_path):
    model = ScriptedModel([call("get_pet_by_id", {"pet_id": 99999999}), AIMessage(content="That pet does not exist.")])
    outcome = try_it.ask(server_path, [try_it.system_message("x")], "Get pet 99999999", model)
    assert outcome["trace"][0]["error"] is True
    assert "failed with HTTP 404" in model.seen[1][-1].content


def test_tools_that_change_data_are_offered_only_when_allowed(server_path):
    model = ScriptedModel([AIMessage(content="ok")])
    outcome = try_it.ask(server_path, [try_it.system_message("x")], "hi", model)
    assert model.offered == ["find_pets_by_status", "get_pet_by_id"] and outcome["hidden_tools"] == 1

    model = ScriptedModel([AIMessage(content="ok")])
    try_it.ask(server_path, [try_it.system_message("x")], "hi", model, allow_changes=True)
    assert "delete_pet" in model.offered


def test_a_hidden_tool_is_refused_even_if_the_model_asks_for_it(server_path):
    model = ScriptedModel([call("delete_pet", {"pet_id": 4}), AIMessage(content="I cannot delete pets here.")])
    outcome = try_it.ask(server_path, [try_it.system_message("x")], "Delete pet 4", model)
    assert outcome["trace"][0]["error"] is True and "not available here" in outcome["trace"][0]["result"]


def test_the_conversation_carries_over_to_the_next_question(server_path):
    conversation = [try_it.system_message("x")]
    try_it.ask(server_path, conversation, "first question", ScriptedModel([AIMessage(content="first answer")]))
    model = ScriptedModel([AIMessage(content="second answer")])
    try_it.ask(server_path, conversation, "second question", model)
    contents = [m.content for m in model.seen[0]]
    assert contents[1:] == ["first question", "first answer", "second question"]


def test_a_model_that_never_stops_calling_tools_is_cut_off(server_path):
    replies = [call("find_pets_by_status", {"status": "sold"}, f"c{i}") for i in range(try_it.MAX_STEPS)]
    outcome = try_it.ask(server_path, [try_it.system_message("x")], "loop", ScriptedModel(replies))
    assert outcome["answer"].startswith("Stopped after") and len(outcome["trace"]) == try_it.MAX_STEPS


def test_the_server_never_sees_the_groq_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret")
    monkeypatch.setenv("PETSTORE_API_KEY", "for-the-api")
    env = try_it.server_environment(["PETSTORE_API_KEY"])
    assert env.get("PETSTORE_API_KEY") == "for-the-api" and "GROQ_API_KEY" not in env


def test_a_failing_llm_is_reported_with_its_real_reason(server_path):
    class Broken:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            raise RuntimeError("Groq is down")

    with pytest.raises(BaseException) as raised:
        try_it.ask(server_path, [try_it.system_message("x")], "hi", Broken())
    # The MCP client wraps it in "unhandled errors in a TaskGroup"; the panel shows the cause.
    assert try_it.describe_error(raised.value) == "RuntimeError: Groq is down"
