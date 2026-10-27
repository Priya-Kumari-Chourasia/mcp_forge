"""The fixed part of every generated server: names, credentials, packaging, and the
final check with a real MCP client. The end-to-end test runs a packaged server as
its own process and calls its tools over MCP against a small fake API."""
import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import nodes
import server_template as T

META = {"title": "Swagger Petstore - OpenAPI 3.0", "base_url": "https://petstore.test/v3",
        "auth": [{"scheme": "petstore_auth", "type": "bearer"},
                 {"scheme": "api_key", "type": "header", "name": "api_key"}]}


# ----------------------------------------------------------------------
# Names and credentials
# ----------------------------------------------------------------------

def test_tool_names_are_snake_case_identifiers():
    assert T.python_name("findPetsByStatus") == "find_pets_by_status"
    assert T.python_name("issues/list-for-repo") == "issues_list_for_repo"
    assert T.python_name("GET /pet/{petId}") == "get_pet_pet_id"
    assert T.python_name("import") == "import_"           # a Python keyword
    assert T.python_name("2fa/verify") == "t_2fa_verify"   # cannot start with a digit


def test_every_endpoint_gets_a_unique_name_that_cannot_clash_with_the_header():
    endpoints = [{"method": "GET", "path": "/a", "operation_id": "getThing"},
                 {"method": "GET", "path": "/b", "operation_id": "getThing"},
                 {"method": "GET", "path": "/server"},
                 {"method": "GET", "path": "/c", "operation_id": "server"}]
    T.assign_tool_names(endpoints)
    assert [e["name"] for e in endpoints] == ["get_thing", "get_thing_2", "get_server", "server_tool"]


def test_environment_variable_names():
    assert T.env_prefix(META) == "PETSTORE"
    assert T.env_prefix({"title": "GitHub v3 REST API"}) == "GITHUB"
    assert T.env_prefix({"title": "", "base_url": "https://api.bookstore.example/v2"}) == "BOOKSTORE"
    assert T.auth_env_names(META) == ["PETSTORE_TOKEN", "PETSTORE_API_KEY"]
    basic = {"title": "Shop", "auth": [{"scheme": "b", "type": "basic"}]}
    assert T.auth_env_names(basic) == ["SHOP_USERNAME", "SHOP_PASSWORD"]


def test_the_auth_option_for_docs_that_do_not_declare_it():
    assert T.parse_auth_option("bearer") == {"scheme": "bearer", "type": "bearer"}
    assert T.parse_auth_option("header:X-Api-Key")["name"] == "X-Api-Key"
    with pytest.raises(ValueError):
        T.parse_auth_option("magic")


def test_claude_desktop_config_has_an_empty_slot_for_each_credential(tmp_path):
    config = T.claude_desktop_config(META, str(tmp_path / "server.py"))
    entry = config["mcpServers"]["swagger-petstore-openapi-3-0"]
    assert entry["command"] == sys.executable
    assert entry["env"] == {"PETSTORE_TOKEN": "", "PETSTORE_API_KEY": ""}    # empty = not sent


# ----------------------------------------------------------------------
# Choosing endpoints
# ----------------------------------------------------------------------

ENDPOINTS = [{"method": "GET", "path": "/pet/findByStatus", "tags": ["pet"]},
             {"method": "GET", "path": "/pet/{petId}", "tags": ["pet"]},
             {"method": "GET", "path": "/store/inventory", "tags": ["store"]},
             {"method": "GET", "path": "/user/login", "tags": ["user"]}]


def test_endpoints_can_be_chosen_by_tag_and_path():
    chosen, note = nodes._select(ENDPOINTS, {"tags": ["pet", "store"]})
    assert [e["path"] for e in chosen] == ["/pet/findByStatus", "/pet/{petId}", "/store/inventory"]
    assert note == "3 of 4 endpoint(s) match the chosen tags/paths"
    chosen, _ = nodes._select(ENDPOINTS, {"tags": ["pet"], "paths": ["/pet/find*"]})
    assert [e["path"] for e in chosen] == ["/pet/findByStatus"]


def test_large_apis_are_capped_and_the_report_says_so():
    chosen, note = nodes._select(ENDPOINTS, {"max_tools": 2})
    assert len(chosen) == 2 and "kept the first 2 of 4" in note
    assert nodes._select(ENDPOINTS, {}) == (ENDPOINTS, None)          # under the default cap: untouched


# ----------------------------------------------------------------------
# A packaged server, run for real
# ----------------------------------------------------------------------

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
TOOL_ENDPOINTS = [{"name": "find_pets_by_status", "method": "GET"}, {"name": "get_pet_by_id", "method": "GET"},
                  {"name": "delete_pet", "method": "DELETE"}]


@pytest.fixture
def fake_api():
    """A tiny HTTP server standing in for the Petstore: one list endpoint, 404 for the rest.
    It records what each request carried, so the test can check the credentials."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen.append({"path": self.path, "api_key": self.headers.get("api_key"),
                         "authorization": self.headers.get("Authorization")})
            if self.path.startswith("/v3/pet/findByStatus"):
                code, body = 200, [{"id": 1, "name": "doggie", "status": "available"}]
            else:
                code, body = 404, {"message": "Pet not found"}
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v3", seen
    server.shutdown()


def call_tools_over_stdio(server_path, env, calls):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=[server_path], env=env)
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name: t for t in (await session.list_tools()).tools}
                results = [await session.call_tool(name, args) for name, args in calls]
                return tools, results

    return asyncio.run(go())


def test_a_packaged_server_works_end_to_end_over_mcp(tmp_path, fake_api):
    base_url, seen = fake_api
    code, info = T.package_server(TOOLS, TOOL_ENDPOINTS, META)
    server_path = tmp_path / "server.py"
    server_path.write_text(code, encoding="utf-8")

    env = dict(os.environ, PETSTORE_BASE_URL=base_url, PETSTORE_API_KEY="key-123", PETSTORE_TOKEN="tok-456")
    tools, (found, missing, bad_arg) = call_tools_over_stdio(str(server_path), env, [
        ("find_pets_by_status", {"status": "available"}),
        ("get_pet_by_id", {"pet_id": 999}),
        ("find_pets_by_status", {"status": "lost"}),
    ])

    # The tools carry the schema and hints an AI client relies on.
    assert tools["find_pets_by_status"].input_schema["properties"]["status"]["enum"] == ["available", "pending", "sold"]
    assert tools["find_pets_by_status"].annotations.read_only_hint is True
    assert tools["delete_pet"].annotations.destructive_hint is True

    # A JSON array comes back intact (the find-by-status bug, fixed).
    assert found.is_error is False and found.structured_content == {
        "result": [{"id": 1, "name": "doggie", "status": "available"}]}
    # An HTTP error reaches the AI with its status code, not as "Error executing tool".
    assert missing.is_error is True and "failed with HTTP 404" in missing.content[0].text
    # A value outside the enum is rejected before any request is sent.
    assert bad_arg.is_error is True and len(seen) == 2

    # Credentials came from environment variables, not from tool arguments.
    assert seen[0] == {"path": "/v3/pet/findByStatus?status=available", "api_key": "key-123",
                       "authorization": "Bearer tok-456"}


def test_the_mcp_client_check_passes_for_a_good_server_and_fails_for_a_broken_one(tmp_path):
    code, info = T.package_server(TOOLS, TOOL_ENDPOINTS, META)
    good = tmp_path / "good.py"
    good.write_text(code, encoding="utf-8")
    assert T.check_with_mcp_client(str(good), [t["name"] for t in info]) == {
        "passed": True, "tools_listed": 3, "missing": [], "error": None}

    broken = tmp_path / "broken.py"
    broken.write_text("raise SystemExit('crashed on start')\n", encoding="utf-8")
    result = T.check_with_mcp_client(str(broken), ["find_pets_by_status"], timeout=30)
    assert result["passed"] is False and result["error"]
