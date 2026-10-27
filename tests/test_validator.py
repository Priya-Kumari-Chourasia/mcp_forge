"""The validator must fail code that is valid Python but would not work as a tool.

The code under test is exactly what gets packaged: the server template's header
(with _request) followed by the generated tool functions."""
from server_template import render_header
from validator_agent import validate_code

BASE = "https://api.example.com/v3"
HEADER = render_header({"title": "Pet API", "base_url": BASE, "auth": []})

ENDPOINTS = [
    {"method": "GET", "path": "/pet/findByStatus", "base_url": BASE, "description": "find", "returns": "array",
     "parameters": [{"name": "status", "in": "query", "required": True, "type": "string",
                     "enum": ["available", "pending", "sold"]}]},
    {"method": "GET", "path": "/pet/{petId}", "base_url": BASE, "description": "get",
     "parameters": [{"name": "petId", "in": "path", "required": True, "type": "integer"}]},
    {"method": "DELETE", "path": "/pet/{petId}", "base_url": BASE, "description": "delete",
     "parameters": [{"name": "petId", "in": "path", "required": True, "type": "integer"}]},
    {"method": "POST", "path": "/pet", "base_url": BASE, "description": "add", "parameters": [],
     "request_body": {"content_type": "application/json", "required": True, "fields": ["name"]}},
]

TOOLS = '''
def find_pets_by_status(
    status: Annotated[Literal["available", "pending", "sold"], Field(description="Status to filter by")],
) -> Union[dict, list, str]:
    """Finds pets by status."""
    return _request("GET", "/pet/findByStatus", params={"status": status})


def get_pet(pet_id: Annotated[int, Field(description="ID of the pet")]) -> Union[dict, list, str]:
    """Find pet by ID."""
    return _request("GET", "/pet/{petId}", path_params={"petId": pet_id})


def delete_pet(pet_id: Annotated[int, Field(description="ID of the pet")]) -> Union[dict, list, str]:
    """Deletes a pet."""
    return _request("DELETE", "/pet/{petId}", path_params={"petId": pet_id})


def add_pet(body: Annotated[dict, Field(description="The pet: name (required)")]) -> Union[dict, list, str]:
    """Adds a pet."""
    return _request("POST", "/pet", json=body)
'''
GOOD = HEADER + TOOLS


def errors_for(code, endpoints=ENDPOINTS):
    result = validate_code(code, endpoints)
    return result["passed"], [e["error"] for e in result["errors"]], result["errors"]


def test_correct_code_passes():
    passed, errors, _ = errors_for(GOOD)
    assert passed, errors


def test_a_list_response_with_a_too_narrow_return_type_is_caught():
    """The bug found by testing in Claude Desktop: find-by-status returns a JSON array,
    and a tool annotated -> Union[dict, str] makes the MCP SDK reject the result."""
    bad = GOOD.replace(') -> Union[dict, list, str]:\n    """Finds pets by status."""',
                       ') -> Union[dict, str]:\n    """Finds pets by status."""')
    passed, errors, structured = errors_for(bad)
    assert not passed
    assert any("find_pets_by_status() returned a JSON array that its return type does not allow" in e
               for e in errors)
    assert structured[0]["function"] == "find_pets_by_status"


def test_a_required_query_parameter_that_is_not_sent_is_caught():
    bad = GOOD.replace('params={"status": status}', 'params={}')
    passed, errors, _ = errors_for(bad)
    assert not passed
    assert any("find_pets_by_status() did not send the required query parameter 'status'" in e for e in errors)


def test_a_documented_parameter_without_an_argument_is_caught():
    endpoints = [dict(ep) for ep in ENDPOINTS]
    endpoints[0]["parameters"] = endpoints[0]["parameters"] + [
        {"name": "limit", "in": "query", "required": False, "type": "integer"}]
    passed, errors, _ = errors_for(GOOD, endpoints)
    assert not passed
    assert errors == ["find_pets_by_status() has no argument for the documented query parameter 'limit'"]


def test_a_required_body_that_is_not_sent_is_caught():
    bad = GOOD.replace('_request("POST", "/pet", json=body)', '_request("POST", "/pet")')
    passed, errors, _ = errors_for(bad)
    assert not passed
    assert any("add_pet() did not send the required request body" in e for e in errors)


def test_undefined_variable_is_caught():
    bad = GOOD.replace('path_params={"petId": pet_id})\n\n\ndef delete_pet',
                       'path_params={"petId": petId})\n\n\ndef delete_pet')
    passed, errors, _ = errors_for(bad)
    assert not passed
    assert any("get_pet() raised NameError" in e for e in errors)


def test_wrong_http_method_is_caught():
    bad = GOOD.replace('_request("DELETE"', '_request("POST"')
    passed, errors, structured = errors_for(bad)
    assert not passed
    assert any("delete_pet() sent POST" in e for e in errors)
    assert any(e["endpoint_index"] == 2 for e in structured)        # ...and the endpoint left without a tool


def test_missing_endpoint_is_caught():
    bad = GOOD[:GOOD.index("def delete_pet")] + GOOD[GOOD.index("def add_pet"):]
    passed, errors, _ = errors_for(bad)
    assert not passed
    assert errors == ["no working tool for endpoint DELETE /pet/{petId}"]


def test_using_requests_directly_is_caught():
    """A tool that bypasses _request also bypasses authentication and error handling."""
    bad = GOOD.replace('return _request("DELETE", "/pet/{petId}", path_params={"petId": pet_id})',
                       'import requests\n    return requests.delete(f"https://api.example.com/v3/pet/{pet_id}").json()')
    passed, errors, _ = errors_for(bad)
    assert not passed
    assert any("delete_pet() used the requests library directly" in e for e in errors)


def test_duplicate_function_names_are_caught():
    passed, errors, _ = errors_for(GOOD + "\ndef get_pet(pet_id: int):\n    pass\n")
    assert not passed and "defined more than once" in errors[0]


def test_helper_functions_are_not_tools():
    """_request (from the header) is a helper: it is neither called as a tool nor reported."""
    passed, errors, _ = errors_for(GOOD + "\n\ndef _format(x):\n    return x\n")
    assert passed, errors


def test_generated_code_never_runs_inside_the_validator_process(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    side_effect = 'open("leaked.txt", "w").write("x")\n' + GOOD
    validate_code(side_effect, ENDPOINTS)
    assert not (tmp_path / "leaked.txt").exists()


def test_hanging_code_is_killed(monkeypatch):
    import validator_agent
    monkeypatch.setattr(validator_agent, "BEHAVIOUR_TIMEOUT_SECONDS", 3)
    passed, errors, _ = errors_for("import time\ntime.sleep(60)\n" + GOOD)
    assert not passed and "timed out" in errors[0]


def test_code_that_cannot_be_registered_as_an_mcp_tool_is_caught():
    """`any` is a built-in function, not a type. The function runs fine as plain Python,
    but the MCP SDK cannot build a schema from it, so the packaged server would crash on start."""
    bad = GOOD.replace('Field(description="ID of the pet")]) -> Union[dict, list, str]:\n    """Find pet by ID."""',
                       'Field(description="ID of the pet")]) -> any:\n    """Find pet by ID."""')
    passed, errors, structured = errors_for(bad)
    assert not passed
    assert any("get_pet() cannot be registered as an MCP tool" in e for e in errors)
    # Its endpoint is then left without a working tool.
    assert [e["function"] for e in structured] == ["get_pet", None]


def test_generated_code_cannot_start_other_programs_or_read_the_api_key(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "secret-value")
    sneaky = GOOD.replace('    """Deletes a pet."""\n',
                          '    """Deletes a pet."""\n'
                          '    import os, subprocess\n'
                          '    assert "GROQ_API_KEY" not in os.environ, "key leaked"\n'
                          '    subprocess.run(["python", "-c", "print(1)"])\n')
    passed, errors, _ = errors_for(sneaky)
    assert not passed
    assert any("starting other programs is disabled" in e for e in errors)
    assert not any("key leaked" in e for e in errors)
