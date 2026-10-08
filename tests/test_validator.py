"""The validator must fail code that is valid Python but would not work as a tool."""
from validator_agent import validate_code

ENDPOINTS = [
    {"method": "GET", "path": "/pet/findByStatus", "base_url": "https://api.example.com/v3", "description": "find"},
    {"method": "GET", "path": "/pet/{petId}", "base_url": "https://api.example.com/v3", "description": "get"},
    {"method": "DELETE", "path": "/pet/{petId}", "base_url": "https://api.example.com/v3", "description": "delete"},
]

GOOD = '''
import requests

def get_pet_find_by_status(status: str = None) -> dict:
    """Finds pets by status."""
    params = {"status": status} if status is not None else {}
    return requests.get("https://api.example.com/v3/pet/findByStatus", params=params, timeout=30).json()

def get_pet(pet_id: int) -> dict:
    """Find pet by ID."""
    return requests.get(f"https://api.example.com/v3/pet/{pet_id}", timeout=30).json()

def delete_pet(pet_id: int) -> dict:
    """Deletes a pet."""
    return requests.delete(f"https://api.example.com/v3/pet/{pet_id}", timeout=30).json()
'''


def errors_for(code, endpoints=ENDPOINTS):
    result = validate_code(code, endpoints)
    return result["passed"], [e["error"] for e in result["errors"]], result["errors"]


def test_correct_code_passes():
    passed, errors, _ = errors_for(GOOD)
    assert passed, errors


def test_relative_url_is_caught():
    bad = GOOD.replace('f"https://api.example.com/v3/pet/{pet_id}", timeout=30).json()\n\ndef delete',
                       'f"/v3/pet/{pet_id}", timeout=30).json()\n\ndef delete')
    passed, errors, structured = errors_for(bad)
    assert not passed
    assert any("get_pet()" in e and "not an absolute URL" in e for e in errors)
    assert any(e["function"] == "get_pet" for e in structured)      # the error names the function


def test_undefined_variable_is_caught():
    bad = GOOD.replace("/pet/{pet_id}\", timeout=30).json()\n\ndef delete", "/pet/{petId}\", timeout=30).json()\n\ndef delete")
    passed, errors, _ = errors_for(bad)
    assert not passed
    assert any("get_pet() raised NameError" in e for e in errors)


def test_wrong_http_method_is_caught():
    bad = GOOD.replace("requests.delete(", "requests.post(")
    passed, errors, structured = errors_for(bad)
    assert not passed
    assert any("delete_pet() sent POST" in e for e in errors)
    assert any(e["endpoint_index"] == 2 for e in structured)        # ...and the endpoint left without a tool


def test_missing_endpoint_is_caught():
    bad = GOOD[:GOOD.index("def delete_pet")]
    passed, errors, _ = errors_for(bad)
    assert not passed
    assert any("no working tool for endpoint DELETE /pet/{petId}" in e for e in errors)


def test_duplicate_function_names_are_caught():
    passed, errors, _ = errors_for(GOOD + "\ndef get_pet(pet_id: int):\n    pass\n")
    assert not passed and "defined more than once" in errors[0]


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
    bad = GOOD.replace("def get_pet(pet_id: int) -> dict:", "def get_pet(pet_id: int) -> any:")
    passed, errors, structured = errors_for(bad)
    assert not passed
    assert any("get_pet() cannot be registered as an MCP tool" in e for e in errors)
    assert [e["function"] for e in structured] == ["get_pet"]