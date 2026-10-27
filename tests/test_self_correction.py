"""
Proves the loop is self-correcting, end to end through the real graph
and a real validator agent (started on a local port by the fixture):

  attempt 1 -> the LLM writes a tool with a bug
  validator -> rejects it and says exactly what is wrong
  attempt 2 -> the LLM is shown that error + its previous code, and fixes it

The LLM here is scripted so the test is deterministic. Crucially, the
scripted LLM only returns the fix IF the validator's error is in the
prompt - so the test fails if the error ever stops being fed back.
"""
import json

import nodes
from graph import run

SPEC = json.dumps({
    "openapi": "3.0.0",
    "servers": [{"url": "https://api.example.com"}],
    "paths": {"/users/{id}": {"get": {
        "summary": "Get a user",
        "parameters": [{"name": "id", "in": "path", "required": True, "schema": {"type": "integer"}}],
    }}},
})

BUGGY = '''def get_users_id(id: int) -> Union[dict, list, str]:
    """Get a user."""
    return _request("GET", f"/user/{id}")
'''
FIXED = BUGGY.replace('f"/user/{id}"', 'f"/users/{id}"')
THE_ERROR = "get_users_id() sent GET https://api.example.com/user/1, which matches no documented endpoint"


class ScriptedLLM:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        saw_the_error = THE_ERROR in prompt and BUGGY in prompt
        return FIXED if saw_the_error else BUGGY


def test_failed_on_attempt_1_fixed_on_attempt_2(tmp_path, monkeypatch, validator_agent_url):
    monkeypatch.chdir(tmp_path)
    llm = ScriptedLLM()
    monkeypatch.setattr(nodes, "call_llm", llm)

    result = run(SPEC)

    assert result["test_passed"] is True
    assert result["codegen_attempt"] == 2

    first, second = result["attempt_history"]
    assert first["passed"] is False
    assert any(THE_ERROR in e for e in first["errors"])
    assert second["passed"] is True

    # The retry prompt contained the validator's error and the previous code.
    assert "FAILED validation" not in llm.prompts[0]
    assert THE_ERROR in llm.prompts[1] and BUGGY in llm.prompts[1]

    # The packaged server is the FIXED code, and the report records both attempts.
    server_code = open(result["output_path"]).read()
    assert '_request("GET", f"/users/{id}")' in server_code and "/user/{id}" not in server_code
    report = json.load(open(tmp_path / "generated" / "https-api-example-com" / "validation_report.json"))
    assert [a["passed"] for a in report["attempts"]] == [False, True]
    assert report["mcp_client_check"]["passed"] is True          # a real MCP client connected to server.py


def test_only_the_failing_batch_is_regenerated():
    batches = ["def a():\n    pass\n", "def b():\n    pass\n", "def c():\n    pass\n"]
    errors = [
        {"function": "b", "endpoint_index": None, "error": "b() raised NameError"},
        {"function": None, "endpoint_index": 2 * nodes.BATCH_SIZE, "error": "no working tool for endpoint GET /c"},
    ]
    assert nodes.errors_by_batch(errors, batches) == {
        1: ["b() raised NameError"],
        2: ["no working tool for endpoint GET /c"],
    }


def test_gives_up_when_the_retry_budget_runs_out(tmp_path, monkeypatch, validator_agent_url):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nodes, "call_llm", lambda prompt: BUGGY)     # an LLM that never learns

    result = run(SPEC)

    assert result["test_passed"] is False
    assert result["codegen_attempt"] == result["max_code_attempts"] == 2


def test_two_batches_only_the_broken_one_is_regenerated(tmp_path, monkeypatch, validator_agent_url):
    """16 endpoints -> 2 batches. The bug is in batch 2, so the retry makes ONE LLM call, not two."""
    import re
    monkeypatch.chdir(tmp_path)
    spec = json.dumps({
        "openapi": "3.0.0", "servers": [{"url": "https://api.example.com"}],
        "paths": {f"/r{i}": {"get": {"summary": f"resource {i}"}} for i in range(16)},
    })
    calls = []

    def llm(prompt):
        calls.append(prompt)
        endpoints_section = prompt.split("Your previous code")[0]
        is_retry = "FAILED validation" in prompt
        code = ""
        for n in re.findall(r'"path": "/r(\d+)"', endpoints_section):
            method = "POST" if (n == "15" and not is_retry) else "GET"       # the bug: r15 uses the wrong method
            code += f'\ndef get_r{n}() -> Union[dict, list, str]:\n    """resource {n}"""\n    return _request("{method}", "/r{n}")\n'
        return code

    monkeypatch.setattr(nodes, "call_llm", llm)

    result = run(spec)

    assert result["orchestration_plan"] == "complex" and result["max_code_attempts"] == 4
    assert result["test_passed"] is True
    assert [a["batches"] for a in result["attempt_history"]] == [[1, 2], [2]]
    assert len(calls) == 3                      # 2 batches, then 1 regeneration
    assert "get_r15() sent POST https://api.example.com/r15" in calls[2]
