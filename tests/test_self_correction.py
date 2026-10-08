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

BUGGY = '''import requests

def get_users(id: int) -> dict:
    """Get a user."""
    return requests.get(f"/users/{id}", timeout=30).json()
'''
FIXED = BUGGY.replace('f"/users/{id}"', 'f"https://api.example.com/users/{id}"')


class ScriptedLLM:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        saw_the_error = "not an absolute URL" in prompt and BUGGY in prompt
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
    assert any("not an absolute URL" in e for e in first["errors"])
    assert second["passed"] is True

    # The retry prompt contained the validator's error and the previous code.
    assert "FAILED validation" not in llm.prompts[0]
    assert "not an absolute URL" in llm.prompts[1] and BUGGY in llm.prompts[1]

    # The packaged server is the FIXED code, and the report records both attempts.
    assert "https://api.example.com/users/" in open(result["output_path"]).read()
    report = json.load(open(tmp_path / "generated" / "https-api-example-com" / "validation_report.json"))
    assert [a["passed"] for a in report["attempts"]] == [False, True]


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
        code = "import requests\n"
        for n in re.findall(r'"path": "/r(\d+)"', endpoints_section):
            base = "" if (n == "15" and not is_retry) else "https://api.example.com"
            code += f'\ndef get_r{n}() -> dict:\n    """resource {n}"""\n    return requests.get("{base}/r{n}", timeout=30).json()\n'
        return code

    monkeypatch.setattr(nodes, "call_llm", llm)

    result = run(spec)

    assert result["orchestration_plan"] == "complex" and result["max_code_attempts"] == 4
    assert result["test_passed"] is True
    assert [a["batches"] for a in result["attempt_history"]] == [[1, 2], [2]]
    assert len(calls) == 3                      # 2 batches, then 1 regeneration
    assert "get_r15() called '/r15'" in calls[2]
