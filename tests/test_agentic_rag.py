"""
Proves the prose path is agentic RAG, end to end through the real graph:

  - the WHOLE document is covered (an endpoint far past the old 6,000-character cut-off is found)
  - retrieval is graded, and when the context is insufficient the query is rewritten and retried
  - the details found on the second try end up in the generated tool

The LLM is scripted so the test is deterministic. Its "judgement" is a simple rule
(does the context contain the passage that describes the endpoint?), which is enough
to exercise every edge of the loop.
"""
import json
import re

import nodes
import rag_nodes
from graph import run
from rag import DocIndex, chunk_text

BILLING = ("Billing overview. Invoices are issued monthly and can be downloaded from the dashboard. "
           "Contact support for refunds, tax forms and purchase orders.\n\n")
# Text that LOOKS relevant to a generic "endpoint description parameters" query but says nothing useful.
CONVENTIONS = ("Documentation conventions. Every endpoint description lists its parameters, and each "
               "endpoint is relative to the base URL. Parameters are shown in a table under the endpoint.\n\n")

DOCS = (
    "Widget API guide\n\nThe base URL for every request is https://api.widgets.test/v1\n\n"
    "Listing widgets: GET /widgets returns every widget in your account. "
    "It accepts an optional query parameter named limit.\n\n"
    + CONVENTIONS * 45 +                              # ~9,000 characters before the next endpoint
    "Endpoint summary table\n| DELETE /widgets/{id} |\n\n"
    + BILLING * 30 +
    "Removing a widget\n\nPermanently removes one widget. The identifier goes in the address. "
    "An optional query flag called force skips the recycle bin.\n\n"
    + BILLING * 10
)

DETAIL_PASSAGES = {
    "GET /widgets": "optional query parameter named limit",
    "DELETE /widgets/{id}": "optional query flag called force",
}
EXTRACTED = {
    "GET /widgets": {"description": "Lists every widget.", "base_url": "https://api.widgets.test/v1",
                     "parameters": [{"name": "limit", "in": "query", "required": False, "type": "integer"}]},
    "DELETE /widgets/{id}": {"description": "Permanently removes one widget.", "base_url": "",
                             "parameters": [{"name": "id", "in": "path", "required": True, "type": "string"},
                                            {"name": "force", "in": "query", "required": False, "type": "boolean"}]},
}


def scripted_llm(prompt):
    if "Analyze these API docs" in prompt:
        return "simple"

    if "List every API endpoint" in prompt:
        found = re.findall(r"\b(GET|POST|PUT|DELETE|PATCH) (/[\w/{}]+)", prompt.split("Documentation excerpt:")[1])
        return json.dumps([{"method": m, "path": p} for m, p in found])

    if "Is the context sufficient" in prompt:
        name = re.search(r"documenting the endpoint (\w+ \S+) using", prompt).group(1)
        context = prompt.split("Context:")[1]
        if DETAIL_PASSAGES[name] in context:
            method, path = name.split(" ")
            return json.dumps({"sufficient": True, "endpoint": {"method": method, "path": path, **EXTRACTED[name]}})
        return json.dumps({"sufficient": False, "missing": "what the endpoint does and its parameters",
                           "next_query": "removing a widget permanently force flag recycle bin"})

    # code generation: one function per endpoint in the prompt
    endpoints = json.loads(prompt.split("Endpoints:\n")[1].split("\nYour previous code")[0])
    code = "import requests\n"
    for ep in endpoints:
        args = [p["name"] for p in ep["parameters"] if p["in"] == "path"]
        name = (ep["method"] + re.sub(r"\W+", "_", ep["path"])).lower().strip("_")
        url = ep["base_url"] + ep["path"]
        code += (f'\ndef {name}({", ".join(a + ": str" for a in args)}) -> dict:\n    """{ep["description"]}"""\n'
                 f'    return requests.{ep["method"].lower()}(f"{url}", timeout=30).json()\n')
    return code


def test_prose_docs_go_through_retrieve_grade_rewrite(tmp_path, monkeypatch, validator_agent_url):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nodes, "call_llm", scripted_llm)
    monkeypatch.setattr(rag_nodes, "call_llm", scripted_llm)
    assert DOCS.index("DELETE /widgets/{id}") > 6000          # beyond what the old truncation ever saw

    result = run(DOCS)

    endpoints = {f"{e['method']} {e['path']}": e for e in result["endpoints"]}
    assert set(endpoints) == {"GET /widgets", "DELETE /widgets/{id}"}

    # The DELETE endpoint needed a second, rewritten query before its context was good enough.
    rounds = [t for t in result["rag_trace"] if t["endpoint"] == "DELETE /widgets/{id}"]
    assert [t["sufficient"] for t in rounds] == [False, True]
    assert rounds[0]["query"] != rounds[1]["query"]
    assert rounds[1]["query"] == "removing a widget permanently force flag recycle bin"

    # What the second retrieval found made it into the endpoint definition...
    assert [p["name"] for p in endpoints["DELETE /widgets/{id}"]["parameters"]] == ["id", "force"]
    # ...and the base URL, stated once at the top of the docs, was shared with it.
    assert endpoints["DELETE /widgets/{id}"]["base_url"] == "https://api.widgets.test/v1"

    # The whole pipeline then generated tools that passed the validator agent.
    assert result["test_passed"] is True
    report = json.load(open(tmp_path / "generated" / "https-api-widgets-test-v1" / "validation_report.json"))
    assert len(report["retrieval"]) == 3                       # 1 round for GET, 2 for DELETE


def test_gives_up_after_max_rounds_but_keeps_the_endpoint(tmp_path, monkeypatch, validator_agent_url):
    monkeypatch.chdir(tmp_path)

    def never_satisfied(prompt):
        if "Is the context sufficient" in prompt:
            return json.dumps({"sufficient": False, "missing": "everything", "next_query": "try again"})
        return scripted_llm(prompt)

    monkeypatch.setattr(nodes, "call_llm", never_satisfied)
    monkeypatch.setattr(rag_nodes, "call_llm", never_satisfied)

    result = run("The API lives at https://x.test and has one endpoint: GET /ping")

    assert len(result["rag_trace"]) == rag_nodes.MAX_RETRIEVAL_ROUNDS
    assert result["endpoints"] == [{"method": "GET", "path": "/ping", "description": "GET /ping",
                                    "base_url": "", "parameters": []}]


def test_openapi_specs_skip_rag_entirely(tmp_path, monkeypatch, validator_agent_url):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(nodes, "call_llm", scripted_llm)
    monkeypatch.setattr(rag_nodes, "call_llm", lambda prompt: 1 / 0)    # must never be called
    spec = json.dumps({"openapi": "3.0.0", "servers": [{"url": "https://api.example.com"}],
                       "paths": {"/ping": {"get": {"summary": "ping"}}}})
    result = run(spec)
    assert result["rag_trace"] == [] and result["test_passed"] is True


def test_chunks_overlap_and_cover_the_whole_text():
    text = "".join(f"line {i}\n" for i in range(600))
    chunks = chunk_text(text, size=500, overlap=100)
    assert all(len(c) <= 500 for c in chunks)
    assert all(f"line {i}\n".strip() in "\n".join(chunks) for i in range(600))
    assert chunks[0][-30:] in chunks[0] and chunks[1][:20] in chunks[0]      # neighbours share text


def test_index_search_returns_the_relevant_chunk_first():
    index = DocIndex(DOCS)
    assert "optional query flag called force" in index.search("force flag recycle bin removing a widget", k=1)[0]
