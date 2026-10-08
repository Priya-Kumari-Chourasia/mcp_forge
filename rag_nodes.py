"""
AGENTIC RAG - how prose documentation is turned into endpoint definitions.

Plain RAG retrieves once and uses whatever comes back. Here the pipeline
checks its own retrieval and tries again when it is not good enough:

    index_docs -> discover_endpoints -> retrieve -> grade_retrieval
                                           ^              |
                                           |              +-- sufficient   -> next endpoint (or done)
                                           |              |
                                           +- rewrite_query <-- insufficient

  index_docs          chunk the whole document and embed it into ChromaDB
  discover_endpoints  scan the document window by window and list every
                      "METHOD /path" it mentions (so nothing is cut off)
  retrieve            for ONE endpoint, fetch the most relevant chunks
  grade_retrieval     the LLM judges: is this context enough to describe the
                      endpoint? If yes it fills in the details. If no it says
                      what is missing and proposes a better search query.
  rewrite_query       adopt that better query and retrieve again; the new chunks
                      are ADDED to the ones already found for this endpoint

OpenAPI specs skip all of this - they are already structured, so they are
parsed directly in spec_parser.py.
"""

import json
import re
from collections import Counter

from state import ForgeState
from llm import call_llm
from rag import DocIndex

DISCOVERY_WINDOW = 6000     # characters of docs per discovery call
DISCOVERY_OVERLAP = 300
TOP_K = 4                   # chunks retrieved per query
MAX_RETRIEVAL_ROUNDS = 3    # retrieve -> grade attempts per endpoint
MAX_DOC_CHARS = 60000       # prose docs longer than this are cut, to bound LLM calls and run time
_index = None               # the DocIndex for the current run (not JSON-serialisable, so kept out of the state)


def _parse_json(text: str, fallback):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _normalise_path(path: str) -> str:
    """/users/:id and /users/<id> both become /users/{id}."""
    path = re.sub(r":(\w+)", r"{\1}", path.strip())
    return re.sub(r"<(\w+)>", r"{\1}", path)


def index_docs_node(state: ForgeState) -> ForgeState:
    global _index
    if len(state["raw_docs"]) > MAX_DOC_CHARS:
        print(f"RAG 1 (index_docs) -> docs are {len(state['raw_docs'])} characters; "
                f"using the first {MAX_DOC_CHARS} to keep the run within the LLM rate limits")
        state["raw_docs"] = state["raw_docs"][:MAX_DOC_CHARS]
    _index = DocIndex(state["raw_docs"])
    print(f"RAG 1 (index_docs) -> {len(state['raw_docs'])} characters split into {len(_index.chunks)} chunk(s) and embedded")
    return state


def discover_endpoints_node(state: ForgeState) -> ForgeState:
    """Finds WHICH endpoints exist. Details come later, one endpoint at a time."""
    docs = state["raw_docs"]
    step = DISCOVERY_WINDOW - DISCOVERY_OVERLAP
    windows = [docs[i:i + DISCOVERY_WINDOW] for i in range(0, max(len(docs), 1), step)]

    found = {}
    for number, window in enumerate(windows, start=1):
        print(f"  scanning window {number}/{len(windows)} for endpoints...", flush=True)
        prompt = f"""List every API endpoint mentioned in the documentation excerpt below.

Return ONLY a JSON array of objects with "method" and "path", for example:
[{{"method": "GET", "path": "/users/{{id}}"}}]
Write path parameters in curly braces. If there are no endpoints, return [].

Documentation excerpt:
{window}"""
        for item in _parse_json(call_llm(prompt), []):
            if isinstance(item, dict) and item.get("method") and item.get("path"):
                method, path = item["method"].upper(), _normalise_path(item["path"])
                found[(method, path)] = {"method": method, "path": path}

    state["rag_candidates"] = list(found.values())
    state["rag_cursor"] = 0
    state["rag_round"] = 0
    state["rag_query"] = None
    state["endpoints"] = []
    print(f"RAG 2 (discover_endpoints) -> {len(found)} endpoint(s) found across {len(windows)} window(s): "
          f"{[f'{m} {p}' for m, p in found]}")
    return state


def route_after_discovery(state: ForgeState) -> str:
    return "retrieve" if state["rag_candidates"] else "none_found"


def retrieve_node(state: ForgeState) -> ForgeState:
    candidate = state["rag_candidates"][state["rag_cursor"]]
    # First round: a default query built from the endpoint. Later rounds: the rewritten query.
    query = state.get("rag_query") or f"{candidate['method']} {candidate['path']} endpoint description parameters base URL"
    state["rag_query"] = query
    state["rag_round"] = state.get("rag_round", 0) + 1

    # A rewritten query ADDS to what earlier rounds found for this endpoint instead of
    # replacing it, so a useful chunk from round 1 is not lost in round 2.
    earlier = (state.get("rag_context") or []) if state["rag_round"] > 1 else []
    fresh = [chunk for chunk in _index.search(query, k=TOP_K) if chunk not in earlier]
    state["rag_context"] = earlier + fresh
    print(f"RAG 3 (retrieve) {candidate['method']} {candidate['path']} round {state['rag_round']} -> query: \"{query}\"")
    return state


def grade_retrieval_node(state: ForgeState) -> ForgeState:
    """Asks the LLM to judge the retrieved context, and to extract the endpoint if it can."""
    candidate = state["rag_candidates"][state["rag_cursor"]]
    context = "\n\n---\n\n".join(state["rag_context"])
    prompt = f"""You are documenting the endpoint {candidate['method']} {candidate['path']} using ONLY the context below.

Is the context sufficient to say what this endpoint does and to NAME its parameters or body fields?
Judge leniently: you do not need data types, required flags or response details. If the context
says what the endpoint is for and names its inputs (or it clearly has none), that is sufficient.

Describe the endpoint like this:
{{"method": "{candidate['method']}", "path": "{candidate['path']}",
  "description": "one sentence",
  "base_url": "the API base URL if the context states it, else an empty string",
  "parameters": [{{"name": "...", "in": "path or query", "required": true, "type": "string"}}],
  "request_body": {{"content_type": "application/json", "fields": ["..."]}}}}
Leave out "request_body" if the endpoint takes no body. If a type is not stated use "string";
path parameters are required, query parameters are optional unless the context says otherwise.

If sufficient, return:
{{"sufficient": true, "endpoint": <the description above>}}

If NOT sufficient, return your best partial description anyway, plus what to search for next:
{{"sufficient": false, "endpoint": <the description above, as far as the context allows>,
  "missing": "what information is missing", "next_query": "a better search query to find it"}}

Return ONLY the JSON object.

Context:
{context}"""
    verdict = _parse_json(call_llm(prompt), {})
    if not isinstance(verdict, dict):
        verdict = {}
    partial = verdict.get("endpoint") if isinstance(verdict.get("endpoint"), dict) else None
    sufficient = bool(verdict.get("sufficient")) and partial is not None
    out_of_rounds = state["rag_round"] >= MAX_RETRIEVAL_ROUNDS
    name = f"{candidate['method']} {candidate['path']}"

    trace = state.get("rag_trace", [])
    trace.append({
        "endpoint": name, "round": state["rag_round"], "query": state["rag_query"],
        "sufficient": sufficient, "missing": verdict.get("missing"), "partial": partial,
    })
    state["rag_trace"] = trace

    if sufficient or out_of_rounds:
        if sufficient:
            endpoint = {**partial, "method": candidate["method"], "path": candidate["path"]}
            print("  GRADE: sufficient -> endpoint extracted")
        else:
            # Out of rounds. Use the best partial description any round produced, so
            # what WAS found is not thrown away; fall back to a bare definition only
            # if no round produced anything.
            partials = [t["partial"] for t in trace if t["endpoint"] == name and t.get("partial")]
            best = partials[-1] if partials else {}
            endpoint = {"description": name, "base_url": "", "parameters": [],
                        **best, "method": candidate["method"], "path": candidate["path"]}
            if not endpoint.get("description"):
                endpoint["description"] = name
            kind = "the best partial description" if partials else "a minimal definition"
            print(f"  GRADE: still insufficient after {MAX_RETRIEVAL_ROUNDS} rounds -> keeping {kind}")
        state["endpoints"] = state["endpoints"] + [endpoint]
        state["rag_cursor"] += 1
        state["rag_round"] = 0
        state["rag_query"] = None
        state["rag_next_query"] = None
        if state["rag_cursor"] >= len(state["rag_candidates"]):
            _fill_missing_base_urls(state["endpoints"])
    else:
        state["rag_next_query"] = verdict.get("next_query")
        print(f"  GRADE: insufficient (missing: {verdict.get('missing')})")
    return state


def _fill_missing_base_urls(endpoints):
    """The base URL is usually stated once, so some endpoints' context won't include it.
    Give those the base URL the other endpoints agreed on."""
    known = Counter(ep["base_url"] for ep in endpoints if ep.get("base_url"))
    if known:
        most_common = known.most_common(1)[0][0]
        for ep in endpoints:
            if not ep.get("base_url"):
                ep["base_url"] = most_common


def route_after_retrieval_grade(state: ForgeState) -> str:
    if state["rag_round"] > 0:
        return "rewrite"                                   # same endpoint, try a better query
    if state["rag_cursor"] < len(state["rag_candidates"]):
        return "next_endpoint"
    return "done"


def rewrite_query_node(state: ForgeState) -> ForgeState:
    candidate = state["rag_candidates"][state["rag_cursor"]]
    suggestion = (state.get("rag_next_query") or "").strip()
    if not suggestion or suggestion == state["rag_query"]:
        # The grader gave nothing new: fall back to a broader query of our own.
        suggestion = f"{candidate['path']} request parameters response example"
    state["rag_query"] = suggestion
    state["rag_next_query"] = None
    print(f"RAG 4 (rewrite_query) -> \"{suggestion}\"")
    return state