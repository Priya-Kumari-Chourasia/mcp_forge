"""
BEHAVIOUR CHECK - run by the validator in a separate process.

Static checks can tell us the generated code is valid Python. They cannot
tell us whether a tool, when an AI client calls it, would send the right
request and hand back a result the client accepts. This script answers that
WITHOUT touching the real API:

  1. Replace the function httpx uses to send a request with a recorder.
     Nothing leaves the machine; the request is recorded and a fake 200
     response comes back, shaped like the docs say (a JSON array or object).
  2. Register every tool on a real MCP server, exactly as server.py will.
  3. Call every tool THROUGH that server (server.call_tool), with dummy
     arguments built from the tool's own input schema. This runs the same
     argument checks and return-value checks a real client triggers - a tool
     annotated `-> Union[dict, str]` that gets a JSON array back fails here,
     the way find-by-status failed in Claude Desktop.
  4. Compare each request with the endpoint the docs describe: method, path,
     the required query parameters and the request body - and check the tool
     has an argument for every documented parameter.

So a generated delete_pet() is exercised safely: we learn it would send
DELETE https://.../pet/1, and no pet is deleted.

Usage: python behaviour_check.py <code.py> <endpoints.json> <result.json>
The result file holds {"errors": [...], "calls": {...}}.
"""

import ast
import asyncio
import importlib.util
import inspect
import json
import os
import re
import socket
import sys
from urllib.parse import parse_qs, urlsplit

import httpx
import requests

DUMMY_BASE_URL = "https://api.test.invalid"   # used when the docs give no base URL

recorded_calls = []      # each: {"library", "method", "url", "body"}
ENDPOINTS = []           # set in main(); the recorder uses it to shape fake responses

# One asyncio event loop for the whole check, created in main() BEFORE the
# network is blocked. On Windows, starting an event loop opens a local socket
# pair, which the network block would otherwise refuse.
_loop = None


# ----------------------------------------------------------------------
# Fake HTTP
# ----------------------------------------------------------------------

def _fake_payload(method, url):
    """A JSON array for endpoints documented to return one, otherwise a JSON object."""
    index = find_endpoint(method, url, ENDPOINTS)
    if index is not None and ENDPOINTS[index].get("returns") == "array":
        return [{"id": 1}]
    return {"id": 1}


def _record_httpx(self, request, **kwargs):
    """Stands in for httpx.Client.send: records the request instead of sending it."""
    method, url = request.method.upper(), str(request.url)
    recorded_calls.append({"library": "httpx", "method": method, "url": url, "body": request.content})
    return httpx.Response(200, json=_fake_payload(method, url), request=request)


class FakeRequestsResponse:
    status_code = 200
    ok = True
    text = "{}"
    content = b"{}"
    headers = {"Content-Type": "application/json"}

    def json(self):
        return {}

    def raise_for_status(self):
        return None


def _record_requests(self, method, url, **kwargs):
    """Stands in for requests: a tool should not use it (see the check below)."""
    recorded_calls.append({"library": "requests", "method": str(method).upper(), "url": str(url), "body": b""})
    return FakeRequestsResponse()


def _no_network(*args, **kwargs):
    raise RuntimeError("real network access is disabled during validation")


def _no_processes(*args, **kwargs):
    raise RuntimeError("starting other programs is disabled during validation")


def _block_process_spawning():
    """Generated tools only need to build an HTTP request, so they have no reason
    to start other programs. Blocking it stops code from escaping the network
    block by launching a fresh, unpatched process. (This raises the bar; it is
    not a complete sandbox.)"""
    import subprocess
    # The MCP SDK refers to subprocess.Popen when it is imported, so import it
    # (for the registration check later) BEFORE Popen is replaced.
    import mcp.server.mcpserver  # noqa: F401
    subprocess.Popen = _no_processes
    for name in ("system", "popen", "execv", "execve", "execvp", "execvpe", "execl", "execle",
                 "execlp", "execlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "spawnl",
                 "spawnle", "spawnlp", "spawnlpe", "fork", "forkpty", "posix_spawn", "posix_spawnp",
                 "startfile"):
        if hasattr(os, name):
            setattr(os, name, _no_processes)


# ----------------------------------------------------------------------
# Matching a request to a documented endpoint
# ----------------------------------------------------------------------

def endpoint_pattern(endpoint: dict):
    """Regex for the full URL of an endpoint; {petId} matches any one path segment."""
    path = re.escape((endpoint.get("path") or "").rstrip("/"))
    path = re.sub(r"\\\{[^}]*\\\}", "[^/?#]+", path)      # {petId} style
    path = re.sub(r"(?<=/):\w+", "[^/?#]+", path)            # :petId style (common in prose docs)
    base = (endpoint.get("base_url") or "").rstrip("/")
    if base.startswith(("http://", "https://")):
        return re.compile("^" + re.escape(base) + path + "$")
    return re.compile("^" + re.escape(DUMMY_BASE_URL) + path + "$")


def find_endpoint(method: str, url: str, endpoints: list):
    """Index of the endpoint this call belongs to, or None.
    /pet/findByStatus matches both '/pet/findByStatus' and '/pet/{petId}',
    so when several match we prefer the one with the fewest {placeholders}."""
    clean_url = url.split("?")[0].split("#")[0].rstrip("/")
    matches = [
        i for i, ep in enumerate(endpoints)
        if ep.get("method", "").upper() == method and endpoint_pattern(ep).match(clean_url)
    ]
    if not matches:
        return None
    return min(matches, key=lambda i: endpoints[i]["path"].count("{") + endpoints[i]["path"].count(":"))


# ----------------------------------------------------------------------
# Calling a tool the way an MCP client does
# ----------------------------------------------------------------------

def dummy_value(schema: dict):
    """A harmless argument that satisfies one property of a tool's input schema."""
    if not isinstance(schema, dict):
        return "x"
    if "const" in schema:
        return schema["const"]
    if schema.get("enum"):
        return schema["enum"][0]
    for key in ("anyOf", "oneOf"):
        if schema.get(key):
            options = [s for s in schema[key] if s.get("type") != "null"] or schema[key]
            return dummy_value(options[0])
    kind = schema.get("type")
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return True
    if kind == "array":
        return [dummy_value(schema.get("items") or {"type": "string"})]
    if kind == "object" or "$ref" in schema:
        return {}
    return "x"


def dummy_arguments(input_schema: dict) -> dict:
    """Fills every REQUIRED argument; optional ones keep their defaults."""
    properties = input_schema.get("properties") or {}
    return {name: dummy_value(properties.get(name, {})) for name in input_schema.get("required") or []}


def _root_cause(error):
    while error.__cause__ is not None:
        error = error.__cause__
    return error


def describe_failure(name: str, error) -> str:
    """Turns what the MCP SDK raised into a message the LLM can act on."""
    root = _root_cause(error)
    title = getattr(root, "title", "") or ""
    if title.endswith("Output"):
        returned = "a JSON array" if "input_value=[" in str(root) else "a value"
        return (f"{name}() returned {returned} that its return type does not allow "
                f"(annotate it -> Union[dict, list, str])")
    if title.endswith("Arguments"):
        first = str(root).splitlines()
        return f"{name}() rejected arguments that match its own input schema: {' '.join(first[:3])}"
    return f"{name}() raised {type(root).__name__}: {root}"


def _normalise(name: str) -> str:
    """petId, pet_id and pet-id all become petid, so a snake_case argument
    matches the documented camelCase name."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def check_request(name, func, call, endpoint) -> list:
    """Checks the request a tool sent against the endpoint it matched."""
    errors = []
    query = parse_qs(urlsplit(call["url"]).query, keep_blank_values=True)
    for p in endpoint.get("parameters") or []:
        if p.get("in") == "query" and p.get("required") and p["name"] not in query:
            errors.append(f"{name}() did not send the required query parameter '{p['name']}'")
    body = endpoint.get("request_body") or {}
    if body.get("required") and not call["body"]:
        errors.append(f"{name}() did not send the required request body")

    # Every documented parameter needs an argument, or the AI can never set it.
    arguments = {_normalise(a) for a in inspect.signature(func).parameters}
    for p in endpoint.get("parameters") or []:
        if p.get("in") in ("path", "query", "header") and _normalise(p["name"]) not in arguments:
            errors.append(f"{name}() has no argument for the documented {p['in']} parameter '{p['name']}'")
    if body and "body" not in arguments:
        errors.append(f"{name}() has no `body` argument for the endpoint's request body")
    return errors


def register_tools(server, module, function_names):
    """Registers every function on the MCP server, exactly as the packaged
    server.py will. A function can be valid Python yet unusable as an MCP tool:
    the SDK builds each tool's input and output schema from the type hints, so
    a hint that is not a real type (for example `-> any` instead of `-> Any`)
    makes registration crash."""
    errors, registered = [], []
    for name in function_names:
        try:
            server.tool()(getattr(module, name))
            registered.append(name)
        except Exception as e:
            reason = str(e).strip().splitlines()[0]
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() cannot be registered as an MCP tool - check its type hints "
                                    f"({type(e).__name__}: {reason})"})
    return registered, errors


def main(code_path: str, endpoints_path: str, result_path: str):
    global _loop
    with open(endpoints_path, encoding="utf-8") as f:
        ENDPOINTS[:] = json.load(f)
    with open(code_path, encoding="utf-8") as f:
        source = f.read()

    errors = []          # each: {"function": name or None, "endpoint_index": int or None, "error": text}
    calls_by_function = {}
    _loop = asyncio.new_event_loop()

    # Swap out real HTTP before any generated code runs.
    httpx.Client.send = _record_httpx
    requests.sessions.Session.request = _record_requests
    socket.socket.connect = _no_network
    socket.create_connection = _no_network
    os.environ.setdefault("API_BASE_URL", DUMMY_BASE_URL)
    _block_process_spawning()

    try:
        spec = importlib.util.spec_from_file_location("generated_tools", code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        errors.append({"function": None, "endpoint_index": None,
                       "error": f"{type(e).__name__} while loading the code: {e}"})
        _write(result_path, errors, calls_by_function)
        return

    # The tools: public top-level functions. Names starting with _ (like _request) are helpers.
    function_names = [n.name for n in ast.parse(source).body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not n.name.startswith("_")]

    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer("validation")
    registered, registration_errors = register_tools(server, module, function_names)
    errors.extend(registration_errors)

    listed = {tool.name: tool for tool in _loop.run_until_complete(server.list_tools())}
    covered = set()

    for name in registered:
        if name not in listed:
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() was registered but is missing from the MCP server's tool list"})
            continue

        recorded_calls.clear()
        try:
            _loop.run_until_complete(server.call_tool(name, dummy_arguments(listed[name].input_schema or {})))
        except ToolError as e:
            errors.append({"function": name, "endpoint_index": None, "error": describe_failure(name, e)})
            continue
        except Exception as e:
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() raised {type(e).__name__}: {e}"})
            continue

        calls_by_function[name] = [(c["method"], c["url"]) for c in recorded_calls]
        if not recorded_calls:
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() made no HTTP request (it must call _request)"})
            continue

        call = recorded_calls[0]
        if call["library"] == "requests":
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() used the requests library directly; call _request(...) instead, "
                                    f"which adds the base URL, authentication and error handling"})
            continue
        if not call["url"].startswith(("http://", "https://")):
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() called '{call['url']}', which is not an absolute URL"})
            continue

        index = find_endpoint(call["method"], call["url"], ENDPOINTS)
        if index is None:
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() sent {call['method']} {call['url']}, which matches no documented "
                                    f"endpoint (check the HTTP method and the path)"})
            continue

        # The endpoint HAS a tool; any problems with it are reported against the tool,
        # not repeated as "no working tool for endpoint ...".
        covered.add(index)
        for problem in check_request(name, getattr(module, name), call, ENDPOINTS[index]):
            errors.append({"function": name, "endpoint_index": None, "error": problem})

    for i, ep in enumerate(ENDPOINTS):
        if i not in covered:
            errors.append({"function": None, "endpoint_index": i,
                           "error": f"no working tool for endpoint {ep.get('method')} {ep.get('path')}"})

    _write(result_path, errors, calls_by_function)


def _write(result_path, errors, calls_by_function):
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"errors": errors, "calls": calls_by_function}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
