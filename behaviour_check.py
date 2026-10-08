"""
BEHAVIOUR CHECK - run by the validator in a separate process.

Static checks can tell us the generated code is valid Python. They cannot
tell us whether a tool, when called, would hit the right URL. This script
answers that question WITHOUT touching the real API:

  1. Replace the function `requests` uses to send HTTP with a recorder.
     Nothing leaves the machine; we just note (method, url) and hand back
     a fake 200 response.
  2. Call every generated function with dummy arguments.
  3. Compare what each function tried to call against the endpoints the
     docs describe.
  4. Register every function on a real MCP server and list its tools, to
     confirm the code works as MCP tools and not just as Python functions.

So a generated delete_pet() is exercised safely: we learn it would send
DELETE https://.../pet/x, and no pet is deleted.

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

import requests

DUMMY_BASE_URL = "https://api.test.invalid"   # used when the docs give no base URL

recorded_calls = []

# One asyncio event loop for the whole check, created in main() BEFORE the
# network is blocked. On Windows, starting an event loop opens a local socket
# pair, which the network block would otherwise refuse.
_loop = None


class FakeResponse:
    status_code = 200
    ok = True
    text = "{}"
    content = b"{}"
    headers = {"Content-Type": "application/json"}

    def json(self):
        return {}

    def raise_for_status(self):
        return None


def _record_instead_of_sending(self, method, url, **kwargs):
    recorded_calls.append((str(method).upper(), str(url)))
    return FakeResponse()


def _no_network(*args, **kwargs):
    raise RuntimeError("real network access is disabled during validation")


def dummy_value(param: inspect.Parameter):
    """A harmless argument of roughly the right type."""
    by_type = {int: 1, float: 1.0, bool: True, dict: {}, list: [], str: "x"}
    return by_type.get(param.annotation, "x")


def call_with_dummy_args(func):
    """Fills every REQUIRED parameter; optional ones keep their defaults."""
    args, kwargs = [], {}
    for param in inspect.signature(func).parameters.values():
        if param.default is not inspect.Parameter.empty:
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if param.kind == param.KEYWORD_ONLY:
            kwargs[param.name] = dummy_value(param)
        else:
            args.append(dummy_value(param))
    result = func(*args, **kwargs)
    if inspect.iscoroutine(result):
        _loop.run_until_complete(result)


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


def check_mcp_registration(module, function_names):
    """Registers every function on a real MCPServer, exactly as the packaged
    server.py will, then asks the server for its tool list.

    A function can be valid Python and send the right request, yet still be
    unusable as an MCP tool: the SDK builds each tool's input and output
    schema from the type hints, so a hint that is not a real type (for
    example `-> any` instead of `-> Any`) makes registration crash."""
    from mcp.server.mcpserver import MCPServer

    errors = []
    server = MCPServer("validation")
    registered = []
    for name in function_names:
        try:
            server.tool()(getattr(module, name))
            registered.append(name)
        except Exception as e:
            reason = str(e).strip().splitlines()[0]
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() cannot be registered as an MCP tool - check its type hints "
                                    f"({type(e).__name__}: {reason})"})

    listed = {tool.name for tool in _loop.run_until_complete(server.list_tools())}
    for name in registered:
        if name not in listed:
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() was registered but is missing from the MCP server's tool list"})
    return errors


def main(code_path: str, endpoints_path: str, result_path: str):
    with open(endpoints_path, encoding="utf-8") as f:
        endpoints = json.load(f)
    with open(code_path, encoding="utf-8") as f:
        source = f.read()

    errors = []          # each: {"function": name or None, "endpoint_index": int or None, "error": text}
    calls_by_function = {}

    global _loop
    _loop = asyncio.new_event_loop()

    # Swap out real HTTP before any generated code runs.
    requests.sessions.Session.request = _record_instead_of_sending
    socket.socket.connect = _no_network
    socket.create_connection = _no_network
    os.environ.setdefault("API_BASE_URL", DUMMY_BASE_URL)

    try:
        spec = importlib.util.spec_from_file_location("generated_tools", code_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        errors.append({"function": None, "endpoint_index": None,
                       "error": f"{type(e).__name__} while loading the code: {e}"})
        _write(result_path, errors, calls_by_function)
        return

    function_names = [n.name for n in ast.parse(source).body
                      if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    covered = set()

    for name in function_names:
        recorded_calls.clear()
        try:
            call_with_dummy_args(getattr(module, name))
        except Exception as e:
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() raised {type(e).__name__}: {e}"})
            continue

        calls_by_function[name] = list(recorded_calls)
        if not recorded_calls:
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() made no HTTP request through the `requests` library"})
            continue

        method, url = recorded_calls[0]
        if not url.startswith(("http://", "https://")):
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() called '{url}', which is not an absolute URL "
                                    f"(it must start with the base_url)"})
            continue

        index = find_endpoint(method, url, endpoints)
        if index is None:
            errors.append({"function": name, "endpoint_index": None,
                           "error": f"{name}() sent {method} {url}, which matches no documented endpoint "
                                    f"(check the HTTP method, the base_url and the path)"})
        else:
            covered.add(index)

    errors.extend(check_mcp_registration(module, function_names))

    for i, ep in enumerate(endpoints):
        if i not in covered:
            errors.append({"function": None, "endpoint_index": i,
                           "error": f"no working tool for endpoint {ep.get('method')} {ep.get('path')}"})

    _write(result_path, errors, calls_by_function)


def _write(result_path, errors, calls_by_function):
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({"errors": errors, "calls": calls_by_function}, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])