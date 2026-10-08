"""
Deterministic OpenAPI / Swagger parsing - no LLM involved.

A structured spec already IS the answer, so asking an LLM to "extract" it
would only add cost and a chance of mistakes. This file walks the spec and
produces the same endpoint dictionaries the LLM path produces for prose docs.
"""

import json
from urllib.parse import urljoin

import yaml

HTTP_METHODS = ("get", "post", "put", "delete", "patch")


def is_openapi_spec(raw_docs: str):
    """Returns the parsed spec dict if raw_docs is an OpenAPI/Swagger spec
    (JSON or YAML), otherwise None."""
    data = None
    try:
        data = json.loads(raw_docs)
    except (json.JSONDecodeError, TypeError):
        try:
            data = yaml.safe_load(raw_docs)
        except Exception:
            return None
    if isinstance(data, dict) and ("openapi" in data or "swagger" in data) and "paths" in data:
        return data
    return None


def _resolve_ref(spec: dict, obj):
    """Follows a local '$ref': '#/components/...' pointer one hop.
    Anything that isn't a local reference is returned unchanged."""
    if not isinstance(obj, dict) or "$ref" not in obj:
        return obj
    ref = obj["$ref"]
    if not ref.startswith("#/"):
        return obj
    target = spec
    for part in ref[2:].split("/"):
        if not isinstance(target, dict) or part not in target:
            return obj
        target = target[part]
    return target


def get_base_url(spec: dict, source_url: str = None) -> str:
    """Works out the absolute base URL every endpoint path is appended to.

    OpenAPI 3 keeps it in servers[0].url, and that value is allowed to be
    RELATIVE (the Petstore spec says just "/api/v3"). A relative URL only
    makes sense relative to where the spec was downloaded from, so we
    resolve it against source_url. Swagger 2 uses schemes + host + basePath.
    """
    base_url = ""
    if spec.get("servers"):
        server = spec["servers"][0]
        base_url = server.get("url", "")
        for name, var in (server.get("variables") or {}).items():
            base_url = base_url.replace("{" + name + "}", str(var.get("default", "")))
    elif spec.get("host"):
        scheme = (spec.get("schemes") or ["https"])[0]
        base_url = f"{scheme}://{spec['host']}{spec.get('basePath', '')}"

    if not base_url.startswith(("http://", "https://")) and source_url:
        base_url = urljoin(source_url, base_url)
    return base_url.rstrip("/")


def _request_body(spec: dict, details: dict, params: list):
    """Describes the request body, if the endpoint takes one."""
    # OpenAPI 3
    body = _resolve_ref(spec, details.get("requestBody"))
    if body:
        content = body.get("content") or {}
        content_type = "application/json" if "application/json" in content else next(iter(content), "application/json")
        schema = _resolve_ref(spec, (content.get(content_type) or {}).get("schema") or {})
        return {
            "content_type": content_type,
            "required": body.get("required", False),
            "fields": list((schema.get("properties") or {}).keys()),
        }
    # Swagger 2: the body is a parameter with in == "body"
    for p in params:
        if p.get("in") == "body":
            schema = _resolve_ref(spec, p.get("schema") or {})
            return {
                "content_type": "application/json",
                "required": p.get("required", False),
                "fields": list((schema.get("properties") or {}).keys()),
            }
    return None


def extract_endpoints_from_openapi(spec: dict, source_url: str = None):
    """Walks the spec's 'paths' object and returns one dict per endpoint."""
    base_url = get_base_url(spec, source_url)
    endpoints = []
    for path, path_item in spec.get("paths", {}).items():
        path_item = _resolve_ref(spec, path_item)
        # Parameters can be declared once on the path and shared by all its methods.
        shared_params = [_resolve_ref(spec, p) for p in path_item.get("parameters", [])]

        for method, details in path_item.items():
            if method.lower() not in HTTP_METHODS:
                continue
            own_params = [_resolve_ref(spec, p) for p in details.get("parameters", [])]

            # Method-level parameters override path-level ones with the same name + location.
            merged = {(p.get("name"), p.get("in")): p for p in shared_params + own_params}
            params = [
                {
                    "name": p.get("name"),
                    "in": p.get("in"),
                    "required": p.get("required", False),
                    "type": (p.get("schema") or {}).get("type") or p.get("type") or "string",
                }
                for p in merged.values() if p.get("in") != "body"
            ]

            endpoint = {
                "method": method.upper(),
                "path": path,
                "base_url": base_url,
                "description": details.get("summary") or details.get("description") or f"{method.upper()} {path}",
                "parameters": params,
            }
            body = _request_body(spec, details, list(merged.values()))
            if body:
                endpoint["request_body"] = body
            endpoints.append(endpoint)
    return endpoints
