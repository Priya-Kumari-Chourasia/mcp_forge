"""
Deterministic OpenAPI / Swagger parsing - no LLM involved.

A structured spec already IS the answer, so asking an LLM to "extract" it
would only add cost and a chance of mistakes. This file walks the spec and
produces the same endpoint dictionaries the LLM path produces for prose docs.

Besides method, path and parameters, it keeps what makes a GOOD tool, not
just a working one:
  - parameter descriptions and allowed values (enums), so the AI client knows
    what to pass
  - whether the endpoint returns a JSON array or a JSON object
  - the API's title and its authentication schemes, so the generated server
    can read credentials from environment variables instead of asking the AI
    for them
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


# ----------------------------------------------------------------------
# Authentication
# ----------------------------------------------------------------------

def get_auth_schemes(spec: dict) -> list:
    """The ways this API accepts credentials, normalised to five types:

        header / query / cookie   an API key sent under `name`
        bearer                    "Authorization: Bearer <token>" (also used for OAuth2:
                                  the user supplies an access token they already have)
        basic                     "Authorization: Basic ..." from a username and password

    OpenAPI 3 lists them under components.securitySchemes, Swagger 2 under
    securityDefinitions. Schemes we cannot send (for example HTTP digest) are
    skipped rather than half-supported."""
    raw = (spec.get("components") or {}).get("securitySchemes") or spec.get("securityDefinitions") or {}
    schemes = []
    for key, scheme in raw.items():
        scheme = _resolve_ref(spec, scheme) or {}
        kind = (scheme.get("type") or "").lower()
        if kind == "apikey" and scheme.get("in") in ("header", "query", "cookie") and scheme.get("name"):
            schemes.append({"scheme": key, "type": scheme["in"], "name": scheme["name"]})
        elif kind == "http" and (scheme.get("scheme") or "").lower() == "bearer":
            schemes.append({"scheme": key, "type": "bearer"})
        elif (kind == "http" and (scheme.get("scheme") or "").lower() == "basic") or kind == "basic":
            schemes.append({"scheme": key, "type": "basic"})
        elif kind in ("oauth2", "openidconnect"):
            schemes.append({"scheme": key, "type": "bearer"})
    return schemes


def _is_credential_parameter(param: dict, auth_schemes: list) -> bool:
    """True for a parameter that carries a credential. Some specs declare the
    API key BOTH as a security scheme and as an ordinary header parameter (the
    Petstore's delete endpoint does). As a tool argument the AI would have to
    supply the secret, so those parameters are dropped: the generated server
    adds credentials itself, from environment variables."""
    name = (param.get("name") or "").lower()
    where = param.get("in")
    if where == "header" and name == "authorization":
        return True
    return any(s["type"] == where and s["name"].lower() == name
               for s in auth_schemes if s["type"] in ("header", "query", "cookie"))


def get_api_meta(spec: dict, source_url: str = None) -> dict:
    """Facts about the API as a whole, used to build the server around the tools."""
    return {
        "title": ((spec.get("info") or {}).get("title") or "").strip(),
        "base_url": get_base_url(spec, source_url),
        "auth": get_auth_schemes(spec),
    }


# ----------------------------------------------------------------------
# Parameters, bodies and responses
# ----------------------------------------------------------------------

def _describe_parameter(spec: dict, p: dict) -> dict:
    """One parameter as the code generator sees it. Optional keys (description,
    enum, items) are only added when the spec states them."""
    schema = _resolve_ref(spec, p.get("schema") or {})
    param = {
        "name": p.get("name"),
        "in": p.get("in"),
        "required": bool(p.get("required", False)) or p.get("in") == "path",
        "type": schema.get("type") or p.get("type") or "string",
    }
    description = (p.get("description") or schema.get("description") or "").strip()
    if description:
        param["description"] = description
    enum = schema.get("enum") or p.get("enum")
    if enum:
        param["enum"] = enum
    if param["type"] == "array":
        items = _resolve_ref(spec, schema.get("items") or p.get("items") or {})
        param["items"] = items.get("type") or "string"
        if items.get("enum"):
            param["enum"] = items["enum"]          # an array of allowed values
    return param


def _body_from_schema(spec: dict, schema: dict, content_type: str, required: bool) -> dict:
    schema = _resolve_ref(spec, schema or {})
    # Some bodies are a LIST of objects (Petstore's /user/createWithList). The fields
    # described are then those of each item, and the tool's body argument is a list.
    is_array = schema.get("type") == "array"
    if is_array:
        schema = _resolve_ref(spec, schema.get("items") or {})
    properties = schema.get("properties") or {}
    required_fields = set(schema.get("required") or [])
    body = {
        "content_type": content_type,
        "required": bool(required),
        "fields": list(properties.keys()),
    }
    details = []
    for name, prop in properties.items():
        prop = _resolve_ref(spec, prop) or {}
        detail = {"name": name, "type": prop.get("type") or ("object" if prop.get("properties") else "string")}
        if name in required_fields:
            detail["required"] = True
        if prop.get("description"):
            detail["description"] = prop["description"].strip()
        if prop.get("enum"):
            detail["enum"] = prop["enum"]
        details.append(detail)
    if details:
        body["field_details"] = details
    if is_array:
        body["type"] = "array"
    return body


def _request_body(spec: dict, details: dict, params: list):
    """Describes the request body, if the endpoint takes one."""
    # OpenAPI 3
    body = _resolve_ref(spec, details.get("requestBody"))
    if body:
        content = body.get("content") or {}
        content_type = "application/json" if "application/json" in content else next(iter(content), "application/json")
        schema = (content.get(content_type) or {}).get("schema") or {}
        return _body_from_schema(spec, schema, content_type, body.get("required", False))

    # Swagger 2: the body is a parameter with in == "body"...
    for p in params:
        if p.get("in") == "body":
            return _body_from_schema(spec, p.get("schema") or {}, "application/json", p.get("required", False))

    # ...or a set of form fields, each a parameter with in == "formData".
    form = [p for p in params if p.get("in") == "formData"]
    if form:
        schema = {"properties": {p["name"]: {"type": p.get("type") or "string",
                                             "description": p.get("description") or ""} for p in form},
                  "required": [p["name"] for p in form if p.get("required")]}
        return _body_from_schema(spec, schema, "application/x-www-form-urlencoded",
                                 any(p.get("required") for p in form))
    return None


def _response_shape(spec: dict, details: dict):
    """'array' or 'object' for the endpoint's success response, None if the spec
    does not say. The validator shapes its fake responses to match, so a tool
    that cannot return a list is caught before a real API ever returns one."""
    responses = details.get("responses") or {}
    for code in ("200", "201", "202", "2XX", "default"):
        response = _resolve_ref(spec, responses.get(code) or responses.get(int(code) if code.isdigit() else code))
        if not response:
            continue
        schema = response.get("schema")                                   # Swagger 2
        if schema is None:                                                # OpenAPI 3
            content = response.get("content") or {}
            media = content.get("application/json") or next(iter(content.values()), {})
            schema = (media or {}).get("schema")
        schema = _resolve_ref(spec, schema or {})
        if schema.get("type") == "array":
            return "array"
        if schema.get("type") == "object" or schema.get("properties"):
            return "object"
        return None
    return None


def extract_endpoints_from_openapi(spec: dict, source_url: str = None):
    """Walks the spec's 'paths' object and returns one dict per endpoint."""
    base_url = get_base_url(spec, source_url)
    auth_schemes = get_auth_schemes(spec)
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
                _describe_parameter(spec, p)
                for p in merged.values()
                if p.get("in") not in ("body", "formData") and not _is_credential_parameter(p, auth_schemes)
            ]

            endpoint = {
                "method": method.upper(),
                "path": path,
                "base_url": base_url,
                "description": details.get("summary") or details.get("description") or f"{method.upper()} {path}",
                "parameters": params,
            }
            if details.get("operationId"):
                endpoint["operation_id"] = details["operationId"]     # becomes the tool's name
            if details.get("tags"):
                endpoint["tags"] = details["tags"]                    # lets the user pick endpoints by tag
            body = _request_body(spec, details, list(merged.values()))
            if body:
                endpoint["request_body"] = body
            returns = _response_shape(spec, details)
            if returns:
                endpoint["returns"] = returns
            endpoints.append(endpoint)
    return endpoints
