import json

def is_openapi_spec(raw_docs: str):
    """Returns the parsed spec dict if raw_docs is genuinely an OpenAPI/
    Swagger JSON spec, otherwise None."""
    try:
        data = json.loads(raw_docs)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict) and ("openapi" in data or "swagger" in data) and "paths" in data:
        return data
    return None

def extract_endpoints_from_openapi(spec: dict):
    """Deterministically walks a spec's 'paths' object - no LLM guessing,
    since the spec is already structured data."""
    base_url = ""
    if spec.get("servers"):
        base_url = spec["servers"][0].get("url", "")
    endpoints = []
    for path, methods in spec.get("paths", {}).items():
        for method, details in methods.items():
            if method.lower() not in ("get", "post", "put", "delete", "patch"):
                continue
            params = [
                {"name": p.get("name"), "required": p.get("required", False), "in": p.get("in")}
                for p in details.get("parameters", [])
            ]
            endpoints.append({
                "method": method.upper(), "path": path, "base_url": base_url,
                "description": details.get("summary") or details.get("description") or f"{method.upper()} {path}",
                "parameters": params,
            })
    return endpoints