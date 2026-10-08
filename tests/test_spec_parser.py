"""The spec parser must give the generator everything it needs to build a correct request."""
from spec_parser import is_openapi_spec, extract_endpoints_from_openapi

PETSTORE_LIKE = {
    "openapi": "3.0.4",
    "servers": [{"url": "/api/v3"}],            # relative, exactly like the real Petstore spec
    "paths": {
        "/pet": {
            "post": {
                "summary": "Add a new pet",
                "requestBody": {"required": True, "content": {
                    "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}}},
            }
        },
        "/pet/{petId}": {
            "parameters": [{"name": "petId", "in": "path", "required": True, "schema": {"type": "integer"}}],
            "get": {"summary": "Find pet by ID"},
            "delete": {"summary": "Deletes a pet", "parameters": [{"$ref": "#/components/parameters/ApiKey"}]},
        },
    },
    "components": {
        "schemas": {"Pet": {"type": "object", "properties": {"name": {}, "status": {}}}},
        "parameters": {"ApiKey": {"name": "api_key", "in": "header", "required": False}},
    },
}
SOURCE = "https://petstore3.swagger.io/api/v3/openapi.json"


def by_key(endpoints):
    return {(e["method"], e["path"]): e for e in endpoints}


def test_relative_server_url_is_resolved_against_the_spec_url():
    endpoints = extract_endpoints_from_openapi(PETSTORE_LIKE, SOURCE)
    assert {e["base_url"] for e in endpoints} == {"https://petstore3.swagger.io/api/v3"}


def test_path_level_parameters_are_inherited_by_every_method():
    eps = by_key(extract_endpoints_from_openapi(PETSTORE_LIKE, SOURCE))
    assert eps[("GET", "/pet/{petId}")]["parameters"] == [
        {"name": "petId", "in": "path", "required": True, "type": "integer"}]


def test_ref_parameters_are_resolved():
    eps = by_key(extract_endpoints_from_openapi(PETSTORE_LIKE, SOURCE))
    names = [p["name"] for p in eps[("DELETE", "/pet/{petId}")]["parameters"]]
    assert names == ["petId", "api_key"]


def test_request_body_is_described():
    eps = by_key(extract_endpoints_from_openapi(PETSTORE_LIKE, SOURCE))
    assert eps[("POST", "/pet")]["request_body"] == {
        "content_type": "application/json", "required": True, "fields": ["name", "status"]}
    assert "request_body" not in eps[("GET", "/pet/{petId}")]


def test_swagger2_host_and_basepath():
    spec = {"swagger": "2.0", "host": "api.example.com", "basePath": "/v1", "schemes": ["https"],
            "paths": {"/ping": {"get": {"summary": "ping"}}}}
    assert extract_endpoints_from_openapi(spec)[0]["base_url"] == "https://api.example.com/v1"


def test_yaml_specs_are_detected_and_prose_is_not():
    yaml_spec = "openapi: 3.0.0\npaths:\n  /ping:\n    get:\n      summary: ping\n"
    assert is_openapi_spec(yaml_spec)["paths"]["/ping"]["get"]["summary"] == "ping"
    assert is_openapi_spec("GET /ping - health check endpoint") is None
