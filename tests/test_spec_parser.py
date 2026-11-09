"""The spec parser must give the generator everything it needs to build a correct request."""
from spec_parser import is_openapi_spec, extract_endpoints_from_openapi, get_api_meta

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
        "content_type": "application/json", "required": True, "fields": ["name", "status"],
        "field_details": [{"name": "name", "type": "string"}, {"name": "status", "type": "string"}]}
    assert "request_body" not in eps[("GET", "/pet/{petId}")]


def test_swagger2_host_and_basepath():
    spec = {"swagger": "2.0", "host": "api.example.com", "basePath": "/v1", "schemes": ["https"],
            "paths": {"/ping": {"get": {"summary": "ping"}}}}
    assert extract_endpoints_from_openapi(spec)[0]["base_url"] == "https://api.example.com/v1"


SPEC_FILE = "https://raw.githubusercontent.com/example/weather/main/openapi/forecast.yml"


def test_servers_declared_on_the_path_are_used():
    """Open-Meteo's specs put `servers` under each path, not at the top of the file."""
    spec = {"openapi": "3.0.0", "info": {"title": "Weather"},
            "paths": {"/v1/forecast": {"servers": [{"url": "https://api.weather.test"}],
                                       "get": {"summary": "forecast"}}}}
    assert get_api_meta(spec, SPEC_FILE)["base_url"] == "https://api.weather.test"
    assert extract_endpoints_from_openapi(spec, SPEC_FILE)[0]["base_url"] == "https://api.weather.test"

    on_the_operation = {"openapi": "3.0.0", "paths": {"/v1/forecast": {"get": {
        "summary": "forecast", "servers": [{"url": "https://{region}.weather.test",
                                            "variables": {"region": {"default": "eu"}}}]}}}}
    assert get_api_meta(on_the_operation, SPEC_FILE)["base_url"] == "https://eu.weather.test"


def test_with_no_server_the_spec_files_own_url_is_never_the_base_url():
    """No server anywhere means "the host that serves the spec" (OpenAPI 3: url "/").
    It used to come out as the spec file's full URL, so every tool called .../forecast.yml/v1/..."""
    spec = {"openapi": "3.0.0", "paths": {"/ping": {"get": {"summary": "ping"}}}}
    assert get_api_meta(spec, SPEC_FILE)["base_url"] == "https://raw.githubusercontent.com"
    assert get_api_meta(spec)["base_url"] == ""         # nothing to resolve against: the server will ask for one
    swagger = {"swagger": "2.0", "basePath": "/v1", "paths": {"/ping": {"get": {"summary": "ping"}}}}
    assert get_api_meta(swagger, "https://docs.example.com/specs/api.json")["base_url"] == "https://docs.example.com/v1"


def test_yaml_specs_are_detected_and_prose_is_not():
    yaml_spec = "openapi: 3.0.0\npaths:\n  /ping:\n    get:\n      summary: ping\n"
    assert is_openapi_spec(yaml_spec)["paths"]["/ping"]["get"]["summary"] == "ping"
    assert is_openapi_spec("GET /ping - health check endpoint") is None


# ----------------------------------------------------------------------
# What makes a tool usable by an AI client, not just runnable
# ----------------------------------------------------------------------

DETAILED = {
    "openapi": "3.0.0",
    "info": {"title": "Swagger Petstore - OpenAPI 3.0"},
    "servers": [{"url": "https://petstore.test/v3"}],
    "paths": {
        "/pet/findByStatus": {"get": {
            "operationId": "findPetsByStatus", "tags": ["pet"], "summary": "Finds pets by status",
            "parameters": [{"name": "status", "in": "query", "required": True, "description": "Status to filter by",
                            "schema": {"type": "string", "enum": ["available", "pending", "sold"]}}],
            "responses": {"200": {"content": {"application/json": {
                "schema": {"type": "array", "items": {"$ref": "#/components/schemas/Pet"}}}}}},
        }},
        "/pet/{petId}": {"delete": {
            "operationId": "deletePet", "summary": "Deletes a pet",
            "parameters": [{"name": "api_key", "in": "header", "schema": {"type": "string"}},
                           {"name": "petId", "in": "path", "required": True, "schema": {"type": "integer"}}],
        }},
        "/store/inventory": {"get": {
            "summary": "Inventory", "responses": {"200": {"content": {"application/json": {
                "schema": {"type": "object", "additionalProperties": {"type": "integer"}}}}}},
        }},
    },
    "components": {
        "schemas": {"Pet": {"type": "object", "properties": {"name": {"type": "string"}}}},
        "securitySchemes": {
            "petstore_auth": {"type": "oauth2", "flows": {}},
            "api_key": {"type": "apiKey", "name": "api_key", "in": "header"},
        },
    },
}


def test_parameter_descriptions_and_allowed_values_are_kept():
    eps = by_key(extract_endpoints_from_openapi(DETAILED))
    assert eps[("GET", "/pet/findByStatus")]["parameters"] == [
        {"name": "status", "in": "query", "required": True, "type": "string",
         "description": "Status to filter by", "enum": ["available", "pending", "sold"]}]


def test_the_response_shape_is_recorded():
    eps = by_key(extract_endpoints_from_openapi(DETAILED))
    assert eps[("GET", "/pet/findByStatus")]["returns"] == "array"
    assert eps[("GET", "/store/inventory")]["returns"] == "object"
    assert "returns" not in eps[("DELETE", "/pet/{petId}")]          # the spec does not say


def test_operation_ids_and_tags_are_kept():
    ep = by_key(extract_endpoints_from_openapi(DETAILED))[("GET", "/pet/findByStatus")]
    assert ep["operation_id"] == "findPetsByStatus" and ep["tags"] == ["pet"]


def test_auth_schemes_are_read_and_credentials_are_never_tool_arguments():
    meta = get_api_meta(DETAILED)
    assert meta["title"] == "Swagger Petstore - OpenAPI 3.0"
    assert meta["auth"] == [{"scheme": "petstore_auth", "type": "bearer"},
                            {"scheme": "api_key", "type": "header", "name": "api_key"}]
    # The api_key header is a credential, so it is not offered to the AI as an argument.
    delete = by_key(extract_endpoints_from_openapi(DETAILED))[("DELETE", "/pet/{petId}")]
    assert [p["name"] for p in delete["parameters"]] == ["petId"]


def test_swagger2_form_fields_become_a_form_body():
    spec = {"swagger": "2.0", "host": "api.example.com", "paths": {"/pet/{petId}": {"post": {
        "summary": "Update with form",
        "parameters": [{"name": "petId", "in": "path", "required": True, "type": "integer"},
                       {"name": "name", "in": "formData", "type": "string", "required": True}]}}}}
    ep = extract_endpoints_from_openapi(spec)[0]
    assert [p["name"] for p in ep["parameters"]] == ["petId"]
    assert ep["request_body"]["content_type"] == "application/x-www-form-urlencoded"
    assert ep["request_body"]["fields"] == ["name"] and ep["request_body"]["required"] is True


def test_a_body_that_is_a_list_is_marked_as_an_array():
    """Petstore's /user/createWithList takes a LIST of users; the tool must accept a list."""
    spec = {"openapi": "3.0.0", "servers": [{"url": "https://x.test"}],
            "paths": {"/user/createWithList": {"post": {
                "summary": "Create users",
                "requestBody": {"content": {"application/json": {"schema": {
                    "type": "array", "items": {"$ref": "#/components/schemas/User"}}}}}}}},
            "components": {"schemas": {"User": {"type": "object", "properties": {"username": {"type": "string"}}}}}}
    body = extract_endpoints_from_openapi(spec)[0]["request_body"]
    assert body["type"] == "array" and body["fields"] == ["username"]
