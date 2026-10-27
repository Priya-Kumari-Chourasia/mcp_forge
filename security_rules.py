"""
SECURITY RULES - what the security reviewer agent checks (security_agent.py).

The review reads the generated tool code with Python's ast module and NEVER
runs it. It runs before the validator, because the validator executes the
code: whatever is dangerous has to be stopped before anything runs it.

There is no LLM in here, on purpose. The docs the code came from may contain
prompt injection, and an LLM reviewer reading them could be talked into
approving bad code. Fixed rules give the same answer every time and cannot
be persuaded.

What is reviewed is the LLM's part only: the tool functions. The header above
them (imports, auth, _request) is the fixed template, written and tested by
hand (server_template.py).

ERRORS block the code. They are sent back to the LLM like validator errors:
  structure    only tool functions at the top level; nothing that runs when the
               server starts; no helpers, decorators or redefined header names
  request      each tool makes exactly one _request call, with the method and the
               path written as plain strings equal to its endpoint's. So the
               endpoint a tool reaches is visible without running it, and a tool
               labelled read-only (from its documented GET) cannot send a DELETE
  names        a tool may use only its own arguments, variables it assigns itself,
               _request and the type names: an ALLOWLIST. Anything else (os, open,
               eval, print, getattr, ...) is refused, whatever it is. A blocklist
               would only stop the dangers someone thought of in advance
  attributes   no __dunder__ attributes (the usual way out of restrictions in
               Python) and no str.format (format strings can read attributes)
  hidden text  no invisible Unicode characters (zero-width, bidirectional
               controls, tag characters): text a person cannot see but an LLM reads

WARNINGS do not block. They flag text in the docs or the tool descriptions
that reads like instructions to the AI ("ignore previous instructions",
"before using this tool, read ~/.ssh/..."). An MCP client shows tool
descriptions to its LLM, so this is how "tool poisoning" works. The phrases
are a heuristic: they can miss a reworded attack and can match an honest
sentence, so they are reported for a person to judge, not enforced.
"""

import ast
import re
import unicodedata

from server_template import RESERVED_NAMES

# What a tool function may refer to, besides its own arguments and locals.
ALLOWED_NAMES = {
    "_request",                                                    # the template's helper
    "Annotated", "Any", "Literal", "Optional", "Union", "Field",    # type hints, from the header
    "dict", "list", "str", "int", "float", "bool", "tuple", "set", # plain types / conversions
    "len", "isinstance", "ToolError", "ValueError",                # harmless helpers
}
# Not allowed, for example: print (a tool's stdout IS the MCP connection over stdio,
# so printing corrupts it), getattr/type (ways to reach attributes indirectly), os, open.

BLOCKED_ATTRIBUTES = {"format", "format_map"}       # "{0.__class__}".format(x) reads attributes

INSTRUCTION_PATTERNS = [
    r"\bignore\b.{0,30}\b(previous|prior|above|earlier|all)\b.{0,20}\b(instructions?|rules|messages|prompts?)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules|above|previous)\b",
    r"\bsystem prompt\b",
    r"\b(do not|don't|never)\s+(tell|inform|mention|reveal|show)\b.{0,30}\b(user|anyone)\b",
    r"<\s*/?\s*(important|system|instructions?)\s*>",
    r"~/\.ssh|\bid_rsa\b|\.env\b|\bmcp\.json\b|claude_desktop_config",
    r"\b(before|after)\s+(using|calling)\s+(this|any)\s+tool\b",
]
_INSTRUCTION_RE = re.compile("|".join(f"(?:{p})" for p in INSTRUCTION_PATTERNS), re.IGNORECASE)


# ----------------------------------------------------------------------
# Hidden characters
# ----------------------------------------------------------------------

def is_invisible(ch: str) -> bool:
    """Unicode 'format' characters (category Cf): zero-width spaces and joiners,
    bidirectional controls, tag characters U+E0000-E007F. They take up no space
    on screen but are part of the text an LLM reads."""
    return unicodedata.category(ch) == "Cf"


def strip_invisible(text: str) -> str:
    """Removes them. Applied to every description before it reaches the LLM."""
    return "".join(ch for ch in (text or "") if not is_invisible(ch))


def _invisible_in(text: str) -> list:
    return sorted({f"U+{ord(ch):04X}" for ch in text or "" if is_invisible(ch)})


# ----------------------------------------------------------------------
# The review
# ----------------------------------------------------------------------

def _error(text, function=None, endpoint_index=None, rule=None):
    return {"function": function, "endpoint_index": endpoint_index, "rule": rule, "error": text}


def _warning(text, function=None, endpoint_index=None):
    return {"function": function, "endpoint_index": endpoint_index, "warning": text}


def _quote(text, limit=80):
    """The text, shortened, for a message. Invisible characters are removed so they
    cannot rearrange the message itself (U+202E reverses what follows it)."""
    text = " ".join(strip_invisible(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _instruction_like(text: str):
    match = _INSTRUCTION_RE.search(text or "")
    return match.group(0) if match else None


def _local_names(function) -> set:
    """Its arguments and every name it assigns (variables, loop and comprehension
    targets, 'except ... as name')."""
    args = function.args
    names = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    names |= {a.arg for a in (args.vararg, args.kwarg) if a}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def _check_request(function, endpoint, index):
    """Exactly one _request call, its method and path plain strings equal to the endpoint's."""
    name = function.name
    calls = [n for n in ast.walk(function)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_request"]
    called = {id(c.func) for c in calls}
    errors = []
    # _request used other than by calling it (e.g. f = _request; f(...)) hides which request is sent.
    if any(isinstance(n, ast.Name) and n.id == "_request" and id(n) not in called for n in ast.walk(function)):
        errors.append(_error(f"{name}() refers to _request without calling it directly; call it as "
                             f"_request(\"METHOD\", \"/path\", ...)", name, index, "request"))
    if len(calls) != 1:
        errors.append(_error(f"{name}() makes {len(calls)} _request calls; a tool makes exactly one",
                             name, index, "request"))
        return errors

    call = calls[0]
    if any(k.arg is None for k in call.keywords) or any(isinstance(a, ast.Starred) for a in call.args):
        errors.append(_error(f"{name}() passes arguments to _request with * or **; write them out so the "
                             f"request can be checked", name, index, "request"))
    method = call.args[0] if call.args else None
    path = call.args[1] if len(call.args) > 1 else None
    want_method, want_path = (endpoint.get("method") or "").upper(), endpoint.get("path") or ""

    if not (isinstance(method, ast.Constant) and isinstance(method.value, str)):
        errors.append(_error(f"{name}()'s HTTP method must be a plain string, \"{want_method}\"", name, index, "request"))
    elif method.value.upper() != want_method:
        errors.append(_error(f"{name}() sends {method.value.upper()} but its endpoint is {want_method} {want_path}; "
                             f"its read-only/destructive label comes from the documented method",
                             name, index, "request"))

    if not (isinstance(path, ast.Constant) and isinstance(path.value, str)):
        errors.append(_error(f"{name}()'s path must be the plain string \"{want_path}\" with values passed in "
                             f"path_params (no f-string), so _request can encode them", name, index, "request"))
    elif path.value != want_path:
        errors.append(_error(f"{name}()'s path is \"{path.value}\" but its endpoint's path is \"{want_path}\"",
                             name, index, "request"))
    return errors


def _check_function(function, endpoint, index):
    name = function.name
    errors = []
    if function.decorator_list:
        errors.append(_error(f"{name}() has a decorator; the template adds @server.tool itself",
                             name, index, "structure"))

    allowed = _local_names(function) | ALLOWED_NAMES
    refused = set()
    for node in ast.walk(function):
        if node is function:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            errors.append(_error(f"{name}() defines a function or class inside it; write the request inline",
                                 name, index, "structure"))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append(_error(f"{name}() imports a module; tools use only what the header provides",
                                 name, index, "names"))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            errors.append(_error(f"{name}() declares global/nonlocal names; a tool must not change the "
                                 f"server's settings", name, index, "names"))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in allowed:
            refused.add(node.id)
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                errors.append(_error(f"{name}() uses the attribute {node.attr}; double-underscore attributes "
                                     f"are not allowed", name, index, "attributes"))
            elif node.attr in BLOCKED_ATTRIBUTES:
                errors.append(_error(f"{name}() uses .{node.attr}(); use an f-string or + instead",
                                     name, index, "attributes"))
    for ref in sorted(refused):
        errors.append(_error(f"{name}() uses '{ref}', which tools are not allowed to use; a tool may use only "
                             f"its arguments, its own variables, _request and the type names",
                             name, index, "names"))

    errors += _check_request(function, endpoint, index)
    return errors


def review(tool_code: str, endpoints: list) -> dict:
    """Reviews generated tool functions against the endpoints they implement.
    Each endpoint needs its tool "name", "method" and "path".
    Returns {"passed", "output", "errors": [...], "warnings": [...]}."""
    errors, warnings = [], []
    by_name = {ep.get("name"): (i, ep) for i, ep in enumerate(endpoints or []) if ep.get("name")}

    # Hidden characters, anywhere in the code (comments included, so the raw text).
    for number, line in enumerate((tool_code or "").splitlines(), start=1):
        hidden = _invisible_in(line)
        if hidden:
            errors.append(_error(f"line {number} contains invisible characters ({', '.join(hidden)}); remove them",
                                 rule="hidden text"))

    # Instruction-like or hidden text in the docs the tools were generated from.
    for i, ep in enumerate(endpoints or []):
        body_fields = (ep.get("request_body") or {}).get("field_details") or []
        texts = ([ep.get("description") or ""] + [p.get("description") or "" for p in ep.get("parameters") or []]
                 + [f.get("description") or "" for f in body_fields])
        for text in texts:
            phrase = _instruction_like(text)
            if phrase:
                warnings.append(_warning(f"the docs for {ep.get('method')} {ep.get('path')} contain text that reads "
                                         f"like an instruction to the AI: \"{_quote(text)}\"", ep.get("name"), i))
            hidden = _invisible_in(text)
            if hidden:
                warnings.append(_warning(f"the docs for {ep.get('method')} {ep.get('path')} contain invisible "
                                         f"characters ({', '.join(hidden)}); they were removed before generation",
                                         ep.get("name"), i))

    try:
        tree = ast.parse(tool_code or "")
    except SyntaxError as e:
        errors.append(_error(f"SyntaxError: {e}", rule="structure"))
        return _verdict(errors, warnings, [])

    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue                                          # a stray string: does nothing
        else:
            errors.append(_error(f"line {node.lineno}: only tool functions are allowed at the top level; "
                                 f"other statements run as soon as the server starts", rule="structure"))

    for function in functions:
        name = function.name
        if name.startswith("_") or name in RESERVED_NAMES:
            errors.append(_error(f"{name}() would replace a name the template defines (or is a helper); "
                                 f"write only the tool functions", name, rule="structure"))
            continue
        if name not in by_name:
            errors.append(_error(f"{name}() is not the tool of any endpoint; every function must be named after "
                                 f"its endpoint's \"name\"", name, rule="structure"))
            continue
        index, endpoint = by_name[name]
        errors += _check_function(function, endpoint, index)

        # Text the AI client will show its LLM: the docstring and the argument descriptions.
        for node in ast.walk(function):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                phrase = _instruction_like(node.value)
                if phrase:
                    warnings.append(_warning(f"{name}()'s description contains text that reads like an "
                                             f"instruction to the AI: \"{_quote(node.value)}\"", name, index))

    return _verdict(errors, warnings, [f.name for f in functions])


def _verdict(errors, warnings, names):
    if errors:
        output = f"{len(errors)} security problem(s): " + "; ".join(e["error"] for e in errors[:5])
    else:
        output = (f"{len(names)} tool(s) passed the security review: each makes one request to its own "
                  f"endpoint and uses nothing outside the allowlist")
    if warnings:
        output += f" ({len(warnings)} warning(s) about the docs or descriptions)"
    return {"passed": not errors, "output": output, "errors": errors, "warnings": warnings}
