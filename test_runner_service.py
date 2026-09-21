import ast
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/tasks/send", methods=["POST"])
def run_test():
    task = request.get_json()
    task_id = task["id"]
    code = task["message"]["parts"][0]["text"]

    print(f"[test-runner] received task {task_id}")
    result = _validate_code(code)

    return jsonify({
        "id": task_id,
        "status": {"state": "completed" if result["passed"] else "failed"},
        "artifacts": [{"parts": [{"type": "text", "text": result["output"]}]}],
    })


def _validate_code(code: str) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"passed": False, "output": f"SyntaxError: {e}"}

    top_level_functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    function_names = [n.name for n in top_level_functions]

    if not function_names:
        return {"passed": False, "output": "No functions were defined in the generated code."}

    # Catch duplicate top-level names BEFORE they silently overwrite each other.
    # This matters specifically because batched generation means independent
    # LLM calls can each define a same-named helper without knowing about
    # each other - Python keeps only the last one and says nothing.
    seen = set()
    duplicates = set()
    for name in function_names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        return {
            "passed": False,
            "output": f"Duplicate function definitions found (later ones silently overwrite earlier ones): {sorted(duplicates)}"
        }

    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return {"passed": False, "output": f"{type(e).__name__} while defining functions: {e}"}

    not_callable = [name for name in function_names if not callable(namespace.get(name))]
    if not_callable:
        return {"passed": False, "output": f"Defined but not callable: {not_callable}"}

    return {"passed": True, "output": f"{len(function_names)} function(s) defined and valid: {function_names}"}

if __name__ == "__main__":
    print("Test-runner agent listening on http://localhost:8000")
    app.run(port=8000)