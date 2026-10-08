"""
PIPELINE RUNNER - runs the graph step by step for the web UI.

graph.py runs the pipeline in one go and prints as it works. A UI needs
progress, so this module runs the same graph with LangGraph's stream()
and yields one EVENT per node:

    {"node": "retrieve", "log": "<what that node printed>", "state": {...}}

The UI draws whatever the latest event says. A list of events can also be
saved to a JSON file and replayed later with no LLM calls (see demo_runs/).
"""

import contextlib
import json
import os
import queue
import re
import threading
import time

import requests

import nodes
from graph import build_graph, initial_state, RUN_CONFIG
from spec_parser import is_openapi_spec

RECORDINGS_DIR = "demo_runs"

# Only one live run at a time: the RAG index and the fault injector are
# module-level, so two simultaneous runs would interfere with each other.
RUN_LOCK = threading.Lock()

# The parts of the state the UI shows.
SHOWN_KEYS = [
    "orchestration_plan", "max_code_attempts", "endpoints", "rag_candidates", "rag_trace",
    "codegen_attempt", "attempt_history", "test_passed", "test_output", "output_path",
]

_validator_server = None


def ensure_validator_running() -> dict:
    """Returns the validator's Agent Card, starting the agent in a background
    thread first if nothing is answering at its address. This lets the UI run
    as a single process (one command locally, one app on a hosting service)."""
    global _validator_server
    card_url = nodes.VALIDATOR_AGENT_URL.rstrip("/") + "/.well-known/agent-card.json"
    try:
        return requests.get(card_url, timeout=2).json()
    except requests.RequestException:
        pass

    from urllib.parse import urlparse
    from werkzeug.serving import make_server
    import validator_agent

    address = urlparse(nodes.VALIDATOR_AGENT_URL)
    _validator_server = make_server(address.hostname or "127.0.0.1", address.port or 8000,
                                    validator_agent.app, threaded=True)
    threading.Thread(target=_validator_server.serve_forever, daemon=True).start()
    return requests.get(card_url, timeout=5).json()


class BugInjector:
    """DEMO ONLY. Wraps the LLM and breaks the FIRST piece of generated code on
    purpose (it strips the host from one URL), so the self-correcting loop has
    something to correct. The validator's error and the fix on the next attempt
    are then real."""

    def __init__(self, real_llm):
        self.real_llm = real_llm
        self.injected = False

    def __call__(self, prompt: str) -> str:
        code = self.real_llm(prompt)
        is_first_codegen = prompt.startswith(nodes.CODEGEN_INSTRUCTIONS) and "FAILED validation" not in prompt
        if is_first_codegen and not self.injected:
            broken, count = re.subn(r"https?://[A-Za-z0-9.\-:]+", "", code, count=1)
            if count:
                self.injected = True
                return broken
        return code


class RunCancelled(Exception):
    pass


class _LiveLog:
    """Stands in for sys.stdout while the pipeline runs. It keeps everything
    the nodes print, remembers WHEN the last line arrived (so the UI can tell
    "working" from "stuck"), and stops the run if the viewer has gone away."""

    def __init__(self):
        self.lock = threading.Lock()
        self.text = ""
        self.last_activity = time.time()
        self.cancelled = threading.Event()

    def write(self, data):
        if self.cancelled.is_set():
            raise RunCancelled()          # the nodes print often, so this ends the run promptly
        with self.lock:
            self.text += data
            if data.strip():
                self.last_activity = time.time()
        return len(data)

    def flush(self):
        pass

    def take(self):
        """Returns the text printed since the last take()."""
        with self.lock:
            text, self.text = self.text, ""
        return text

    def tail(self, lines=6):
        with self.lock:
            return "\n".join(self.text.strip().splitlines()[-lines:])


def stream_run(raw_docs: str, source_url: str = None, inject_bug: bool = False, heartbeat_seconds: float = 1.0):
    """Runs the pipeline and yields one event per node.

    The graph runs in a worker thread. While a node is busy (several LLM calls,
    or waiting out a rate limit) this generator yields "progress" events about
    once a second with the node's latest output and how long it has been
    silent, so the UI can show that the run is alive."""
    agent_card = ensure_validator_running()
    yield {
        "node": "start", "log": "",
        "state": {"input_kind": "OpenAPI spec" if is_openapi_spec(raw_docs) else "Prose docs",
                  "input_chars": len(raw_docs), "agent_card": agent_card, "bug_injected": inject_bug},
    }

    live_log = _LiveLog()
    updates = queue.Queue()
    real_llm = nodes.call_llm

    def work():
        try:
            if inject_bug:
                nodes.call_llm = BugInjector(real_llm)
            with contextlib.redirect_stdout(live_log):
                for update in build_graph().stream(initial_state(raw_docs, source_url), RUN_CONFIG,
                                                   stream_mode="updates"):
                    updates.put(("update", update))
            updates.put(("done", None))
        except RunCancelled:
            updates.put(("done", None))
        except Exception as error:
            updates.put(("error", error))
        finally:
            nodes.call_llm = real_llm

    started = time.time()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        while True:
            try:
                kind, payload = updates.get(timeout=heartbeat_seconds)
            except queue.Empty:
                yield {"node": "progress", "log": "", "state": {}, "progress": {
                    "tail": live_log.tail(),
                    "idle_seconds": int(time.time() - live_log.last_activity),
                    "elapsed_seconds": int(time.time() - started)}}
                continue

            if kind == "done":
                return
            if kind == "error":
                raise payload
            for node, state in payload.items():
                shown = {key: state.get(key) for key in SHOWN_KEYS}
                if node == "package_server" and state.get("output_path") and os.path.exists(state["output_path"]):
                    with open(state["output_path"], encoding="utf-8") as f:
                        shown["server_code"] = f.read()
                yield {"node": node, "log": live_log.take(), "state": shown}
    finally:
        # Reached when the run ends OR when the UI stops listening (Stop button,
        # closed tab). Cancelling makes the worker quit instead of spending LLM calls.
        live_log.cancelled.set()


# ----------------------------------------------------------------------
# Recordings: a saved run that can be replayed without any LLM calls
# ----------------------------------------------------------------------

def recording_path(slug: str, inject_bug: bool) -> str:
    return os.path.join(RECORDINGS_DIR, f"{slug}{'_bug' if inject_bug else ''}.json")


def save_recording(slug: str, inject_bug: bool, events: list) -> str:
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    path = recording_path(slug, inject_bug)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=1)
    return path


def load_recording(slug: str, inject_bug: bool):
    path = recording_path(slug, inject_bug)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)