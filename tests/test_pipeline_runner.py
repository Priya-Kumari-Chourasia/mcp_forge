"""The UI's runner must report every step, and its demo switch must trigger a real correction."""
import nodes
import pipeline_runner as runner

DOCS = "GET https://catfact.ninja/fact - returns a random cat fact"


def run(monkeypatch, tmp_path, validator_agent_url, inject_bug):
    monkeypatch.chdir(tmp_path)
    return list(runner.stream_run(DOCS, None, inject_bug))


def test_one_event_per_node_ending_with_the_packaged_server(monkeypatch, tmp_path, validator_agent_url):
    events = run(monkeypatch, tmp_path, validator_agent_url, inject_bug=False)
    assert events[0]["node"] == "start"
    assert events[0]["state"]["agent_card"]["name"] == "MCP-Forge Validator"
    assert events[-1]["node"] == "package_server"
    assert "@server.tool()" in events[-1]["state"]["server_code"]
    assert [a["passed"] for a in events[-1]["state"]["attempt_history"]] == [True]


def test_demo_switch_breaks_attempt_1_and_the_loop_repairs_it(monkeypatch, tmp_path, validator_agent_url):
    real_llm = nodes.call_llm
    events = run(monkeypatch, tmp_path, validator_agent_url, inject_bug=True)
    first, second = events[-1]["state"]["attempt_history"]
    assert first["passed"] is False and "not an absolute URL" in first["errors"][0]
    assert second["passed"] is True
    assert nodes.call_llm is real_llm                    # the injector is removed afterwards


def test_recordings_round_trip(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    events = [{"node": "start", "log": "", "state": {"input_kind": "Prose docs"}}]
    runner.save_recording("example", True, events)
    assert runner.load_recording("example", True) == events
    assert runner.load_recording("example", False) is None


def test_progress_events_arrive_while_a_node_is_busy(monkeypatch, tmp_path, validator_agent_url):
    """A slow LLM call must not look like a frozen run: heartbeats report how long it has been silent."""
    import time
    monkeypatch.chdir(tmp_path)
    real_llm = nodes.call_llm

    def slow_llm(prompt):
        time.sleep(0.5)
        return real_llm(prompt)

    monkeypatch.setattr(nodes, "call_llm", slow_llm)
    events = list(runner.stream_run(DOCS, None, False, heartbeat_seconds=0.1))

    heartbeats = [e for e in events if e["node"] == "progress"]
    assert heartbeats and all({"tail", "idle_seconds", "elapsed_seconds"} <= set(e["progress"]) for e in heartbeats)
    assert [e for e in events if e["node"] != "progress"][-1]["node"] == "package_server"


def test_abandoning_the_run_stops_the_pipeline(monkeypatch, tmp_path, validator_agent_url):
    """Pressing Stop must not leave a background run spending LLM calls."""
    import time
    monkeypatch.chdir(tmp_path)
    calls = []
    real_llm = nodes.call_llm

    def counting_llm(prompt):
        calls.append(prompt)
        time.sleep(0.2)
        return real_llm(prompt)

    monkeypatch.setattr(nodes, "call_llm", counting_llm)
    run = runner.stream_run(DOCS, None, False, heartbeat_seconds=0.05)
    for event in run:
        if event["node"] == "orchestrator":
            break
    run.close()                              # what happens when the viewer presses Stop
    time.sleep(1.5)
    assert not (tmp_path / "generated").exists()        # it never got as far as packaging