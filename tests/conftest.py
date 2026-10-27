import os
import sys

import pytest

# Tests always use the offline stub LLM + hash embeddings, and import modules from the project root.
os.environ["USE_REAL_LLM"] = "false"
os.environ["EMBEDDINGS"] = "hash"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def validator_agent_base_url():
    """Starts the real validator agent (FastAPI + uvicorn) on a free local port for the test session."""
    import validator_agent
    base_url, server = validator_agent.serve_in_thread("127.0.0.1", 0)
    yield base_url
    server.should_exit = True


@pytest.fixture(scope="session")
def security_agent_base_url():
    """Starts the real security reviewer agent the same way."""
    import security_agent
    base_url, server = security_agent.serve_in_thread("127.0.0.1", 0)
    yield base_url
    server.should_exit = True


@pytest.fixture
def pipeline_agents(validator_agent_base_url, security_agent_base_url, monkeypatch):
    """Points the pipeline at both agents."""
    import nodes
    monkeypatch.setattr(nodes, "VALIDATOR_AGENT_URL", validator_agent_base_url)
    monkeypatch.setattr(nodes, "SECURITY_AGENT_URL", security_agent_base_url)
    return {"validator": validator_agent_base_url, "security": security_agent_base_url}
