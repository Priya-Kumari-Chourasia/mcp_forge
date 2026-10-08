import os
import sys
import threading

import pytest
from werkzeug.serving import make_server

# Tests always use the offline stub LLM + hash embeddings, and import modules from the project root.
os.environ["USE_REAL_LLM"] = "false"
os.environ["EMBEDDINGS"] = "hash"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def validator_agent_base_url():
    """Starts the real validator agent on a free local port for the test session."""
    import validator_agent
    server = make_server("127.0.0.1", 0, validator_agent.app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.fixture
def validator_agent_url(validator_agent_base_url, monkeypatch):
    """Points the pipeline at that agent."""
    import nodes
    monkeypatch.setattr(nodes, "VALIDATOR_AGENT_URL", validator_agent_base_url)
    return validator_agent_base_url
