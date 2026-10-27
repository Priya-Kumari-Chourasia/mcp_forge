# One Dockerfile, two images (build "targets"), from the same code:
#
#   agents - for the security reviewer and the validator, the containers that handle
#            LLM-written code. Only requirements-agents.txt and the six files the
#            agents use: no web UI, no LLM client, no vector database.
#   app    - the Streamlit UI and the LangGraph pipeline (everything).
#
# docker-compose.yml builds both and decides how each container runs (networks,
# limits, read-only filesystem). Build one by hand with:  docker build --target app .

# python:3.13-slim = Debian with Python 3.13 and little else. The tag follows 3.13.x
# patch releases, so a rebuild picks up security fixes.
FROM python:3.13-slim AS base

# No .pyc files (the agents' filesystem is read-only), log lines appear immediately,
# and no pip download cache left inside the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Every container runs as this ordinary user, never as root.
RUN useradd --create-home --uid 10001 forge
WORKDIR /app


FROM base AS agents
# Requirements first, code second: Docker caches each step, so changing the code
# does not reinstall the packages.
COPY requirements-agents.txt ./
RUN pip install -r requirements-agents.txt
# Owned by root, run as forge: the agents cannot change their own code.
COPY a2a_server.py security_agent.py security_rules.py validator_agent.py behaviour_check.py server_template.py ./
USER forge


FROM base AS app
COPY requirements-agents.txt requirements.txt ./
RUN pip install -r requirements.txt
# The app writes generated servers (generated/) and recordings (demo_runs/), so it owns its
# folder: --chown covers the copied files, and the chown covers /app itself (WORKDIR made it
# as root), so the app can create new folders in it.
COPY --chown=forge:forge . .
RUN chown forge:forge /app
USER forge
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
