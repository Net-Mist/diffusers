FROM nvidia/cuda:12.9.1-devel-ubuntu24.04
ARG DEBIAN_FRONTEND=noninteractive

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0


# Git is usefull for vscode
RUN apt update && apt install -y pipx git && \
    rm -rf /var/lib/apt/lists/*
RUN pipx install uv

WORKDIR /python_env
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=dev_env/uv.lock,target=uv.lock \
    --mount=type=bind,source=dev_env/pyproject.toml,target=pyproject.toml \
    /root/.local/bin/uv sync --locked --no-install-project --no-dev

ENV PATH="/python_env/.venv/bin:/root/.local/bin/:$PATH"
# WORKDIR /workspace
