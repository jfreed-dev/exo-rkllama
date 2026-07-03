# exo for RK3588 (aarch64) with the RKLLM runtime baked in.
# Build on an arm64 host or with buildx: docker buildx build --platform linux/arm64 .
# Status: DRAFT — first real build happens once an RK1 node is up (issue #5); the
# rkllm-runtime path inside the airockchip/rknn-llm repo moves between releases, so
# verify RKLLM_RUNTIME_PATH against the tag before building.

FROM node:22-slim AS dashboard
WORKDIR /src/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ ./
RUN npm run build

FROM python:3.13-slim-bookworm AS runtime
ARG RKLLM_VERSION=release-v1.2.3
# Path within airockchip/rknn-llm at RKLLM_VERSION; adjust per release
# (1.3.x moved it under rkllm-runtime/runtime/).
ARG RKLLM_RUNTIME_PATH=rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so

# git: uv fetches git dependencies (mflux); cmake/perl: native -sys crates in the
# exo_rs dependency tree.
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential pkg-config git cmake perl \
  && rm -rf /var/lib/apt/lists/*

# Rust toolchain: the exo_rs bindings build from source on aarch64.
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH="/root/.cargo/bin:${PATH}"

# RKLLM runtime library (closed-source, from the airockchip release tree).
RUN curl -fsSL -o /usr/lib/librkllmrt.so \
  "https://github.com/airockchip/rknn-llm/raw/${RKLLM_VERSION}/${RKLLM_RUNTIME_PATH}"

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY . .
COPY --from=dashboard /src/dashboard/build /app/dashboard/build
# mlx-cpu extra: pyproject pins the linux-aarch64 mlx wheel; CPU fallback engine only,
# the NPU path goes through the rkllm engine.
RUN uv sync --extra mlx-cpu --no-dev

ENV EXO_DASHBOARD_DIR=/app/dashboard/build
ENV EXO_RKLLM_BACKEND=ctypes
ENV RKLLM_LIB_PATH=/usr/lib/librkllmrt.so
EXPOSE 52415

CMD ["uv", "run", "exo"]
