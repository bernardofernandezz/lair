# syntax=docker/dockerfile:1.7
# Targets: `cuda` (default, NVIDIA GPU) and `cpu`.

FROM oven/bun:1.4.4 AS web
WORKDIR /web
COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile
COPY web/ ./
RUN bun run build


FROM rust:1-bookworm AS build-cpu
RUN apt-get update && apt-get install -y --no-install-recommends cmake clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY src src
COPY skills skills
COPY --from=web /web/dist web/dist
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target,id=lair-target-cpu \
    cargo build --release && cp target/release/lair /lair

FROM debian:bookworm-slim AS cpu
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-cpu /lair /usr/local/bin/lair
EXPOSE 8080
ENTRYPOINT ["lair", "serve", "--host", "0.0.0.0", "--models", "/models"]


FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build-cuda
RUN apt-get update && apt-get install -y --no-install-recommends cmake clang libclang-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo PATH=/usr/local/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
# GPU generation to compile kernels for (86 = RTX 30xx). Fewer architectures, much faster build.
ARG CUDA_ARCH=86
ENV CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}
# No GPU driver during the build; link against the toolkit's stub libcuda instead.
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY src src
COPY skills skills
COPY --from=web /web/dist web/dist
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target,id=lair-target-cuda \
    cargo build --release --features cuda && cp target/release/lair /lair

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS cuda
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-cuda /lair /usr/local/bin/lair
EXPOSE 8080
ENTRYPOINT ["lair", "serve", "--host", "0.0.0.0", "--models", "/models"]
