# syntax=docker/dockerfile:1.7
# Targets: `cuda` (default, NVIDIA GPU) and `cpu`.
#
# BUILD_JOBS caps parallel compile jobs. Each CUDA compiler process can take over 1 GB of RAM, so
# using every core can push a desktop into swap; the build also runs at the lowest CPU priority.
ARG BUILD_JOBS=4

FROM oven/bun:1.4.4 AS web
WORKDIR /web
COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile
COPY web/ ./
RUN bun run build


FROM rust:1-bookworm AS build-cpu
ARG BUILD_JOBS
ENV CARGO_BUILD_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
RUN apt-get update && apt-get install -y --no-install-recommends cmake clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
# Dependencies first, in their own layer: it is rebuilt only when Cargo.toml/Cargo.lock change, so a
# remote cache (see release.yml) skips llama.cpp's build when only lair's own code changed.
COPY Cargo.toml Cargo.lock build.rs ./
RUN mkdir -p src web/dist && echo "fn main() {}" > src/main.rs \
    && nice -n 19 cargo build --release && rm -rf src
COPY src src
COPY skills skills
COPY --from=web /web/dist web/dist
RUN touch src/main.rs && nice -n 19 cargo build --release && cp target/release/lair /lair

FROM debian:bookworm-slim AS cpu
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-cpu /lair /usr/local/bin/lair
EXPOSE 8080
ENTRYPOINT ["lair", "serve", "--host", "0.0.0.0", "--models", "/models"]


FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS build-cuda
ARG BUILD_JOBS
ENV CARGO_BUILD_JOBS=${BUILD_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
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
# Dependencies first, as in build-cpu: this layer holds the CUDA kernels, the slow part of the build.
COPY Cargo.toml Cargo.lock build.rs ./
RUN mkdir -p src web/dist && echo "fn main() {}" > src/main.rs \
    && nice -n 19 cargo build --release --features cuda && rm -rf src
COPY src src
COPY skills skills
COPY --from=web /web/dist web/dist
RUN touch src/main.rs && nice -n 19 cargo build --release --features cuda && cp target/release/lair /lair

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS cuda
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-cuda /lair /usr/local/bin/lair
EXPOSE 8080
ENTRYPOINT ["lair", "serve", "--host", "0.0.0.0", "--models", "/models"]
