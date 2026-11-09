# lair

Run open-weight LLMs on your own machine. lair is a single Rust binary that loads GGUF models with llama.cpp and serves a web chat and an HTTP API. Prompts, replies and models stay on your machine.

- **Any GGUF model.** Point lair at a folder and every `.gguf` in it shows up in the model picker. One model is in memory at a time, and picking another swaps it.
- **Reasoning you can control.** Models that think before answering get three modes: off, light or deep. The reasoning is always kept separate from the answer, both in the UI and in the API.
- **Standard chat API.** `/v1/chat/completions` (with streaming) and `/v1/models` use the common chat-completions format, so existing editors and tools can connect.
- **Web UI included.** The chat UI is built into the binary: markdown answers, a model picker, reasoning modes, and inline 3D scenes.
- **GPU or CPU.** CUDA build for NVIDIA cards, plain build for everything else.

## Quick start

With Docker and an NVIDIA GPU ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) required):

```bash
LAIR_MODELS=~/models docker compose up --build
```

Then open http://localhost:8080. The first build compiles the CUDA kernels and takes a few minutes.

No NVIDIA GPU:

```bash
LAIR_MODELS=~/models docker compose --profile cpu up --build lair-cpu
```

| Variable      | Default    | Purpose                                                              |
| ------------- | ---------- | -------------------------------------------------------------------- |
| `LAIR_MODELS` | `./models` | Folder with `.gguf` files, searched recursively and mounted read-only |
| `LAIR_PORT`   | `8080`     | Host port. Bound to localhost only                                   |
| `LAIR_TOKEN`  | unset      | When set, the UI and API ask for this token                          |
| `CUDA_ARCH`   | `86`       | GPU generation to compile for: 75 = RTX 20xx, 86 = 30xx, 89 = 40xx   |

To reach lair from other devices, set `LAIR_TOKEN`, change the port mapping in `compose.yaml` to `"${LAIR_PORT:-8080}:8080"`, and put HTTPS in front of it.

## Build from source

Requires Rust, bun, cmake and clang.

```bash
(cd web && bun install && bun run build)   # the UI is embedded at compile time
cargo build --release --features cuda      # drop --features cuda for a CPU-only build
```

```bash
./target/release/lair serve -m ~/models                 # web UI + API on 127.0.0.1:8080
./target/release/lair chat  -m ~/models --reasoning low # chat in the terminal
```

Run `lair serve --help` for every option: context size, GPU layers, sampling defaults, host and port.

## Reasoning modes

For models that emit a `<think>` block before answering:

| UI        | `reasoning_effort` | Behavior                                    |
| --------- | ------------------ | ------------------------------------------- |
| Desligado | `none`             | Answers straight away                       |
| Leve      | `low`              | Thinks for up to ~512 tokens, then answers  |
| Profundo  | `high` (default)   | Thinks as long as it needs                  |

The API returns the reasoning in `reasoning_content` and the answer in `content`, in full responses and in streamed deltas.

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "<id from /v1/models>",
  "reasoning_effort": "low",
  "messages": [{ "role": "user", "content": "Is 221 prime?" }]
}'
```

## Skills

Skills are prompt packages that lair adds to a request when the user's message asks for them. Every other message reaches the model unchanged.

The built-in skill, **Cenas 3D**, runs when a message mentions "3D" or "três dimensões". The answer then includes an interactive three.js scene, with orbit, reset and fullscreen. To use local `.glb`/`.gltf` models in scenes, pass `--assets <folder>` or set `LAIR_ASSETS`.

To add your own skills, create one folder per skill and pass the parent folder with `--skills-dir`, or set `LAIR_SKILLS`. A skill with the same id as a built-in replaces it.

```
~/lair-skills/pirate/skill.json   { "name": "Pirata", "when": ["pirata", "arr"] }
~/lair-skills/pirate/prompt.md    Responda como um pirata do século XVIII.
```

A skill runs when one of its `when` phrases appears as a whole word in the last user message. Case and accents are ignored, and at most one skill runs per reply. Streamed replies announce it with an SSE event named `lair.skill`. Skills are trusted prompt files, so only point lair at folders you control. `--no-skills` turns them off.

## Project layout

```
src/engine.rs   model catalog, generation, reasoning modes, <think> splitting
src/server.rs   HTTP API, token auth, embedded UI
src/skills.rs   skill loading and triggering
src/assets.rs   safe access to local glTF files
src/chat.rs     terminal chat
web/            React UI (Vite); web/src/scene renders 3D scenes
skills/         built-in skills, embedded in the binary
```

Tests: `cargo test` and `cd web && bun test`. To run the model-backed test, set `LAIR_TEST_MODEL=/path/to/model.gguf` and run `cargo test -- --include-ignored`.
