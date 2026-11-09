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

## Conversations

Conversations are saved on the server, in SQLite under `--data` (`LAIR_DATA`, default `~/.local/share/lair`; a named volume in Docker), so they show up on every device that opens lair. The sidebar lists and searches them by title and content.

Type `@` in a message to cite another conversation. Its most recent part is added to the prompt as quoted context, so the model can use what was said there.

## Agent mode

Turn on **Agente** in the composer and the model can act instead of only writing text. It calls tools, lair runs them and hands back the result, up to 8 steps per reply. Each step shows up in the chat as a card.

| Tool | What it does | Asks you first |
| --- | --- | --- |
| `list_dir`, `read_file` | Look at files in the workspace | no |
| `write_file`, `edit_file`, `delete_file` | Change files in the workspace | yes |
| `run_command` | Run a shell command in the workspace (60 s limit) | yes |
| `web_search`, `fetch_url` | Search the web through SearXNG, read a page | yes |
| `search_conversations`, `read_conversation` | Look through your saved conversations | no |

Before running a tool that changes something or goes online, lair shows the exact arguments and waits for **Permitir**, **Sempre nesta conversa** or **Negar**. File paths can't leave the workspace (`--workspace` / `LAIR_WORKSPACE`, default `<data>/workspace`). Shell commands start in the workspace but aren't sandboxed beyond that. Outside Docker they run as your user, so read each one before you allow it.

Web search uses a SearXNG instance you run. `docker compose --profile web up` starts one with lair; otherwise point `--searxng` / `LAIR_SEARXNG` at yours (JSON output must be enabled, as in `searxng/settings.yml`).

lair asks the model to call tools as `<tool_call>{"name": …, "arguments": {…}}</tool_call>`, and also understands the `[tool(arg='…')]` calls some models (LFM, Llama 3.2) write natively. Small models sometimes skip a step. Asking again, or switching to a larger model, usually helps.

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

Skills are prompt packages that lair adds to a request when the user's message asks for them. Every other message reaches the model unchanged. Manage them in the **Skills** panel at the bottom of the sidebar: switch them on and off, edit them, create new ones, or install them from the network.

The built-in skill, **Cenas 3D**, runs when a message mentions "3D" or "três dimensões". The answer then includes an interactive three.js scene, with orbit, reset and fullscreen. To use local `.glb`/`.gltf` models in scenes, pass `--assets <folder>` or set `LAIR_ASSETS`.

A skill is a folder with two files. A skill runs when one of its `when` phrases appears as a whole word in the last user message. Case and accents are ignored, and at most one skill runs per reply.

```
pirate/skill.json   { "name": "Pirata", "description": "Fala como um pirata.", "when": ["pirata", "arr"] }
pirate/prompt.md    Responda como um pirata do século XVIII.
```

- **Install** from a `.zip`, a GitHub folder (`github.com/owner/repo/tree/branch/folder`), or the URL of a folder holding those two files. Only `skill.json` and `prompt.md` are read from a download.
- **Browse** an index: point `--skills-index` / `LAIR_SKILLS_INDEX` at a JSON file like `{ "skills": [{ "id", "name", "description", "url" }] }`. lair ships without a default index.
- **Create** skills in the panel, or let the agent write one with `save_skill`. They're saved under `<data>/skills`. A skill with a built-in's id replaces it until you delete the copy.
- `--skills-dir` / `LAIR_SKILLS` adds a read-only folder of skills; `--no-skills` turns them all off.

Streamed replies announce the skill in use with an SSE event named `lair.skill`. Skills are trusted prompt text, and one could try to steer the agent, so install only from sources you trust. Risky agent actions still wait for your approval.

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
