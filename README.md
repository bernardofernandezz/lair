# lair

Your own AI on your own hardware. lair is a single Rust binary that loads GGUF models with llama.cpp and serves a web chat and an HTTP API. Prompts, replies and models stay on your machine.

- **Any GGUF model.** Point lair at a folder and every `.gguf` in it shows up in the model picker. One model is in memory at a time, and picking another swaps it.
- **Reasoning you can control.** Models that think before answering get three modes: off, light or deep. The reasoning is always kept separate from the answer, both in the UI and in the API.
- **Standard chat API.** `/v1/chat/completions` (with streaming) and `/v1/models` use the common chat-completions format, so existing editors and tools can connect.
- **Web UI included.** The chat UI is built into the binary: markdown answers, a model picker, reasoning modes, and inline 3D scenes.
- **Conversations that last.** Saved on the server, searchable, and citable from other conversations with `@`.
- **An agent when you want one.** The model can read and write files in a workspace, run commands, search the web through your own SearXNG, and look through past conversations, asking before anything risky.
- **Skills.** Prompt packages that kick in on trigger words; create them in the UI or install them from a URL.
- **Your machines as one network.** Nodes on your Tailscale network find each other and share their models.
- **GPU or CPU.** CUDA build for NVIDIA cards, plain build for everything else.

## Quick start

Linux (x86_64) and macOS (Apple Silicon):

```bash
curl -fsSL https://raw.githubusercontent.com/bernardofernandezz/lair/main/install.sh | sh
lair serve
```

Then open http://localhost:8080. The first time, with no models yet, lair downloads a small reasoning model (Qwen3 4B, ~2.5 GB) and shows the progress. On Linux the installer picks the Vulkan build when a GPU driver is present (NVIDIA, AMD or Intel; no CUDA needed), and the CPU build otherwise. On a Mac it uses Metal.

```bash
lair run              # chat in the terminal (lair run qwen picks a model by name)
lair pull <url>       # download a .gguf, e.g. https://huggingface.co/<org>/<repo>/resolve/main/<file>.gguf
lair list             # models in ~/.local/share/lair/models
lair serve --help     # every option: context size, GPU layers, token, port...
```

Models live in `~/.local/share/lair/models`. Drop any `.gguf` there, or pass `-m <folder>`.

## Docker

You need Docker. For an NVIDIA GPU, also the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

```bash
curl -O https://raw.githubusercontent.com/bernardofernandezz/lair/main/compose.yaml
docker compose up                                 # NVIDIA GPU via CUDA (RTX 20xx to 50xx)
docker compose --profile cpu up lair-cpu          # no NVIDIA GPU
```

Then open http://localhost:8080. On the first run, with no model in `./models`, lair downloads a small reasoning model (Qwen3 4B, ~2.5 GB) into it and shows the progress in the logs. Later runs start right away.

To use other models, drop `.gguf` files into `./models` (or point `LAIR_MODELS` at a folder that has them) and restart. Every model in the folder shows up in the picker.

| Variable             | Default    | Purpose                                                                 |
| -------------------- | ---------- | ----------------------------------------------------------------------- |
| `LAIR_MODELS`        | `./models` | Folder with `.gguf` files, searched recursively                          |
| `LAIR_DEFAULT_MODEL` | Qwen3 4B   | URL of the `.gguf` downloaded when the folder is empty; `""` never downloads |
| `LAIR_PORT`          | `8080`     | Host port. Bound to localhost only                                      |
| `LAIR_TOKEN`         | unset      | When set, the UI and API ask for this token                             |

To reach lair from other devices, set `LAIR_TOKEN`, change the port mapping in `compose.yaml` to `"${LAIR_PORT:-8080}:8080"`, and put HTTPS in front of it.

The agent's web search (`--profile web`) needs `searxng/settings.yml`, so it runs from a clone of this repo.

### Building the image yourself

From a clone, add `--build`: `docker compose up --build`. The first build compiles the CUDA kernels and takes a few minutes. It runs at low priority so the desktop stays usable.

| Variable          | Default | Purpose                                                            |
| ----------------- | ------- | ------------------------------------------------------------------ |
| `CUDA_ARCH`       | `86`    | GPU generation to compile for: 75 = RTX 20xx, 86 = 30xx, 89 = 40xx, 120 = 50xx |
| `LAIR_BUILD_JOBS` | `4`     | Parallel compile jobs; use 2 on a machine with little free RAM     |

## Build from source

Requires Rust, bun, cmake and clang. For Vulkan, also the Vulkan headers, `glslc` and SPIR-V headers (Arch: `vulkan-headers shaderc spirv-headers`; Ubuntu 24.04: `libvulkan-dev glslc spirv-headers`).

```bash
(cd web && bun install && bun run build)   # the UI is embedded at compile time
cargo build --release --features vulkan    # or cuda, metal, or none for CPU only
```

```bash
./target/release/lair serve -m ~/models                 # web UI + API on 127.0.0.1:8080
./target/release/lair chat  -m ~/models --reasoning low # chat in the terminal
```

Run `lair serve --help` for every option: context size, GPU layers, sampling defaults, host and port.

## Conversations

Conversations are saved on the server, in SQLite under `--data` (`LAIR_DATA`, default `~/.local/share/lair`; a named volume in Docker), so they show up on every device that opens lair. The sidebar lists and searches them by title and content.

Type `@` in a message to cite another conversation. Its most recent part is added to the prompt as quoted context, so the model can use what was said there.

## Your machines, one network

Run lair on several machines in your [Tailscale](https://tailscale.com) network and each node lists the others' models next to its own, as `model@machine`. Pick one and the conversation stays on the machine you're using, while the text is generated on the other. That machine only generates; history, skills and agent tools stay where the conversation lives.

- **Discovery:** lair asks the local `tailscaled` (over `/var/run/tailscale/tailscaled.sock`, or `--tailscale-socket`) which machines are online and checks each for lair on the same port, every minute. Add machines outside the tailnet by hand with `--peer host:port` (`LAIR_PEERS`, comma-separated).
- **Trust:** a request coming from a machine in your tailnet is verified with `tailscale whois` and needs no token. Everything else follows `LAIR_TOKEN` as usual.
- **Names:** each node goes by its Tailscale host name; override it with `--name` / `LAIR_NAME`.
- **Docker:** `docker compose -f compose.yaml -f compose.tailnet.yaml up` shares the host network and mounts the Tailscale socket. The node then listens on all of the host's interfaces, so set `LAIR_TOKEN` to keep the rest of your LAN out.

Nodes talk to each other through lair's own API (`/lair/v1/node`, `/lair/v1/generate`) over the tailnet's encrypted links. No outside service is involved.

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
src/engine.rs        local GGUF models: generation, reasoning modes, <think> splitting
src/models.rs        every usable model, local or on a peer, behind one streaming generate
src/peers.rs         other lair nodes on the tailnet: discovery, whois trust, remote generation
src/store.rs         conversations in SQLite, with full-text search
src/turn.rs          one conversation turn: history, references, skills, the agent loop
src/agent/           tool-call protocol and parser, tools, approvals
src/skills/          skill catalog, install from URL/zip/GitHub, index
src/server/          HTTP: chat-completions format (compat.rs), lair's own API (native.rs), auth, UI
src/jail.rs          keeps paths inside a folder (workspace, assets)
web/                 React UI (Vite): chat, sidebar, agent cards, skills panel, 3D scenes
skills/              built-in skills, embedded in the binary
```

Tests: `cargo test` and `cd web && bun test`. To run the model-backed test, set `LAIR_TEST_MODEL=/path/to/model.gguf` and run `cargo test -- --include-ignored`.
