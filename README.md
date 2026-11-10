<div align="center">

# lair

**Your own AI on your own hardware.**

One install, any GGUF model: web chat, terminal, agent and a standard chat API.<br>
Prompts, replies and models stay on machines you own.

[![Release](https://img.shields.io/github/v/release/bernardofernandezz/lair?style=flat-square)](https://github.com/bernardofernandezz/lair/releases)
[![License](https://img.shields.io/github/license/bernardofernandezz/lair?style=flat-square)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-linux%20%7C%20macOS-555?style=flat-square)

[Install](#install) · [Usage](#usage) · [Features](#features) · [Configuration](#configuration) · [Build from source](#build-from-source)

</div>

---

lair is a single Rust binary built on [llama.cpp](https://github.com/ggml-org/llama.cpp). Point it at a folder of `.gguf` files and you get a chat UI in the browser, a chat in the terminal and an HTTP API that editors and tools already know how to talk to. Turn on agent mode and the model can work with files, run commands and search the web, always asking before it does anything risky.

## Install

Linux (x86_64) and macOS (Apple Silicon):

```bash
curl -fsSL https://raw.githubusercontent.com/bernardofernandezz/lair/main/install.sh | sh
lair serve
```

Open http://localhost:8080 and start talking. The first run downloads a small reasoning model (Qwen3 4B, about 2.5 GB) so you have something to chat with right away.

The installer puts `lair` in `~/.local/bin` and doesn't need root. On Linux it picks the Vulkan build when a GPU driver is present (NVIDIA, AMD and Intel all work, no CUDA required) and the CPU build otherwise. Macs use Metal.

Prefer containers? See [Docker](#docker).

## Usage

```bash
lair serve            # web UI and API on http://localhost:8080
lair run              # chat in the terminal
lair run llama        # ...with the model whose name contains "llama"
lair pull <url>       # download a .gguf from any URL
lair list             # see what you have
```

Models live in `~/.local/share/lair/models`. Drop any `.gguf` file in there, or point lair somewhere else with `-m <folder>`. Hugging Face is the usual place to find them; `lair pull` takes the file's download link:

```bash
lair pull https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf
```

Anything that speaks the chat-completions format can use lair as a backend:

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen3-4B-Q4_K_M",
  "messages": [{ "role": "user", "content": "Is 221 prime?" }]
}'
```

## Features

|  |  |
| --- | --- |
| **Any GGUF model** | Every `.gguf` in the folder shows up in the model picker. One stays in memory and picking another swaps it. |
| **Reasoning you control** | Models that think before answering get three modes: off, light or deep. The thinking is kept apart from the answer, in the UI and in the API. |
| **Conversations that stay** | Saved on the server in SQLite, searchable, and citable from another chat with `@`. |
| **Agent mode** | The model reads and writes files, runs commands, searches the web and digs through past chats. You approve each risky step. |
| **Skills** | Prompt packages that kick in on trigger words. Write them in the UI or install them from a URL. |
| **Your machines, one network** | lair nodes on your Tailscale network find each other and share their models. |
| **Standard API** | `/v1/chat/completions` with streaming, and `/v1/models`. |

### Agent mode

Switch on **Agente** in the composer and the model can act instead of only writing. It calls a tool, lair runs it and hands back the result, up to 8 steps per reply. Each step shows up in the chat as a card.

| Tool | What it does | Asks first |
| --- | --- | :---: |
| `list_dir`, `read_file` | Look at files in the workspace | |
| `write_file`, `edit_file`, `delete_file` | Change files in the workspace | ✓ |
| `run_command` | Run a shell command in the workspace (60 s limit) | ✓ |
| `web_search`, `fetch_url` | Search through your SearXNG, read a page | ✓ |
| `search_conversations`, `read_conversation` | Look through saved conversations | |

Before a tool changes something or goes online, lair shows the exact arguments and waits for you to allow it once, allow it for the rest of the conversation, or deny it. File paths can't leave the workspace (`<data>/workspace` by default).

> [!WARNING]
> Shell commands start in the workspace but aren't sandboxed beyond that. Outside Docker they run as your user, so read each one before you allow it.

Web search needs a [SearXNG](https://github.com/searxng/searxng) instance with JSON output on. `docker compose --profile web up` starts one next to lair, or pass `--searxng <url>` to use yours.

Small models sometimes skip a step. Asking again, or switching to a bigger model, usually sorts it out.

### Reasoning modes

For models that write a `<think>` block before answering:

| In the UI | `reasoning_effort` | What happens |
| --- | --- | --- |
| Desligado | `none` | Answers straight away |
| Leve | `low` | Thinks for up to ~512 tokens, then answers |
| Profundo | `high` (default) | Thinks as long as it needs |

The API returns the thinking in `reasoning_content` and the answer in `content`, both in full responses and in streamed deltas.

### Conversations

Chats are stored on the server, so they follow you to any device that opens lair. The sidebar searches them by title and content. Type `@` in a message to cite another conversation: its latest part goes into the prompt as quoted context.

### Skills

A skill is a folder with two files. It runs when one of its `when` phrases shows up as a whole word in your message (case and accents don't matter), and at most one runs per reply.

```
pirate/skill.json   { "name": "Pirata", "description": "Fala como um pirata.", "when": ["pirata", "arr"] }
pirate/prompt.md    Responda como um pirata do século XVIII.
```

Manage them in the **Skills** panel at the bottom of the sidebar. You can install one from a `.zip`, a GitHub folder (`github.com/owner/repo/tree/branch/folder`) or any URL that serves those two files, and only those two files are ever read from a download. lair ships with one, **Cenas 3D**, which answers "3D" requests with an interactive three.js scene.

Skills are prompt text, and a malicious one could try to steer the agent. Install them only from sources you trust.

<details>
<summary>More on skills</summary>

- **Browse an index:** point `--skills-index` at a JSON file shaped like `{ "skills": [{ "id", "name", "description", "url" }] }`. There's no default index.
- **Let the agent write one:** it can save skills with the `save_skill` tool. Created skills live in `<data>/skills`, and one with a built-in's id replaces it until you delete the copy.
- **Extra folders:** `--skills-dir` adds a read-only folder of skills; `--no-skills` turns them all off.
- **3D assets:** `--assets <folder>` lets scenes load your local `.glb`/`.gltf` models.
- **API:** streamed replies announce the skill in use with an SSE event named `lair.skill`.

</details>

### Your machines, one network

Run lair on a few machines in your [Tailscale](https://tailscale.com) network and each one lists the others' models next to its own, as `model@machine`. Pick one and the conversation stays where you are while the text is generated over there. History, skills and agent tools never leave the machine you're using.

<details>
<summary>How it works</summary>

- **Discovery:** every minute lair asks the local `tailscaled` which machines are online and checks each one for lair on the same port. Add machines outside the tailnet with `--peer host:port`.
- **Trust:** requests from your tailnet are verified with `tailscale whois` and need no token. Everything else follows `LAIR_TOKEN`.
- **Names:** a node goes by its Tailscale host name, or whatever you pass to `--name`.
- **Docker:** `docker compose -f compose.yaml -f compose.tailnet.yaml up` shares the host network and the Tailscale socket. The node then listens on every interface of the host, so set `LAIR_TOKEN`.

Nodes talk through lair's own API (`/lair/v1/node`, `/lair/v1/generate`) over the tailnet's encrypted links. No outside service is involved.

</details>

## Configuration

`lair serve --help` lists every option. The ones you're most likely to touch:

| Variable | Default | What it does |
| --- | --- | --- |
| `LAIR_TOKEN` | unset | Require this token on the UI and API |
| `LAIR_DATA` | `~/.local/share/lair` | Conversations, skills and the agent's workspace |
| `LAIR_DEFAULT_MODEL` | Qwen3 4B | `.gguf` URL fetched when there are no models; `""` turns it off |
| `LAIR_SEARXNG` | unset | SearXNG URL for the agent's web search |
| `LAIR_PEERS` | unset | Extra lair nodes, as `host:port`, comma-separated |

Flags only: `--port` (8080), `--host` (127.0.0.1), `--ctx` (8192 tokens), `--gpu-layers` (99, `0` for CPU only), `--reasoning` (`high`).

lair listens on localhost only by default. To reach it from other devices, set `LAIR_TOKEN`, bind it with `--host 0.0.0.0` and put HTTPS in front.

## Docker

You need Docker, plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) for a GPU.

```bash
curl -O https://raw.githubusercontent.com/bernardofernandezz/lair/main/compose.yaml
docker compose up                          # NVIDIA GPU through CUDA (RTX 20xx to 50xx)
docker compose --profile cpu up lair-cpu   # no NVIDIA GPU
```

Models go in `./models` (or wherever `LAIR_MODELS` points), and the port stays on localhost (`LAIR_PORT`, default 8080). The web search profile needs `searxng/settings.yml`, so run that one from a clone.

To build the image yourself, clone the repo and add `--build`. The first build compiles the CUDA kernels and takes a few minutes. Set `CUDA_ARCH` to your card (75 = RTX 20xx, 86 = 30xx, 89 = 40xx, 120 = 50xx) to make it faster, and `LAIR_BUILD_JOBS=2` if the machine is short on RAM.

## Build from source

You'll need Rust, [bun](https://bun.sh), cmake and clang. For Vulkan, also the Vulkan headers, `glslc` and the SPIR-V headers:

```bash
sudo pacman -S vulkan-headers shaderc spirv-headers          # Arch
sudo apt install libvulkan-dev glslc spirv-headers           # Ubuntu 24.04
```

```bash
(cd web && bun install && bun run build)    # the UI is embedded at compile time
cargo build --release --features vulkan     # or cuda, metal, or nothing for CPU only
./target/release/lair serve
```

Tests run with `cargo test` and `cd web && bun test`. The test that loads a real model is skipped by default; set `LAIR_TEST_MODEL=/path/to/model.gguf` and run `cargo test -- --include-ignored`.

<details>
<summary>Project layout</summary>

```
src/engine.rs    local GGUF models: generation, reasoning modes, <think> splitting
src/models.rs    every usable model, local or on a peer, behind one streaming generate
src/peers.rs     other lair nodes on the tailnet: discovery, whois trust, remote generation
src/store.rs     conversations in SQLite, with full-text search
src/turn.rs      one conversation turn: history, references, skills, the agent loop
src/agent/       tool-call protocol and parser, tools, approvals
src/skills/      skill catalog, installs from URL, zip or GitHub, index
src/server/      HTTP: chat-completions format, lair's own API, auth, UI
src/jail.rs      keeps paths inside a folder (workspace, assets)
web/             React UI: chat, sidebar, agent cards, skills panel, 3D scenes
skills/          built-in skills, embedded in the binary
```

</details>

## License

[MIT](LICENSE)
