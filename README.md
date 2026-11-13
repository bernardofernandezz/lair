<div align="center">

# lair

**Your own AI on your own hardware.**

One install, any GGUF model: lair reads your hardware, picks the best model for it on Hugging Face and tunes how it runs.<br>
Web chat, terminal, agent, memory, and the OpenAI and Ollama APIs your tools already speak.<br>
All your machines become one AI. Prompts, replies and models stay on hardware you own.

[![Release](https://img.shields.io/github/v/release/bernardofernandezz/lair?style=flat-square)](https://github.com/bernardofernandezz/lair/releases)
[![License](https://img.shields.io/github/license/bernardofernandezz/lair?style=flat-square)](LICENSE)
![Platforms](https://img.shields.io/badge/platforms-linux%20%7C%20macOS%20%7C%20windows-555?style=flat-square)

[Install](#install) · [Usage](#usage) · [The right model](#the-right-model-for-your-machine) · [Use it from other apps](#use-it-from-other-apps) · [Features](#features) · [Configuration](#configuration) · [Build from source](#build-from-source)

</div>

---

lair is a single Rust binary built on [llama.cpp](https://github.com/ggml-org/llama.cpp). Point it at a folder of `.gguf` files and you get a chat UI in the browser, a chat in the terminal and an HTTP API that editors and tools already know how to talk to, in both the OpenAI and the Ollama formats. Turn on agent mode and the model can work with files, run commands and search the web, always asking before it does anything risky. Run it on a few machines in your Tailscale network and each one can use the others' models.

## Install

Linux (x86_64 and arm64) and macOS (Apple Silicon):

```bash
curl -fsSL https://raw.githubusercontent.com/bernardofernandezz/lair/main/install.sh | sh
lair serve
```

With [Homebrew](https://brew.sh) (macOS and Linux):

```bash
brew tap bernardofernandezz/lair https://github.com/bernardofernandezz/lair
brew install bernardofernandezz/lair/lair
```

Windows 10 or 11 (x86_64), in PowerShell:

```powershell
irm https://raw.githubusercontent.com/bernardofernandezz/lair/main/install.ps1 | iex
lair serve
```

Open http://localhost:8080 and start talking. The first run looks at your hardware, asks Hugging Face which model runs best on it (6 GB at most; bigger ones are yours to choose) and downloads that one, so you have something to chat with right away. Offline, or if Hugging Face doesn't answer, it falls back to Qwen3 4B (about 2.5 GB).

The installer puts `lair` in `~/.local/bin` (`%LOCALAPPDATA%\Programs\lair` on Windows) and doesn't need root. On Linux and Windows it picks the Vulkan build when a GPU driver is present (NVIDIA, AMD and Intel all work, no CUDA required) and the CPU build otherwise (always on arm64 Linux). Macs use Metal.

### Windows

- **Native** is the way to use your GPU. Data lives in `%LOCALAPPDATA%\lair`, and the agent's commands run in PowerShell. Other nodes on your tailnet aren't found on their own yet; name them with `--peer`.
- **WSL** works with the Linux installer, and your Windows browser reaches it at http://localhost:8080. It gets the CPU build: Vulkan inside WSL is emulated and slower than the CPU. For an NVIDIA GPU from WSL, use the CUDA image in [Docker](#docker).
- lair needs the [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe). Most machines already have it; install it if Windows says `VCRUNTIME140.dll`, `MSVCP140.dll` or `VCOMP140.dll` is missing.

Prefer containers? See [Docker](#docker).

## Usage

```bash
lair serve            # web UI and API on http://localhost:8080
lair run              # chat in the terminal
lair run llama        # ...with the model whose name contains "llama"
lair pull <repo>      # download from Hugging Face: the file that fits this machine best
lair list             # see what you have
lair rm <name>        # delete a model
lair bench            # measure how fast a model runs here
lair update           # switch to the latest release
```

Models live in `~/.local/share/lair/models` (`%LOCALAPPDATA%\lair\models` on Windows). Drop any `.gguf` file in there, or point lair somewhere else with `-m <folder>`. The easiest way to get more is **Settings → Models** in the web UI: it suggests models that fit your GPU or RAM, searches Hugging Face, and downloads them without a restart. From the terminal, `lair pull` does the same:

```bash
lair pull Qwen/Qwen3-8B-GGUF            # the quant that fits your GPU or RAM
lair pull Qwen/Qwen3-8B-GGUF:Q5_K_M     # ...or the one you name
lair pull Qwen3-8B-GGUF                 # a bare name is searched on Hugging Face
lair pull https://example.com/m.gguf    # any .gguf link
```

Vision models come with their projector (`mmproj`), so they can see images right away. Downloads from Hugging Face are checked against the hash it publishes, an interrupted one continues where it stopped when you start it again, and models split into several files come down whole. Gated and private repos need a token: set `HF_TOKEN`, or save one in **Settings → General**.

## The right model for your machine

lair works out what to run and how, so you don't have to read model cards and count gigabytes.

- **It reads the hardware.** CPU, cores, total and free RAM, each GPU with its total and free memory, and which backend generation runs on (CUDA, Vulkan, Metal or the CPU). **Settings → Models** shows what it found.
- **It asks Hugging Face what is good right now.** The suggestions are not a fixed list: lair takes what is trending and most downloaded among GGUF repos, adds the ones it vouches for, and judges each against your machine. The **best pick** is the biggest model from a publisher lair trusts that runs fully on your machine, fast, at 4 bits or better; trending breaks ties. Every suggestion says why it is where it is.
- **It tells you the speed before you download.** Once any model has run here, lair knows how fast this machine moves weights and estimates tokens per second for models you don't have yet (shown with a `~`). Each generation it measures makes the estimates better; `lair bench` measures on purpose. The estimate assumes speed follows model size, which holds for ordinary models and is off for mixture-of-experts ones (faster than estimated) and very small ones.
- **It tunes each model.** How many layers go to the GPU and how much context a model gets are fitted per model to the memory that is free when it loads, from 8,192 tokens up to 32,768. `--gpu-layers`, `--ctx` and `--threads` override that for every model.
- **It keeps several models in memory.** As many as fit. Different models answer at the same time; when a new one doesn't fit, the one idle longest makes room. **Settings → Models** shows what is loaded and lets you unload or delete.
- **`auto` picks per message.** Choose **Auto** in the model picker (or send `"model": "auto"` to the API) and each request goes to the biggest model that is fast right now, on this machine or another of yours, taking into account what is already loaded and how busy each machine is. A message with images goes to a local model that sees. The reply says which model answered and why.

## Use it from other apps

lair speaks three APIs on the same port, so most apps that run local models work with it unchanged.

**Apps made for Ollama:** point them at `http://localhost:8080` instead of `http://localhost:11434`, or start lair with `--port 11434` so their default address works (stop Ollama first if it runs there). lair answers `/api/chat`, `/api/generate`, `/api/embed`, `/api/embeddings`, `/api/tags`, `/api/show`, `/api/ps`, `/api/pull`, `/api/delete` and `/api/version`. `/api/pull` and `lair pull` take Hugging Face repos (`Qwen/Qwen3-4B-GGUF`, `hf.co/owner/repo:QUANT`) and the common names from Ollama's library (`qwen3`, `llama3.2:3b`, `gemma3`, `deepseek-r1:7b`, `gpt-oss`…), each mapped to a fixed repo from a known publisher.

**Coming from Ollama?** The models it already downloaded show up in lair under the same names, read from `~/.ollama/models` (or `OLLAMA_MODELS`) without a second copy. lair never deletes them; Ollama's vision models load without their image projector.

```bash
curl localhost:8080/api/chat -d '{
  "model": "Qwen3-4B-Q4_K_M",
  "messages": [{ "role": "user", "content": "Is 221 prime?" }]
}'
```

**Apps made for OpenAI:** set the base URL to `http://localhost:8080/v1` and any API key. lair answers `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` and `/v1/models`.

```bash
curl localhost:8080/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Qwen3-4B-Q4_K_M",
  "messages": [{ "role": "user", "content": "Is 221 prime?" }]
}'
```

**Apps made for Anthropic, Claude Code included:** set the base URL to `http://localhost:8080` and any API key. lair answers `/v1/messages` (streaming, tools, thinking, images) and `/v1/messages/count_tokens`. A model named `claude-…` means `auto`; name a lair model to pick one.

```bash
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_API_KEY=local ANTHROPIC_MODEL=Qwen3-8B-Q4_K_M claude
```

Claude Code's prompt alone is about 17,000 tokens, so start lair with `--ctx 32768` and use a GPU: on a CPU the first reply takes minutes.

All of them accept `tools` (function calling) and images, which is what coding agents and chat front-ends rely on:

- **Tools:** pass `tools` as usual and calls come back as `tool_calls`, several in one reply when the model makes several. Any model works: lair asks for calls in plain text and parses them, so it doesn't depend on the model's chat template supporting tools. `tool_choice` is honoured as an instruction to the model.
- **Structured output:** `response_format` with `json_object` or a `json_schema` (OpenAI), or `format` (Ollama), constrains the answer to valid JSON matching the schema, token by token.
- **Embeddings:** with an embedding model installed (lair suggests one in **Settings → General**), `/v1/embeddings` and `/api/embed` return vectors made on this machine.
- **Sampling and stops:** `stop`, `top_k`, `min_p`, a repeat penalty and `seed` are honoured, and `stream_options.include_usage` adds token counts to a stream. Responses carry timings, tokens per second included.
- **Images:** send them inline (`data:image/...;base64,` URLs in OpenAI's format, `images` in Ollama's) to a vision model. Links to images are refused, so the server never fetches anything a client points it at.
- **Reasoning:** `reasoning_effort` (OpenAI) or `think` (Ollama) turns thinking off, light or deep, and it comes back in `reasoning_content` or `thinking`.

If you set `LAIR_TOKEN`, clients must send it as `Authorization: Bearer <token>`.

## Features

|  |  |
| --- | --- |
| **Any GGUF model** | Every `.gguf` in the folder shows up in the model picker. As many as fit stay in memory, and different ones answer at the same time. |
| **The right model, tuned** | lair reads your hardware, ranks what is on Hugging Face for it, estimates each model's speed before you download, and fits GPU layers and context per model. |
| **Auto** | One model id that picks, per message, the biggest model that is fast right now across your machines. |
| **Memory** | Your chats and workspace files are indexed on your machine and found again by meaning. |
| **Reasoning you control** | Models that think before answering get three modes: off, light or deep. The thinking is kept apart from the answer, in the UI and in the API. |
| **Conversations that stay** | Saved on the server in SQLite, searchable, and citable from another chat with `@`. |
| **Agent mode** | The model reads and writes files, runs commands, searches the web and digs through past chats. You approve each risky step. |
| **Skills** | Prompt packages that kick in on trigger words. Write them in the UI or install them from a URL. |
| **Your machines, one network** | lair nodes on your Tailscale network find each other and share their models. |
| **Works with your tools** | OpenAI (`/v1`), Anthropic (`/v1/messages`) and Ollama (`/api`) compatible, with streaming, tool calling, images, structured output and embeddings. |

### Agent mode

Switch on **Agent** in the composer and the model can act instead of only writing. It calls a tool, lair runs it and hands back the result, up to 20 steps per reply (`--max-steps` or `LAIR_MAX_STEPS` changes that). Each step shows up in the chat as a card.

| Tool | What it does | Asks first |
| --- | --- | :---: |
| `list_dir`, `read_file` | Look at files in the workspace | |
| `write_file`, `edit_file`, `delete_file` | Change files in the workspace | ✓ |
| `run_command` | Run a shell command in the workspace (60 s limit) | ✓ |
| `web_search`, `fetch_url` | Search through your SearXNG, read a page | ✓ |
| `search_conversations`, `read_conversation` | Look through saved conversations | |
| `search_memory` | Find passages of past chats and workspace files by meaning | |

Before a tool changes something or goes online, lair shows the exact arguments and waits for you to allow it once, allow it for the rest of the conversation, or deny it. File paths can't leave the workspace (`<data>/workspace` by default).

> [!WARNING]
> Shell commands start in the workspace but aren't sandboxed beyond that. Outside Docker they run as your user, so read each one before you allow it.

Web search needs a [SearXNG](https://github.com/searxng/searxng) instance with JSON output on. `docker compose --profile web up` starts one next to lair, or pass `--searxng <url>` to use yours.

Small models sometimes skip a step. Asking again, or switching to a bigger model, usually sorts it out.

### Reasoning modes

For models that write a `<think>` block before answering:

| In the UI | `reasoning_effort` | What happens |
| --- | --- | --- |
| Off | `none` | Answers straight away |
| Light | `low` | Thinks for up to ~512 tokens, then answers |
| Deep | `high` (default) | Thinks as long as it needs |

The API returns the thinking in `reasoning_content` and the answer in `content`, both in full responses and in streamed deltas.

### Conversations

Chats are stored on the server, so they follow you to any device that opens lair. The sidebar searches them by title and content. Type `@` in a message to cite another conversation: its latest part goes into the prompt as quoted context.

### Memory

lair keeps an index of your chats and of the documents in the agent's workspace (text, code, Markdown, PDFs), on your machine, in the same SQLite file as the conversations. Only what changed is read again.

- **By words, always.** The agent's `search_memory` tool finds passages that share words with a question.
- **By meaning, with an embedding model.** **Settings → General** offers to download one (bge-m3, 0.4 GB, multilingual). Passages are then also found when they say the same thing in other words, and a chat is shown the passages clearly related to your question before it answers. The chat tells you when it did, and from which chat or file.
- **Yours to control.** Turn memory off, or make it forget everything, in **Settings → General**. Deleting a conversation removes it from memory.

What memory recalls goes into the prompt as information, with the same warning lair puts on cited chats. A page the agent saved to the workspace can still come back in a later chat, so the agent's approvals stay as they are.

### Skills

A skill is a folder with two files. It runs when one of its `when` phrases shows up as a whole word in your message (case and accents don't matter), and at most one runs per reply.

```
pirate/skill.json   { "name": "Pirate", "description": "Talks like a pirate.", "when": ["pirate", "arr"] }
pirate/prompt.md    Answer like an 18th-century pirate.
```

Manage them in **Settings → Skills**. You can install one from a `.zip`, a GitHub folder (`github.com/owner/repo/tree/branch/folder`) or any URL that serves those two files, and only those two files are ever read from a download. lair ships with one, **3D scenes**, which answers "3D" requests with an interactive three.js scene.

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
| `LAIR_DATA` | `~/.local/share/lair`, `%LOCALAPPDATA%\lair` on Windows | Conversations, skills and the agent's workspace |
| `LAIR_DEFAULT_MODEL` | `auto` | What a first run downloads when there are no models: `auto` picks the best for this machine, a `.gguf` URL names one, `""` turns it off |
| `HF_TOKEN` | unset | Hugging Face token, for gated or private models |
| `LAIR_SEARXNG` | unset | SearXNG URL for the agent's web search |
| `LAIR_PEERS` | unset | Extra lair nodes, as `host:port`, comma-separated |

Flags only: `--port` (8080), `--host` (127.0.0.1), `--reasoning` (`high`), and three that are fitted per model unless you set them: `--ctx` (context in tokens), `--gpu-layers` (`0` for CPU only) and `--threads`.

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

On Windows: Rust (MSVC), bun, cmake and LLVM (`winget install Kitware.CMake LLVM.LLVM`), plus the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) for `--features vulkan`.

```bash
(cd web && bun install && bun run build)    # the UI is embedded at compile time
cargo build --release --features vulkan     # or cuda, metal, or nothing for CPU only
./target/release/lair serve
```

Tests run with `cargo test` and `cd web && bun test`. The tests that load real models are skipped by default; set `LAIR_TEST_MODEL=/path/to/a-thinking-model.gguf` and `LAIR_TEST_EMBED_MODEL=/path/to/an-embedding-model.gguf`, and run `cargo test --release -- --include-ignored`.

<details>
<summary>Project layout</summary>

```
src/engine.rs    local GGUF models: the pool in memory, per-model fitting, generation, embeddings
src/models.rs    every usable model, local or on a peer, behind one streaming generate; `auto`
src/hw.rs        what this machine has: CPU, memory, GPUs, backend
src/hub.rs       Hugging Face: what is trending, what fits, which file to take
src/download.rs  downloads that resume and are verified; removing models
src/speed.rs     measured speeds, estimates for other models, the `auto` policy
src/memory.rs    chats and workspace files indexed and found again by words and meaning
src/peers.rs     other lair nodes on the tailnet: discovery, whois trust, remote generation
src/store.rs     conversations and memory in SQLite, with full-text search
src/turn.rs      one conversation turn: history, references, skills, the agent loop
src/agent/       tool-call protocol and parser, tools, approvals
src/skills/      skill catalog, installs from URL, zip or GitHub, index
src/server/      HTTP: OpenAI and Ollama formats, lair's own API, auth, UI
src/jail.rs      keeps paths inside a folder (workspace, assets)
web/             React UI: chat, sidebar, agent cards, skills panel, 3D scenes
skills/          built-in skills, embedded in the binary
```

</details>

## License

[MIT](LICENSE)
