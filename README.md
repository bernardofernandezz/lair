# lair

Run LLMs locally (GGUF via llama.cpp) with a web chat and an OpenAI-compatible API. Nothing leaves your machine.

## Run with Docker

```bash
LAIR_MODELS=~/models docker compose up --build
```

Open http://localhost:8080. Every `.gguf` under `LAIR_MODELS` (searched recursively) shows up in the model picker. Only one model sits in memory at a time; picking another one swaps it.

| Variable      | Default    | What it does                                                          |
| ------------- | ---------- | --------------------------------------------------------------------- |
| `LAIR_MODELS` | `./models` | Folder with your `.gguf` files, mounted read-only                     |
| `LAIR_PORT`   | `8080`     | Port on the host, bound to localhost only                             |
| `LAIR_TOKEN`  | unset      | When set, the UI and API ask for this token                           |
| `CUDA_ARCH`   | `86`       | GPU generation to compile for (75 = RTX 20xx, 86 = 30xx, 89 = 40xx)   |

If the container restarts with `no usable .gguf chat models in ["/models"]`, `LAIR_MODELS` points at a folder with no `.gguf` files.

- The default image uses an NVIDIA GPU through the NVIDIA Container Toolkit (CDI). The first build compiles the CUDA kernels and takes a while.
- No NVIDIA GPU: `docker compose --profile cpu up --build lair-cpu`.
- To reach it from other devices, set `LAIR_TOKEN`, change the port mapping in `compose.yaml` to `"${LAIR_PORT:-8080}:8080"`, and put HTTPS in front of it.

## Reasoning

Models that think (they emit a `<think>` block) get three modes, chosen in the UI or sent as `reasoning_effort`:

| Mode      | `reasoning_effort` | Behavior                                         |
| --------- | ------------------ | ------------------------------------------------ |
| Desligado | `none`             | Answers straight away                            |
| Leve      | `low`              | Thinks for up to ~512 tokens, then has to answer |
| Profundo  | `high` (default)   | Thinks as long as it needs                       |

The reasoning never reaches the answer text. The API returns it in `reasoning_content`, next to `content`, both in full responses and in streamed deltas.

## Skills

Skills are small prompt packages added to a request when the user's message asks for them. The built-in one is **Cenas 3D**: a message that mentions "3D" or "três dimensões" gets an answer with an interactive three.js canvas inline in the chat, with orbit, reset and fullscreen controls. Other messages are sent to the model untouched.

Scenes can use local `.glb`/`.gltf` files: pass `--assets <folder-or-file>` (repeatable) or `LAIR_ASSETS`, and the UI loads them from `/v1/assets/<id>`.

To add your own skills, point lair at a folder (with Docker, mount it and set `LAIR_SKILLS` to the mount):

```bash
./target/release/lair serve -m ~/models --skills-dir ~/lair-skills
```

Each skill is a folder with two files. A skill with the same id as a built-in replaces it.

```json
// ~/lair-skills/pirate/skill.json
{ "name": "Pirata", "when": ["pirata", "arr"] }
```

```markdown
<!-- ~/lair-skills/pirate/prompt.md -->
Responda como um pirata do século XVIII, sem quebrar o personagem.
```

A skill runs when one of its `when` phrases appears as a whole word in the last user message (case and accents are ignored); at most one per reply. Skills are trusted prompt files: only point `LAIR_SKILLS` at a folder you control. When a skill runs, streamed replies start with an SSE event named `lair.skill` (`{"id":"3d","name":"Cenas 3D"}`) and full replies include `"lair": {"skills": [...]}`. `--no-skills` turns them off.

## Without Docker

Needs Rust, bun, cmake and clang.

```bash
(cd web && bun install && bun run build)   # the UI is embedded into the binary
cargo build --release                      # add --features cuda for NVIDIA GPUs
./target/release/lair serve -m ~/models
./target/release/lair chat -m ~/models --reasoning low
```
