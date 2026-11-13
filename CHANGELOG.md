# Changelog

## Unreleased

- Anthropic API: `/v1/messages` (streaming, tools, thinking, images) and `/v1/messages/count_tokens`,
  so Claude Code and other Anthropic clients run on local models. `x-api-key` is accepted as the
  token, and `claude-…` model names mean `auto`.

## 1.0.0

lair now works out what to run and how. It reads the hardware, ranks what is on Hugging Face for it,
estimates speed before a download, fits each model to the memory that is free, keeps several models
in memory, and can pick the model per message.

### The right model for your machine

- **Hardware:** CPU, cores, total and free RAM (macOS included), each GPU with total and free
  memory, the backend in use and whether lair runs inside WSL. Shown in Settings → Models.
- **Live recommendations:** the suggestions come from what is trending and most downloaded on
  Hugging Face plus the repos lair curates, each judged against the machine and listed with its
  reasons. The best pick is the biggest model from a trusted publisher that runs fully and fast at
  4 bits or better. The fixed list of nine repos remains as the offline fallback.
- **First run:** `lair serve` on an empty folder downloads the best model for the machine (6 GB at
  most) instead of always Qwen3 4B, which is now the fallback.
- **Speed estimates:** every generation is timed; from those measurements lair estimates tokens per
  second for models not installed yet. `lair bench` measures on purpose.
- **Per-model tuning:** GPU layers and context size are fitted per model when it loads (llama.cpp's
  own fit), from 8,192 up to 32,768 tokens of context. `--gpu-layers` and `--ctx` no longer default
  to 99 and 8192; set them to override the fit for every model. New `--threads`.
- **Several models in memory:** as many as fit; different models generate at the same time, and the
  one idle longest is unloaded when a new one needs the room.
- **`auto`:** a model id that picks, per request, among local models and the ones on other nodes.
  Nodes now tell each other how busy and how fast they are.

### Downloads

- Resume: an interrupted download continues from where it stopped.
- Verified: files are checked against the SHA-256 Hugging Face publishes before they become models.
- Gated and private repos, with `HF_TOKEN` or a token saved in Settings (sent to huggingface.co only).
- Models split into several files are listed and downloaded as one.
- `lair rm`, and delete and unload buttons in Settings → Models.

### APIs

- OpenAI: `/v1/completions`, `/v1/embeddings`, `/v1/models/{id}`; `response_format` (`json_object`,
  `json_schema`), `stop`, `top_k`, `min_p`, `repeat_penalty`, `tool_choice`,
  `stream_options.include_usage`; several tool calls in one reply; real token counts on replies that
  end in a tool call; timings on every response.
- Ollama: `/api/pull`, `/api/ps`, `/api/delete`, `/api/embed`, `/api/embeddings`; `format`,
  `keep_alive`, `raw`, durations; `/api/tags` and `/api/show` report family, parameter count and
  quantization read from the GGUF.

### Memory

- Chats and workspace documents are indexed in SQLite and searched by words and, with an embedding
  model, by meaning. The agent has a `search_memory` tool, and a chat is shown clearly related
  passages before it answers. On by default; turn it off or clear it in Settings → General.

### Changed

- The server starts with no models (download one from the UI) instead of exiting.
- Text the backend sends to models and API clients is in English throughout.
- `DELETE /lair/v1/hub/downloads` takes `repo` and `file` as query parameters; downloads are told
  apart by repo as well as file name.
