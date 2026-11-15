# Contributing

Bug reports, fixes and new integrations are welcome. For anything bigger than a fix, open an issue first so we can agree on the shape before you write it.

## Build and test

You need Rust, [bun](https://bun.sh), cmake and clang ([more in the README](README.md#build-from-source)).

```bash
(cd web && bun install && bun run build)   # the UI is embedded at compile time
cargo build
cargo run -- serve
```

CI runs exactly this, and a pull request needs all of it green:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
(cd web && bun test)
```

Tests that load a real model are skipped unless you point them at one:
`LAIR_TEST_MODEL=/path/to/a-thinking-model.gguf cargo test --release -- --include-ignored`.

## How the code is written

- **Small and direct.** Use what the codebase already has before adding a helper, and the standard library before adding a dependency.
- **Comments say why**, not what. The module docs at the top of each file explain how its part works.
- **Every new branch of logic gets a test**, next to the code (`#[cfg(test)] mod tests`) or in `src/server/tests.rs` for the HTTP API.
- **Text users read is plain English**: errors say what went wrong and what to do.
- **Nothing leaves the machine** unless the user asked for it. Models, prompts and replies stay local. Don't add telemetry.

[The project layout](README.md#build-from-source) says where each part lives.

## Commits and pull requests

Commits follow [Conventional Commits](https://www.conventionalcommits.org) (`feat(api): …`, `fix: …`, `docs: …`). Keep a pull request to one change, say how you tested it, and add a line to `CHANGELOG.md` under **Unreleased** when users will notice it.
