# Security

lair runs models, shell commands and file edits on your machine, so security bugs matter here.

## Reporting a vulnerability

Report it privately through [GitHub's vulnerability reporting](https://github.com/bernardofernandezz/lair/security/advisories/new), not in a public issue. Say what an attacker can do, and how to reproduce it. You'll get an answer within a week.

## What counts

- Reaching the API or UI without the token when `LAIR_TOKEN` is set, or from outside the tailnet when trust is limited to it.
- An agent tool running without approval, or a path escaping the workspace.
- A download that isn't checked against its hash, or a model, skill or URL that makes lair fetch or run something it shouldn't.
- Anything that sends prompts, replies or files off the machine without being asked.

Not a vulnerability: commands you approved doing what they say (the README warns that `run_command` isn't sandboxed outside Docker), or a skill you installed steering the model.

Only the latest release gets fixes.
