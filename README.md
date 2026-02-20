# OneContext

OneContext is an Agent Self-Managed Context layer, it gives your team a ***unified context for ALL AI Agents***, so anyone / any agent can pick up from the same context.

## Key Features
1. Run your Agents with OneContext and it records your Agents trajectory.
2. Share Agent Context so Anyone can Talk to it on Slack.
3. Load the Context to Agents, so Anyone can Continue from the Same Point.

## Quick Install

**Option A: From npm (recommended for end users)**
```bash
npm i -g onecontext-ai
```
This will automatically install the latest `aline-ai` Python package using the best available Python package manager (`uv` > `pipx` > `pip3` > `pip`).

**Option B: From source (for development)**
```bash
git clone https://github.com/Datar-Tech/OneContext.git
cd OneContext/python
pip install -e ".[dev]"
```

**Prerequisites**
* Node.js >= 16 (for npm install)
* Python 3.11+ with one of: `uv`, `pipx`, `pip3`, or `pip`

## Quick Start
Run:
```bash
onecontext
```

## Usage
After installation, the following commands are available:
```bash
onecontext-ai <command> [args...]
onecontext <command> [args...]
oc <command> [args...]
```
All three commands are equivalent and proxy to the underlying Python CLI.

**Examples**
```bash
# Check version
onecontext version

# Show help
onecontext --help

# Short alias
oc version
```

## Project Structure

```
OneContext/
├── npm/                # Node.js CLI wrapper (proxies to Python CLI)
│   ├── bin/            # CLI entry points (onecontext-ai, onecontext, oc)
│   ├── run.js          # Command resolver and process spawner
│   └── postinstall.js  # Auto-installs Python package on npm install
│
├── python/             # Core Python package (aline-ai / realign)
│   ├── pyproject.toml  # Python package configuration
│   └── realign/        # Main module (~57,000 lines)
│       ├── cli.py              # CLI entry point (Typer)
│       ├── adapters/           # Agent adapters (Claude, Codex, Gemini)
│       ├── claude_hooks/       # Claude Code hooks integration
│       ├── codex_hooks/        # Codex hooks integration
│       ├── commands/           # CLI subcommands
│       ├── dashboard/          # TUI dashboard (Textual)
│       ├── db/                 # SQLite database layer
│       ├── events/             # Event & summarizer system
│       ├── triggers/           # Turn detection triggers
│       └── ...                 # Auth, config, LLM client, etc.
│
├── assets/             # Documentation screenshots
├── Documentation.md    # Detailed usage guide
└── LICENSE             # MIT License
```

## Updating the Python Package
The npm wrapper installs the latest `onecontext` on `npm` install.
For normal users, use the unified upgrade command:
```bash
# Recommended
onecontext update
```

If upgrade routing is broken, repair once and retry:
```bash
onecontext doctor --fix-upgrade
onecontext update
```
Use `npm rebuild onecontext-ai` only when the npm wrapper links/cached paths are stale.

## Troubleshooting
If the commands aren't found after installation:
1. Repair and update: `onecontext doctor --fix-upgrade && onecontext update`
2. Rebuild wrapper links if needed: `npm rebuild onecontext-ai`
3. Check that `onecontext` is on your PATH: `which onecontext`

## Issue & Feedback
Feel free to raise an issue ticket for any problems you encounter, or for any features or improvements you'd like to see in the future 🙂

## Update
***14-02-2026: v0.8.3 Release - Import your past Codex/Claude sessions as Context, so to keep working from that context across sessions, devices, and agents***

07-02-2026: First Release
