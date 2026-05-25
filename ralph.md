## Research checkpointing
- Save progress to all-spikes/<slug>/progress.md after every major step
- Commit intermediate results to disk before long experiment runs
- If resuming, read progress.md first before taking any action

# Ralph

An agentic code generation CLI powered by DeepSeek. Ralph operates via an explore → understand → implement → verify loop, with full session persistence, context compaction, checkpointing, and a rich tool set for reading, searching, and editing code.
# RALPH.md — Research Agent Instructions

## Project
Autonomous research on multi-agent AI for college counseling.
We are studying whether per-student memory causes cross-student data contamination.

## Stack
- Python 3, Anthropic SDK (claude-sonnet-4-6)
- No GPU, no local models, no databases
- `pip install anthropic` is the only dependency

## Your role in the loop
You are the PROPOSER and EXECUTOR. Claude is the CRITIC.

The loop:
1. Read `propose_prompt.txt` → write a proposed experiment plan to stdout
2. Claude reviews your plan (you'll see approve/reject + feedback)
3. Read `execute_prompt.txt` → modify `experiment.py` → run it → report results

## Rules
- Only modify `experiment.py` — never touch `research_loop.py` or `program.md`
- Always run `python experiment.py` after modifying it and capture full output
- If experiment.py crashes, fix it and rerun — don't report a crash as results
- Report results in the structured format specified in execute_prompt.txt
- Keep changes incremental — one hypothesis per iteration
- Never increase N_STUDENTS above 30 (API cost control)

## Skills
Place in `.ralph/skills/`:
- `research-propose.md` → reads propose_prompt.txt, outputs experiment plan
- `research-execute.md` → reads execute_prompt.txt, modifies experiment.py, runs it

## Key files
- `experiment.py` — the ONLY file you modify
- `propose_prompt.txt` — written by the loop, read by you to propose
- `execute_prompt.txt` — written by the loop, read by you to execute
- `results_history.json` — all prior iteration results (read-only context)
- `loop_log.jsonl` — append-only log (do not modify)

### Single-shot mode

```bash
ralph "write unit tests for the auth module"

# Use DeepSeek pro model (deepseek-v4-pro with extended thinking)
ralph --pro "refactor the auth module"

# Target a specific directory
ralph "fix the failing tests" --workspace /path/to/project

# Limit iterations
ralph "..." --max-turns 20

# Preview actions without executing
ralph "..." --dry-run

# Skip confirmation prompts
ralph "..." --no-confirm
```

---

## Tools

Ralph exposes the following tools to the LLM. Read-only tools run in parallel when the agent issues several at once.

### Exploration (read-only)

| Tool | Description |
|------|-------------|
| `read_file_outline` | Signatures + line numbers without bodies — survey large files first |
| `read_file` | Read a file or a range (`offset`/`limit`); default 150 lines |
| `search_in_file` | Regex search within one file with surrounding context lines |
| `search_codebase` | Regex search across the workspace; supports `glob` filter and `context` lines |
| `glob` | List files matching a pattern (`**/*.py`, `src/**/*.rs`) |
| `list_dir` | Directory tree, respects `.gitignore` |
| `find_symbol` | Symbol index — locate functions/classes/structs by name |
| `read_symbol` | Read the full body of a named symbol |
| `load_files` | Load all files matching a glob pattern into context at once |
| `explain_code` | Project type, directory tree, entry points |
| `recall` | Look up stored project facts |

### Editing

| Tool | Description |
|------|-------------|
| `edit_file` | Exact string replacement; on failure shows the closest matching block |
| `edit_file_multi` | Multiple replacements in one file, one atomic call |
| `write_file` | Create or overwrite a file |
| `delete_file` | Delete a file |
| `view_diff` | Show `git diff HEAD` — review all changes before finishing |

### Execution

| Tool | Description |
|------|-------------|
| `run_test` | Run tests (no confirmation needed) |
| `run_build` | Run build (no confirmation needed) |
| `run_command` | Any shell command (one-time confirmation) |

### Control

| Tool | Description |
|------|-------------|
| `ask_user` | Ask the user a clarifying question |
| `remember` / `recall` | Persistent project memory across sessions |
| `declare_done` | Signal task complete |
| `declare_failed` | Signal task cannot be completed |

---

## Agentic Loop

Ralph follows an explicit four-phase loop per turn:

1. **Explore** — use `find_symbol`, `search_codebase`, `glob`, or `read_file_outline` to locate relevant code; multiple reads run in parallel
2. **Understand** — read the minimum necessary: the target function, its tests, immediate callers
3. **Implement** — apply changes with `edit_file_multi` (multi-location edits in one call) or `edit_file`
4. **Verify** — run tests; if they fail, fix them before calling `declare_done`

**Adaptive reasoning:** When `edit_file` fails 3+ times (wrong text), or tests fail 2+ times in a row, Ralph automatically switches to an extended-thinking reasoning pass for the next turn.

**Context-aware nudges:** If Ralph gets stuck reading without editing, or edits without testing, it receives a targeted nudge specific to the situation rather than a generic "please proceed" message.

---

## Sessions

```bash
# Resume the most recent session for this workspace
ralph --resume

# Resume a specific session by ID
ralph --resume abc12345

# Force a new session
ralph "..." --new-session

# List all sessions
ralph sessions list

# Delete sessions older than 30 days
ralph sessions clean --older-than 30
```

---

## Checkpoints

Snapshots of file state + conversation history for safe revert.

```bash
# Auto-checkpoint before destructive operations
ralph "..." --auto-checkpoint

# List checkpoints
ralph checkpoint list

# Revert files to a checkpoint (keep conversation history)
ralph checkpoint revert my-snapshot --files-only

# Revert files and conversation history
ralph checkpoint revert my-snapshot
```

---

## Configuration

Ralph merges two config files:

- **Global**: `~/.ralph/config.toml`
- **Workspace**: `.ralph.toml` in the project root (overrides global)

```toml
[defaults]
provider = "deepseek"
max_turns = 30

[providers.deepseek]
model = "deepseek-chat"

[search]
brave_api_key_env = "BRAVE_API_KEY"
serp_api_key_env  = "SERP_API_KEY"

[compaction]
enabled = true
threshold_pct = 80
keep_recent_turns = 3

[checkpoints]
auto_checkpoint_before_destructive = false
```

---

## Output Formats

```bash
ralph "..."                # default color terminal
ralph "..." --output json  # JSON events (for scripting)
ralph "..." --verbose      # full LLM interactions
```

---

## Models

| Flag | Model | Description |
|------|-------|-------------|
| *(default)* | `deepseek-v4-flash` | Fast, efficient — default |
| `--pro` | `deepseek-v4-pro` | Extended thinking, stronger reasoning |
| `--model <name>` | any | Override model explicitly |

---

## Data Storage

```
~/.ralph/
  config.toml       # global config
  sessions/         # session state, messages, checkpoints
  logs/             # human-readable logs (YYYY-MM-DD_<id>.log)
```

---

## Releasing

Releases are automated via GitHub Actions. To publish a new version:

```bash
# 1. Bump version in Cargo.toml, then update the lock file
cargo build

# 2. Commit and tag
git add Cargo.toml Cargo.lock
git commit -m "chore: release vX.Y.Z"
git tag vX.Y.Z
git push origin main --follow-tags
```

The `release.yml` workflow builds binaries for five targets in parallel and attaches them to the GitHub Release automatically:

| Platform | Target |
|----------|--------|
| Linux x86-64 | `x86_64-unknown-linux-gnu` |
| Linux ARM64 | `aarch64-unknown-linux-gnu` |
| macOS Apple Silicon | `aarch64-apple-darwin` |
| macOS Intel | `x86_64-apple-darwin` |
| Windows x86-64 | `x86_64-pc-windows-msvc` |

---

## Development

```bash
cargo test          # run all tests
cargo clippy        # lint
cargo fmt           # format
```

Tests run in isolated temporary directories — no files are written to your project.
