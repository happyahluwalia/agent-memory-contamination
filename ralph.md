## Research checkpointing
- Save progress to all-spikes/<slug>/progress.md after every major step
- Commit intermediate results to disk before long experiment runs
- If resuming, read progress.md first before taking any action

# Ralph

An agentic code generation CLI powered by DeepSeek. Ralph operates via an explore → understand → implement → verify loop, with full session persistence, context compaction, checkpointing, and a rich tool set for reading, searching, and editing code.

## Install

### cargo-binstall (recommended)

```bash
cargo binstall ralph-coder
```

Downloads the pre-built binary for your platform — no compilation required. The installed command is `ralph`. Install `cargo-binstall` first if you don't have it:

```bash
cargo install cargo-binstall
```

### Build from source

```bash
cargo install ralph-coder
```

Compiles from source via crates.io. Requires Rust 1.75+ — install via [rustup](https://rustup.rs).

---

## API Keys

```bash
export DEEPSEEK_API_KEY=sk-...   # required

# Optional — enables web search
export BRAVE_API_KEY=BSA...      # Brave Search
export SERP_API_KEY=...          # SerpAPI (fallback)
```

Web search activates automatically when either key is set. Without a search key Ralph works offline using codebase search only.

---

## Usage

### Interactive mode

Launch without a prompt for a persistent chat session:

```bash
ralph
```

Type tasks at the `>>` prompt. Context (files read, edits made, history) carries across tasks in the same session.

```
>> explain the structure of this codebase
>> add error handling to the database module
>> run the tests and fix any failures
>> exit
```

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
