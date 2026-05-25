# Memory Contamination in Multi-Agent AI College Counseling

**Does per-student memory in shared AI infrastructure cause cross-student data leakage?**

This repository contains the code, experiments, and paper for our CAISc 2026 submission. We study whether multi-tenant college counseling AI agents — where each student gets a per-session memory agent on shared infrastructure — exhibit cross-student data contamination, and under what conditions.

> **Core claim:** Per-student memory in shared-infrastructure agents causes cross-student data contamination, degrading accuracy and consistency — especially when student profiles are similar.

---

## Key findings

- With explicit student stats in the system prompt: **0% contamination** (memory agent acts as a clean anchor)
- With generic system prompts relying on conversation history: contamination emerges — agents confuse GPA, SAT scores, and extracurriculars across students
- High-similarity student cohorts (same major, state, GPA range) show higher contamination than diverse cohorts
- The shared sliding-window agent propagates injected wrong stats to subsequent students in the same session

---

## Research methodology

This study uses an **autonomous research loop** inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch). Two AI systems collaborate:

```
Ralph (DeepSeek) proposes experiment → Claude critiques plan
       ↓ approved                              ↓ rejected (with feedback)
Ralph modifies experiment.py & runs it    ←────┘
       ↓
Claude scores results (1-10)
       ↓ score ≥ 6            ↓ score < 6
git commit (kept)         git revert (discarded)
       ↓
Next iteration
```

| autoresearch | this project |
|---|---|
| `train.py` | `all-spikes/memory-contamination/experiment.py` |
| GPU val_bpb | `contamination_rate` + `accuracy_score` |
| Claude Code agent | Ralph (DeepSeek) |
| Runs overnight | Runs until 20 iters or loop complete |

The loop is **fully automated** — no manual steps. Ralph proposes and executes via subprocess; Claude reviews and scores via API.

---

## Experiment design

Each iteration runs two agent architectures on the same set of synthetic students:

| Architecture | Description |
|---|---|
| **Memory agent** | Per-student conversation history in system prompt. Accumulates context across rounds. |
| **Shared agent** | Single sliding window across all students. Later students see earlier students' history. |

Synthetic students are generated with configurable similarity conditions (`high_similarity`, `low_similarity`, `mixed`). A **poison-pill** injection design explicitly tests memory persistence: wrong stats (fabricated GPA, SAT, ECs) are injected into a student's conversation history, then we measure whether the agent incorporates them in subsequent rounds and whether they propagate to other students in the shared agent.

**Metrics reported per experiment:**
- `personalization_score` (1–5)
- `accuracy_score` (1–5)
- `hallucination_score` (1–5, higher = less hallucination)
- `consistency_score` (1–5)
- `contamination_rate` (0.0–1.0)
- `poison_incorporation_rate` (r2: immediate, r3: delayed)
- `poison_propagation_rate` (shared agent only)

---

## Reproduce

### Requirements

- Docker
- Anthropic API key
- Ralph API key (for the DeepSeek proposer/executor agent)

### Setup

```bash
# Clone
git clone https://github.com/happyahluwalia/agent-memory-contamination
cd agent-memory-contamination

# Add API keys
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and RALPH_API_KEY

# Build Docker environment
docker compose build
docker compose run --rm research

# Inside container
cd /research/project
git config user.email "you@example.com"
git config user.name "Your Name"
git add all-spikes/memory-contamination/experiment.py
git commit -m "baseline experiment.py"
```

### Run the loop

```bash
# Full run (up to 20 iterations, paper auto-written at end)
python scripts/research.py

# Short test (3 iterations)
python scripts/research.py --max-iter 3

# Resume after a crash
python scripts/research.py --resume
```

The loop writes `all-spikes/memory-contamination/paper_draft.md` automatically when complete.

### Run a single experiment directly

```bash
cd all-spikes/memory-contamination
python experiment.py
```

---

## Repository structure

```
scripts/
├── research.py              ← autonomous loop orchestrator
├── experiment.py            ← baseline experiment template
└── write_paper_prompt.py    ← generates paper-writing prompt from results

all-spikes/memory-contamination/
├── experiment.py            ← the file Ralph modifies each iteration
├── results_history.json     ← all iteration records (kept + rejected)
├── best_result.json         ← current best kept result
├── paper_draft.md           ← auto-generated paper (after loop completes)
└── results_*.json           ← raw per-experiment outputs

program.md                   ← research agenda and hypothesis space
voila.md                     ← research philosophy and paper-writing guide
ralph.md                     ← Ralph agent standing instructions
docker-compose.yml
dockerfile
requirements.txt
```

---

## Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | Claude API — used for critique, scoring, evaluation, and paper writing |
| `RALPH_API_KEY` | Ralph API — used for experiment proposal and execution |

---

## Citation

```bibtex
@inproceedings{ahluwalia2026memory,
  title     = {Memory Contamination in Multi-Agent AI College Counseling:
               A Study of Per-Student Agent Architecture},
  author    = {Ahluwalia, Harpreet},
  booktitle = {Conference for AI Scientists (CAISc) 2026},
  year      = {2026},
  url       = {https://github.com/happyahluwalia/agent-memory-contamination}
}
```

---

## AI involvement

This research was conducted using an autonomous AI loop. Ralph (DeepSeek) proposed and executed all experiment modifications. Claude (Anthropic) critiqued experiment plans, scored results, and wrote the paper draft. Human involvement was limited to: defining the research question, setting evaluation thresholds, reviewing the paper draft, and repository maintenance.

Full AI Involvement Checklist included in `paper_draft.md` per CAISc 2026 submission requirements.

---

## Credits

- **[autovoila](https://github.com/paraschopra/autovoila)** by [Paras Chopra](https://github.com/paraschopra) — the autonomous research loop scaffolding this project is built on
- **[Ralph](https://ralphy-server.fly.dev/)** — the DeepSeek-powered coding agent that proposes and executes experiments