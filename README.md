# Autonomous Research Loop
### College Counseling Multi-Agent Study

Karpathy-style autonomous research loop adapted for API-based experiments.

```
DeepSeek/Ralph proposes → Claude critiques → Ralph executes → Claude scores → keep/revert → repeat
```

Analogous to autoresearch:
| autoresearch | this project |
|---|---|
| `train.py` | `experiment.py` |
| `program.md` | `program.md` |
| GPU val_bpb | Claude contamination_rate score |
| Claude Code agent | Ralph (DeepSeek via ralph-server) |
| loops overnight | loops until 20 iters or 5 kept |

---

## Setup

```bash
# 1. Docker (nothing on your Mac)
docker compose build
docker compose run --rm research

# 2. Inside container — env vars come from .env via docker-compose
# Verify they loaded:
echo $ANTHROPIC_API_KEY | head -c 20
echo $RALPH_API_KEY | head -c 20

# 3. Move to project dir (all scripts run from here)
cd /research/project

# 4. Commit baseline so git revert works (loop auto-bootstraps experiment.py)
git add scripts/experiment.py
git commit -m "baseline experiment.py" --allow-empty
```

---

## Run the loop

```bash
cd /research/project

# Standard run (up to 20 iterations)
python scripts/research.py

# Short test run (3 iterations)
python scripts/research.py --max-iter 3

# Resume after a crash
python scripts/research.py --resume
```

---

## How each iteration works

**Step 1 — Ralph proposes**
The loop writes `propose_prompt.txt` and pauses.
Open a second terminal and run Ralph:
```bash
cd all-spikes/memory-contamination
ralph
# At the prompt:
> /research-propose
```
Copy Ralph's output, paste it into the loop terminal, press Ctrl+D.

**Step 2 — Claude reviews**
Automatic. Claude scores the plan 1-10 and either approves or rejects.
If rejected, the loop goes back to step 1 with Claude's feedback baked in.

**Step 3 — Ralph executes**
The loop writes `execute_prompt.txt` and pauses again.
In the Ralph terminal:
```bash
> /research-execute
```
Paste results back, press Ctrl+D.

**Step 4 — Claude scores**
Automatic. Claude scores results 1-10.
If score >= 6: `git commit` (kept).
If score < 6: `git checkout -- experiment.py` (reverted).

---

## Files

```
all-spikes/memory-contamination/
├── experiment.py          ← Ralph modifies this each iteration
├── program.md             ← research agenda (read-only)
├── RALPH.md               ← Ralph's standing instructions
├── .ralph/skills/
│   ├── research-propose.md
│   └── research-execute.md
├── loop_log.jsonl         ← append-only event log
├── results_history.json   ← all iteration records
└── best_result.json       ← current best

scripts/
└── research_loop.py       ← the orchestrator (don't edit)
```

---

## Stop conditions

The loop ends when:
- 20 iterations complete, OR
- 5 experiments are "kept" (paper-ready), OR
- You Ctrl+C (safe — resumes with `--resume`)

---

## After the loop — writing the paper

Once you have 5+ kept experiments, the paper structure is clear:
- Results table = all kept iterations
- Core finding = contamination_rate delta (memory vs shared)
- Key condition = high_similarity amplifies the effect
- Mitigation = whichever iteration tested student-ID injection

Use autovoila's `voila.md` to write the paper with Claude Code / Ralph,
pointing it at `results_history.json` and `loop_log.jsonl` as source material.