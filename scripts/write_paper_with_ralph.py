#!/usr/bin/env python3
"""
Generate a research paper using Ralph (DeepSeek) — no Claude credits needed.

Reads all completed results_iteration*.json files directly and builds a
comprehensive prompt, then invokes Ralph to write the paper as paper_draft.md.

Usage:
  python scripts/write_paper_with_ralph.py
"""

import json
import shutil
import subprocess
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPIKE_DIR    = PROJECT_ROOT / "all-spikes" / "memory-contamination"
PAPER_FILE   = SPIKE_DIR / "paper_draft.md"
PROMPT_FILE  = SPIKE_DIR / "paper_prompt.txt"


# ─── Collect completed results ─────────────────────────────────────────────────

def load_all_results() -> list:
    """Read all completed results_iteration*.json files. Skip tiny/incomplete ones."""
    results = []
    for f in sorted(SPIKE_DIR.glob("results_iteration*.json")):
        if f.stat().st_size < 10_000:
            print(f"  Skipping {f.name} (too small — likely incomplete)")
            continue
        try:
            d = json.loads(f.read_text())
            d["_source_file"] = f.name
            results.append(d)
            print(f"  Loaded {f.name} ({f.stat().st_size // 1024}K)")
        except Exception as e:
            print(f"  ERROR loading {f.name}: {e}")
    return results


def summarize_experiment(d: dict) -> str:
    exp = d.get("experiment", d.get("_source_file", "?"))
    summary = d.get("summary", {})
    config = d.get("config", {})

    lines = [f"### {exp}"]
    if config:
        lines.append(f"Config: {json.dumps(config)}")

    # Contamination metrics
    contam_lines = []
    for k, v in summary.items():
        if isinstance(v, (int, float)):
            contam_lines.append(f"  {k}: {v}")
    if contam_lines:
        lines.append("Key metrics:")
        lines.extend(contam_lines[:30])  # cap to avoid prompt bloat

    return "\n".join(lines)


# ─── Build paper prompt ────────────────────────────────────────────────────────

PAPER_PROMPT_TEMPLATE = """You are an AI research assistant writing a scientific paper.

Write the full paper as a Markdown document and save it to:
  all-spikes/memory-contamination/paper_draft.md

═══════════════════════════════════════════════════════════════════════════════
PAPER TITLE:
  "Memory Contamination in Multi-Agent AI College Counseling:
   A Study of Per-Student Agent Architecture"

VENUE: CAISc 2026 (Conference for AI Scientists)
TARGET LENGTH: 8 pages (excluding appendix, references, checklists)
═══════════════════════════════════════════════════════════════════════════════

CORE CLAIM:
  Shared-history AI agents serving multiple students in the same context window
  cause spontaneous cross-student data contamination. Per-student memory agents
  (isolated conversation history per student) produce zero contamination.
  Name-tagging intervention on shared agents eliminates contamination entirely.

DOMAIN:
  College counseling AI systems (inspired by lumne.ai production context).
  Synthetic student profiles with varied GPA, SAT, ECs, state, major.
  Two agent architectures tested:
    - Memory agent: isolated per-student conversation history
    - Shared agent: sliding window across ALL students in sequence

EXPERIMENT HISTORY (chronological):
{experiment_summaries}

KEY FINDINGS TO HIGHLIGHT:
1. Shared agent contamination vs. memory agent (iter 5-6, 12):
   - Shared agent: 22-31% contamination rate
   - Memory agent: 0% contamination (confirmed in iter 12)
   - Clean 3-condition replication (iter 12): shared=29%, memory=0%, shared+tags=0%

2. Low-similarity students more contaminated than high-similarity (iter 12):
   - shared_passive high-sim contamination: 0%
   - shared_passive low-sim contamination: 58%
   (Counter-intuitive: low-sim students receive more specific, incorrect data)

3. Context accumulation drives contamination (iter 10 pollinator):
   - When students explicitly share stats, contamination hits 92% in shared agent
   - Passive probes (students don't share stats) show 12-29% contamination

4. Name-tagging intervention works (iter 11 + iter 12):
   - Adding [Student: name] tags to user turns eliminates contamination entirely
   - shared_tagged_passive: 0% vs shared_passive: 29% (iter 12)

5. Poison-pill injection (iter 5-6):
   - Fabricated stats injected into one student's history propagate to others
   - Shared agent propagation rate: 22-31%
   - Memory agent: isolated, 0% propagation

STRUCTURE TO WRITE:
1. Abstract (150 words max)
2. Introduction — motivate from production AI tutoring context; state core claim
3. Related Work — cite: Bloom 2-sigma problem, LLM hallucination in high-stakes
   domains, RAG vs memory tradeoffs, multi-agent context pollution
4. Methodology — synthetic student profiles, two agent architectures,
   evaluation protocol (LLM-as-judge with Claude claude-sonnet-4-6),
   contamination detection (regex + LLM), experimental conditions
5. Results — use the experiment data above; include markdown tables for:
   - Memory vs Shared contamination rates (main result)
   - Effect of context accumulation on contamination
   - Name-tagging intervention effect
   - Similarity moderation (high vs low sim)
6. Discussion — why low-sim students are MORE contaminated (specificity of
   leaked data is more identifiable); practical implications for AI advising
7. Limitations — N sizes modest, LLM-as-judge bias, synthetic vs real students,
   single model (Claude) tested
8. Conclusion — restate core claim with numbers

APPENDIX:
A. System prompts used (memory agent, shared agent, shared+tags agent)
B. Evaluator prompt (LLM-as-judge)
C. Example contamination event (student name, round, leaked attribute, source)
D. AI Involvement Checklist (per CAISc 2026 requirements):
   - Ralph (DeepSeek): proposed and executed all experiment modifications
   - Claude (Anthropic): critiqued plans, scored results, served as agent under test
   - Human involvement: research question, evaluation thresholds, paper review
E. Reproducibility Checklist

STYLE RULES:
- No em dashes in body text (use commas or restructure)
- No AI-sounding hedges ("it is worth noting", "it is important to note")
- Scope claims to evidence — N sizes are modest, acknowledge this
- Use past tense for results
- Acknowledge Claude Sonnet 4 (claude-sonnet-4-6) in Acknowledgements
- Acknowledge Ralph (DeepSeek) and the autovoila framework

After writing the paper, confirm by printing:
  PAPER WRITTEN: all-spikes/memory-contamination/paper_draft.md
"""


def build_prompt(results: list) -> str:
    summaries = "\n\n".join(summarize_experiment(d) for d in results)
    return PAPER_PROMPT_TEMPLATE.format(experiment_summaries=summaries)


# ─── Run Ralph ─────────────────────────────────────────────────────────────────

def run_ralph(prompt: str, cwd: Path, timeout: int = 1800) -> str:
    sessions_dir = cwd / ".ralph" / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)

    print(f"\n[ralph] Starting paper writing (timeout={timeout}s)...")
    proc = subprocess.Popen(
        ["ralph", "--no-confirm", prompt],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(cwd)
    )
    output_lines = []
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            output_lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"\n[ralph] Timed out after {timeout}s")
    if proc.returncode and proc.returncode != 0:
        print(f"[ralph] Exited with code {proc.returncode}")
    return "".join(output_lines)


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PAPER GENERATION — using Ralph (DeepSeek, no Claude needed)")
    print("=" * 60)

    print("\nLoading completed experiment results...")
    results = load_all_results()
    print(f"\nLoaded {len(results)} completed experiments")

    prompt = build_prompt(results)
    PROMPT_FILE.write_text(prompt)
    print(f"\nPaper prompt written to: {PROMPT_FILE}")
    print(f"Prompt length: {len(prompt)} chars\n")

    print("Invoking Ralph to write the paper...")
    output = run_ralph(prompt, cwd=PROJECT_ROOT)

    if PAPER_FILE.exists():
        size = PAPER_FILE.stat().st_size
        print(f"\n{'='*60}")
        print(f"SUCCESS: {PAPER_FILE} ({size // 1024}K)")
        print(f"{'='*60}")
    else:
        print(f"\nWARNING: {PAPER_FILE} not found. Check Ralph output above.")
        print("You can manually run Ralph with the prompt at:")
        print(f"  {PROMPT_FILE}")


if __name__ == "__main__":
    main()
