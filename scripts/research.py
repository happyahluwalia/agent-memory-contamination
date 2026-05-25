#!/usr/bin/env python3
"""
Karpathy-style autonomous research loop for college counseling multi-agent study.

Loop: Ralph(DeepSeek) proposes experiment → Claude critiques plan → 
      Ralph executes → Claude scores results → keep/revert → repeat

Analogous to autoresearch:
  - program.md     = research context + instructions
  - experiment.py  = the thing Ralph modifies each iteration
  - this script    = the loop that calls both agents and gates progress

Usage:
  python scripts/research.py                  # run until max_iterations
  python scripts/research.py --resume         # pick up from last checkpoint
  python scripts/research.py --max-iter 5     # short run to test
"""

import os
import json
import shutil
import argparse
import subprocess
import datetime
from pathlib import Path
from anthropic import Anthropic

# ─── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR          = Path(__file__).parent
PROJECT_ROOT        = SCRIPT_DIR.parent
SPIKE_DIR           = PROJECT_ROOT / "all-spikes" / "memory-contamination"
TEMPLATE_EXPERIMENT = SCRIPT_DIR / "experiment.py"
LOG_FILE            = SPIKE_DIR / "loop_log.jsonl"
RESULTS_FILE        = SPIKE_DIR / "results_history.json"
EXPERIMENT_FILE     = SPIKE_DIR / "experiment.py"
BEST_FILE           = SPIKE_DIR / "best_result.json"

MAX_ITERATIONS = 20
APPROVE_THRESHOLD = 6        # Claude must score plan >= 6/10 to proceed
KEEP_THRESHOLD    = 6        # Claude must score results >= 6/10 to keep

client = Anthropic()

RESEARCH_CONTEXT = """
We are studying multi-agent AI systems for college counseling.
Specifically: does per-student memory in multi-tenant AI agents cause 
cross-student data contamination, degrading accuracy and consistency?

Prior result (N=6, 2 agents):
| Metric         | Memory Agent | Shared Agent |
|----------------|:------------:|:------------:|
| Personalization| 3.83         | 3.75         |
| Accuracy       | 3.33         | 3.67         |
| Hallucination  | 2.92         | 3.50         |
| Consistency    | 3.00         | 4.00         |
Memory recall was 1.58/5, with agents confusing student names, SAT scores, ECs.

The CORE CLAIM we are trying to validate/strengthen:
"Per-student memory in shared-infrastructure agents causes cross-student 
data contamination, degrading accuracy and consistency — especially when 
student profiles are similar to each other."

All experiments use the Anthropic API (claude-sonnet-4-6).
No GPU. No human eval. Keep total disk < 15GB.
"""

# ─── Helpers ───────────────────────────────────────────────────────────────────

def log(entry: dict):
    entry["timestamp"] = datetime.datetime.now().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\n[loop] {entry.get('event', '?')} — {entry.get('summary', '')}")

def load_results_history() -> list:
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    return []

def save_results_history(history: list):
    RESULTS_FILE.write_text(json.dumps(history, indent=2))

def load_best() -> dict:
    if BEST_FILE.exists():
        return json.loads(BEST_FILE.read_text())
    return {"score": 0, "iteration": 0, "metrics": {}}

def save_best(best: dict):
    BEST_FILE.write_text(json.dumps(best, indent=2))

def git_commit(msg: str):
    subprocess.run(["git", "add", str(EXPERIMENT_FILE)], cwd=str(PROJECT_ROOT), capture_output=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=str(PROJECT_ROOT), capture_output=True)

def git_revert():
    rel = EXPERIMENT_FILE.relative_to(PROJECT_ROOT)
    subprocess.run(["git", "checkout", "--", str(rel)], cwd=str(PROJECT_ROOT), capture_output=True)

def load_current_iteration() -> int:
    history = load_results_history()
    return len(history)

def run_ralph(prompt: str, cwd: Path, timeout: int = 900) -> str:
    """Invoke ralph in single-shot mode, streaming output live while capturing it.
    Resets the ralph session directory first to prevent context carry-over between iterations.
    """
    # Force a fresh session each call — session carry-over causes the execute step
    # to inherit the propose context and get confused about what file to read.
    sessions_dir = cwd / ".ralph" / "sessions"
    if sessions_dir.exists():
        shutil.rmtree(sessions_dir)

    print(f"\n[ralph] Starting: {prompt[:80]}...")
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

# ─── Claude as Critic ──────────────────────────────────────────────────────────

def claude_critique_plan(proposed_plan: str, iteration: int, history: list) -> dict:
    """Claude reviews Ralph's proposed experiment plan. Returns {score, feedback, approved}."""
    
    history_summary = "\n".join([
        f"  Iter {r['iteration']}: {r.get('hypothesis','?')} → score {r.get('claude_score','?')}/10, kept={r.get('kept',False)}"
        for r in history[-5:]  # last 5 only
    ]) or "  (none yet)"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=f"""You are a rigorous AI research critic reviewing experiment plans.
Context: {RESEARCH_CONTEXT}

Recent experiment history:
{history_summary}

Your job: Review the proposed experiment plan and score it 1-10.
Respond ONLY with valid JSON like:
{{
  "score": 7,
  "approved": true,
  "feedback": "Good N increase. However suggest also tracking ...",
  "suggested_changes": "Add a 'similar_students' condition where ...",
  "novelty": "What's new vs prior iterations in 1 sentence"
}}

Score rubric:
- 1-4: Reject (too similar to prior iter, won't advance the claim, or flawed methodology)
- 5: Borderline — suggest changes
- 6-7: Approve with minor suggestions  
- 8-10: Strong, proceed immediately
""",
        messages=[{
            "role": "user",
            "content": f"Iteration {iteration}. Proposed experiment plan:\n\n{proposed_plan}"
        }]
    )
    
    raw = response.content[0].text.strip()
    # strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[loop] JSON parse error in plan critique: {e}\nRaw: {raw[:300]}")
        return {"score": 5, "approved": False, "feedback": f"Claude returned unparseable JSON: {e}",
                "suggested_changes": "", "novelty": ""}
    result["approved"] = result["score"] >= APPROVE_THRESHOLD
    return result

def claude_score_results(results_text: str, hypothesis: str, iteration: int) -> dict:
    """Claude scores the experiment results. Returns {score, keep, findings, next_hypothesis}."""
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=f"""You are a rigorous AI research evaluator.
Context: {RESEARCH_CONTEXT}

Your job: Evaluate experiment results and decide if they advance our core claim.
Respond ONLY with valid JSON like:
{{
  "score": 7,
  "keep": true,
  "key_findings": "Memory agents showed 40% higher contamination rate when student profiles had >80% similarity",
  "supports_claim": true,
  "statistical_concern": "N=15 is better but still small for the subgroup analysis",
  "next_hypothesis": "Test whether contamination rate correlates with profile similarity distance (cosine sim of embeddings)",
  "paper_contribution": "This strengthens Section 3.2 — add contamination-vs-similarity scatter plot"
}}

Score rubric:
- 1-4: Don't keep (results are noise, methodology flawed, or claim not advanced)
- 5: Borderline
- 6-10: Keep (results genuinely advance the claim)
""",
        messages=[{
            "role": "user", 
            "content": f"Iteration {iteration}. Hypothesis: {hypothesis}\n\nResults:\n{results_text}"
        }]
    )
    
    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[loop] JSON parse error in results scoring: {e}\nRaw: {raw[:300]}")
        return {"score": 5, "keep": False, "key_findings": f"JSON parse error: {e}",
                "supports_claim": False, "statistical_concern": "",
                "next_hypothesis": "", "paper_contribution": ""}
    result["keep"] = result["score"] >= KEEP_THRESHOLD
    return result

# ─── Ralph as Proposer + Executor ─────────────────────────────────────────────

def ralph_propose_experiment(iteration: int, history: list, claude_feedback: str = "") -> str:
    """Invoke ralph to propose the next experiment. Fully automated."""
    feedback_section = f"\nClaude's feedback on last proposal:\n{claude_feedback}\n" if claude_feedback else ""

    history_summary = "\n".join([
        f"  Iter {r['iteration']}: {r.get('hypothesis','?')} → kept={r.get('kept',False)}, finding: {r.get('key_findings','?')}"
        for r in history[-3:]
    ]) or "  (first iteration)"

    propose_prompt = f"""ITERATION {iteration} — PROPOSE NEXT EXPERIMENT

{RESEARCH_CONTEXT}

Experiments run so far:
{history_summary}
{feedback_section}

Your task: Propose the next experiment to run in experiment.py.
Think like a scientist: what's the most informative next test given what we know?

Output a concise plan covering:
1. HYPOTHESIS: One specific, falsifiable claim to test
2. CHANGES: What to modify in experiment.py (be specific about code changes)
3. METRICS: Exactly what to measure and how
4. N: How many synthetic students / trials
5. EXPECTED RUNTIME: Approximate API calls needed
6. WHY NOW: Why is this the right next experiment given prior results?
"""

    propose_file = SPIKE_DIR / "propose_prompt.txt"
    propose_file.write_text(propose_prompt)

    print(f"\n{'='*60}")
    print(f"ITERATION {iteration} — RALPH PROPOSES")
    print(f"{'='*60}")

    # Pass the full prompt content directly — avoids ralph needing to read a file,
    # which can fail if it inherits a confused session state.
    full_prompt = (
        propose_prompt
        + "\n\nOutput your proposal now with all six sections: "
          "HYPOTHESIS, CHANGES, METRICS, N, EXPECTED RUNTIME, WHY NOW."
    )
    return run_ralph(full_prompt, cwd=SPIKE_DIR)

def ralph_execute_experiment(plan: str, iteration: int, critique: dict) -> str:
    """Invoke ralph to execute the approved experiment plan. Fully automated."""
    execute_prompt = f"""ITERATION {iteration} — EXECUTE APPROVED EXPERIMENT

Approved plan:
{plan}

Claude's critique and suggestions:
Score: {critique['score']}/10
Feedback: {critique['feedback']}
Suggested changes: {critique.get('suggested_changes', 'none')}

NOW EXECUTE:
1. Modify experiment.py according to the plan (incorporating Claude's suggestions)
2. Run it: python experiment.py
3. Capture ALL output — metrics table, any errors, observations
4. Report results in this format:
   HYPOTHESIS TESTED: ...
   METRICS: (table)
   KEY OBSERVATION: ...
   CONTAMINATION RATE: X% (if measured)
   SURPRISES: ...
   RAW OUTPUT: (paste stdout)
"""

    execute_file = SPIKE_DIR / "execute_prompt.txt"
    execute_file.write_text(execute_prompt)

    print(f"\n{'='*60}")
    print(f"ITERATION {iteration} — RALPH EXECUTES (plan approved {critique['score']}/10)")
    print(f"{'='*60}")

    full_prompt = (
        execute_prompt
        + "\n\nYour task now: modify experiment.py as specified above, run `python experiment.py`, "
          "fix any errors and rerun until it succeeds. Report results in the exact format: "
          "HYPOTHESIS TESTED, METRICS table, KEY OBSERVATION, CONTAMINATION RATE, SURPRISES, RAW OUTPUT."
    )
    return run_ralph(full_prompt, cwd=SPIKE_DIR, timeout=900)

# ─── Paper Writer ──────────────────────────────────────────────────────────────

def write_paper(history: list):
    """Use Claude to draft the research paper from accumulated results."""
    kept = [r for r in history if r.get('kept')]
    print(f"\n{'='*60}")
    print(f"WRITING PAPER — {len(kept)} kept experiments, {len(history)} total iterations")
    print(f"{'='*60}")

    kept_summary = "\n\n".join([
        f"Experiment {r['iteration']} (scored {r['claude_score']}/10):\n"
        f"  Hypothesis: {r['hypothesis']}\n"
        f"  Finding: {r['key_findings']}\n"
        f"  Supports claim: {r.get('supports_claim', '?')}\n"
        f"  Paper contribution: {r.get('paper_contribution', 'not specified')}"
        for r in kept
    ]) or "No experiments were kept. Paper covers methodology and negative results."

    all_results = "\n\n".join([
        f"Iteration {r['iteration']} (kept={r.get('kept',False)}, score={r['claude_score']}/10):\n"
        f"  Hypothesis: {r['hypothesis']}\n"
        f"  Finding: {r['key_findings']}"
        for r in history
    ])

    paper_prompt = f"""Write a research paper for CAISc 2026 based on the following experimental results.

TITLE: Memory Contamination in Multi-Agent AI College Counseling: A Study of Per-Student Agent Architecture

CORE CLAIM: Per-student memory in shared-infrastructure AI agents causes cross-student data contamination, degrading accuracy and consistency — especially when student profiles are similar.

DOMAIN: lumne.ai production college counseling platform. Two agent architectures compared:
- Memory agent: per-student conversation history in system prompt
- Shared agent: sliding window over shared history across all students

ALL EXPERIMENT RESULTS ({len(history)} iterations, {len(kept)} kept):
{all_results}

KEPT RESULTS (the evidence base):
{kept_summary}

Write the full paper with these sections (be concise but complete):

# Abstract
[150 words max. State claim, method, key finding, implication]

# 1. Introduction
[Motivate from lumne.ai context. State the contamination problem. Preview findings.]

# 2. Related Work
[Bloom 2-sigma tutoring, LLM multi-agent coordination, hallucination in high-stakes domains, RAG vs memory tradeoffs]

# 3. Methodology
[Synthetic student profiles, two agent architectures, evaluation protocol, metrics: personalization/accuracy/hallucination/consistency/contamination_rate]

# 4. Results
[Report from kept experiments. Use markdown tables. Be honest about N sizes.]

# 5. Discussion
[What conditions drive contamination. Practical implications for AI advising system design. Negative results are informative too.]

# 6. Limitations
[Synthetic students, LLM-as-judge bias, small N, single model tested]

# 7. Conclusion
[One paragraph. Restate claim with appropriate caveats.]

# Appendix A: Prompts
[System prompts for both agents, evaluator prompt]

# Appendix B: AI Involvement Checklist
[Fill out for CAISc 2026 submission]

Be direct. No hedging phrases. Scope claims to what the evidence supports."""

    print("[paper] Calling Claude to write draft...")

    sections = []
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8000,
        messages=[{"role": "user", "content": paper_prompt}]
    )
    sections.append(response.content[0].text)

    paper_text = "\n\n".join(sections)
    out_file = SPIKE_DIR / "paper_draft.md"
    out_file.write_text(paper_text)

    print(f"[paper] Draft written to: {out_file}")
    print(f"[paper] Word count approx: {len(paper_text.split())}")


# ─── Main Loop ─────────────────────────────────────────────────────────────────

def run_loop(max_iterations: int, resume: bool):
    SPIKE_DIR.mkdir(parents=True, exist_ok=True)

    if not EXPERIMENT_FILE.exists():
        if TEMPLATE_EXPERIMENT.exists():
            shutil.copy(TEMPLATE_EXPERIMENT, EXPERIMENT_FILE)
            print(f"[loop] Bootstrapped experiment.py from {TEMPLATE_EXPERIMENT}")
        else:
            print(f"[loop] ERROR: No experiment.py at {EXPERIMENT_FILE} or {TEMPLATE_EXPERIMENT}")
            return

    history = load_results_history()
    best = load_best()
    start_iter = load_current_iteration() if resume else 0
    
    if resume:
        print(f"[loop] Resuming from iteration {start_iter}")
    
    claude_feedback = ""  # feedback from last rejected plan
    
    for iteration in range(start_iter, start_iter + max_iterations):
        print(f"\n{'#'*60}")
        print(f"# RESEARCH LOOP — ITERATION {iteration + 1}/{start_iter + max_iterations}")
        print(f"# Best score so far: {best['score']}/10 (iter {best['iteration']})")
        print(f"{'#'*60}")
        
        # ── STEP 1: Ralph proposes ──────────────────────────────────────
        proposed_plan = ralph_propose_experiment(iteration + 1, history, claude_feedback)
        
        if not proposed_plan.strip():
            print("[loop] Empty proposal — skipping iteration")
            continue
        
        log({"event": "proposal", "iteration": iteration+1, 
             "summary": proposed_plan[:200]})
        
        # ── STEP 2: Claude critiques the plan ───────────────────────────
        print(f"\n[loop] Claude reviewing plan...")
        critique = claude_critique_plan(proposed_plan, iteration + 1, history)
        
        print(f"\n{'─'*40}")
        print(f"CLAUDE PLAN REVIEW — Score: {critique['score']}/10")
        print(f"Approved: {critique['approved']}")
        print(f"Feedback: {critique['feedback']}")
        if critique.get('suggested_changes'):
            print(f"Suggestions: {critique['suggested_changes']}")
        print(f"{'─'*40}")
        
        log({"event": "plan_critique", "iteration": iteration+1,
             "score": critique['score'], "approved": critique['approved'],
             "summary": critique['feedback'][:200]})
        
        if not critique['approved']:
            print(f"[loop] Plan rejected ({critique['score']}/10 < {APPROVE_THRESHOLD}). Looping back.")
            claude_feedback = f"Plan was rejected (score {critique['score']}/10). {critique['feedback']} {critique.get('suggested_changes','')}"
            continue
        
        # ── STEP 3: Ralph executes ──────────────────────────────────────
        hypothesis = proposed_plan.split('\n')[0]  # fallback: first line
        for line in proposed_plan.split('\n'):
            if 'HYPOTHESIS' in line.upper() and ':' in line:
                hypothesis = line.split(':', 1)[1].strip()
                break
        results_text = ralph_execute_experiment(proposed_plan, iteration + 1, critique)
        
        if not results_text.strip():
            print("[loop] Empty results — skipping scoring")
            continue
        
        log({"event": "execution", "iteration": iteration+1,
             "summary": results_text[:200]})
        
        # ── STEP 4: Claude scores results ───────────────────────────────
        print(f"\n[loop] Claude scoring results...")
        evaluation = claude_score_results(results_text, hypothesis, iteration + 1)
        
        print(f"\n{'─'*40}")
        print(f"CLAUDE RESULTS REVIEW — Score: {evaluation['score']}/10")
        print(f"Keep: {evaluation['keep']}")
        print(f"Key findings: {evaluation['key_findings']}")
        print(f"Supports claim: {evaluation.get('supports_claim', '?')}")
        print(f"Next hypothesis: {evaluation.get('next_hypothesis', '?')}")
        print(f"{'─'*40}")
        
        # ── STEP 5: Keep or revert ──────────────────────────────────────
        record = {
            "iteration": iteration + 1,
            "hypothesis": hypothesis,
            "proposed_plan": proposed_plan[:500],
            "plan_score": critique['score'],
            "results_summary": results_text[:500],
            "claude_score": evaluation['score'],
            "key_findings": evaluation['key_findings'],
            "supports_claim": evaluation.get('supports_claim', False),
            "kept": evaluation['keep'],
            "next_hypothesis": evaluation.get('next_hypothesis', ''),
            "paper_contribution": evaluation.get('paper_contribution', '')
        }
        
        if evaluation['keep']:
            git_commit(f"iter-{iteration+1}: {evaluation['key_findings'][:60]}")
            if evaluation['score'] > best['score']:
                best = {"score": evaluation['score'], "iteration": iteration+1,
                       "findings": evaluation['key_findings']}
                save_best(best)
            print(f"[loop] ✅ KEPT — score {evaluation['score']}/10")
            claude_feedback = ""  # reset
        else:
            git_revert()
            print(f"[loop] ❌ REVERTED — score {evaluation['score']}/10 < {KEEP_THRESHOLD}")
            claude_feedback = f"Results scored {evaluation['score']}/10 and were reverted. {evaluation.get('statistical_concern','')}"
        
        history.append(record)
        save_results_history(history)
        
        log({"event": "iteration_complete", "iteration": iteration+1,
             "kept": evaluation['keep'], "score": evaluation['score'],
             "summary": evaluation['key_findings'][:200]})
        
        # ── Check if we have enough for a paper ────────────────────────
        kept_count = sum(1 for r in history if r.get('kept'))
        if kept_count >= 5:
            print(f"\n[loop] 🎉 {kept_count} kept experiments — milestone reached, continuing to collect more data...")

    # ── End of loop summary ─────────────────────────────────────────────
    final_history = load_results_history()
    kept = [r for r in final_history if r.get('kept')]
    print(f"\n{'='*60}")
    print(f"LOOP COMPLETE — {len(final_history)} iterations run")
    print(f"Kept: {len(kept)} experiments")
    print(f"Best score: {best['score']}/10 (iter {best['iteration']})")
    if kept:
        print(f"\nTop findings:")
        for r in kept:
            print(f"  Iter {r['iteration']}: {r['key_findings']}")
    print(f"\nAll results: {RESULTS_FILE}")
    print(f"Loop log:    {LOG_FILE}")
    write_paper(final_history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous research loop")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()
    
    run_loop(max_iterations=args.max_iter, resume=args.resume)