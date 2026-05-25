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

All experiments use the Anthropic API (claude-sonnet-4-20250514).
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

# ─── Claude as Critic ──────────────────────────────────────────────────────────

def claude_critique_plan(proposed_plan: str, iteration: int, history: list) -> dict:
    """Claude reviews Ralph's proposed experiment plan. Returns {score, feedback, approved}."""
    
    history_summary = "\n".join([
        f"  Iter {r['iteration']}: {r.get('hypothesis','?')} → score {r.get('claude_score','?')}/10, kept={r.get('kept',False)}"
        for r in history[-5:]  # last 5 only
    ]) or "  (none yet)"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
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
        model="claude-sonnet-4-20250514",
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
    """
    In the real setup, this is where you'd invoke Ralph interactively.
    We simulate by writing a prompt file that Ralph reads via RALPH.md skill.
    Returns the proposed plan as a string.
    
    In practice: ralph reads propose_prompt.txt and writes its plan to proposed_plan.txt
    """
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
    print(f"Prompt written to: {propose_file}")
    print("\nRun Ralph now to generate the experiment plan:")
    print(f"  cd {SPIKE_DIR} && ralph  (then type: /research-propose)")
    print("\nWhen Ralph is done, paste the proposed plan, then press Ctrl+D:")
    
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    return "\n".join(lines)

def ralph_execute_experiment(plan: str, iteration: int, critique: dict) -> str:
    """
    Ralph executes the approved experiment plan.
    Returns the results as a string.
    """
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
    print(f"Execute prompt written to: {execute_file}")
    print("\nRun Ralph now to execute the experiment:")
    print(f"  cd {SPIKE_DIR} && ralph  (then type: /research-execute)")
    print("\nPaste Ralph's results output, then press Ctrl+D:")
    
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    return "\n".join(lines)

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
            print(f"\n[loop] 🎉 {kept_count} kept experiments — enough to write the paper!")
            print(f"[loop] Run: python scripts/write_paper_prompt.py")
    
    # ── End of loop summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"LOOP COMPLETE — {load_current_iteration()} iterations run")
    kept = [r for r in load_results_history() if r.get('kept')]
    print(f"Kept: {len(kept)} experiments")
    print(f"Best score: {best['score']}/10 (iter {best['iteration']})")
    print(f"\nTop findings:")
    for r in kept:
        print(f"  Iter {r['iteration']}: {r['key_findings']}")
    print(f"\nAll results: {RESULTS_FILE}")
    print(f"Loop log:    {LOG_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous research loop")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--max-iter", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()
    
    run_loop(max_iterations=args.max_iter, resume=args.resume)