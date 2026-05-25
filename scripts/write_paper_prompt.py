#!/usr/bin/env python3
"""
Generate a paper-writing prompt from accumulated loop results.
Run this once you have 5+ kept experiments.

Usage:
  python scripts/write_paper_prompt.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SPIKE_DIR    = PROJECT_ROOT / "all-spikes" / "memory-contamination"
RESULTS_FILE = SPIKE_DIR / "results_history.json"
BEST_FILE    = SPIKE_DIR / "best_result.json"
OUT_FILE     = SPIKE_DIR / "paper_prompt.txt"


def main():
    if not RESULTS_FILE.exists():
        print(f"No results found at {RESULTS_FILE}. Run the research loop first.")
        return

    history = json.loads(RESULTS_FILE.read_text())
    kept    = [r for r in history if r.get("kept")]
    best    = json.loads(BEST_FILE.read_text()) if BEST_FILE.exists() else {}

    print(f"Total iterations : {len(history)}")
    print(f"Kept experiments : {len(kept)}")
    print(f"Best score       : {best.get('score', '?')}/10 (iter {best.get('iteration', '?')})")

    if len(kept) < 3:
        print("\nWARNING: Fewer than 3 kept experiments. Consider running more iterations.")

    kept_summary = "\n\n".join([
        f"Experiment {r['iteration']} (plan score {r.get('plan_score','?')}/10, result score {r['claude_score']}/10):\n"
        f"  Hypothesis: {r['hypothesis']}\n"
        f"  Finding: {r['key_findings']}\n"
        f"  Supports claim: {r.get('supports_claim', '?')}\n"
        f"  Paper contribution: {r.get('paper_contribution', 'not specified')}"
        for r in kept
    ])

    all_next_hypotheses = "\n".join([
        f"  - Iter {r['iteration']}: {r.get('next_hypothesis', '')}"
        for r in kept if r.get('next_hypothesis')
    ])

    prompt = f"""You are writing a research paper for submission to CAISc 2026 (Conference for AI Scientists).

TITLE: "Memory Contamination in Multi-Agent AI College Counseling: A Study of Per-Student Agent Architecture"

CORE CLAIM:
Per-student memory in shared-infrastructure AI agents causes cross-student data contamination,
degrading accuracy and consistency — especially when student profiles are similar to each other.

DOMAIN CONTEXT:
- lumne.ai: production college counseling platform with per-student agents
- Anthropic API (claude-sonnet-4-6) used for all agents
- Synthetic student profiles (varied GPA, demographics, intended major, state)
- Two agent architectures compared: memory agent (per-student context) vs shared agent (sliding window)

KEPT EXPERIMENT RESULTS ({len(kept)} experiments):
{kept_summary}

CLAUDE'S SUGGESTED NEXT HYPOTHESES (for Related Work or Limitations framing):
{all_next_hypotheses}

PRIOR BASELINE (iteration 0, N=6):
| Metric         | Memory | Shared |
|----------------|--------|--------|
| Personalization| 3.83   | 3.75   |
| Accuracy       | 3.33   | 3.67   |
| Hallucination  | 2.92   | 3.50   |
| Consistency    | 3.00   | 4.00   |
Memory recall: 1.58/5 — agents confused student names, SAT scores, extracurriculars.

WRITING INSTRUCTIONS:
Using the draft-format/ LaTeX template, write an 8-page paper (excluding appendix, references, checklists).

Structure:
1. Abstract (150 words max)
2. Introduction — motivate from lumne.ai production context; state core claim clearly
3. Related Work — cite: Bloom 2-sigma problem (AI tutoring), Du et al. 2023 (multi-agent debate),
   LLM hallucination in high-stakes domains, RAG vs memory tradeoffs
4. Methodology — synthetic student profiles, two agent architectures, evaluation protocol
5. Results — use the kept experiment data above; include tables and the key conditions
6. Discussion — similarity as a moderator; practical implications for AI advising system design
7. Limitations — N sizes, LLM-as-judge bias, synthetic vs real students
8. Conclusion

Then add APPENDIX with:
- All prompts used (system prompts for both agents, evaluator prompt)
- Example agent trajectory showing contamination event
- Full results tables for all kept experiments
- Completed AI Involvement Checklist
- Completed Reproducibility and Responsibility Checklist

STYLE RULES:
- No em dashes in body text
- No AI-sounding hedging phrases ("it is worth noting", "it is important to")
- Acknowledge Claude Sonnet 4 (claude-sonnet-4-6) in acknowledgements
- Claims must be scoped to what the evidence supports (N sizes are modest — say so)
"""

    OUT_FILE.write_text(prompt)
    print(f"\nPaper prompt written to: {OUT_FILE}")
    print(f"\nNext step: run Claude Code or cco pointing at voila.md, then paste this prompt.")
    print(f"  cco -p \"Read voila.md, then read {OUT_FILE} and write the paper\"")


if __name__ == "__main__":
    main()
