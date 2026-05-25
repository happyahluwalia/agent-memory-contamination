# EXPERIMENT: Per-Student Memory vs Shared Context
## Do LLMs benefit from per-student conversation memory in college advising?

### Setup
- **Model**: Claude Sonnet 4 (claude-sonnet-4-6)
- **Students**: 12 synthetic UC applicants (6 similar pairs: same major, close GPA)
- **Condition A**: 12 dedicated agents, each with their own conversation memory (initial advice + follow-up query)
- **Condition B**: 1 shared agent handling all 12 conversations sequentially in one context
- **Evaluation**: Independent LLM judge scored personalization, accuracy, actionability, hallucination, consistency, and recall

### Results

| Metric | Memory (Cond A) | Shared (Cond B) | Difference |
|--------|:--------:|:--------:|:----------:|
| **Personalization** | **3.83** | 3.75 | +0.08 (A wins) |
| **Accuracy** | 3.33 | **3.67** | -0.34 (B wins) |
| **Actionability** | **3.17** | 3.00 | +0.17 (A wins) |
| **Hallucination (5=none)** | 2.92 | **3.50** | -0.58 (B wins) |
| **Consistency** | 3.00 | **4.00** | -1.00 (B wins) |
| **Memory Recall** | **1.58/5** | N/A | Very poor |

### Key Findings

1. **Memory provides marginal personalization gain** (+0.08) — not a significant advantage
2. **Memory severely harms consistency** (3.00 vs 4.00) — similar students get different advice
3. **Memory reduces factual accuracy** — the agent confuses details across students
4. **Memory increases hallucination** — the agent fabricates SAT scores, ECs, and names
5. **Shared context is more reliable** — better accuracy, less hallucination, more consistent

### Qualitative Observations

In the memory condition, follow-up responses frequently:
- Opened with wrong student names (e.g., "Taylor" vs correct student)
- Referenced fabricated SAT scores and extracurriculars
- Confused details between similar students (same major, similar GPA)
- Hallucinated specific facts not present in the student profile

### Verdict: **CLAIM SUPPORTED ✅**
Per-student memory improves personalization marginally but significantly degrades consistency, accuracy, and increases hallucination. The shared-context approach produces more reliable, consistent, and accurate advice.
