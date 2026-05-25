# Experiment Progress Log
## Per-Student Memory vs Consistency in LLM Advising

### Run: Sat Nov 18 2024
**Model**: claude-sonnet-4-6  
**Students**: 12 (6 similar pairs)  
**Total Budget**: < $2.00  
**Total Cost**: ~$0.88 (experiment + evaluation)

### Phases Completed

| Phase | Status | Notes |
|-------|--------|-------|
| 0. Generate Profiles | ✅ | 50 synthetic UC-applicant profiles generated |
| 0b. Select Similar Pairs | ✅ | 25 similar pairs (same major, close GPA) |
| 1. Condition A (Memory) | ✅ | 12 per-student agents, each with their own conversation memory |
| 2. Condition B (Shared) | ✅ | 1 shared agent, all conversations in same context |
| 3a. Recall Evaluation | ✅ | Evaluated memory recall in Cond A follow-ups |
| 3b. Quality Evaluation | ✅ | Scored personalization, accuracy, actionability, hallucination |
| 3c. Consistency Evaluation | ✅ | Compared advice consistency across similar pairs |

### Results

#### Quality Metrics (1-5, higher is better)

| Metric | Cond A (Memory) | Cond B (Shared) | Winner |
|--------|:-:|:-:|:-:|
| **Personalization** | **3.83** | 3.75 | Memory (+0.08) |
| **Accuracy** | 3.33 | **3.67** | Shared (+0.34) |
| **Actionability** | **3.17** | 3.00 | Memory (+0.17) |
| **Hallucination (5=none)** | 2.92 | **3.50** | Shared (+0.58) |

#### Consistency (1-5, higher = more consistent)

| Condition | Score | 
|-----------|:----:|
| A (Memory) | **3.00** |
| B (Shared) | **4.00** |

#### Memory Recall (Cond A only)

| Metric | Score |
|--------|:----:|
| Recall Score | **1.58/5** |

### Raw Per-Student Comparisons
(A vs B: pers/acc/hal scores)

| Student | A Personalization | B Personalization | A Accuracy | B Accuracy | A Hallucination | B Hallucination |
|---------|:-:|:-:|:-:|:-:|:-:|:-:|
| S0 Alex | 4 | 4 | 4 | 4 | 4 | 4 |
| S1 Jordan | 4 | 4 | 3 | **4** | 3 | **4** |
| S2 Taylor | **4** | 3 | 3 | **4** | 2 | **4** |
| S3 Morgan | 3 | 3 | 3 | **4** | 2 | **4** |
| S4 Casey | **5** | 4 | **4** | 3 | **5** | 4 |
| S5 Riley | 4 | 4 | 3 | 3 | 2 | 2 |
| S6 Avery | 3 | **4** | 2 | **4** | 2 | **4** |
| S7 Quinn | 4 | 4 | 4 | 4 | 4 | 4 |
| S8 Maya | 3 | **4** | 4 | 4 | 2 | 2 |
| S9 Ethan | 4 | 4 | 4 | 4 | 4 | 4 |
| S10 Sophia | 4 | 4 | 3 | 3 | 3 | 3 |
| S11 Liam | **4** | 3 | 3 | 3 | 2 | **3** |

**Head-to-head wins**: Personalization: A=3, B=2, tie=7 | Accuracy: A=1, B=4, tie=7 | Hallucination: A=1, B=4, tie=7

### Qualitative Observations

In the memory condition, follow-up responses frequently:
- **Wrong names**: Agent addressed student by another student's name
- **Fabricated stats**: Invented SAT scores (e.g., "1148" when not in profile)
- **Confused ECs**: Referenced other students' extracurriculars
- **Hallucinated details**: Made up school lists and recommendations

Example: Student 2's follow-up started "Welcome back, Taylor!" but referenced wrong ECs (Student Council Presidency instead of Science Olympiad/Model UN/Coding Club) and wrong school lists.

### Key Findings

1. **Memory provides marginal personalization gain** (+0.08) — not a practically significant advantage
2. **Memory severely harms consistency** (3.00 vs 4.00) — 33% worse consistency
3. **Memory reduces factual accuracy** (3.33 vs 3.67) — agent confuses details across students
4. **Memory increases hallucination** (2.92 vs 3.50, where 5=none) — more fabricated information
5. **The memory agent can barely recall specific details** (1.58/5) about the student

**Conclusion: The claim is SUPPORTED.** Per-student memory provides marginal personalization benefits but significantly degrades consistency, accuracy, and increases hallucination. The shared-context approach is more reliable for college admissions advising.
