# program.md — Research Agenda

Analogous to Karpathy's autoresearch program.md.
This file describes the research goal and constraints.
The loop reads this. Neither Ralph nor Claude should modify it.

## Core Claim
"Per-student memory in shared-infrastructure AI agents causes 
cross-student data contamination, degrading accuracy and consistency — 
especially when student profiles are similar to each other."

## Domain
College counseling AI (lumne.ai context):
- Each student gets a per-session agent
- Agents have access to UC/college admissions data via MCP tools
- Common student profiles: CA resident, CS or pre-med, high GPA

## What "better" means (the metric, analogous to val_bpb)
Primary: **contamination_rate** — lower is better for memory agent
Secondary: **accuracy_score** — higher is better
Both must move in the hypothesized direction for a result to be "kept"

## Experiment constraints
- All API calls to claude-sonnet-4-20250514
- Max N_STUDENTS = 30 per run
- No GPU, no local models
- Each iteration must complete in < 30 minutes
- Results must be reproducible (set random seeds)

## Hypothesis space to explore (in rough priority order)
1. Does profile similarity increase contamination rate? (baseline → high_similarity condition)
2. Does sliding window size affect contamination in shared agent?
3. Does explicit student-ID injection in system prompt reduce contamination?
4. Does multi-turn depth (N_ROUNDS) increase contamination rate?
5. Can we measure contamination via embedding similarity of responses?
6. Does a "memory summary" approach outperform raw conversation history?

## What a paper-ready result looks like
- At least 5 kept experiments
- Contamination rate difference between memory and shared is statistically meaningful (N≥20)
- At least one condition that shows the effect getting stronger (e.g., high similarity)
- At least one mitigation tested (e.g., explicit ID injection)