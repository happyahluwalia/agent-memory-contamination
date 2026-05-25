## Experiment Proposal — Iteration 3

### 1. HYPOTHESIS
**Cross-student contamination in memory agents emerges when the system prompt is stripped of student-specific statistics, forcing the agent to rely on conversation history alone.** Specifically: when the memory agent's system prompt is generic (no explicit GPA/SAT/ECs listed), the agent will show a contamination rate >0% after 2+ rounds, with high-similarity students showing higher rates than low-similarity. The shared agent's quality degradation for low-similarity students (observed in v2: accuracy 2.67 vs 4.00) will replicate and be confirmed as a *profile-majority bias* effect.

This is based on the v2 null result: zero contamination with per-student stats in the system prompt. The most parsimonious explanation is that the explicit stats in the system prompt act as an unerasable anchor, making memory-agent contamination trivially easy to avoid. Real-world memory agents won't have each student's stats in the system prompt — they'll have a generic system prompt and student-specific info injected into the conversation history.

### 2. CHANGES

**a) Generic system prompt for memory agent** — Remove per-student stats from the system prompt. Instead, the system prompt is just: "You are a college counselor helping students with their applications." Student profile info only appears in the user's messages (the questions include GPA, SAT, etc.). This forces the agent to rely on conversation history, mimicking real-world memory agents where facts are in stored messages, not the system prompt.

**b) Merge high-sim and low-sim into one agent run** — Instead of running memory agent separately for high-sim then low-sim, run ALL 6 students through each agent (as v2 did). This tests within-agent contamination.

**c) Increase to 3 rounds** — Each round adds 2 questions (counseling + probing). Over 3 rounds, the agent sees each student 6 times, accumulating ~18 turns of conversation per student. This creates enough conversational context for memory to potentially blur.

**d) Add a "cross-talk" round** — In round 2, have one high-sim student ask about another high-sim student's activities (e.g., Alan asks: "My friend Brian is also applying to CS programs — what advice can you give him?"). This directly pressures the agent to distinguish between similar profiles.

**e) Two contamination detection methods** — (1) Claude-based eval (as before) AND (2) explicit regex/string checks for stat conflicts (e.g., does Alan's response ever mention SAT=1480, which is Brian's SAT?). This catches contamination the evaluator might miss.

**f) Track "favorability bias"** — For the shared agent, measure whether response quality (personalization, accuracy) correlates with how similar the current student is to the majority profile type. Score a "majority distance" metric (0 = same as majority profile, 1+ = increasingly different).

### 3. METRICS

| Metric | Definition | Range |
|--------|-----------|-------|
| personalization | References this student's specific stats/ECs | 1–5 |
| accuracy | Admissions advice factually correct | 1–5 |
| hallucination | Invents stats/programs (higher = better) | 1–5 |
| consistency | Coherent with own prior advice across rounds | 1–5 |
| contamination_rate | Fraction of responses mentioning another student's stats | 0.0–1.0 |
| stat_conflict_rate | Fraction of responses where agent attributes another student's exact stat (GPA/SAT) to current student (regex-checked) | 0.0–1.0 |
| majority_distance_corr | Correlation between response quality and distance from majority profile type | -1 to 1 |

**Primary test**: Does contamination_rate > 0 in the memory agent with generic system prompt after 3 rounds?

**Secondary test**: Does the contamination rate increase more for high-similarity students than low-similarity?

**Tertiary test**: Does the shared agent show lower quality for diverse students (replicating v2's 2.67 accuracy finding)?

### 4. N
- **Students**: 6 (same 3 high-sim + 3 low-sim from v2)
- **Rounds**: 3 per student × 2 questions per round = 6 questions per student
- **Agents**: memory + shared (2)
- **Total response calls**: 6 × 3 × 2 = 36
- **Total eval calls**: 12 (1 per student per agent, evaluating combined 3-round response)
- **Total API calls**: 36 + 12 = 48
- **Expected runtime**: ~3 minutes (48 calls × ~4s avg = 192s)

### 5. EXPECTED RUNTIME
**~180–240 seconds.** 48 API calls × ~4s avg = 192s. This exceeds the 120s tool timeout, so the experiment should be run in the background (nohup + sleep polling) as was done successfully in v2.

### 6. WHY NOW
The v2 null result is the most informative result we've gotten: **zero contamination with explicit per-student stats in the system prompt.** This tells us Claude Sonnet 4 is very good at not confusing student profiles *when their stats are always visible in the prompt.* But this is a weak test of the contamination hypothesis — it's like testing whether a calculator makes arithmetic errors by giving it the answer key.

The real-world scenario is different: a memory agent has a **generic system prompt** and student-specific facts are embedded in conversation history. The agent must *remember* which facts belong to which student, not just *read* them from the prompt. This is where contamination is expected to occur — when the model has to track multiple similar profiles across dozens of conversation turns.

This experiment directly tests the mechanism of contamination (memory load vs. prompt anchoring) rather than just testing whether it exists. If contamination still doesn't appear with a generic system prompt after 3 rounds, that would be a much stronger falsification of the core claim. If it does appear, we'll have identified the key condition and can design more targeted experiments in iteration 4.
