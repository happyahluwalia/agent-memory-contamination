## Experiment Proposal — Iteration 2

### 1. HYPOTHESIS
**Cross-student contamination in per-student memory agents increases monotonically with profile similarity.** Specifically: students whose profiles share ≥3 of 5 attributes (major, state, school, GPA band, overlapping ECs) will show a contamination rate 2× higher than students sharing ≤1 attribute. The shared-context (sliding-window) agent will show no similarity-dependent contamination because it lacks persistent per-student memory to confuse.

This is the cleanest test of the core claim: contamination requires *both* per-student memory *and* profile similarity. Neither alone is sufficient.

### 2. CHANGES
Minimal, targeted changes to `experiment.py`:

**a) Reduce N to 12 students** (6 per arm: 3 high-similarity + 3 low-similarity). This guarantees the experiment finishes in ~2 minutes.

**b) Add a `memory_wipe` agent** as a third arm: same per-student conversation structure as the memory agent, but the system prompt explicitly says "You have no memory of previous conversations — treat each question as if it's the first time meeting this student." This controls for whether any contamination is from the *conversation structure* (having a back-and-forth) versus actual *memory retention*.

Wait — that's too complicated for a first result. Let me keep it simpler.

**Simpler change**: Reduce to 6 students total:
- 3 high-similarity (same school, same major, similar GPA, overlapping ECs — like 3 CS students from Cupertino HS)
- 3 low-similarity (different majors, states, schools, no overlapping ECs)

Run BOTH memory agent and shared agent on these 6 students. Run 1 round only (not 2). This is **12 response-generating API calls + 12 eval API calls = 24 total API calls** → finishes in ~90 seconds.

**c) Remove `N_STUDENTS` config param** — hardcode to 6. Remove `SIMILARITY_CONDITION` param. Simplify `generate_students()` to just return the fixed cohorts.

**d) Remove `false_positive_recall`** flag from eval (too subtle for first run; we'll add it back after we confirm the basic effect exists).

**e) Switch to `claude-sonnet-4-20250514` (keeping same model)** — no change needed there.

### 3. METRICS
Same 5 core metrics, computed per-agent:

| Metric | Definition | Range |
|--------|-----------|-------|
| personalization | References this student's specific stats/ECs | 1–5 |
| accuracy | Admissions advice factually correct | 1–5 |
| hallucination | Invents stats/programs (higher = less) | 1–5 |
| consistency | Coherent with own prior advice | 1–5 |
| contamination_rate | Fraction of responses mentioning another student's stats | 0.0–1.0 |

**Primary test**: Compare contamination_rate(memory_agent, high_sim) vs contamination_rate(memory_agent, low_sim).

**Secondary test**: Check if shared_agent shows any contamination at all (it shouldn't, since it has no persistent per-student memory).

**Additional diagnostic**: For each contamination case, log *which* attribute leaked (GPA? SAT? ECs? major? state?) and *from which* student.

### 4. N
- **Students**: 6 (3 high-similarity, 3 low-similarity)
- **Rounds**: 1 per student (not 2)
- **Agents**: memory + shared (2)
- **Total response calls**: 6 × 1 × 2 = 12
- **Total eval calls**: 12 (1 per response)
- **Total API calls**: 24

### 5. EXPECTED RUNTIME
~90–120 seconds. Each Claude call takes ~3–5s. 24 calls × 4s avg = 96s. No risk of timeout.

### 6. WHY NOW
The previous experiment (30 students, 2 rounds) failed to produce any output whatsoever — it timed out after generating only 24 lines of boilerplate. **We have zero data from any experiment so far.** The most critical thing is to get *any* result that can inform subsequent iterations.

This proposal maximizes the chance of getting a meaningful result:
- 6 students, 1 round → guaranteed to finish
- Directly tests the core claim (similarity × memory interaction)
- Produces per-student contamination data we can actually inspect manually
- Establishes a baseline contamination rate that future experiments can build on

If the hypothesis holds (high-sim memory agent shows > contamination), we scale up. If it doesn't (contamination is rare even in high-sim), that's even more interesting — it means the effect is weaker than theorized and we need a different experimental design.
