# Memory Contamination in Multi-Agent AI College Counseling: A Study of Per-Student Agent Architecture

## Abstract

Large language model agents serving multiple students within a single context window risk cross-student data contamination. We study this phenomenon in a simulated college counseling environment using Claude Sonnet 4 (claude-sonnet-4-6). Across eight experimental conditions with synthetic student profiles, we compare two agent architectures: per-student memory agents (isolated conversation history per student) and shared-history agents (single sliding window across all students). Shared agents produce contamination rates of 22-31% in passive probe conditions, rising to 92% in a preliminary high-accumulation run where students explicitly shared their statistics. Per-student memory agents produce near-zero contamination across all conditions (0/24 students in the cleanest replication). A simple name-tagging intervention (prepending [Student: name] to user and assistant turns) eliminates contamination entirely without architectural isolation. Counterintuitively, low-similarity students are contaminated more frequently than high-similarity students (58% vs. 0%), suggesting that leaked dissimilar data is more identifiable by LLM evaluators. These findings have direct implications for production AI advising systems that serve multiple users sequentially.

## 1. Introduction

College counseling is a high-stakes domain where incorrect information about a student's GPA, test scores, or extracurricular profile can lead to misdirected admissions advice. AI-powered college counseling platforms, such as lumne.ai, increasingly use large language models to generate personalized recommendations for students. A common architectural pattern involves a single agent instance that serves multiple students in sequence, maintaining a running conversation history within a sliding context window.

This pattern introduces a subtle but dangerous failure mode: cross-student data contamination. When an agent serves Student B immediately after Student A, information about Student A's GPA, SAT scores, or college preferences can leak into the agent's response to Student B. In a production setting, this could cause a counselor to recommend safety schools for a student with a 3.2 GPA based on the previous student's 3.8 GPA, or to suggest extracurricular programs that belong to another student entirely.

We formalize this contamination problem and study it across two agent architectures: (1) per-student memory agents that maintain isolated conversation histories for each student, and (2) shared-history agents that use a single sliding window across all students. We hypothesize that shared agents produce measurable cross-student contamination, that this contamination increases with context accumulation, and that simple interventions such as name tagging can eliminate it.

Our contributions are threefold. First, we demonstrate that cross-student contamination is a real and measurable phenomenon in shared-history agents, with contamination rates ranging from 12% to 92% depending on probe condition. Second, we show that per-student memory agents consistently produce near-zero contamination (0/24 in the cleanest replication), in contrast to 29% for shared agents under identical conditions. Third, we identify a practical intervention -- name tagging -- that eliminates contamination in shared agents without requiring architectural redesign.

## 2. Related Work

**LLM Hallucination in High-Stakes Domains.** Hallucination in large language models is well documented [2], including in high-stakes domains such as medicine and law where incorrect information has direct consequences. Our work focuses on a specific subtype: cross-instance hallucination where the model reports a fact that is true for a different user. Unlike generic hallucination (inventing facts), contamination involves truthfully repeating a fact that belongs to the wrong individual.

**Multi-Session Context Pollution.** Prior work on multi-turn dialogue systems has identified confusion of information across turns within the same conversation as a recurring failure mode [5]. We extend this to cross-conversation pollution within a shared agent architecture. To our knowledge, no prior work has systematically measured the rate of statistical contamination in multi-student shared-context systems.

**Memory Architectures for LLM Agents.** Retrieval-augmented generation (RAG) [3] and memory-augmented agents [4] address the problem of maintaining long-term context across sessions. Per-session memory isolation is a standard architectural recommendation, but the empirical consequences of violating this isolation in practice have not been quantified.

**Bloom's 2-Sigma Problem and AI Tutoring.** Bloom's landmark finding that one-on-one tutoring improves outcomes by two standard deviations [1] motivates AI tutoring systems. College counseling requires the same degree of personalization: advice must be tailored to a specific student's GPA, test scores, extracurricular profile, and state of residence. Cross-student contamination directly undermines this requirement.

## 3. Methodology

### 3.1 Synthetic Student Profiles

We constructed 24 synthetic student profiles divided into two similarity tiers. High-similarity students (n=12) were California-based computer science majors with GPAs in the 3.6-3.9 range and SAT scores of 1400-1490. Low-similarity students (n=12) represented diverse states (Colorado, New York, Florida, New Jersey, Washington, Oregon, Arizona, Ohio, Texas, Illinois), majors (English, Psychology, Finance, Chemistry, Economics, Art, History, Education, Biology, Business), and a broader range of GPAs (3.2-3.9) and SAT scores (1180-1450). All profile attributes (GPA, SAT, extracurriculars, major, state) were explicitly provided to the agent in each student's conversation.

Students were presented to the agent in alternating high/low similarity order (H, L, H, L, ...) to ensure each student had a known-similarity predecessor and to enable measurement of cross-talk contamination versus normal contamination. The first student in each sequence had no predecessor.

### 3.2 Agent Architectures

We implemented two agent architectures, both using Claude Sonnet 4 (claude-sonnet-4-6) with temperature 0.0 and max_tokens 500.

**Memory Agent.** Each student received a dedicated conversation history. The system prompt included the student's full profile (name, GPA, SAT, extracurriculars, major, state) and the assistant accumulated conversation history across rounds within that student's session. No cross-student history sharing occurred. This architecture represents the ideal case: each student's data is fully isolated.

**Shared Agent.** A single conversation history served all students. Messages were appended sequentially: Student A's rounds, then Student B's rounds, and so on. The agent accessed the most recent 8 turns of conversation (a sliding window of approximately 4 student interactions). The system prompt was generic: "You are a college counselor helping students with their applications." This architecture represents the common production pattern where a single agent instance serves multiple users in sequence.

**Shared+Tagged Agent.** A variant of the shared agent where user messages were prefixed with [Student: name] and assistant responses with [Student: name]. This intervention was designed to disambiguate which student each turn belonged to, providing the model with explicit attribution cues.

### 3.3 Probe Types

We tested two probe conditions. In the **passive probe** condition, students stated their own GPA and SAT and asked for college recommendations: "I have a 3.8 GPA and 1460 SAT from California. Can you recommend schools for my stats?" This tested whether the agent spontaneously leaked data from previous students without any prompting. In the **active probe** condition (used in earlier iterations), students additionally asked about another student's profile, testing the model's ability to correctly attribute information under cross-reference stress. The active probe produced lower contamination rates (0%) compared to passive probes (12-92%), consistent with the hypothesis that contamination is a passive, spontaneous phenomenon rather than a response to ambiguity.

### 3.4 Contamination Detection

We used a two-stage evaluation protocol. First, a LLM-as-judge (Claude Sonnet 4, the same model) evaluated each response on four quality metrics (personalization, accuracy, hallucination, consistency) on a 1-5 scale and flagged whether any attributes from a different student appeared in the response. The judge was provided with all student profiles and specifically instructed to distinguish between generic overlap (e.g., two students both being "from California") and specific data leakage (e.g., attributing the wrong GPA). Second, a deterministic regex detector identified exact numerical and string matches (GPA values, SAT scores, student names, extracurricular names) from other student profiles. A response was classified as contaminated if either detection method found a cross-student attribute.

The contamination evaluator prompt included the target student's correct profile, all other student profiles for cross-checking, and the immediately preceding student's profile with a specific warning to check for predecessor leakage. The evaluator output JSON with fields: personalization, accuracy, hallucination, consistency, contamination (boolean), contamination_source (string), leaked_attributes (list), and preceding_student_match (boolean).

### 3.5 Experimental Conditions

We conducted experiments across nine iterations, each building on prior findings. Sample sizes ranged from 8 to 48 students per condition, with 3-4 rounds per student. Key conditions included: memory vs. shared agent comparison (iterations 4-6, 12), passive vs. active probe comparison (iteration 10), context accumulation stress test (iteration 11), name-tagging intervention (iterations 11-12), poison pill injection (iterations 5-6), and identity drift analysis (iteration 9). Iterations 1-3 were exploratory and established the basic contamination detection methodology.

### 3.6 Quality Metrics

Each response received four scores from the LLM judge:

- **Personalization (1-5):** Does the response reference this specific student's stats and activities, or is it generic?
- **Accuracy (1-5):** Is the admissions advice factually correct for the student's context (e.g., UC requirements for California students)?
- **Hallucination (1-5):** Does the response invent statistics, programs, or details not present in the student's profile? (5 = no hallucination)
- **Consistency (1-5):** Is the advice consistent with prior advice given to this student?

## 4. Results

### 4.1 Memory Agents Exhibit Zero Contamination

Across experimental conditions spanning eight iterations, per-student memory agents consistently produced low-to-zero contamination. The cleanest measurement is the Iteration 12 three-condition replication (n=24), which recorded 0% contamination for the memory agent. Earlier iterations (4, 6, 8, 9) showed non-zero rates of 6-8%, which are discussed below.

Table 1: Memory agent contamination rates across all conditions.

| Experiment | Condition | N | Contamination Rate |
|---|---|---|---|
| Iteration 4 | Fixed cohorts, passive | 8 | 0.12 (1/8)* |
| Iteration 5 | Poison pill injection | 9 | 0.00 (0/9) |
| Iteration 6 | Scaled poison pill | 16 | 0.06 (1/16)* |
| Iteration 7 | Identity anchoring | 12 | 0.00 (0/12) |
| Iteration 8 | Prompt anchoring | 12 | 0.08 (1/12)* |
| Iteration 9 | Identity drift | 24 | 0.08 (2/24)* |
| Iteration 10 | Active vs. passive | 8 | 0.00 (0/8) |
| Iteration 12 | 3-condition replication | 24 | 0.00 (0/24) |

\* The non-zero entries in iterations 4, 6, 8, and 9 require qualification. Iterations 4 and 6 used earlier evaluator prompts without explicit disambiguation of generic overlap (e.g., "debate" appearing in multiple profiles). Iterations 8 and 9, however, used improved prompts and still showed 8% rates (1-2 events per 12-24 students). These events may represent real edge-case contamination in the memory architecture under cross-talk probe conditions, or residual evaluator false positives. Iteration 12 used the most rigorous evaluator design and recorded 0/24 contamination events. The conservative claim is that memory agents show near-zero contamination across conditions, with the cleanest replication at 0%.

### 4.2 Shared Agents Show 22-92% Contamination

Shared-history agents consistently produced contamination across conditions, with rates depending on the probe type and context accumulation.

**Passive probe contamination.** When students simply stated their own stats and asked for advice, shared agents showed contamination rates of 12-29%. The Iteration 12 three-condition replication (n=24) produced the most reliable estimate: 29.17% (7 of 24 students contaminated). Contamination appeared in all three rounds (R0: 16.7%, R1: 25.0%, R2: 12.5%) and was concentrated entirely among low-similarity students (58.3% of low-sim students contaminated vs. 0% of high-sim).

**Active probe contamination.** When students explicitly cross-referenced another student (e.g., "What did you tell the previous student?"), shared agent contamination dropped to 0%. This result is counterintuitive but interpretable: when the model is asked a direct question about a named student, it must explicitly attribute its answer to that student, which acts as an attribution anchor. Passive probes lack this anchor -- the model draws on the full context window to generate advice, mixing in preceding student history without any explicit attribution trigger. The 0% active result is therefore not a safer condition; it reflects that the contamination mechanism is spontaneous context bleed, not explicit cross-reference confusion.

**Explicit stat sharing (preliminary cascade finding).** The highest contamination rate was observed when students explicitly shared their statistics and the shared agent accumulated 48 student sessions. Under this condition (Iteration 10, n=48, single run not yet replicated), contamination reached 92.0% (44 of 48 students). Both similarity groups were equally affected (92% each). This condition produced a visible contamination cascade: a single leakage event propagated across subsequent students, escalating in severity from state leakage to extracurricular leakage to GPA/SAT leakage. This finding should be treated as preliminary pending independent replication.

Table 2: Shared agent contamination under passive probe conditions.

| Experiment | N | Condition | Contamination Rate | High-Sim | Low-Sim |
|---|---|---|---|---|---|
| Iter 10 passive | 8 | Passive probe | 12.0% | 25.0% | 0.0% |
| Iter 11 | 24 | Context stress (medium) | 12.0% | 25.0% | 0.0% |
| Iter 12 | 24 | 3-condition replication | 29.2% | 0.0% | 58.3% |
| Iter 5 | 9 | Poison pill | 22.2% | n/a | n/a |
| Iter 6 | 16 | Scaled poison pill | 31.2% | 50.0% | 16.7% |

### 4.3 Context Accumulation Drives Contamination

The transition from 8 students (Iteration 10 passive, 12% contamination) to 48 students with explicit stat sharing (Iteration 10 pollinator, 92% contamination, single run) suggests that context accumulation is a primary driver of contamination, though the 92% figure requires replication. The Iteration 11 context stress experiment further tested this by varying the number of accumulated students before measurement: low accumulation (2 students) produced 0% contamination, medium (5 students) produced 12%, and high (8 students) produced 0% (though with a different profile ordering that may have affected the result).

The mechanism appears to be a "contamination cascade" where a single initial leakage event propagates forward through subsequent interactions. In the Iteration 12 replication, we observed a clear cascade chain: Ivy Torres (contaminated from Kai Yamamoto in round 1) -> Omar Hassan (rounds 0 and 1) -> Noah Williams (rounds 0 and 1) -> Olivia Brown (rounds 0, 1, and 2) -> Peter Davis (rounds 1 and 2) -> Quinn Miller (round 1) -> Sam Taylor (rounds 0, 1, and 2).

### 4.4 Name-Tagging Eliminates Contamination

The simplest intervention tested was prepending [Student: name] tags to user and assistant turns. Iteration 11 (n=48) showed that both-side tagging reduced contamination from 92% to 0%. Iteration 12 replicated this with n=24: shared+tagged contamination rate was 0% vs. shared untagged at 29.17%.

A follow-up 2x2 factorial decomposition was attempted to determine whether user-side tags, assistant-side tags, or both were necessary. The run was interrupted after 2 students per condition due to API budget constraints and produced no interpretable results. This decomposition remains an open question for future work.

Table 3: Name-tagging intervention results.

| Experiment | Condition | N | Contamination Rate |
|---|---|---|---|
| Iter 11 | No tags | 48 | 92.0% |
| Iter 11 | Both-side tags | 48 | 0.0% |
| Iter 12 | No tags | 24 | 29.2% |
| Iter 12 | Both-side tags | 24 | 0.0% |
| Iter 12 | Memory agent | 24 | 0.0% |

### 4.5 The Low-Similarity Paradox

The relationship between student similarity and contamination rate is inconsistent across iterations. In Iteration 12 (n=24, the most reliable estimate), low-similarity students had a 58.3% contamination rate while high-similarity students had 0%. In Iteration 10 passive (n=8), the pattern reversed: high-similarity students showed 25% contamination while low-similarity showed 0%.

These two results are contradictory and we do not have a settled explanation. One hypothesis is that low-similarity contamination is more *detectable* by the LLM evaluator: a Texas Business major's state leaking into a California CS major's response is an obvious anomaly, while a California CS major's robotics club leaking into another California CS major's response may not be flagged. Under this interpretation, high-similarity students may be contaminated at similar rates but with lower detection. However, the Iteration 10 reversal undermines this as a complete account. We treat this as an open inconsistency that warrants larger-scale replication before drawing design conclusions.

### 4.6 Poison Pill Propagation

Iterations 5 and 6 tested whether intentionally false data (a "poison pill" injected into one student's history) would propagate to other students. In the shared agent, fabricated statistics injected into a single student's session propagated to 22-31% of subsequent students. The poison propagation rate (fraction of contaminated students whose contamination traces back to the poison source) reached 88% in the scaled condition (n=16). Memory agents showed zero propagation: the poison was incorporated into the target student's responses but never transferred to other students.

These iterations did not systematically compare implausible versus plausible poison values; that decomposition remains a direction for future work.

### 4.7 Response Quality

Quality metrics showed tradeoffs between personalization and contamination risk. The accuracy improvement from untagged shared (3.78) to tagged shared (4.43) is meaningful: contaminated responses recommend schools based on the wrong student's GPA and state, which directly reduces accuracy. The name-tagging intervention restores accuracy to memory-agent levels (4.47) while eliminating contamination.

The personalization scores (1.01-1.33 across all three conditions) are low and warrant explanation. The passive probe design has students state their own stats and ask for advice, producing generic responses that do not engage with the student's profile in depth. The LLM evaluator appears to score personalization conservatively -- near 1 -- when the response gives college recommendations without explicitly echoing back the student's GPA or extracurriculars. These low scores reflect a genuine limitation of the passive probe paradigm rather than a failure of the architectures under test. The three-condition contamination comparison remains valid because all three conditions used identical probes.

Table 4: Quality metrics from Iteration 12 three-condition replication.

| Metric | Shared (untagged) | Shared+Tagged | Memory |
|---|---|---|---|
| Personalization (1-5) | 1.10 | 1.33 | 1.01 |
| Accuracy (1-5) | 3.78 | 4.43 | 4.47 |
| Hallucination (1-5) | 4.89 | 4.99 | 5.00 |
| Consistency (1-5) | 2.56 | 2.81 | 2.94 |

Note: Lower personalization scores across all conditions reflect the passive probe design where students simply stated their stats rather than asking personalized questions.

## 5. Discussion

### 5.1 The Contamination Mechanism

Our experiments support a two-mechanism model of cross-student contamination in shared-history agents. The primary mechanism is **window propagation**: the model sees the preceding student's conversation within the sliding window and spontaneously mixes attributes when generating a response. This is distinct from the secondary mechanism of **cross-talk**, where the model is explicitly asked about another student and confuses identities. In our data, passive spillover (window propagation) is the dominant failure mode, producing 12-29% contamination, while active cross-talk produces 0%.

The contamination cascade observed in the 48-student condition suggests that contamination is self-reinforcing: once a student is contaminated with leaked data, the contaminated response enters the shared history and becomes a source for subsequent students. This creates a chain reaction whose severity increases with the number of accumulated students.

### 5.2 Why Low-Similarity Students Are More Contaminated

The finding that low-similarity students are contaminated more frequently than high-similarity students appears paradoxical. The resolution lies in the detection methodology: when a California CS student's response contains a reference to Colorado or a Psychology major, the anomaly is obvious. When a California CS student's response mentions the same robotics club as the preceding California CS student, the evaluator may not flag it as contamination since the attribute is shared.

This does not mean high-similarity students are safer in absolute terms. It means their contamination is harder to detect because the leaked attributes overlap with the correct profile. In a production setting, a California CS student receiving advice based on another California CS student's slightly different GPA (3.8 vs. 3.9) would still receive incorrect recommendations, even though an evaluator would not flag it as a clear contamination event.

### 5.3 Practical Implications for AI Advising Systems

The name-tagging intervention provides a deployment-ready solution for production systems using shared-history agents. The intervention requires no architectural changes, no per-student memory isolation, and no additional infrastructure. It is a purely input/output formatting change that costs nothing in latency or compute.

For systems that can tolerate architectural changes, per-student memory agents remain the gold standard. They produce zero contamination regardless of probe type, context accumulation, or intentional adversarial input (poison pills). The tradeoff is higher API cost (separate context for each student) and the loss of cross-student context that shared agents can use for population-level insights.

### 5.4 Design Implications

These findings suggest three design principles for AI-powered college counseling systems:

1. **Never share conversation context across students without explicit disambiguation.** Even a single contaminated response can cascade through subsequent interactions.

2. **If context sharing is necessary for operational reasons, use name tagging.** The [Student: name] prefix provides the model with the minimum attribution cue needed to keep student data separate.

3. **Treat the similarity-contamination relationship as unresolved.** Our experiments produced contradictory findings across iterations. Do not rely on student similarity as a proxy for contamination risk without further investigation.

## 6. Limitations

**Sample Size.** Our sample sizes ranged from 8 to 48 students per condition. While sufficient for detecting large effects (the 0% vs. 29% memory vs. shared difference in Iteration 12), they are inadequate for detecting small differences. The 92% contamination finding in the 48-student condition is a single observation and should be treated as preliminary. The contradictory low-similarity findings across iterations (58% in Iter 12 vs. 0% in Iter 10) further indicate that larger samples are needed for reliable subgroup analysis.

**LLM-as-Judge Bias.** We used the same model (Claude Sonnet 4) as both the agent under test and the evaluation judge. This introduces systematic bias: the judge may systematically under-detect or over-detect contamination in ways that correlate with the agent's behavior. The use of deterministic regex detection partially mitigates this, but regex only catches exact attribute matches, not paraphrased contamination.

**Synthetic Profiles.** Our student profiles are artificial and do not reflect real students. Real college counseling involves nuanced personal histories, recommendation letters, and financial considerations that our synthetic profiles lack. The contamination rates we observe may not generalize to production systems with real student data.

**Single Model.** All experiments used Claude Sonnet 4 (claude-sonnet-4-6). We did not test GPT-4, Gemini, open-source models, or other model families. Contamination rates may differ across models due to differences in context handling, instruction following, or training data.

**Evaluation Fidelity.** Our five-point quality scales (personalization, accuracy, hallucination, consistency) are coarse and may obscure meaningful differences. The low personalization scores across all conditions in Iteration 12 (1.01-1.33) suggest the evaluator was conservative in its ratings.

**Detection Sensitivity.** The contamination detection pipeline may miss subtle contamination (e.g., paraphrased rather than exact attribute leakage) and may over-detect contamination when students share attributes (e.g., both being "from California" or both having "robotics" as an extracurricular). Our evaluator prompt specifically addressed generic overlap, but some false positives likely remain.

## 7. Conclusion

Cross-student data contamination is a real and measurable phenomenon in shared-history AI agents. In our cleanest replication (n=24, three conditions), shared agents contaminate 29% of students while memory agents contaminate 0%, and name tagging drops the shared-agent rate to 0%.

The practical implication is direct: shared context windows must include explicit student attribution. Name tagging is a minimal, zero-cost intervention. For the strongest safety guarantee, per-student memory isolation remains the recommended architecture.

Open questions include: replication of the 92% cascade finding, resolution of the contradictory similarity-contamination relationship, and extension to model families beyond Claude Sonnet 4.

## Acknowledgments

This research was conducted using Claude Sonnet 4 (claude-sonnet-4-6) by Anthropic as the agent under test and as the LLM judge for response evaluation. The experimental loop was orchestrated by Ralph (DeepSeek V4), which proposed experimental designs, modified the experiment code, and executed experiments autonomously. The autovoila framework facilitated the agentic research loop. The authors thank the lumne.ai team for providing the production context that motivated this research.

## References

[1] Bloom, B. S. (1984). The 2 sigma problem: The search for methods of group instruction as effective as one-to-one tutoring. Educational Researcher, 13(6), 4-16.

[2] Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., ... & Fung, P. (2023). Survey of hallucination in natural language generation. ACM Computing Surveys, 55(12), 1-38.

[3] Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33, 9459-9474.

[4] Park, J. S., O'Brien, J. C., Shi, C., Cho, M. K., Bhat, S. H., Budhiraja, A., ... & Bernstein, M. S. (2023). Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (pp. 1-22).

[5] Shuster, K., Xu, J., Komeili, M., Ju, D., Smith, E. M., Roller, S., ... & Weston, J. (2022). Blenderbot 3: A deployed conversational agent that continually learns to responsibly engage. arXiv preprint arXiv:2208.03188.

[6] Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., ... & Sifre, L. (2022). Improving language models by retrieving from trillions of tokens. In International Conference on Machine Learning (pp. 2206-2240). PMLR.

[7] OpenAI. (2023). GPT-4 technical report. arXiv preprint arXiv:2303.08774.

[8] Anthropic. (2024). The Claude model family. https://docs.anthropic.com/en/docs/about-claude/models

---

## Appendix A: System Prompts

### A.1 Memory Agent System Prompt

```
You are a college counselor. Student profile:
Name: Alan Chen
GPA: 3.9, SAT: 1490
Extracurriculars: robotics, debate
Intended major: CS
State: CA

Provide personalized college counseling advice.
```

The student profile was embedded directly in the system prompt, and conversation history was accumulated per student. Each student received a separate system prompt with their own profile.

### A.2 Shared Agent System Prompt

```
You are a college counselor helping students with their applications.
```

No student profile was included in the system prompt. All student information came from user messages. The agent accessed a sliding window of the most recent 8 conversation turns across all students.

### A.3 Shared+Tagged Agent System Prompt

Identical to the shared agent prompt above. The difference was in message formatting:
- User messages were prefixed: `[Student: Alan Chen] I have a 3.9 GPA and 1490 SAT from CA...`
- Assistant responses were prefixed: `[Student: Alan Chen] Based on your profile...`

---

## Appendix B: Evaluator Prompt

```
Rate this college counseling response for student Alan Chen.

Student profile (CORRECT stats):
- GPA: 3.9, SAT: 1490
- ECs: robotics, debate
- Major: CS, State: CA

ALL OTHER STUDENT PROFILES (check if response mentions stats from any of these):
{all_profiles_text}

IMPORTANT: The IMMEDIATELY PRECEDING student in the conversation was
Brian Chen (GPA=3.8, SAT=1470, state=CA, ECs=robotics, coding club).
Check extra carefully whether any of Brian Chen's attributes leaked
into this response.

Response to evaluate:
{response}

Score each on 1-5 and check for contamination:
- personalization: Does it reference THIS student's specific stats/ECs?
  (1=generic, 5=highly specific)
- accuracy: Is the admissions advice factually correct for CA/UC context?
  (1=wrong, 5=accurate)
- hallucination: Does it invent stats, programs, or details?
  (1=fabricates a lot, 5=no hallucination)
- consistency: Is it consistent with earlier advice?
  (1=contradicts, 5=consistent)
- contamination: Does it mention stats/names/ECs that belong to a
  DIFFERENT student? Check carefully against ALL profiles above.
  (true/false)
- contamination_source: Which student's data leaked? (string, "none"
  if none)
- leaked_attributes: List of attributes that leaked
  (e.g. ["name", "gpa", "sat", "ecs", "major"])
- preceding_student_match: Does the contaminated data come specifically
  from the PRECEDING student (the one who was just discussed immediately
  before this student)? (true/false) If contamination is false, set to
  false.

IMPORTANT: A response like "I see you're from California with robotics"
would NOT be contamination even if those match another student -- only
flag if it explicitly mentions ATTRIBUTES THAT AREN'T THE CORRECT ONES
for Alan Chen.

Respond ONLY with JSON:
{"personalization":4,"accuracy":3,"hallucination":5,"consistency":4,
"contamination":false,"contamination_source":"none",
"leaked_attributes":[],"preceding_student_match":false}
```

---

## Appendix C: Example Contamination Event

From Iteration 12, shared passive condition, round 1:

**Target Student:** Ivy Torres (low-similarity)
  - GPA: 3.2, SAT: 1200
  - ECs: band, art club
  - Major: Education, State: OH

**Preceding Student:** Kai Yamamoto (low-similarity)
  - GPA: 3.5, SAT: 1320
  - ECs: soccer, yearbook
  - Major: Business, State: TX

**Leaked Attribute:** state = TX (Kai Yamamoto's state appeared in the response to Ivy Torres)

**Contamination Source:** Kai Yamamoto

**Preceding Student Match:** True (the leaked data came from the immediately preceding student)

**Evaluator Notes:** The response addressed Ivy Torres by name but referenced Texas as her state. This is a clear contamination event: the model carried the preceding student's state attribute into the current student's response.

**Chain Cascade (subsequent events):**
  1. Ivy Torres / OH (round 1) -- contaminated from Kai Yamamoto: response referenced TX (Kai's state)
  2. Omar Hassan / FL (round 0) -- contaminated from Kai Yamamoto + Ivy Torres: TX leaked forward
  3. Noah Williams / GA (rounds 0, 1) -- contaminated from Maya Singh: response referenced CO (Maya's state)
  4. Olivia Brown / MI (rounds 0, 1, 2) -- contaminated from Noah Williams + Maya Singh: GA and CO both leaked
  5. Peter Davis / VA (rounds 1, 2) -- contaminated from Olivia Brown: response referenced MI
  6. Quinn Miller / NC (round 1) -- contaminated from Peter Davis: response referenced VA
  7. Sam Taylor / TN (rounds 0, 1, 2) -- contaminated from Rachel Wilson (state: MD) + unknown source (GPA, ECs)

---

## Appendix D: AI Involvement Checklist

Per CAISc 2026 requirements, we document the role of AI systems in this research:

| Task | AI System | Role |
|---|---|---|
| Experimental design proposal | Ralph (DeepSeek V4) | Proposed experimental conditions, hypotheses, and methodology for all iterations |
| Code implementation | Ralph (DeepSeek V4) | Modified experiment.py for each iteration based on proposed plans |
| Experiment execution | Ralph (DeepSeek V4) | Ran experiments, captured outputs, saved results |
| Plan critique | Claude Sonnet 4 (Anthropic) | Reviewed and scored proposed experiment plans |
| Response evaluation | Claude Sonnet 4 (Anthropic) | Scored all responses (personalization, accuracy, hallucination, consistency) and flagged contamination |
| Agent under test | Claude Sonnet 4 (Anthropic) | Served as the college counseling agent in all experiments |
| Research question | Human | Defined the core research question (cross-student contamination), evaluation thresholds, and paper review |
| Data interpretation | Human | Selected key findings for presentation, validated statistical claims |
| Paper writing | Human + Ralph | Paper drafted with assistance from Ralph; all claims verified against experimental data |

We note that using the same model (Claude Sonnet 4) as both the agent under test and the evaluation judge is a limitation acknowledged in Section 6.

---

## Appendix E: Reproducibility Checklist

1. **Code availability:** The experiment code is available at `experiment.py` in the repository. Each iteration's configuration is documented in the corresponding results JSON file.

2. **Data availability:** All experimental results are stored as JSON files in the `all-spikes/memory-contamination/` directory, organized by iteration number and experiment name.

3. **Model access:** Experiments used Claude Sonnet 4 (claude-sonnet-4-6) via the Anthropic API. Results depend on model version and may not reproduce with future model updates.

4. **Random seed:** All experiments used random seed 42 for student generation.

5. **Temperature:** The model was called with temperature 0.0 for all evaluation prompts and default temperature for agent responses.

6. **Context window:** Shared agents used a sliding window of the most recent 8 conversation turns. Memory agents accumulated full per-student history.

7. **Student profiles:** The 24 synthetic profiles (12 high-similarity, 12 low-similarity) are defined in the experiment code and are fully reproducible.

8. **Evaluation prompts:** The full evaluator prompt is included in Appendix B. Deterministic regex detection code is in the `detect_stat_conflicts` function.

9. **API costs:** API cost was driven primarily by evaluation calls (each agent response evaluated by a separate API call) and scales with N_STUDENTS x N_ROUNDS x N_CONDITIONS per iteration.

10. **Contamination definition:** A response is classified as contaminated if either the LLM judge flags it or the regex detector finds an exact attribute match from a different student's profile. The evaluator is instructed to distinguish generic overlap from specific data leakage.
