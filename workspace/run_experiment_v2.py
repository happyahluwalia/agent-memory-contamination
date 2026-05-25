#!/usr/bin/env python3
"""
Experiment runner v2 — SCALED DOWN: 10 students, efficient calls to stay under $2.

"Per-student memory in agents improves personalization but hurts consistency across similar students"

Requires: ANTHROPIC_API_KEY env var
"""

import json
import os
import sys
import time

from anthropic import Anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)

client = Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-sonnet-4-6"

# Load profiles — use first 10 plus their similar pairs
with open("/research/workspace/synthetic_student_profiles.json") as f:
    all_profiles = json.load(f)

with open("/research/workspace/similar_pairs.json") as f:
    all_similar_pairs = json.load(f)

# Select 10 students + their paired students (to get consistency pairs)
# Use first 10 indices (0-9) for main eval, and their pairs (which go up to ~19)
selected_indices = set()
selected_pairs = []
for i, j in all_similar_pairs[:10]:  # Take first 10 pairs
    selected_indices.add(i)
    selected_indices.add(j)
    selected_pairs.append((i, j))

selected_indices = sorted(list(selected_indices))[:12]  # Cap at 12 students
profiles = [all_profiles[i] for i in selected_indices]

# Filter to relevant pairs
student_set = set(i for i in selected_indices)
similar_pairs = [(i, j) for i, j in all_similar_pairs if i in student_set and j in student_set]
# Remap indices
index_map = {old: new for new, old in enumerate(selected_indices)}
similar_pairs = [(index_map[i], index_map[j]) for i, j in similar_pairs]

NUM_STUDENTS = len(profiles)
print(f"Using {NUM_STUDENTS} students, {len(similar_pairs)} similar pairs")

# Cost tracking
total_input_tokens = 0
total_output_tokens = 0
COST_PER_INPUT = 3.0 / 1_000_000
COST_PER_OUTPUT = 15.0 / 1_000_000

def track_cost(input_tokens, output_tokens, label=""):
    global total_input_tokens, total_output_tokens
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    cost = (input_tokens * COST_PER_INPUT) + (output_tokens * COST_PER_OUTPUT)
    current_total = total_input_tokens * COST_PER_INPUT + total_output_tokens * COST_PER_OUTPUT
    print(f"  [{label}] In:{input_tokens} Out:{output_tokens} ${cost:.4f}  (cumulative: ${current_total:.4f})")
    return cost

def print_cost_summary():
    total = total_input_tokens * COST_PER_INPUT + total_output_tokens * COST_PER_OUTPUT
    print(f"\n{'='*50}")
    print(f"Total cost: ${total:.4f}")
    print(f"Input tokens: {total_input_tokens}, Output tokens: {total_output_tokens}")

# Prompts
ADVISING_SYSTEM_PROMPT = """You are an expert college admissions counselor with deep knowledge of UC admissions. Provide accurate, personalized, and specific college advising advice. Base your recommendations on the student's profile and actual UC admissions data. Do NOT fabricate specific admissions statistics or policies. If you are unsure, state that clearly."""

def make_advising_prompt(student):
    return f"""Student Profile:
- Name: {student['first_name']} {student['last_name']}
- Demographics: {student['demographics']}
- State: {student['state']}
- Intended Major: {student['intended_major']}
- GPA (unweighted): {student['gpa']}
- SAT: {student['sat'] if student['sat'] else 'Not submitted'}
- First Generation: {'Yes' if student['first_generation'] else 'No'}
- Extracurriculars: {', '.join(student['extracurriculars'][:4])}
- Essay Theme: {student['essay_theme']}
- Personal Goal: {student['goal']}

Based on this profile, provide:
1. **Reach Schools** (2-3 with reasons)
2. **Target Schools** (2-3 with reasons)
3. **Safety Schools** (2-3 with reasons)
4. **Top 3 Recommendations** to strengthen their application
5. **Major-Specific Advice** for {student['intended_major']}

Base recommendations on actual UC admissions knowledge. Say if uncertain."""

# CONDITION A: Per-Student Memory
def condition_a(student):
    prompt = make_advising_prompt(student)
    messages = [{"role": "user", "content": prompt}]
    
    resp = client.messages.create(
        model=MODEL, max_tokens=1500,
        system=ADVISING_SYSTEM_PROMPT,
        messages=messages
    )
    advice = resp.content[0].text
    track_cost(resp.usage.input_tokens, resp.usage.output_tokens, label=f"A-init-{student['id']}")
    
    # Follow-up: test memory
    messages.append({"role": "assistant", "content": advice})
    messages.append({"role": "user", "content": "I'm back for another session. What was the key advice we discussed? And what is the single most important thing I should do next, specific to my profile?"})
    
    resp2 = client.messages.create(
        model=MODEL, max_tokens=800,
        system=ADVISING_SYSTEM_PROMPT,
        messages=messages
    )
    follow_up = resp2.content[0].text
    track_cost(resp2.usage.input_tokens, resp2.usage.output_tokens, label=f"A-follow-{student['id']}")
    
    return {"student_id": student["id"], "condition": "A_per_agent_memory",
            "initial_advice": advice, "follow_up": follow_up,
            "recalled_details": None}  # Will be analyzed later

# CONDITION B: Shared Context (single agent, no per-student memory)
def condition_b(student, context_prefix=""):
    prompt = make_advising_prompt(student)
    
    messages = []
    if context_prefix:
        messages.append({"role": "user", "content": context_prefix})
        messages.append({"role": "assistant", "content": "Understood."})
    
    messages.append({"role": "user", "content": prompt})
    
    # Prune if too long
    total_chars = sum(len(m["content"]) for m in messages)
    if total_chars > 4000:
        messages = messages[-2:]
    
    resp = client.messages.create(
        model=MODEL, max_tokens=1500,
        system=ADVISING_SYSTEM_PROMPT,
        messages=messages
    )
    advice = resp.content[0].text
    track_cost(resp.usage.input_tokens, resp.usage.output_tokens, label=f"B-{student['id']}")
    
    return {"student_id": student["id"], "condition": "B_shared_context",
            "initial_advice": advice}, advice

# EVALUATION
def evaluate_advice(results_a, results_b):
    """Evaluate each pair of advice on personalization and accuracy."""
    evaluators = []
    
    for i, student in enumerate(profiles):
        ra = results_a[i]
        rb = results_b[i]
        
        prompt = f"""Rate the following two college advising responses for the SAME student.

Student: GPA {student['gpa']}, Major: {student['intended_major']}, ECs: {', '.join(student['extracurriculars'][:3])}...
Goal: {student['goal'][:120]}

**ADVICE A** (dedicated agent):
{ra['initial_advice'][:1200]}

**ADVICE B** (shared agent):
{rb['initial_advice'][:1200]}

Rate each 1-5 on:
- personalization: uses THIS student's specific details
- accuracy: correct UC admissions knowledge, no fabricated stats
- actionability: specific actionable recommendations
- hallucination: 5 = none, 1 = severe

Return JSON ONLY:
{{"a": {{"personalization": int, "accuracy": int, "actionability": int, "hallucination": int}},
"b": {{"personalization": int, "accuracy": int, "actionability": int, "hallucination": int}}}}"""
        
        resp = client.messages.create(
            model=MODEL, max_tokens=400,
            system="Respond ONLY with valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        track_cost(resp.usage.input_tokens, resp.usage.output_tokens, label=f"eval-{student['id']}")
        
        try:
            evaluators.append(json.loads(resp.content[0].text))
        except json.JSONDecodeError:
            print(f"  WARNING: eval parse failed for student {student['id']}")
            evaluators.append(None)
        
        if i % 3 == 2:
            time.sleep(0.3)
    
    return evaluators

def evaluate_recall(results_a):
    """Analyze how well Condition A agents remembered student details."""
    print("\n--- Evaluating Memory Recall ---")
    
    for i, result in enumerate(results_a):
        student = profiles[i]
        
        # Check if the follow-up mentions specific details from the student profile
        prompt = f"""Student details:
- Name: {student['first_name']} {student['last_name']}
- GPA: {student['gpa']}, Major: {student['intended_major']}
- ECs: {', '.join(student['extracurriculars'][:3])}
- Goal: {student['goal'][:100]}

Agent's follow-up response (supposed to recall the student):
{result['follow_up'][:800]}

Did the agent correctly recall the student's specific details?
Score 1-5: 1=completely wrong/no recall, 5=perfect recall with specific details.
Return JSON: {{"recall_score": int, "details_recalled": "list what was remembered correctly"}}"""
        
        resp = client.messages.create(
            model=MODEL, max_tokens=300,
            system="Respond ONLY with valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        track_cost(resp.usage.input_tokens, resp.usage.output_tokens, label=f"recall-{student['id']}")
        
        try:
            eval_recall = json.loads(resp.content[0].text)
            result["recalled_details"] = eval_recall
        except json.JSONDecodeError:
            print(f"  WARNING: recall eval parse failed for student {student['id']}")
            result["recalled_details"] = {"recall_score": 0, "details_recalled": "parse failed"}
        
        time.sleep(0.2)
    
    return results_a


def evaluate_consistency(results_a, results_b):
    """Compare consistency across similar pairs."""
    consistency_scores = []
    
    for pair_idx, (i, j) in enumerate(similar_pairs):
        s1, s2 = profiles[i], profiles[j]
        a1, a2 = results_a[i], results_a[j]
        b1, b2 = results_b[i], results_b[j]
        
        prompt = f"""Two similar students received advising. Rate consistency.

Student A: GPA {s1['gpa']}, Major: {s1['intended_major']}, Goal: {s1['goal'][:80]}
Student B: GPA {s2['gpa']}, Major: {s2['intended_major']}, Goal: {s2['goal'][:80]}

Condition A (per-agent memory) advice excerpts:
Student A: {a1['initial_advice'][:600]}
Student B: {a2['initial_advice'][:600]}

Condition B (shared context) advice excerpts:
Student A: {b1['initial_advice'][:600]}
Student B: {b2['initial_advice'][:600]}

Rate consistency (1=very different, 5=near-identical):
Return JSON: {{"condition_a_consistency": int, "condition_b_consistency": int}}"""
        
        resp = client.messages.create(
            model=MODEL, max_tokens=250,
            system="Respond ONLY with valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        track_cost(resp.usage.input_tokens, resp.usage.output_tokens, label=f"cons-{pair_idx}")
        
        try:
            consistency_scores.append(json.loads(resp.content[0].text))
        except json.JSONDecodeError:
            consistency_scores.append(None)
        
        time.sleep(0.2)
    
    return consistency_scores


# =============================================
# MAIN
# =============================================

def main():
    print("=" * 50)
    print("EXPERIMENT: Per-Student Memory vs Consistency")
    print(f"Students: {NUM_STUDENTS}, Pairs: {len(similar_pairs)}")
    print(f"Model: {MODEL}")
    print("=" * 50)
    
    # Phase 1: Condition A
    print("\n--- Phase 1: Per-Student Memory (Condition A) ---")
    results_a = []
    for i, student in enumerate(profiles):
        print(f"\n[{i+1}/{NUM_STUDENTS}] {student['first_name']} {student['last_name']} (GPA: {student['gpa']}, {student['intended_major']})")
        result = condition_a(student)
        results_a.append(result)
        time.sleep(0.2)
    
    with open("/research/workspace/results_a.json", "w") as f:
        json.dump(results_a, f, indent=2, default=str)
    print("\nCondition A results saved.")
    
    # Phase 2: Condition B
    print("\n--- Phase 2: Shared Context (Condition B) ---")
    results_b = []
    last_advice = ""
    for i, student in enumerate(profiles):
        print(f"\n[{i+1}/{NUM_STUDENTS}] {student['first_name']} {student['last_name']}")
        result, advice = condition_b(student, last_advice)
        results_b.append(result)
        last_advice = f"Previous student query about: {student['first_name']} {student['last_name']}"[:300]
        time.sleep(0.2)
    
    with open("/research/workspace/results_b.json", "w") as f:
        json.dump(results_b, f, indent=2, default=str)
    print("\nCondition B results saved.")
    
    # Phase 3: Evaluate recall for Condition A
    print("\n--- Phase 3: Memory Recall Evaluation ---")
    results_a = evaluate_recall(results_a)
    with open("/research/workspace/results_a.json", "w") as f:
        json.dump(results_a, f, indent=2, default=str)
    
    # Phase 4: Evaluate personalization + accuracy
    print("\n--- Phase 4: Personalization & Accuracy ---")
    evaluators = evaluate_advice(results_a, results_b)
    with open("/research/workspace/evaluators.json", "w") as f:
        json.dump(evaluators, f, indent=2, default=str)
    
    # Phase 5: Evaluate consistency
    print("\n--- Phase 5: Consistency Across Similar Pairs ---")
    consistency_scores = evaluate_consistency(results_a, results_b)
    with open("/research/workspace/consistency_scores.json", "w") as f:
        json.dump(consistency_scores, f, indent=2, default=str)
    
    # --- Summary Stats ---
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    valid_evals = [e for e in evaluators if e]
    if valid_evals:
        for cond_key, cond_label in [("a", "Cond A: Per-Student Memory"), ("b", "Cond B: Shared Context")]:
            scores = {k: [e[cond_key][k] for e in valid_evals] for k in ["personalization", "accuracy", "actionability", "hallucination"]}
            print(f"\n{cond_label}:")
            for metric, vals in scores.items():
                print(f"  {metric}: {sum(vals)/len(vals):.2f}/5")
    
    valid_cons = [c for c in consistency_scores if c]
    if valid_cons:
        print(f"\nConsistency (higher = more consistent):")
        a_cons = [c["condition_a_consistency"] for c in valid_cons]
        b_cons = [c["condition_b_consistency"] for c in valid_cons]
        print(f"  Cond A (memory): {sum(a_cons)/len(a_cons):.2f}/5")
        print(f"  Cond B (shared):  {sum(b_cons)/len(b_cons):.2f}/5")
    
    # Recall scores
    recall_scores = [r.get("recalled_details", {}).get("recall_score", 0) for r in results_a]
    if recall_scores:
        print(f"\nMemory Recall (Cond A): {sum(recall_scores)/len(recall_scores):.2f}/5")
    
    print_cost_summary()
    
    # Save summary
    summary = {
        "model": MODEL,
        "num_students": NUM_STUDENTS,
        "num_similar_pairs": len(similar_pairs),
        "total_cost": total_input_tokens * COST_PER_INPUT + total_output_tokens * COST_PER_OUTPUT,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    
    if valid_evals:
        summary["condition_a"] = {k: sum([e["a"][k] for e in valid_evals])/len(valid_evals) for k in ["personalization", "accuracy", "actionability", "hallucination"]}
        summary["condition_b"] = {k: sum([e["b"][k] for e in valid_evals])/len(valid_evals) for k in ["personalization", "accuracy", "actionability", "hallucination"]}
    if valid_cons:
        summary["consistency"] = {
            "condition_a": sum([c["condition_a_consistency"] for c in valid_cons])/len(valid_cons),
            "condition_b": sum([c["condition_b_consistency"] for c in valid_cons])/len(valid_cons)
        }
    if recall_scores:
        summary["recall_score"] = sum(recall_scores)/len(recall_scores)
    
    with open("/research/workspace/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll results saved to /research/workspace/")


if __name__ == "__main__":
    main()
