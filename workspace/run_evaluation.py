#!/usr/bin/env python3
"""
Re-run evaluation only, with better JSON extraction.
"""
import json
import os
import sys
import time
import re
from anthropic import Anthropic

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
client = Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-sonnet-4-6"

with open("/research/workspace/synthetic_student_profiles.json") as f:
    all_profiles = json.load(f)
with open("/research/workspace/similar_pairs.json") as f:
    all_similar_pairs = json.load(f)

with open("/research/workspace/results_a.json") as f:
    results_a = json.load(f)
with open("/research/workspace/results_b.json") as f:
    results_b = json.load(f)

# Determine which students were used
selected_indices = sorted(list(set(
    int(r["student_id"], 16) % 50 for r in results_a  # hacky but works
)))
# Actually just use the order from results
profiles = []
for r in results_a:
    sid = r["student_id"]
    for p in all_profiles:
        if p["id"] == sid:
            profiles.append(p)
            break

NUM_STUDENTS = len(profiles)

# Determine similar pairs
student_ids = [p["id"] for p in profiles]
similar_pairs = []
for i, j in all_similar_pairs:
    if i < len(all_profiles) and j < len(all_profiles):
        pid_i = all_profiles[i]["id"]
        pid_j = all_profiles[j]["id"]
        if pid_i in student_ids and pid_j in student_ids:
            # Get the indices in our subset
            idx_i = student_ids.index(pid_i)
            idx_j = student_ids.index(pid_j)
            similar_pairs.append((idx_i, idx_j))

print(f"Students: {NUM_STUDENTS}, Similar pairs: {len(similar_pairs)}")

def extract_json(text):
    """Extract a JSON object from text, handling markdown code blocks or extra text."""
    # Try to find ```json ... ``` or just { ... }
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    
    # Find the first { and last }
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        text = text[start:end+1]
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

COST_PER_INPUT = 3.0 / 1_000_000
COST_PER_OUTPUT = 15.0 / 1_000_000
total_input_tokens = 0
total_output_tokens = 0

def track_cost(in_t, out_t, label=""):
    global total_input_tokens, total_output_tokens
    total_input_tokens += in_t
    total_output_tokens += out_t
    cost = in_t * COST_PER_INPUT + out_t * COST_PER_OUTPUT
    cum = total_input_tokens * COST_PER_INPUT + total_output_tokens * COST_PER_OUTPUT
    print(f"  [{label}] In:{in_t} Out:{out_t} ${cost:.4f} (cum: ${cum:.4f})")

# Evaluation prompt - more structured
def run_eval():
    """Re-run evaluation with better prompts."""
    evaluators = []
    
    for i, student in enumerate(profiles):
        ra = results_a[i]
        rb = results_b[i]
        
        prompt = f"""You are an automated evaluator. Return ONLY valid JSON, no other text.

Student: GPA {student['gpa']}, Major: {student['intended_major']}, State: {student['state']}
ECs: {', '.join(student['extracurriculars'][:3])}
Goal: {student['goal'][:120]}

ADVICE A (dedicated per-student agent):
{ra['initial_advice'][:1000]}

ADVICE B (shared single agent):
{rb['initial_advice'][:1000]}

Return exactly this JSON structure:
{{"a":{{"personalization":1,"accuracy":1,"actionability":1,"hallucination":1}},"b":{{"personalization":1,"accuracy":1,"actionability":1,"hallucination":1}}}}

Scoring:
- personalization: 1-5 (5=uses this student's specific GPA, major, ECs, goal)
- accuracy: 1-5 (5=correct UC admissions knowledge, no fabricated stats)
- actionability: 1-5 (5=specific actionable recommendations)
- hallucination: 5=none, 1=severe fabricated data

Return ONLY valid JSON, nothing else."""
        
        resp = client.messages.create(
            model=MODEL, max_tokens=300,
            system="You are an evaluator. Respond with ONLY valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        track_cost(resp.usage.input_tokens, resp.usage.output_tokens, label=f"eval-{student['id'][:6]}")
        
        parsed = extract_json(resp.content[0].text)
        if parsed:
            evaluators.append(parsed)
            print(f"  Student {i}: A-pers={parsed['a']['personalization']} B-pers={parsed['b']['personalization']}")
        else:
            print(f"  Student {i}: PARSE FAILED. Raw: {resp.content[0].text[:100]}")
            evaluators.append(None)
        
        if i % 3 == 2:
            time.sleep(0.3)
    
    return evaluators


def run_recall_eval():
    """Re-run recall evaluation."""
    for i, student in enumerate(profiles):
        result = results_a[i]
        
        prompt = f"""You are an evaluator. Return ONLY valid JSON.

Student profile: GPA {student['gpa']}, Major: {student['intended_major']}, ECs: {', '.join(student['extracurriculars'][:3])}, Goal: {student['goal'][:100]}

Agent's follow-up response (supposed to recall the student):
{result['follow_up'][:600]}

Did the agent correctly recall specific details about THIS student?
Score 1-5: 5=perfect recall with specific details, 1=completely generic/no recall

Return ONLY: {{"recall_score": 1, "details_recalled": "brief note on what was remembered"}}"""
        
        resp = client.messages.create(
            model=MODEL, max_tokens=200,
            system="You are an evaluator. Respond with ONLY valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        track_cost(resp.usage.input_tokens, resp.usage.output_tokens, label=f"recall-{student['id'][:6]}")
        
        parsed = extract_json(resp.content[0].text)
        if parsed:
            result["recalled_details"] = parsed
            print(f"  Student {i}: recall={parsed['recall_score']}")
        else:
            print(f"  Student {i}: PARSE FAILED. Raw: {resp.content[0].text[:100]}")
            result["recalled_details"] = {"recall_score": 0, "details_recalled": "parse failed"}
        
        time.sleep(0.3)
    
    return results_a


def run_consistency_eval():
    """Re-run consistency evaluation."""
    consistency_scores = []
    
    for pair_idx, (i, j) in enumerate(similar_pairs):
        s1, s2 = profiles[i], profiles[j]
        a1, a2 = results_a[i], results_a[j]
        b1, b2 = results_b[i], results_b[j]
        
        prompt = f"""You are an evaluator. Return ONLY valid JSON.

Student A: GPA {s1['gpa']}, Major: {s1['intended_major']}
Student B: GPA {s2['gpa']}, Major: {s2['intended_major']}

CONDITION A advice (per-agent memory) excerpts:
A: {a1['initial_advice'][:500]}
B: {a2['initial_advice'][:500]}

CONDITION B advice (shared context) excerpts:
A: {b1['initial_advice'][:500]}
B: {b2['initial_advice'][:500]}

Rate consistency 1-5 (5=near-identical advice, 1=very different):
Return ONLY: {{"condition_a_consistency": 1, "condition_b_consistency": 1}}"""
        
        resp = client.messages.create(
            model=MODEL, max_tokens=200,
            system="You are an evaluator. Respond with ONLY valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        track_cost(resp.usage.input_tokens, resp.usage.output_tokens, label=f"cons-{pair_idx}")
        
        parsed = extract_json(resp.content[0].text)
        if parsed:
            consistency_scores.append(parsed)
            print(f"  Pair {pair_idx}: A-cons={parsed['condition_a_consistency']} B-cons={parsed['condition_b_consistency']}")
        else:
            print(f"  Pair {pair_idx}: PARSE FAILED")
            consistency_scores.append(None)
        
        time.sleep(0.3)
    
    return consistency_scores


# Run all evaluations
print("=== Re-running Evaluations ===\n")

print("\n--- Phase A: Recall Evaluation ---")
results_a = run_recall_eval()
with open("/research/workspace/results_a.json", "w") as f:
    json.dump(results_a, f, indent=2, default=str)

print("\n--- Phase B: Personalization & Accuracy Evaluation ---")
evaluators = run_eval()
with open("/research/workspace/evaluators.json", "w") as f:
    json.dump(evaluators, f, indent=2, default=str)

print("\n--- Phase C: Consistency Evaluation ---")
consistency_scores = run_consistency_eval()
with open("/research/workspace/consistency_scores.json", "w") as f:
    json.dump(consistency_scores, f, indent=2, default=str)

# Summary
valid_evals = [e for e in evaluators if e]
print("\n" + "=" * 50)
print("UPDATED RESULTS SUMMARY")
print("=" * 50)

if valid_evals:
    for cond_key, cond_label in [("a", "Cond A: Per-Student Memory"), ("b", "Cond B: Shared Context")]:
        scores = {}
        for k in ["personalization", "accuracy", "actionability", "hallucination"]:
            vals = [e[cond_key][k] for e in valid_evals]
            scores[k] = sum(vals)/len(vals)
        print(f"\n{cond_label}:")
        for metric, val in scores.items():
            print(f"  {metric}: {val:.2f}/5")

valid_cons = [c for c in consistency_scores if c]
if valid_cons:
    print(f"\nConsistency:")
    a_cons = [c["condition_a_consistency"] for c in valid_cons]
    b_cons = [c["condition_b_consistency"] for c in valid_cons]
    print(f"  Cond A (memory): {sum(a_cons)/len(a_cons):.2f}/5")
    print(f"  Cond B (shared): {sum(b_cons)/len(b_cons):.2f}/5")

recall_scores = [r.get("recalled_details", {}).get("recall_score", 0) for r in results_a]
if recall_scores:
    valid_recall = [s for s in recall_scores if s > 0]
    print(f"\nMemory Recall:")
    print(f"  Cond A: {sum(recall_scores)/len(recall_scores):.2f}/5 (non-zero: {len(valid_recall)}/{len(recall_scores)})")

total_cost = total_input_tokens * COST_PER_INPUT + total_output_tokens * COST_PER_OUTPUT
print(f"\nAdditional evaluation cost: ${total_cost:.4f}")

# Save updated summary
summary = {"model": MODEL, "num_students": NUM_STUDENTS, "num_similar_pairs": len(similar_pairs)}
if valid_evals:
    summary["condition_a"] = {k: sum([e["a"][k] for e in valid_evals])/len(valid_evals) for k in ["personalization", "accuracy", "actionability", "hallucination"]}
    summary["condition_b"] = {k: sum([e["b"][k] for e in valid_evals])/len(valid_evals) for k in ["personalization", "accuracy", "actionability", "hallucination"]}
if valid_cons:
    summary["consistency"] = {"condition_a": sum(a_cons)/len(a_cons), "condition_b": sum(b_cons)/len(b_cons)}
if recall_scores:
    summary["recall_score"] = sum(recall_scores)/len(recall_scores)
summary["total_eval_cost"] = total_cost

with open("/research/workspace/evaluation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved.")
