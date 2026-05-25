#!/usr/bin/env python3
"""
Experiment runner for "Per-student memory in agents improves personalization
but hurts consistency across similar students"

Requires: ANTHROPIC_API_KEY env var
Outputs: results in /research/workspace/
"""

import json
import os
import sys
import time
import hashlib
from datetime import datetime

# Make sure anthropic is available
try:
    from anthropic import Anthropic
except ImportError:
    print("ERROR: anthropic not installed. Run: pip install anthropic")
    sys.exit(1)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("ERROR: ANTHROPIC_API_KEY not set")
    sys.exit(1)

client = Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-sonnet-4-20250514"  # Using Sonnet 4

# Load profiles
with open("/research/workspace/synthetic_student_profiles.json") as f:
    profiles = json.load(f)

with open("/research/workspace/similar_pairs.json") as f:
    similar_pairs = json.load(f)

# --- Cost tracking ---
total_input_tokens = 0
total_output_tokens = 0
COST_PER_INPUT = 3.0 / 1_000_000   # $3 per million input tokens (Sonnet)
COST_PER_OUTPUT = 15.0 / 1_000_000  # $15 per million output tokens (Sonnet)

def track_cost(input_tokens, output_tokens, label=""):
    global total_input_tokens, total_output_tokens
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    cost = (input_tokens * COST_PER_INPUT) + (output_tokens * COST_PER_OUTPUT)
    if label:
        print(f"  [{label}] Input: {input_tokens}, Output: {output_tokens}, Cost: ${cost:.4f}")
    print(f"  Running total: ${(total_input_tokens * COST_PER_INPUT + total_output_tokens * COST_PER_OUTPUT):.4f}")
    return cost

def print_cost_summary():
    total_cost = total_input_tokens * COST_PER_INPUT + total_output_tokens * COST_PER_OUTPUT
    print(f"\n{'='*60}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")

# =============================================
# PROMPTS
# =============================================

ADVISING_SYSTEM_PROMPT = """You are an expert college admissions counselor with deep knowledge of UC (University of California) admissions, including:
- UC GPA calculation (weighted and unweighted, capped/uncapped)
- UC admissions criteria: academic GPA, test scores (optional), ECs, essays, demonstrated interest, major competitiveness
- UC application review process (holistic review by two readers)
- Admissions rates by campus (UC Berkeley, UCLA, UC San Diego, UC Davis, UC Irvine, UC Santa Barbara, etc.)
- Major-specific admissions competitiveness (e.g., CS/Engineering more competitive at certain campuses)

Provide accurate, personalized, and specific college advising advice. Base your recommendations on the student's profile and UC admissions data. Do NOT fabricate specific admissions statistics or policies. If you are unsure, state that clearly rather than making up numbers."""

def make_advising_prompt(student, use_memory_context=True):
    """Construct a prompt for advising a specific student.
    
    If use_memory_context is True, include the student's full profile history.
    If False, just the current query (shared-context condition).
    """
    profile = f"""Student Profile:
- Name: {student['first_name']} {student['last_name']}
- Demographics: {student['demographics']}
- State of Residence: {student['state']}
- Intended Major: {student['intended_major']}
- GPA (unweighted): {student['gpa']}
- SAT: {student['sat'] if student['sat'] else 'Not submitted'}
- ACT: {student['act'] if student['act'] else 'Not submitted'}
- Income Bracket: {student['income_bracket']}
- First Generation College: {'Yes' if student['first_generation'] else 'No'}
- Extracurriculars: {', '.join(student['extracurriculars'])}
- EC Hours/Week: {student['ec_hours_per_week']}
- Essay Theme: {student['essay_theme']}
- Personal Goal: {student['goal']}
"""
    
    query = f"""Based on this student's profile, please provide:

1. **Reach Schools** (list 2-3 with reasons): UC campuses or other colleges where admission is a stretch but possible
2. **Target Schools** (list 2-3 with reasons): Where the student is competitive
3. **Safety Schools** (list 2-3 with reasons): Where admission is likely
4. **Key Recommendations** (3-5 specific actionable suggestions to strengthen their application)
5. **Major-Specific Advice** for {student['intended_major']} applicants

For each recommendation, explain WHY based on the student's specific profile data.

IMPORTANT: Base your recommendations on actual UC admissions knowledge. If you're uncertain about a specific statistic, say so rather than fabricating."""
    
    return profile + "\n" + query


# =============================================
# CONDITION A: Per-Student Memory (dedicated agent per student)
# =============================================

def condition_a_per_agent_memory(student):
    """Each student gets their own agent with memory of their profile."""
    prompt = make_advising_prompt(student, use_memory_context=True)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=ADVISING_SYSTEM_PROMPT,
        messages=messages
    )
    
    advice = response.content[0].text
    track_cost(response.usage.input_tokens, response.usage.output_tokens, label=f"CondA-{student['id']}")
    
    # For condition A, we also do a follow-up to test memory
    follow_up = "Given what you know about this student, what is the single most important thing they should focus on right now to improve their college application? Be specific to this student's profile."
    
    messages.append({"role": "assistant", "content": advice})
    messages.append({"role": "user", "content": follow_up})
    
    response2 = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system=ADVISING_SYSTEM_PROMPT,
        messages=messages
    )
    
    follow_up_advice = response2.content[0].text
    track_cost(response2.usage.input_tokens, response2.usage.output_tokens, label=f"CondA-followup-{student['id']}")
    
    # Simulate another session (next day) — test if memory persists
    messages.append({"role": "assistant", "content": follow_up_advice})
    messages.append({"role": "user", "content": "I'm back for another session. Do you remember my profile? What were we discussing?"})
    
    response3 = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=ADVISING_SYSTEM_PROMPT,
        messages=messages
    )
    
    recall_response = response3.content[0].text
    track_cost(response3.usage.input_tokens, response3.usage.output_tokens, label=f"CondA-recall-{student['id']}")
    
    return {
        "condition": "A_per_agent_memory",
        "student_id": student["id"],
        "initial_advice": advice,
        "follow_up_response": follow_up_advice,
        "recall_response": recall_response,
        "conversation_length": len(messages)
    }


# =============================================
# CONDITION B: Shared-Context Single Agent (no per-student memory)
# =============================================

def condition_b_shared_context(student, preceding_context=""):
    """Single agent seeing all students sequentially."""
    prompt = make_advising_prompt(student, use_memory_context=False)
    
    messages = []
    if preceding_context:
        messages.append({"role": "user", "content": preceding_context})
        messages.append({"role": "assistant", "content": f"Understood. I'll provide advice for this student."})
    
    messages.append({"role": "user", "content": prompt})
    
    # Limit context to avoid blowing up
    if len(messages) > 6:
        messages = messages[-4:]
        # Re-add system context
        full_prompt = "Continuing with college advising. " + prompt
        messages = [{"role": "user", "content": full_prompt}]
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=ADVISING_SYSTEM_PROMPT,
        messages=messages
    )
    
    advice = response.content[0].text
    track_cost(response.usage.input_tokens, response.usage.output_tokens, label=f"CondB-{student['id']}")
    
    return {
        "condition": "B_shared_context",
        "student_id": student["id"],
        "initial_advice": advice,
        "conversation_length": len(messages)
    }, advice


# =============================================
# EVALUATION
# =============================================

def evaluate_results(results_a, results_b):
    """Evaluate the results for personalization and consistency metrics."""
    
    # Strategy: Use a separate Claude call to evaluate
    evaluations = []
    
    for i, student in enumerate(profiles):
        res_a = results_a[i]
        res_b = results_b[i]
        
        eval_prompt = f"""You are evaluating college advising quality. Compare the two pieces of advice below for the SAME student profile.

Student Profile Summary:
- GPA: {student['gpa']}, Major: {student['intended_major']}, State: {student['state']}
- ECs: {', '.join(student['extracurriculars'][:3])}...
- Goal: {student['goal'][:100]}...

**Advice A** (dedicated per-student agent):
{res_a['initial_advice'][:1500]}

**Advice B** (shared-context single agent):
{res_b['initial_advice'][:1500]}

Please rate each on a scale of 1-5 (5 is best):

1. **Personalization**: How specifically does the advice use THIS student's details (GPA, ECs, major, goals)? Generic advice that could apply to anyone scores low.
2. **Accuracy**: Does the advice reflect correct UC admissions knowledge? (No fabricated statistics or policies)
3. **Actionability**: Are the recommendations specific and actionable?
4. **Hallucination**: Are there any fabricated facts, statistics, or policies? Rate 5 = no hallucination, 1 = severe hallucination

Return your evaluation as JSON ONLY:
{{"advice_a": {{"personalization": int, "accuracy": int, "actionability": int, "hallucination": int, "explanation": "brief"}},
"advice_b": {{"personalization": int, "accuracy": int, "actionability": int, "hallucination": int, "explanation": "brief"}}}}"""
        
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            system="You are an expert evaluator of college advising quality. Respond ONLY with valid JSON.",
            messages=[{"role": "user", "content": eval_prompt}]
        )
        
        track_cost(response.usage.input_tokens, response.usage.output_tokens, label=f"Eval-{student['id']}")
        
        try:
            eval_result = json.loads(response.content[0].text)
            evaluators.append(eval_result)
        except json.JSONDecodeError:
            print(f"WARNING: Failed to parse evaluation for student {student['id']}")
            evaluators.append(None)
        
        # Rate limit
        if i % 5 == 4:
            time.sleep(0.5)
    
    return evaluators


def evaluate_consistency(profiles, results_a, results_b, similar_pairs):
    """Evaluate consistency: how similar are recommendations for similar-student pairs?"""
    
    consistency_evaluators = []
    
    for pair_idx, (i, j) in enumerate(similar_pairs):
        s1, s2 = profiles[i], profiles[j]
        a1, a2 = results_a[i], results_a[j]
        b1, b2 = results_b[i], results_b[j]
        
        prompt = f"""Two similar students received college advising. Rate how CONSISTENT the advice is.

Student 1: GPA {s1['gpa']}, Major: {s1['intended_major']}, State: {s1['state']}
Student 2: GPA {s2['gpa']}, Major: {s2['intended_major']}, State: {s2['state']}

**Advice for Student 1 (Condition A - per-agent memory):**
{a1['initial_advice'][:1000]}

**Advice for Student 2 (Condition A - per-agent memory):**
{a2['initial_advice'][:1000]}

**Advice for Student 1 (Condition B - shared context):**
{b1['initial_advice'][:1000]}

**Advice for Student 2 (Condition B - shared context):**
{b2['initial_advice'][:1000]}

Rate CONSISTENCY: For each condition, how similarly were the two students advised?
Consider: Are the same schools recommended? Is the advice similar in tone and content?
Score 1 (very different advice) to 5 (near-identical advice).

Return JSON ONLY:
{{"condition_a_consistency": int, "condition_b_consistency": int,
"a_explanation": "brief", "b_explanation": "brief"}}"""
        
        response = client.messages.create(
            model=MODEL,
            max_tokens=400,
            system="You are an expert evaluator. Respond ONLY with valid JSON.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        track_cost(response.usage.input_tokens, response.usage.output_tokens, label=f"Consistency-{pair_idx}")
        
        try:
            result = json.loads(response.content[0].text)
            consistency_evaluators.append(result)
        except json.JSONDecodeError:
            print(f"WARNING: Failed to parse consistency eval for pair {pair_idx}")
            consistency_evaluators.append(None)
        
        if pair_idx % 5 == 4:
            time.sleep(0.5)
    
    return consistency_evaluators


# =============================================
# MAIN EXPERIMENT
# =============================================

def main():
    global total_input_tokens, total_output_tokens
    
    print("=" * 60)
    print("EXPERIMENT: Per-Student Memory vs Consistency")
    print(f"Profiles: {len(profiles)}, Similar Pairs: {len(similar_pairs)}")
    print(f"Model: {MODEL}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # --- Phase 1: Run Condition A (per-agent memory) ---
    print("\n--- Phase 1: Condition A (Per-Student Memory) ---")
    results_a = []
    for i, student in enumerate(profiles):
        print(f"\nStudent {i+1}/50: {student['first_name']} {student['last_name']} (GPA: {student['gpa']}, Major: {student['intended_major']})")
        result = condition_a_per_agent_memory(student)
        results_a.append(result)
        
        # Save intermediate
        if (i+1) % 10 == 0:
            with open("/research/workspace/results_a_intermediate.json", "w") as f:
                json.dump(results_a, f, indent=2, default=str)
            print(f"  [Checkpoint] Saved {i+1}/50 results for Condition A")
        
        # Rate limit
        time.sleep(0.3)
    
    with open("/research/workspace/results_a.json", "w") as f:
        json.dump(results_a, f, indent=2, default=str)
    
    # --- Phase 2: Run Condition B (shared-context) ---
    print("\n--- Phase 2: Condition B (Shared-Context Single Agent) ---")
    results_b = []
    shared_context = ""
    for i, student in enumerate(profiles):
        print(f"\nStudent {i+1}/50: {student['first_name']} {student['last_name']} (GPA: {student['gpa']}, Major: {student['intended_major']})")
        result, advice_text = condition_b_shared_context(student, shared_context)
        results_b.append(result)
        shared_context = advice_text[:500]  # Keep last advice as context
        
        if (i+1) % 10 == 0:
            with open("/research/workspace/results_b_intermediate.json", "w") as f:
                json.dump(results_b, f, indent=2, default=str)
            print(f"  [Checkpoint] Saved {i+1}/50 results for Condition B")
        
        time.sleep(0.3)
    
    with open("/research/workspace/results_b.json", "w") as f:
        json.dump(results_b, f, indent=2, default=str)
    
    # --- Phase 3: Evaluate ---
    print("\n--- Phase 3: Evaluating Personalization & Accuracy ---")
    evaluators = evaluate_results(results_a, results_b)
    
    with open("/research/workspace/evaluators.json", "w") as f:
        json.dump(evaluators, f, indent=2, default=str)
    
    # --- Phase 4: Evaluate Consistency ---
    print("\n--- Phase 4: Evaluating Consistency Across Similar Pairs ---")
    consistency_scores = evaluate_consistency(profiles, results_a, results_b, similar_pairs)
    
    with open("/research/workspace/consistency_scores.json", "w") as f:
        json.dump(consistency_scores, f, indent=2, default=str)
    
    # --- Summary ---
    print("\n--- Summary ---")
    print_cost_summary()
    
    # Compute average scores
    if evaluators:
        valid_evals = [e for e in evaluators if e]
        if valid_evals:
            a_personalization = [e["advice_a"]["personalization"] for e in valid_evals]
            a_accuracy = [e["advice_a"]["accuracy"] for e in valid_evals]
            a_actionability = [e["advice_a"]["actionability"] for e in valid_evals]
            a_hallucination = [e["advice_a"]["hallucination"] for e in valid_evals]
            
            b_personalization = [e["advice_b"]["personalization"] for e in valid_evals]
            b_accuracy = [e["advice_b"]["accuracy"] for e in valid_evals]
            b_actionability = [e["advice_b"]["actionability"] for e in valid_evals]
            b_hallucination = [e["advice_b"]["hallucination"] for e in valid_evals]
            
            print(f"\nCondition A (Per-Student Memory):")
            print(f"  Personalization: {sum(a_personalization)/len(a_personalization):.2f}/5")
            print(f"  Accuracy: {sum(a_accuracy)/len(a_accuracy):.2f}/5")
            print(f"  Actionability: {sum(a_actionability)/len(a_actionability):.2f}/5")
            print(f"  Hallucination (higher=better): {sum(a_hallucination)/len(a_hallucination):.2f}/5")
            
            print(f"\nCondition B (Shared Context):")
            print(f"  Personalization: {sum(b_personalization)/len(b_personalization):.2f}/5")
            print(f"  Accuracy: {sum(b_accuracy)/len(b_accuracy):.2f}/5")
            print(f"  Actionability: {sum(b_actionability)/len(b_actionability):.2f}/5")
            print(f"  Hallucination (higher=better): {sum(b_hallucination)/len(b_hallucination):.2f}/5")
    
    if consistency_scores:
        valid_cons = [c for c in consistency_scores if c]
        if valid_cons:
            a_cons = [c["condition_a_consistency"] for c in valid_cons]
            b_cons = [c["condition_b_consistency"] for c in valid_cons]
            print(f"\nConsistency Across Similar Pairs:")
            print(f"  Condition A (per-agent memory): {sum(a_cons)/len(a_cons):.2f}/5")
            print(f"  Condition B (shared context):  {sum(b_cons)/len(b_cons):.2f}/5")
    
    # Save final results summary
    summary = {
        "model": MODEL,
        "timestamp": datetime.now().isoformat(),
        "num_students": len(profiles),
        "num_similar_pairs": len(similar_pairs),
        "total_cost": total_input_tokens * COST_PER_INPUT + total_output_tokens * COST_PER_OUTPUT,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }
    with open("/research/workspace/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to /research/workspace/")
    print("=" * 60)


if __name__ == "__main__":
    main()
