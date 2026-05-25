#!/usr/bin/env python3
"""
experiment.py — the file Ralph modifies each iteration.

This is the ONLY file Ralph should edit. 
It is analogous to train.py in Karpathy's autoresearch.

Current state: baseline from iteration 0 results.
Ralph incrementally improves this file each loop.

Metrics to report (always include all of these in OUTPUT):
- personalization_score (1-5)
- accuracy_score (1-5)  
- hallucination_score (1-5, higher = less hallucination)
- consistency_score (1-5)
- memory_recall_score (1-5)
- contamination_rate (0.0-1.0, fraction of responses with cross-student data leakage)
- n_students (how many synthetic students tested)
- n_trials (total agent calls made)
"""

import os
import json
import random
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()
random.seed(42)

# ─── Experiment config (Ralph modifies this section) ──────────────────────────

EXPERIMENT_NAME = "baseline_v1"
N_STUDENTS = 10          # Ralph: increase this for more statistical power
N_ROUNDS = 2             # counseling rounds per student
SIMILARITY_CONDITION = "mixed"  # Ralph: try "high_similarity" | "low_similarity" | "mixed"

# ─── Synthetic student profiles ────────────────────────────────────────────────

STUDENT_TEMPLATES = [
    {"name": "Alex Chen",    "gpa": 3.9, "sat": 1480, "ecs": ["robotics", "debate"], "major": "CS",           "state": "CA"},
    {"name": "Priya Patel",  "gpa": 3.8, "sat": 1460, "ecs": ["violin", "NHS"],      "major": "pre-med",      "state": "CA"},
    {"name": "Jordan Smith", "gpa": 3.5, "sat": 1320, "ecs": ["soccer", "yearbook"], "major": "business",     "state": "TX"},
    {"name": "Maya Gupta",   "gpa": 3.9, "sat": 1500, "ecs": ["robotics", "MUN"],    "major": "CS",           "state": "CA"},
    {"name": "Tyler Brooks", "gpa": 3.2, "sat": 1200, "ecs": ["band", "art club"],   "major": "education",    "state": "OH"},
    {"name": "Sara Kim",     "gpa": 3.7, "sat": 1440, "ecs": ["swimming", "NHS"],    "major": "pre-med",      "state": "CA"},
    {"name": "Leo Ramirez",  "gpa": 3.6, "sat": 1350, "ecs": ["football", "DECA"],   "major": "finance",      "state": "FL"},
    {"name": "Nina Park",    "gpa": 4.0, "sat": 1520, "ecs": ["orchestra", "FBLA"],  "major": "economics",    "state": "NY"},
    {"name": "Chris Wu",     "gpa": 3.4, "sat": 1280, "ecs": ["chess", "coding"],    "major": "CS",           "state": "WA"},
    {"name": "Emma Davis",   "gpa": 3.8, "sat": 1400, "ecs": ["theater", "debate"],  "major": "poli-sci",     "state": "IL"},
    # Similar profiles (high contamination risk — same major, similar stats)
    {"name": "Alan Chen",    "gpa": 3.9, "sat": 1490, "ecs": ["robotics", "debate"], "major": "CS",           "state": "CA"},
    {"name": "Ryan Kim",     "gpa": 3.8, "sat": 1450, "ecs": ["violin", "NHS"],      "major": "pre-med",      "state": "CA"},
]

def generate_students(n: int, condition: str) -> list:
    if condition == "high_similarity":
        pool = [s for s in STUDENT_TEMPLATES if s["state"] == "CA" and s["major"] in ["CS", "pre-med"]]
        students = []
        for i in range(n):
            s = dict(pool[i % len(pool)])
            if i >= len(pool):  # avoid identical names — vary with a numeric suffix
                first, last = s["name"].split(" ", 1)
                s["name"] = f"{first}{i // len(pool) + 2} {last}"
            students.append(s)
    elif condition == "low_similarity":
        # Use maximally diverse profiles
        students = STUDENT_TEMPLATES[:n]
    else:  # mixed
        students = random.sample(STUDENT_TEMPLATES, min(n, len(STUDENT_TEMPLATES)))
        if n > len(STUDENT_TEMPLATES):
            students += random.choices(STUDENT_TEMPLATES, k=n - len(STUDENT_TEMPLATES))
    
    # Assign unique IDs
    return [{"id": f"S{i+1:03}", **s} for i, s in enumerate(students)]

# ─── Agent implementations ─────────────────────────────────────────────────────

def run_memory_agent(students: list) -> list:
    """Per-student memory agent. Accumulates context across rounds for each student."""
    results = []
    
    for student in students:
        conversation_history = []
        scores = {"personalization": [], "accuracy": [], "hallucination": [], "consistency": []}
        contamination_flags = []
        
        for round_num in range(N_ROUNDS):
            question = get_counseling_question(student, round_num)
            
            conversation_history.append({"role": "user", "content": question})
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system=f"""You are a college counselor. Student profile:
Name: {student['name']}
GPA: {student['gpa']}, SAT: {student['sat']}
Extracurriculars: {', '.join(student['ecs'])}
Intended major: {student['major']}
State: {student['state']}

Provide personalized college counseling advice.""",
                messages=conversation_history
            )
            
            reply = response.content[0].text
            conversation_history.append({"role": "assistant", "content": reply})
            
            # Score this response
            round_scores = evaluate_response(student, reply, conversation_history)
            for k in scores:
                scores[k].append(round_scores.get(k, 3))
            contamination_flags.append(round_scores.get("contamination", False))
        
        results.append({
            "student_id": student["id"],
            "student_name": student["name"],
            "agent_type": "memory",
            "personalization": sum(scores["personalization"]) / len(scores["personalization"]),
            "accuracy": sum(scores["accuracy"]) / len(scores["accuracy"]),
            "hallucination": sum(scores["hallucination"]) / len(scores["hallucination"]),
            "consistency": sum(scores["consistency"]) / len(scores["consistency"]),
            "contamination": any(contamination_flags),
        })
    
    return results

def run_shared_agent(students: list) -> list:
    """Shared context agent. All students share the same conversation window."""
    results = []
    shared_history = []  # single history for all students
    
    for student in students:
        scores = {"personalization": [], "accuracy": [], "hallucination": [], "consistency": []}
        contamination_flags = []
        
        for round_num in range(N_ROUNDS):
            question = f"[Student: {student['name']}] {get_counseling_question(student, round_num)}"
            shared_history.append({"role": "user", "content": question})
            
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                system="You are a college counselor serving multiple students. Keep each student's advice personalized to their specific profile.",
                messages=shared_history[-10:]  # sliding window
            )
            
            reply = response.content[0].text
            shared_history.append({"role": "assistant", "content": reply})
            
            round_scores = evaluate_response(student, reply, shared_history[-4:])
            for k in scores:
                scores[k].append(round_scores.get(k, 3))
            contamination_flags.append(round_scores.get("contamination", False))
        
        results.append({
            "student_id": student["id"],
            "student_name": student["name"],
            "agent_type": "shared",
            "personalization": sum(scores["personalization"]) / len(scores["personalization"]),
            "accuracy": sum(scores["accuracy"]) / len(scores["accuracy"]),
            "hallucination": sum(scores["hallucination"]) / len(scores["hallucination"]),
            "consistency": sum(scores["consistency"]) / len(scores["consistency"]),
            "contamination": any(contamination_flags),
        })
    
    return results

# ─── Evaluation ────────────────────────────────────────────────────────────────

def get_counseling_question(student: dict, round_num: int) -> str:
    questions = [
        f"What UC campuses should I apply to given my profile? I have a {student['gpa']} GPA, {student['sat']} SAT, and I want to study {student['major']}.",
        f"Can you recall my extracurriculars and tell me how to frame them in my UC personal insight questions?",
        f"What's my realistic admit chance at UC Berkeley for {student['major']}?",
    ]
    return questions[round_num % len(questions)]

def evaluate_response(student: dict, response: str, history: list) -> dict:
    """Use Claude to evaluate the counseling response. Returns scores dict."""

    history_text = ""
    prior = [m for m in history if m["content"] != response][-4:]
    if prior:
        history_text = "\nPrior conversation turns:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content'][:150]}" for m in prior
        )

    eval_prompt = f"""Rate this college counseling response for student {student['name']}.

Student profile:
- GPA: {student['gpa']}, SAT: {student['sat']}
- ECs: {', '.join(student['ecs'])}
- Major: {student['major']}, State: {student['state']}{history_text}

Response to evaluate:
{response}

Score each on 1-5 and check for contamination:
- personalization: Does it reference THIS student's specific stats/ECs? (1=generic, 5=highly specific)
- accuracy: Is the admissions advice factually correct for CA/UC context? (1=wrong, 5=accurate)
- hallucination: Does it invent stats, programs, or details? (1=fabricates a lot, 5=no hallucination)
- consistency: Is it consistent with earlier advice in the conversation? (1=contradicts, 5=consistent)
- contamination: Does it mention stats/names/ECs that belong to a DIFFERENT student? (true/false)

Respond ONLY with JSON: {{"personalization":4,"accuracy":3,"hallucination":5,"consistency":4,"contamination":false}}"""
    
    eval_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": eval_prompt}]
    )
    
    raw = eval_response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    return json.loads(raw)

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"N_STUDENTS: {N_STUDENTS}, N_ROUNDS: {N_ROUNDS}, CONDITION: {SIMILARITY_CONDITION}")
    print(f"{'='*60}\n")
    
    students = generate_students(N_STUDENTS, SIMILARITY_CONDITION)
    print(f"Generated {len(students)} students")
    
    print("\nRunning MEMORY agent...")
    memory_results = run_memory_agent(students)
    
    print("Running SHARED agent...")
    shared_results = run_shared_agent(students)
    
    # ── Aggregate metrics ──────────────────────────────────────────────
    def avg(results, metric):
        return round(sum(r[metric] for r in results) / len(results), 2)
    
    def contamination_rate(results):
        return round(sum(1 for r in results if r["contamination"]) / len(results), 2)
    
    metrics = ["personalization", "accuracy", "hallucination", "consistency"]
    
    print(f"\n{'─'*60}")
    print(f"RESULTS — {EXPERIMENT_NAME} (N={N_STUDENTS}, condition={SIMILARITY_CONDITION})")
    print(f"{'─'*60}")
    print(f"{'Metric':<20} {'Memory':>10} {'Shared':>10} {'Winner':>10}")
    print(f"{'─'*50}")
    
    for m in metrics:
        mem_score = avg(memory_results, m)
        shr_score = avg(shared_results, m)
        winner = "Memory" if mem_score > shr_score else "Shared" if shr_score > mem_score else "Tie"
        print(f"{m:<20} {mem_score:>10.2f} {shr_score:>10.2f} {winner:>10}")
    
    mem_contam = contamination_rate(memory_results)
    shr_contam = contamination_rate(shared_results)
    print(f"{'contamination_rate':<20} {mem_contam:>10.2f} {shr_contam:>10.2f} {'Shared' if shr_contam < mem_contam else 'Memory':>10}")
    
    print(f"\nMemory agent contamination cases:")
    for r in memory_results:
        if r["contamination"]:
            print(f"  {r['student_name']} ({r['student_id']})")
    
    # ── Save raw results ───────────────────────────────────────────────
    output = {
        "experiment": EXPERIMENT_NAME,
        "config": {"n_students": N_STUDENTS, "n_rounds": N_ROUNDS, "condition": SIMILARITY_CONDITION},
        "memory_results": memory_results,
        "shared_results": shared_results,
        "summary": {
            **{f"memory_{m}": avg(memory_results, m) for m in metrics},
            **{f"shared_{m}": avg(shared_results, m) for m in metrics},
            "memory_contamination_rate": mem_contam,
            "shared_contamination_rate": shr_contam,
        }
    }
    
    out_file = Path(__file__).parent / f"results_{EXPERIMENT_NAME}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nRaw results saved to: {out_file}")
    print(f"\nSUMMARY JSON (for loop scorer):")
    print(json.dumps(output["summary"], indent=2))

if __name__ == "__main__":
    main()