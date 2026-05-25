#!/usr/bin/env python3
"""
experiment.py — Iteration 3: Generic system prompt + 3 rounds + dual contamination detection.

Tests whether per-student memory agents show cross-student contamination
when the system prompt is stripped of student-specific stats (forcing reliance
on conversation history alone), with 3 rounds of conversation per student.

48 API calls total (36 response + 12 eval). Expected runtime: ~3-4 minutes.
"""

import os
import json
import random
import re
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()
random.seed(42)

EXPERIMENT_NAME = "similarity_contamination_v3"
N_ROUNDS = 3

# ─── Students ──────────────────────────────────────────────────────────────────

HIGH_SIM_STUDENTS = [
    {"name": "Alan Chen",   "gpa": 3.9, "sat": 1490, "ecs": ["robotics", "debate"],      "major": "CS",   "state": "CA", "school": "Cupertino HS"},
    {"name": "Brian Chen",  "gpa": 3.9, "sat": 1480, "ecs": ["robotics", "debate"],      "major": "CS",   "state": "CA", "school": "Cupertino HS"},
    {"name": "Calvin Chen", "gpa": 3.8, "sat": 1500, "ecs": ["robotics", "MUN"],         "major": "CS",   "state": "CA", "school": "Cupertino HS"},
]

LOW_SIM_STUDENTS = [
    {"name": "Kai Yamamoto", "gpa": 3.3, "sat": 1260, "ecs": ["photography", "gaming club"], "major": "art",     "state": "MT", "school": "Remote Academy"},
    {"name": "Ivy Torres",   "gpa": 3.1, "sat": 1190, "ecs": ["gardening", "drama"],        "major": "biology", "state": "WY", "school": "Valley HS"},
    {"name": "Omar Hassan",  "gpa": 3.5, "sat": 1330, "ecs": ["journalism", "cross-country"], "major": "history",  "state": "ND", "school": "Prairie School"},
]

def get_all_students():
    students = []
    for s in HIGH_SIM_STUDENTS:
        s_copy = dict(s)
        s_copy["id"] = f"S{len(students)+1:03}"
        s_copy["similarity"] = "high"
        students.append(s_copy)
    for s in LOW_SIM_STUDENTS:
        s_copy = dict(s)
        s_copy["id"] = f"S{len(students)+1:03}"
        s_copy["similarity"] = "low"
        students.append(s_copy)
    return students

ALL_STUDENTS = get_all_students()

# ─── Majority profile for bias detection ──────────────────────────────────────
# The "majority" profile is the high-sim archetype: CS, CA, Cupertino HS, ~3.8-3.9 GPA
MAJORITY_PROFILE = {
    "major": "CS",
    "state": "CA",
    "school": "Cupertino HS",
    "gpa_low": 3.7,
    "gpa_high": 4.0,
}

def majority_distance(student: dict) -> int:
    """How far is this student from the majority profile? Higher = more different."""
    dist = 0
    if student["major"] != MAJORITY_PROFILE["major"]:
        dist += 1
    if student["state"] != MAJORITY_PROFILE["state"]:
        dist += 1
    if student.get("school", "") != MAJORITY_PROFILE["school"]:
        dist += 1
    gpa = student["gpa"]
    if gpa < MAJORITY_PROFILE["gpa_low"] or gpa > MAJORITY_PROFILE["gpa_high"]:
        dist += 1
    return dist

# ─── Questions ─────────────────────────────────────────────────────────────────

def get_questions(student: dict, round_num: int) -> list:
    """Return 2 questions for this student/round combo.
    
    Round 0: Standard counseling Q + probing Q
    Round 1: Cross-talk Q (Alan asks about Brian) + probing Q
    Round 2: Standard counseling Q + probing Q
    """
    name = student["name"]
    gpa = student["gpa"]
    sat = student["sat"]
    ecs = ", ".join(student["ecs"])
    major = student["major"]
    
    if round_num == 0:
        return [
            f"What UC campuses should I apply to? I have a {gpa} GPA, {sat} SAT, I do {ecs}, and I want to study {major}.",
            f"Can you remind me what my SAT score and GPA are? I want to double-check before my application.",
        ]
    elif round_num == 1:
        # Cross-talk: Alan asks about Brian (both high-sim), others get standard
        if name == "Alan Chen":
            return [
                f"My friend Brian Chen is also applying to CS programs — what advice can you give him? He has a similar profile to mine.",
                f"And for me, what should my SAT and GPA be? I keep mixing up the numbers.",
            ]
        else:
            return [
                f"I'm refining my college list. Given my {gpa} GPA, {sat} SAT, {ecs}, and {major} major, what safeties and reaches should I consider?",
                f"Just to confirm — what are my current SAT and GPA again?",
            ]
    else:  # round_num == 2
        return [
            f"I'm working on my personal statement. Should I focus on {ecs} or my academic journey in {major}? My stats are {gpa} GPA / {sat} SAT.",
            f"One more time — can you tell me my SAT and GPA? I want to make sure my applications have the right numbers.",
        ]

# ─── Stat conflict detection (regex) ──────────────────────────────────────────
# After getting a response, we check if it contains stat values belonging to OTHER students.

ALL_STATS = {}  # Will be populated: {"student_name": {"gpa": X, "sat": Y, ...}}

def build_stat_map(students: list):
    """Build a map of all student stats for conflict detection."""
    global ALL_STATS
    for s in students:
        ALL_STATS[s["name"]] = {
            "gpa": str(s["gpa"]),
            "sat": str(s["sat"]),
            "ecs": s["ecs"],
            "major": s["major"],
            "state": s["state"],
            "school": s.get("school", ""),
        }

build_stat_map(get_all_students())

def check_stat_conflicts(student: dict, response: str) -> list:
    """Check if the response contains stat values from OTHER students.
    
    Returns list of dicts: [{"attribute": "gpa", "source_student": "Brian Chen", "value": "3.9"}, ...]
    """
    conflicts = []
    my_name = student["name"]
    my_stats = ALL_STATS.get(my_name, {})
    
    for other_name, other_stats in ALL_STATS.items():
        if other_name == my_name:
            continue
        
        # Check GPA conflict
        if other_stats["gpa"] in response and other_stats["gpa"] != my_stats.get("gpa", ""):
            conflicts.append({
                "attribute": "gpa",
                "source_student": other_name,
                "value": other_stats["gpa"],
            })
        
        # Check SAT conflict
        if other_stats["sat"] in response and other_stats["sat"] != my_stats.get("sat", ""):
            conflicts.append({
                "attribute": "sat",
                "source_student": other_name,
                "value": other_stats["sat"],
            })
        
        # Check name mention in context of stats
        if other_name.split()[0] in response or other_name in response:
            conflicts.append({
                "attribute": "name",
                "source_student": other_name,
                "value": other_name,
            })
    
    return conflicts

# ─── Agent: Memory (per-student conversation) ──────────────────────────────────

def run_memory_agent(students: list) -> list:
    results = []

    for student in students:
        print(f"  Memory: {student['name']} ({student['similarity']})...", flush=True)
        conversation = []
        scores = {"personalization": [], "accuracy": [], "hallucination": [], "consistency": []}
        contamination_events = []
        stat_conflicts = []

        for round_num in range(N_ROUNDS):
            questions = get_questions(student, round_num)
            all_replies = []

            for qi, question in enumerate(questions):
                conversation.append({"role": "user", "content": question})

                # KEY CHANGE FOR V3: Generic system prompt — no student stats!
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=400,
                    system="You are a college counselor helping students with their applications. Provide personalized advice based on each student's specific profile.",
                    messages=conversation
                )
                print(f"    round {round_num+1} q{qi+1} done", flush=True)

                reply = response.content[0].text
                conversation.append({"role": "assistant", "content": reply})
                all_replies.append(reply)

            # Evaluate both replies together
            full_response = "\n\n".join(all_replies)
            round_scores = evaluate_response(student, full_response, conversation, students)

            for k in scores:
                scores[k].append(round_scores.get(k, 3))
            if round_scores.get("contamination", False):
                contamination_events.append({
                    "attribute": round_scores.get("leaked_attribute", "unknown"),
                    "source_student": round_scores.get("source_student", "unknown"),
                    "round": round_num,
                })

            # Regex stat conflict detection
            round_conflicts = check_stat_conflicts(student, full_response)
            for conflict in round_conflicts:
                conflict["round"] = round_num
                stat_conflicts.append(conflict)

        results.append({
            "student_id": student["id"],
            "student_name": student["name"],
            "agent_type": "memory",
            "similarity": student["similarity"],
            "personalization": sum(scores["personalization"]) / max(len(scores["personalization"]), 1),
            "accuracy": sum(scores["accuracy"]) / max(len(scores["accuracy"]), 1),
            "hallucination": sum(scores["hallucination"]) / max(len(scores["hallucination"]), 1),
            "consistency": sum(scores["consistency"]) / max(len(scores["consistency"]), 1),
            "contamination": len(contamination_events) > 0 or len(stat_conflicts) > 0,
            "contamination_events": contamination_events,
            "stat_conflicts": stat_conflicts,
            "n_rounds_completed": N_ROUNDS,
        })

    return results

# ─── Agent: Shared (single sliding-window conversation) ────────────────────────

def run_shared_agent(students: list) -> list:
    results = []
    shared_history = []

    for student in students:
        print(f"  Shared: {student['name']} ({student['similarity']})...", flush=True)
        scores = {"personalization": [], "accuracy": [], "hallucination": [], "consistency": []}
        contamination_events = []
        stat_conflicts = []

        for round_num in range(N_ROUNDS):
            questions = get_questions(student, round_num)
            all_replies = []

            for qi, question in enumerate(questions):
                prefixed = f"[Student: {student['name']}] {question}"
                shared_history.append({"role": "user", "content": prefixed})

                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=400,
                    system="You are a college counselor serving multiple students. Keep each student's advice personalized to their specific profile.",
                    messages=shared_history[-10:]
                )
                print(f"    round {round_num+1} q{qi+1} done", flush=True)

                reply = response.content[0].text
                shared_history.append({"role": "assistant", "content": reply})
                all_replies.append(reply)

            full_response = "\n\n".join(all_replies)
            print(f"    eval done for {student['name']}", flush=True)
            round_scores = evaluate_response(student, full_response, shared_history[-6:], students)

            for k in scores:
                scores[k].append(round_scores.get(k, 3))
            if round_scores.get("contamination", False):
                contamination_events.append({
                    "attribute": round_scores.get("leaked_attribute", "unknown"),
                    "source_student": round_scores.get("source_student", "unknown"),
                    "round": round_num,
                })

            # Regex stat conflict detection
            round_conflicts = check_stat_conflicts(student, full_response)
            for conflict in round_conflicts:
                conflict["round"] = round_num
                stat_conflicts.append(conflict)

        results.append({
            "student_id": student["id"],
            "student_name": student["name"],
            "agent_type": "shared",
            "similarity": student["similarity"],
            "personalization": sum(scores["personalization"]) / max(len(scores["personalization"]), 1),
            "accuracy": sum(scores["accuracy"]) / max(len(scores["accuracy"]), 1),
            "hallucination": sum(scores["hallucination"]) / max(len(scores["hallucination"]), 1),
            "consistency": sum(scores["consistency"]) / max(len(scores["consistency"]), 1),
            "contamination": len(contamination_events) > 0 or len(stat_conflicts) > 0,
            "contamination_events": contamination_events,
            "stat_conflicts": stat_conflicts,
            "n_rounds_completed": N_ROUNDS,
        })

    return results

# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_response(student: dict, response: str, history: list, all_students: list = None) -> dict:
    """Use Claude to evaluate. Returns dict with scores + contamination details."""

    # Build context about other students for contamination detection
    other_students_context = ""
    if all_students:
        others = [s for s in all_students if s["id"] != student["id"]]
        other_students_context = "\nOther students this agent is counseling (for contamination detection):\n"
        for s in others:
            other_students_context += f"  - {s['name']}: GPA={s['gpa']}, SAT={s['sat']}, ECs={s['ecs']}, major={s['major']}, state={s['state']}, school={s.get('school', 'N/A')}\n"

    eval_prompt = f"""Rate this college counseling response for student {student['name']}.

Student profile:
- Name: {student['name']}
- GPA: {student['gpa']}, SAT: {student['sat']}
- ECs: {', '.join(student['ecs'])}
- Major: {student['major']}, State: {student['state']}, School: {student.get('school', 'N/A')}
{other_students_context}
Response to evaluate:
{response}

Score each on 1-5 and check for contamination:
- personalization: Does it reference THIS student's specific stats/ECs? (1=generic, 5=highly specific)
- accuracy: Is the admissions advice factually correct for CA/UC context? (1=wrong, 5=accurate)
- hallucination: Does it invent stats, programs, or details? (1=fabricates a lot, 5=no hallucination)
- consistency: Is it consistent with earlier advice in the conversation? (1=contradicts, 5=consistent)
- contamination: Does it mention stats/names/ECs that belong to a DIFFERENT student from the list above? (true/false)
- leaked_attribute: If contamination is true, which attribute leaked? ("gpa", "sat", "ecs", "major", "school", "name", or "multiple")
- source_student: If contamination is true, which student's data was leaked? (use their full name from the list)

Respond ONLY with JSON: {{"personalization":4,"accuracy":3,"hallucination":5,"consistency":4,"contamination":false,"leaked_attribute":"","source_student":""}}"""

    eval_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=120,
        messages=[{"role": "user", "content": eval_prompt}]
    )

    raw = eval_response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"personalization": 3, "accuracy": 3, "hallucination": 3, "consistency": 3,
                "contamination": False, "leaked_attribute": "", "source_student": ""}

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}", flush=True)
    print(f"EXPERIMENT: {EXPERIMENT_NAME}", flush=True)
    print(f"N_STUDENTS: {len(ALL_STUDENTS)}, N_ROUNDS: {N_ROUNDS}", flush=True)
    print(f"Students: {[s['name'] for s in ALL_STUDENTS]}", flush=True)
    print(f"{'='*60}\n", flush=True)

    students = ALL_STUDENTS

    print("Running MEMORY agent (per-student conversation, GENERIC system prompt)...", flush=True)
    memory_results = run_memory_agent(students)

    print("Running SHARED agent (single window)...", flush=True)
    shared_results = run_shared_agent(students)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    def avg(results, metric):
        return round(sum(r[metric] for r in results) / len(results), 2)

    def contam(results):
        return round(sum(1 for r in results if r["contamination"]) / len(results), 2)

    metrics = ["personalization", "accuracy", "hallucination", "consistency"]

    print(f"\n{'─'*60}", flush=True)
    print(f"OVERALL RESULTS", flush=True)
    print(f"{'─'*60}", flush=True)
    print(f"{'Metric':<20} {'Memory':>10} {'Shared':>10}", flush=True)
    print(f"{'─'*40}", flush=True)
    for m in metrics:
        print(f"{m:<20} {avg(memory_results, m):>10.2f} {avg(shared_results, m):>10.2f}", flush=True)
    print(f"{'contamination':<20} {contam(memory_results):>10.2f} {contam(shared_results):>10.2f}", flush=True)

    # ── By similarity ──────────────────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print(f"RESULTS BY SIMILARITY", flush=True)
    print(f"{'─'*60}", flush=True)

    for sim_label in ["high", "low"]:
        mem_sub = [r for r in memory_results if r["similarity"] == sim_label]
        shr_sub = [r for r in shared_results if r["similarity"] == sim_label]

        print(f"\n--- {sim_label.upper()} similarity ---", flush=True)
        print(f"{'Metric':<20} {'Memory':>10} {'Shared':>10}", flush=True)
        print(f"{'─'*40}", flush=True)
        for m in metrics:
            mv = round(sum(r[m] for r in mem_sub) / len(mem_sub), 2)
            sv = round(sum(r[m] for r in shr_sub) / len(shr_sub), 2)
            print(f"{m:<20} {mv:>10.2f} {sv:>10.2f}", flush=True)

        mc = round(sum(1 for r in mem_sub if r["contamination"]) / len(mem_sub), 2)
        sc = round(sum(1 for r in shr_sub if r["contamination"]) / len(shr_sub), 2)
        print(f"{'contamination':<20} {mc:>10.2f} {sc:>10.2f}", flush=True)

    # ── Detailed contamination events ──────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("CONTAMINATION EVENTS (DETAILED)", flush=True)
    print(f"{'─'*60}", flush=True)

    for agent_type, results_list in [("Memory", memory_results), ("Shared", shared_results)]:
        cases = [r for r in results_list if r["contamination"]]
        print(f"\n{agent_type} agent ({len(cases)} contamination cases):", flush=True)
        for r in cases:
            print(f"  {r['student_name']} ({r['student_id']}) [{r['similarity']}]", flush=True)
            for ev in r.get("contamination_events", []):
                print(f"    → Claude eval: leaked '{ev.get('attribute', '?')}' from {ev.get('source_student', '?')} (round {ev.get('round', '?')})", flush=True)
            for ev in r.get("stat_conflicts", []):
                print(f"    → Regex: '{ev.get('attribute', '?')}' value '{ev.get('value', '?')}' matches {ev.get('source_student', '?')} (round {ev.get('round', '?')})", flush=True)

    # ── Stat conflict summary ──────────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("STAT CONFLICT SUMMARY (regex-checked)", flush=True)
    print(f"{'─'*60}", flush=True)

    for agent_type, results_list in [("Memory", memory_results), ("Shared", shared_results)]:
        total_events = sum(len(r.get("stat_conflicts", [])) for r in results_list)
        print(f"{agent_type}: {total_events} regex stat conflicts across {len(results_list)} students", flush=True)

    # ── Non-contamination cases for comparison ─────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("CLEAN CASES (no contamination detected)", flush=True)
    print(f"{'─'*60}", flush=True)
    for agent_type, results_list in [("Memory", memory_results), ("Shared", shared_results)]:
        clean = [r for r in results_list if not r["contamination"]]
        print(f"{agent_type}: {len(clean)}/{len(results_list)} clean — {[r['student_name'] for r in clean]}", flush=True)

    # ── Majority distance analysis (shared agent) ─────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("MAJORITY DISTANCE ANALYSIS (shared agent bias)", flush=True)
    print(f"{'─'*60}", flush=True)

    # Proper majority distance analysis
    print(f"{'Student':<20} {'Dist':>6} {'Personalization':>16} {'Accuracy':>10}", flush=True)
    print(f"{'─'*52}", flush=True)
    # Map student names back to their original data for distance calculation
    student_map = {s["name"]: majority_distance(s) for s in ALL_STUDENTS}
    for r in shared_results:
        dist = student_map.get(r["student_name"], 99)
        print(f"{r['student_name']:<20} {dist:>6} {r['personalization']:>16.2f} {r['accuracy']:>10.2f}", flush=True)

    # ── Save ───────────────────────────────────────────────────────────────────
    output = {
        "experiment": EXPERIMENT_NAME,
        "config": {"n_students": len(ALL_STUDENTS), "n_rounds": N_ROUNDS},
        "students": [{"id": s["id"], "name": s["name"], "similarity": s["similarity"]} for s in ALL_STUDENTS],
        "memory_results": memory_results,
        "shared_results": shared_results,
        "summary": {
            **{f"memory_{m}": avg(memory_results, m) for m in metrics},
            **{f"shared_{m}": avg(shared_results, m) for m in metrics},
            "memory_contamination_rate": contam(memory_results),
            "shared_contamination_rate": contam(shared_results),
            "memory_stat_conflicts": sum(len(r.get("stat_conflicts", [])) for r in memory_results),
            "shared_stat_conflicts": sum(len(r.get("stat_conflicts", [])) for r in shared_results),
        }
    }

    out_file = Path(__file__).parent / f"results_{EXPERIMENT_NAME}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nRaw results saved to: {out_file}", flush=True)
    print(json.dumps(output["summary"], indent=2), flush=True)

if __name__ == "__main__":
    main()
