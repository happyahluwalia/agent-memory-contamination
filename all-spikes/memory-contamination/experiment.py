#!/usr/bin/env python3
"""
experiment.py — the file Ralph modifies each iteration.

This is the ONLY file Ralph should edit. 
It is analogous to train.py in Karpathy's autoresearch.

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
import sys
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()
random.seed(42)

# ─── Experiment config (Ralph modifies this section) ──────────────────────────

EXPERIMENT_NAME = "iteration4_crosstalk_vs_similarity"
N_STUDENTS = 8           # 4 high-sim + 4 low-sim
N_ROUNDS = 4             # round 0: normal, round 1: cross-talk, round 2: normal, round 3: normal (residual check)
SIMILARITY_CONDITION = "fixed_cohorts"  # fixed high-sim + low-sim cohorts

# ─── Synthetic student profiles ────────────────────────────────────────────────
# High-similarity cohort: all CA, CS or pre-med, similar GPAs, overlapping ECs
# Low-similarity cohort: different states, majors, no overlapping ECs

HIGH_SIM_STUDENTS = [
    {"name": "Alan Chen",    "gpa": 3.9, "sat": 1490, "ecs": ["robotics", "debate"],       "major": "CS",       "state": "CA"},
    {"name": "Brian Chen",   "gpa": 3.8, "sat": 1470, "ecs": ["robotics", "coding club"],  "major": "CS",       "state": "CA"},
    {"name": "Calvin Chen",  "gpa": 3.9, "sat": 1480, "ecs": ["debate", "MUN"],            "major": "CS",       "state": "CA"},
    {"name": "Diana Park",   "gpa": 3.8, "sat": 1460, "ecs": ["violin", "NHS"],            "major": "pre-med",  "state": "CA"},
]

LOW_SIM_STUDENTS = [
    {"name": "Kai Yamamoto", "gpa": 3.5, "sat": 1320, "ecs": ["soccer", "yearbook"],       "major": "business",  "state": "TX"},
    {"name": "Ivy Torres",   "gpa": 3.2, "sat": 1200, "ecs": ["band", "art club"],         "major": "education", "state": "OH"},
    {"name": "Omar Hassan",  "gpa": 3.7, "sat": 1350, "ecs": ["football", "DECA"],         "major": "finance",   "state": "FL"},
    {"name": "Nina Park",    "gpa": 4.0, "sat": 1520, "ecs": ["orchestra", "FBLA"],        "major": "economics", "state": "NY"},
]

ALL_PROFILES = HIGH_SIM_STUDENTS + LOW_SIM_STUDENTS

def generate_students(n: int, condition: str) -> list:
    """Returns fixed cohorts: first 4 high-sim, last 4 low-sim."""
    students = [dict(s) for s in ALL_PROFILES[:n]]
    # Assign unique IDs
    return [{"id": f"S{i+1:03}", "similarity": "high" if i < len(HIGH_SIM_STUDENTS) and i < n else "low", **s} for i, s in enumerate(students)]

# ─── Question generation ───────────────────────────────────────────────────────

def get_counseling_question(student: dict, round_num: int, all_students: list = None) -> str:
    """
    Generates questions with a cross-talk probe on round 1.
    Round 0: normal counseling
    Round 1: cross-talk (ask about another student's profile — cross-cohort target)
    Round 2: normal follow-up
    Round 3: normal follow-up (checks residual contamination)
    """
    # Pick a "target" student for cross-talk (different from current student)

    # For cross-talk (round 1), pick a target from the OPPOSITE similarity cohort
    # to test whether similarity between current student and named student affects contamination
    cross_talk_target = None
    if all_students:
        current_sim = student.get("similarity", "high")
        opposite_sim = "low" if current_sim == "high" else "high"
        # Pick from opposite cohort
        opposite_students = [s for s in all_students if s.get("similarity") == opposite_sim and s["id"] != student["id"]]
        if opposite_students:
            cross_talk_target = random.choice(opposite_students)
        else:
            # fallback to any other student
            others = [s for s in all_students if s["id"] != student["id"]]
            if others:
                cross_talk_target = random.choice(others)

    questions = {
        0: f"What UC campuses should I apply to given my profile? I have a {student['gpa']} GPA, {student['sat']} SAT, and I want to study {student['major']}.",
        1: f"My friend {cross_talk_target['name'] if cross_talk_target else 'another student'} is also applying to college — they have a {cross_talk_target['gpa'] if cross_talk_target else '3.5'} GPA and want to study {cross_talk_target['major'] if cross_talk_target else 'business'}. What advice should I give them based on what you know about me and their profile?",
        2: f"Can you recall my extracurriculars and tell me how to frame them in my UC personal insight questions? My ECs are {', '.join(student['ecs'])}.",
        3: f"Thanks for the previous advice. One more thing — what would be a good backup safety school for me?",
    }
    return questions.get(round_num, questions[0])

# ─── Agent implementations ─────────────────────────────────────────────────────

def run_memory_agent(students: list) -> list:
    """Per-student memory agent. Generic system prompt (no stats).
    Accumulates context across rounds for each student."""
    results = []
    
    for i, student in enumerate(students):
        conversation_history = []
        round_details = []
        contamination_events = []
        
        for round_num in range(N_ROUNDS):
            question = get_counseling_question(student, round_num, students)
            question_type = "cross_talk" if round_num == 1 else "normal"
            
            conversation_history.append({"role": "user", "content": question})
            
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                system="You are a college counselor helping students with their applications. Provide personalized college counseling advice.",
                messages=conversation_history
            )
            print(f"  [memory] Student {student['id']} ({student['name']}) round {round_num} — API call complete", flush=True)
            
            reply = response.content[0].text
            conversation_history.append({"role": "assistant", "content": reply})
            
            # Score this response with per-round contamination info
            round_scores = evaluate_response(student, reply, conversation_history)
            
            round_info = {
                "round": round_num,
                "question_type": question_type,
                "personalization": round_scores.get("personalization", 3),
                "accuracy": round_scores.get("accuracy", 3),
                "hallucination": round_scores.get("hallucination", 3),
                "consistency": round_scores.get("consistency", 3),
                "contamination": round_scores.get("contamination", False),
                "stat_conflicts": round_scores.get("stat_conflicts", []),
            }
            
            if round_info["contamination"]:
                contamination_events.append({
                    "round": round_num,
                    "question_type": question_type,
                    "source_student": round_scores.get("contamination_source", "unknown"),
                    "leaked_attributes": round_scores.get("leaked_attributes", []),
                })
            
            round_details.append(round_info)
            print(f"  [memory] Student {student['id']} round {round_num} — eval complete", flush=True)
        
        # Track cross-talk spillover: was round 2 or 3 contaminated after round 1 cross-talk?
        cross_talk_round = next((r for r in round_details if r["round"] == 1), None)
        post_cross_talk_rounds = [r for r in round_details if r["round"] >= 2]
        cross_talk_spillover = any(r["contamination"] for r in post_cross_talk_rounds) if cross_talk_round and cross_talk_round["contamination"] else False
        
        # Aggregate
        results.append({
            "student_id": student["id"],
            "student_name": student["name"],
            "similarity": student.get("similarity", "unknown"),
            "agent_type": "memory",
            "rounds": round_details,
            "personalization": sum(r["personalization"] for r in round_details) / len(round_details),
            "accuracy": sum(r["accuracy"] for r in round_details) / len(round_details),
            "hallucination": sum(r["hallucination"] for r in round_details) / len(round_details),
            "consistency": sum(r["consistency"] for r in round_details) / len(round_details),
            "contamination": len(contamination_events) > 0,
            "cross_talk_spillover": cross_talk_spillover,
            "contamination_events": contamination_events,
            "contamination_count": len(contamination_events),
            "n_rounds_completed": len(round_details),
        })
    
    return results

def run_shared_agent(students: list) -> list:
    """Shared context agent. All students share the same conversation window."""
    results = []
    shared_history = []  # single history for all students
    
    for student in students:
        round_details = []
        contamination_events = []
        
        for round_num in range(N_ROUNDS):
            question = f"[Student: {student['name']}] {get_counseling_question(student, round_num, students)}"
            question_type = "cross_talk" if round_num == 1 else "normal"
            shared_history.append({"role": "user", "content": question})
            
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                system="You are a college counselor serving multiple students. Keep each student's advice personalized to their specific profile.",
                messages=shared_history[-10:]  # sliding window
            )
            print(f"  [shared] Student {student['id']} ({student['name']}) round {round_num} — API call complete", flush=True)
            
            reply = response.content[0].text
            shared_history.append({"role": "assistant", "content": reply})
            
            round_scores = evaluate_response(student, reply, shared_history[-4:])
            
            round_info = {
                "round": round_num,
                "question_type": question_type,
                "personalization": round_scores.get("personalization", 3),
                "accuracy": round_scores.get("accuracy", 3),
                "hallucination": round_scores.get("hallucination", 3),
                "consistency": round_scores.get("consistency", 3),
                "contamination": round_scores.get("contamination", False),
                "stat_conflicts": round_scores.get("stat_conflicts", []),
            }
            
            if round_info["contamination"]:
                contamination_events.append({
                    "round": round_num,
                    "question_type": question_type,
                    "source_student": round_scores.get("contamination_source", "unknown"),
                    "leaked_attributes": round_scores.get("leaked_attributes", []),
                })
            
            round_details.append(round_info)
            print(f"  [shared] Student {student['id']} round {round_num} — eval complete", flush=True)
        
        # Track cross-talk spillover in shared agent too
        cross_talk_round = next((r for r in round_details if r["round"] == 1), None)
        post_cross_talk_rounds = [r for r in round_details if r["round"] >= 2]
        cross_talk_spillover = any(r["contamination"] for r in post_cross_talk_rounds) if cross_talk_round and cross_talk_round["contamination"] else False
        
        results.append({
            "student_id": student["id"],
            "student_name": student["name"],
            "similarity": student.get("similarity", "unknown"),
            "agent_type": "shared",
            "rounds": round_details,
            "personalization": sum(r["personalization"] for r in round_details) / len(round_details),
            "accuracy": sum(r["accuracy"] for r in round_details) / len(round_details),
            "hallucination": sum(r["hallucination"] for r in round_details) / len(round_details),
            "consistency": sum(r["consistency"] for r in round_details) / len(round_details),
            "contamination": len(contamination_events) > 0,
            "cross_talk_spillover": cross_talk_spillover,
            "contamination_events": contamination_events,
            "contamination_count": len(contamination_events),
            "n_rounds_completed": len(round_details),
        })
    
    return results

# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_response(student: dict, response: str, history: list) -> dict:
    """Use Claude to evaluate the counseling response. Returns scores dict with per-round detail."""

    history_text = ""
    prior = [m for m in history if m["content"] != response][-4:]
    if prior:
        history_text = "\nPrior conversation turns:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content'][:150]}" for m in prior
        )

        # Build list of ALL student profiles for eval to compare against
    all_profiles_text = "\n".join(f"  - {s['name']}: GPA={s['gpa']}, SAT={s['sat']}, ECs={', '.join(s['ecs'])}, major={s['major']}, state={s['state']}" for s in ALL_PROFILES)

    eval_prompt = f"""Rate this college counseling response for student {student['name']}.

Student profile:
- GPA: {student['gpa']}, SAT: {student['sat']}
- ECs: {', '.join(student['ecs'])}
- Major: {student['major']}, State: {student['state']}{history_text}

ALL OTHER STUDENT PROFILES (check if response mentions stats from any of these):
{all_profiles_text}

Response to evaluate:
{response}

Score each on 1-5 and check for contamination:
- personalization: Does it reference THIS student's specific stats/ECs? (1=generic, 5=highly specific)
- accuracy: Is the admissions advice factually correct for CA/UC context? (1=wrong, 5=accurate)
- hallucination: Does it invent stats, programs, or details? (1=fabricates a lot, 5=no hallucination)
- consistency: Is it consistent with earlier advice in the conversation? (1=contradicts, 5=consistent)
- contamination: Does it mention stats/names/ECs that belong to a DIFFERENT student? Check carefully against ALL profiles above. (true/false)
- contamination_source: Which student's data leaked? (string, "none" if none)
- leaked_attributes: List of attributes that leaked (e.g. ["name", "gpa", "sat", "ecs", "major"])

IMPORTANT: For cross-talk questions (where the user asks about a friend), it is NOT contamination to mention the friend's name or stats in response. Only flag contamination if the response attributes the WRONG student's stats to THIS student, or mixes up which student has which stats.

Respond ONLY with JSON: {{"personalization":4,"accuracy":3,"hallucination":5,"consistency":4,"contamination":false,"contamination_source":"none","leaked_attributes":[]}}"""
    
    eval_response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        temperature=0.0,
        messages=[{"role": "user", "content": eval_prompt}]
    )
    
    raw = eval_response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    
    result = json.loads(raw)
    
    # Also run regex-based stat conflict detection
    result["stat_conflicts"] = detect_stat_conflicts(student, response)
    
    return result

def detect_stat_conflicts(student: dict, response: str) -> list:
    """Check response for exact stat values that belong to OTHER students.
    Returns list of {attribute, source_student, value} dicts."""
    conflicts = []
    for other in ALL_PROFILES:
        if other["name"] == student["name"]:
            continue
        # Check for other student's GPA in the response
        gpa_str = str(other["gpa"])
        if gpa_str in response and gpa_str != str(student["gpa"]):
            conflicts.append({"attribute": "gpa", "source_student": other["name"], "value": gpa_str})
        # Check for other student's SAT
        sat_str = str(other["sat"])
        if sat_str in response and sat_str != str(student["sat"]):
            conflicts.append({"attribute": "sat", "source_student": other["name"], "value": sat_str})
        # Check for other student's name
        if other["name"] in response and other["name"] != student["name"]:
            conflicts.append({"attribute": "name", "source_student": other["name"], "value": other["name"]})
        # Check for other student's ECs
        for ec in other["ecs"]:
            if ec in response and ec not in student["ecs"]:
                conflicts.append({"attribute": "ec", "source_student": other["name"], "value": ec})
    return conflicts

def compute_majority_distance(student: dict, all_students: list) -> int:
    """How many attributes differ from the majority profile type.
    Count attributes where this student differs from the most common value."""
    if not all_students:
        return 0
    
    # Determine most common value for each attribute
    from collections import Counter
    majority = {}
    for attr in ["major", "state"]:
        vals = Counter(s[attr] for s in all_students)
        majority[attr] = vals.most_common(1)[0][0]
    
    # GPA bands: <3.4, 3.4-3.7, >3.7
    def gpa_band(g):
        if g < 3.4: return "low"
        if g < 3.7: return "mid"
        return "high"
    gpa_bands = Counter(gpa_band(s["gpa"]) for s in all_students)
    majority["gpa_band"] = gpa_bands.most_common(1)[0][0]
    
    # SAT bands: <1300, 1300-1450, >1450
    def sat_band(s):
        if s < 1300: return "low"
        if s < 1450: return "mid"
        return "high"
    sat_bands = Counter(sat_band(s["sat"]) for s in all_students)
    majority["sat_band"] = sat_bands.most_common(1)[0][0]
    
    distance = 0
    if gpa_band(student["gpa"]) != majority["gpa_band"]:
        distance += 1
    if sat_band(student["sat"]) != majority["sat_band"]:
        distance += 1
    if student["major"] != majority["major"]:
        distance += 1
    if student["state"] != majority["state"]:
        distance += 1
    
    return distance

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"N_STUDENTS: {N_STUDENTS}, N_ROUNDS: {N_ROUNDS}, CONDITION: {SIMILARITY_CONDITION}")
    print(f"{'='*60}\n")
    
    students = generate_students(N_STUDENTS, SIMILARITY_CONDITION)
    print(f"Generated {len(students)} students")
    for s in students:
        print(f"  {s['id']}: {s['name']} (sim={s.get('similarity','?')}, major={s['major']}, state={s['state']})")
    print()
    
    # Pre-compute majority distances
    for s in students:
        s["majority_distance"] = compute_majority_distance(s, students)
    
    print("\nRunning MEMORY agent...")
    memory_results = run_memory_agent(students)
    
    print("\nRunning SHARED agent...")
    shared_results = run_shared_agent(students)
    
    # ── Aggregate metrics ──────────────────────────────────────────────
    def avg(results, metric):
        return round(sum(r[metric] for r in results) / len(results), 2)
    
    def contamination_rate(results):
        return round(sum(1 for r in results if r["contamination"]) / len(results), 2)
    
    def contamination_rate_by_similarity(results, sim):
        subset = [r for r in results if r.get("similarity") == sim]
        if not subset:
            return 0.0
        return round(sum(1 for r in subset if r["contamination"]) / len(subset), 2)
    
    def contamination_rate_by_question_type(results, qtype):
        """Fraction of rounds of a given question type that were contaminated."""
        total = 0
        contaminated = 0
        for r in results:
            for rd in r.get("rounds", []):
                if rd.get("question_type") == qtype:
                    total += 1
                    if rd.get("contamination"):
                        contaminated += 1
        return round(contaminated / total, 2) if total > 0 else 0.0
    
    metrics = ["personalization", "accuracy", "hallucination", "consistency"]
    
    print(f"\n{'─'*60}")
    print(f"RESULTS — {EXPERIMENT_NAME} (N={N_STUDENTS})")
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
    
    # Similarity-breakdown
    mem_high_contam = contamination_rate_by_similarity(memory_results, "high")
    mem_low_contam = contamination_rate_by_similarity(memory_results, "low")
    shr_high_contam = contamination_rate_by_similarity(shared_results, "high")
    shr_low_contam = contamination_rate_by_similarity(shared_results, "low")
    
    print(f"\n{'─'*60}")
    print(f"CONTAMINATION BY SIMILARITY")
    print(f"{'─'*60}")
    print(f"{'Agent':<10} {'High-sim':>10} {'Low-sim':>10} {'Delta(H-L)':>12}")
    print(f"{'Memory':<10} {mem_high_contam:>10.2f} {mem_low_contam:>10.2f} {mem_high_contam - mem_low_contam:>+10.2f}")
    print(f"{'Shared':<10} {shr_high_contam:>10.2f} {shr_low_contam:>10.2f} {shr_high_contam - shr_low_contam:>+10.2f}")
    
    # Question-type breakdown (memory agent only)
    mem_cross_talk_contam = contamination_rate_by_question_type(memory_results, "cross_talk")
    mem_normal_contam = contamination_rate_by_question_type(memory_results, "normal")
    shr_cross_talk_contam = contamination_rate_by_question_type(shared_results, "cross_talk")
    shr_normal_contam = contamination_rate_by_question_type(shared_results, "normal")
    
    print(f"\n{'─'*60}")
    print(f"CONTAMINATION BY QUESTION TYPE")
    print(f"{'─'*60}")
    print(f"{'Agent':<10} {'Cross-talk':>12} {'Normal':>10} {'Delta(C-N)':>12}")
    print(f"{'Memory':<10} {mem_cross_talk_contam:>12.2f} {mem_normal_contam:>10.2f} {mem_cross_talk_contam - mem_normal_contam:>+12.2f}")
    print(f"{'Shared':<10} {shr_cross_talk_contam:>12.2f} {shr_normal_contam:>10.2f} {shr_cross_talk_contam - shr_normal_contam:>+12.2f}")
    
    # Majority distance analysis (shared agent)
    print(f"\n{'─'*60}")
    print(f"MAJORITY DISTANCE ANALYSIS (Shared Agent)")
    print(f"{'─'*60}")
    for r in shared_results:
        s = next((x for x in students if x["id"] == r["student_id"]), None)
        dist = s["majority_distance"] if s else 0
        print(f"  {r['student_name']:20} dist={dist}  pers={r['personalization']:.2f}  acc={r['accuracy']:.2f}  contam={'Y' if r['contamination'] else 'N'}")
    
    # Per-student contamination events
    print(f"\n{'─'*60}")
    print(f"CONTAMINATION EVENTS (Memory Agent)")
    print(f"{'─'*60}")
    for r in memory_results:
        if r["contamination_events"]:
            for e in r["contamination_events"]:
                print(f"  {r['student_name']:20} round={e['round']} type={e['question_type']:10} src={e['source_student']:20} leaked={e['leaked_attributes']}")
            if r.get("cross_talk_spillover"):
                print(f"  {'':>20} ⚠ SPILLOVER: post-cross-talk rounds also contaminated")
    
    print(f"\n{'─'*60}")
    print(f"CONTAMINATION EVENTS (Shared Agent)")
    print(f"{'─'*60}")
    for r in shared_results:
        if r["contamination_events"]:
            for e in r["contamination_events"]:
                print(f"  {r['student_name']:20} round={e['round']} type={e['question_type']:10} src={e['source_student']:20} leaked={e['leaked_attributes']}")
            if r.get("cross_talk_spillover"):
                print(f"  {'':>20} ⚠ SPILLOVER: post-cross-talk rounds also contaminated")
    
    # ── Compute majority distance correlation ──────────────────────────
    shared_dist_pairs = []
    for r in shared_results:
        s = next((x for x in students if x["id"] == r["student_id"]), None)
        if s:
            shared_dist_pairs.append((s["majority_distance"], r["personalization"]))
    
    # Simple ordinal correlation: compare mean pers for dist=0 vs dist>0
    if shared_dist_pairs:
        dist0_pers = [p[1] for p in shared_dist_pairs if p[0] == 0]
        dist_pos_pers = [p[1] for p in shared_dist_pairs if p[0] > 0]
        dist0_mean = round(sum(dist0_pers) / len(dist0_pers), 2) if dist0_pers else 0
        dist_pos_mean = round(sum(dist_pos_pers) / len(dist_pos_pers), 2) if dist_pos_pers else 0
        majority_distance_corr = round(dist0_mean - dist_pos_mean, 2)
    else:
        majority_distance_corr = 0
    
    # ── Save raw results ───────────────────────────────────────────────
    output = {
        "experiment": EXPERIMENT_NAME,
        "config": {"n_students": N_STUDENTS, "n_rounds": N_ROUNDS, "condition": SIMILARITY_CONDITION},
        "students": [{"id": s["id"], "name": s["name"], "similarity": s.get("similarity"), "majority_distance": s.get("majority_distance")} for s in students],
        "memory_results": memory_results,
        "shared_results": shared_results,
        "summary": {
            **{f"memory_{m}": avg(memory_results, m) for m in metrics},
            **{f"shared_{m}": avg(shared_results, m) for m in metrics},
            "memory_contamination_rate": mem_contam,
            "shared_contamination_rate": shr_contam,
            "memory_contamination_high_sim": mem_high_contam,
            "memory_contamination_low_sim": mem_low_contam,
            "shared_contamination_high_sim": shr_high_contam,
            "shared_contamination_low_sim": shr_low_contam,
            "memory_cross_talk_contamination": mem_cross_talk_contam,
            "memory_normal_contamination": mem_normal_contam,
            "shared_cross_talk_contamination": shr_cross_talk_contam,
            "shared_normal_contamination": shr_normal_contam,
            "shared_majority_distance_delta_pers": majority_distance_corr,
            "memory_cross_talk_spillover_count": sum(1 for r in memory_results if r.get("cross_talk_spillover")),
            "shared_cross_talk_spillover_count": sum(1 for r in shared_results if r.get("cross_talk_spillover")),
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
