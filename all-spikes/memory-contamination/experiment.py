#!/usr/bin/env python3
"""
experiment.py — Iteration 9: Question-Type Probe Provocation

Design: Single condition (shared agent + passive probe), 24 students.
3 rounds per student, each round uses a different probe type (within-subjects).
Round 0: Vague probe
Round 1: Self-ref probe
Round 2: Specific probe

Oversampling: Self-ref is the theoretically most interesting condition,
but with within-subjects design, every student gets all 3 types, so we
get N=24 per probe type for maximum statistical power.

Key manipulation: For each student, record the preceding student's similarity
and name. Hypothesis: vague/ambiguous probes cause more contamination because
the agent lacks a clear retrieval target and defaults to the most recently
seen student profile in the context window.
"""

import os, json, random
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()
random.seed(42)

EXPERIMENT_NAME = "iteration9_probe_provocation"
N_STUDENTS = 24
N_ROUNDS = 3

# ─── Student profiles ──────────────────────────────────────────────────────────
HIGH_SIM_STUDENTS = [
    {"name": "Alan Chen",    "gpa": 3.9, "sat": 1490, "ecs": ["robotics", "debate"],            "major": "CS",     "state": "CA"},
    {"name": "Brian Chen",   "gpa": 3.8, "sat": 1470, "ecs": ["robotics", "coding club"],       "major": "CS",     "state": "CA"},
    {"name": "Calvin Chen",  "gpa": 3.9, "sat": 1480, "ecs": ["debate", "MUN"],                 "major": "CS",     "state": "CA"},
    {"name": "Diana Park",   "gpa": 3.7, "sat": 1450, "ecs": ["robotics", "science olympiad"],  "major": "CS",     "state": "CA"},
    {"name": "Eric Wu",      "gpa": 3.8, "sat": 1460, "ecs": ["coding club", "math team"],      "major": "CS",     "state": "CA"},
    {"name": "Felix Li",     "gpa": 3.6, "sat": 1440, "ecs": ["MUN", "debate"],                 "major": "CS",     "state": "CA"},
    {"name": "Grace Kim",    "gpa": 3.7, "sat": 1400, "ecs": ["swim", "NHS"],                   "major": "biology", "state": "CA"},
    {"name": "Henry Liu",    "gpa": 3.8, "sat": 1420, "ecs": ["math team", "orchestra"],        "major": "CS",     "state": "CA"},
    {"name": "Iris Wang",    "gpa": 3.9, "sat": 1480, "ecs": ["debate", "science olympiad"],    "major": "CS",     "state": "CA"},
    {"name": "Jason Park",   "gpa": 3.7, "sat": 1430, "ecs": ["robotics", "chess"],             "major": "CS",     "state": "CA"},
    {"name": "Karen Tan",    "gpa": 3.6, "sat": 1410, "ecs": ["coding club", "piano"],          "major": "CS",     "state": "CA"},
    {"name": "Leo Zhang",    "gpa": 3.8, "sat": 1440, "ecs": ["math team", "MUN"],              "major": "CS",     "state": "CA"},
]

LOW_SIM_STUDENTS = [
    {"name": "Maya Singh",    "gpa": 3.6, "sat": 1330, "ecs": ["theater", "newspaper"],       "major": "english",     "state": "CO"},
    {"name": "Nina Patel",    "gpa": 3.4, "sat": 1280, "ecs": ["dance", "volunteer"],         "major": "psychology",  "state": "NY"},
    {"name": "Omar Hassan",   "gpa": 3.7, "sat": 1350, "ecs": ["football", "DECA"],           "major": "finance",     "state": "FL"},
    {"name": "Priya Sharma",  "gpa": 3.9, "sat": 1450, "ecs": ["math team", "orchestra"],     "major": "chemistry",   "state": "NJ"},
    {"name": "Quinn Miller",  "gpa": 3.5, "sat": 1300, "ecs": ["track", "chess club"],        "major": "economics",   "state": "WA"},
    {"name": "Rachel Wilson", "gpa": 3.3, "sat": 1220, "ecs": ["art club", "yearbook"],       "major": "art",         "state": "OR"},
    {"name": "Sam Taylor",    "gpa": 3.8, "sat": 1390, "ecs": ["basketball", "spanish club"], "major": "history",      "state": "AZ"},
    {"name": "Tina Brown",    "gpa": 3.2, "sat": 1180, "ecs": ["band", "key club"],           "major": "education",   "state": "OH"},
    {"name": "Uma Patel",     "gpa": 3.7, "sat": 1340, "ecs": ["soccer", "volunteer"],        "major": "biology",     "state": "TX"},
    {"name": "Victor Kim",    "gpa": 3.5, "sat": 1310, "ecs": ["baseball", "NHS"],            "major": "business",    "state": "FL"},
    {"name": "Wendy Liu",     "gpa": 3.4, "sat": 1260, "ecs": ["dance", "newspaper"],         "major": "psychology",  "state": "IL"},
    {"name": "Xander Chen",   "gpa": 3.6, "sat": 1330, "ecs": ["theater", "key club"],        "major": "english",     "state": "CO"},
]

ALL_PROFILES = HIGH_SIM_STUDENTS + LOW_SIM_STUDENTS


def generate_students(n: int) -> list:
    """
    Interleave high-sim and low-sim in pairs so every student has
    a preceding student of known similarity. First student has no predecessor.
    Order: H L H L H L ... creating alternating patterns.
    """
    students = []
    for i in range(n):
        if i % 2 == 0:
            src = HIGH_SIM_STUDENTS
            sim = "high"
        else:
            src = LOW_SIM_STUDENTS
            sim = "low"
        s = dict(src[i // 2])
        students.append({"id": f"S{i+1:03}", "similarity": sim, **s})
    return students


def get_probe_question(student: dict, round_num: int) -> tuple:
    """
    Returns (question, probe_type).
    Within-subjects: each student sees all 3 probe types across 3 rounds.
    Round 0: Vague — generic advice request
    Round 1: Self-ref — asks agent to recall their profile (ambiguous retrieval)
    Round 2: Specific — provides their own stats, just asks for recommendations
    """
    if round_num == 0:
        return (
            "Can you give me some general advice about college applications?",
            "vague"
        )
    elif round_num == 1:
        return (
            "Can you tell me about my profile based on what you know about me?",
            "self_ref"
        )
    else:  # round_num == 2
        return (
            f"I have a {student['gpa']} GPA and {student['sat']} SAT from {student['state']}. "
            f"Can you recommend schools for my stats?",
            "specific"
        )


def run_shared_agent(students: list) -> list:
    """
    Shared-history agent. Processes students in order, appending to a single
    context window. Records the preceding student's similarity and name for each
    target student to enable analysis of proximity-driven contamination.
    """

    results = []
    shared_history = []

    for i, student in enumerate(students):
        preceding_student = students[i - 1] if i > 0 else None
        round_details = []
        contamination_events = []

        for round_num in range(N_ROUNDS):
            question, probe_type = get_probe_question(student, round_num)
            tagged_question = f"[Student: {student['name']}] {question}"
            shared_history.append({"role": "user", "content": tagged_question})

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                system=(
                    "You are a college counselor serving multiple students. "
                    "Keep each student's advice personalized to their specific profile. "
                    "If a student gives generic or vague questions, respond helpfully "
                    "but do NOT mix up information between different students."
                ),
                messages=shared_history[-8:]  # sliding window
            )
            print(f"  [shared] {student['id']} ({student['name']}) round {round_num} ({probe_type}) — API call", flush=True)

            reply = response.content[0].text
            shared_history.append({"role": "assistant", "content": reply})

            # Evaluate
            # Pass all prior students so evaluator can check which one leaked
            round_scores = evaluate_response(student, reply, preceding_student)

            contamination = round_scores.get("contamination", False)
            if contamination:
                contamination_events.append({
                    "round": round_num,
                    "probe_type": probe_type,
                    "source_student": round_scores.get("contamination_source", "unknown"),
                    "leaked_attributes": round_scores.get("leaked_attributes", []),
                    "preceding_student_match": round_scores.get("preceding_student_match", False),
                })

            round_details.append({
                "round": round_num,
                "probe_type": probe_type,
                "personalization": round_scores.get("personalization", 3),
                "accuracy": round_scores.get("accuracy", 3),
                "hallucination": round_scores.get("hallucination", 3),
                "consistency": round_scores.get("consistency", 3),
                "contamination": contamination,
                "stat_conflicts": round_scores.get("stat_conflicts", []),
                "contamination_source": round_scores.get("contamination_source", "none"),
                "leaked_attributes": round_scores.get("leaked_attributes", []),
                "preceding_student_match": round_scores.get("preceding_student_match", False),
            })

            print(f"  [shared] {student['id']} round {round_num} — eval complete", flush=True)

        results.append({
            "student_id": student["id"],
            "student_name": student["name"],
            "similarity": student["similarity"],
            "agent_type": "shared",
            "preceding_student_name": preceding_student["name"] if preceding_student else None,
            "preceding_student_similarity": preceding_student["similarity"] if preceding_student else None,
            "rounds": round_details,
            "contamination": len(contamination_events) > 0,
            "contamination_events": contamination_events,
            "contamination_count": len(contamination_events),
            "personalization": sum(r["personalization"] for r in round_details) / len(round_details),
            "accuracy": sum(r["accuracy"] for r in round_details) / len(round_details),
            "hallucination": sum(r["hallucination"] for r in round_details) / len(round_details),
            "consistency": sum(r["consistency"] for r in round_details) / len(round_details),
            "n_rounds_completed": len(round_details),
        })

    return results


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_response(student: dict, response: str, preceding_student: dict = None) -> dict:
    """Use Claude to evaluate the response. Returns scores dict."""

    all_profiles_text = "\n".join(
        f"  - {s['name']}: GPA={s['gpa']}, SAT={s['sat']}, "
        f"ECs={', '.join(s['ecs'])}, major={s['major']}, state={s['state']}"
        for s in ALL_PROFILES
    )

    preceding_clue = ""
    if preceding_student:
        preceding_clue = (
            f"\n\nIMPORTANT — The IMMEDIATELY PRECEDING student in the conversation was {preceding_student['name']} "
            f"(GPA={preceding_student['gpa']}, SAT={preceding_student['sat']}, "
            f"state={preceding_student['state']}, ECs={', '.join(preceding_student['ecs'])}). "
            f"Check extra carefully whether any of {preceding_student['name']}'s attributes leaked into this response."
        )

    eval_prompt = f"""Rate this college counseling response for student {student['name']}.

Student profile (CORRECT stats):
- GPA: {student['gpa']}, SAT: {student['sat']}
- ECs: {', '.join(student['ecs'])}
- Major: {student['major']}, State: {student['state']}

ALL OTHER STUDENT PROFILES (check if response mentions stats from any of these):
{all_profiles_text}
{preceding_clue}

Response to evaluate:
{response}

Score each on 1-5 and check for contamination:
- personalization: Does it reference THIS student's specific stats/ECs? (1=generic, 5=highly specific)
- accuracy: Is the admissions advice factually correct for CA/UC context? (1=wrong, 5=accurate)
- hallucination: Does it invent stats, programs, or details? (1=fabricates a lot, 5=no hallucination)
- consistency: Is it consistent with earlier advice? (1=contradicts, 5=consistent)
- contamination: Does it mention stats/names/ECs that belong to a DIFFERENT student? Check carefully against ALL profiles above. (true/false)
- contamination_source: Which student's data leaked? (string, "none" if none)
- leaked_attributes: List of attributes that leaked (e.g. ["name", "gpa", "sat", "ecs", "major"])
- preceding_student_match: Does the contaminated data come specifically from the PRECEDING student (the one who was just discussed immediately before this student)? (true/false) If contamination is false, set to false.

IMPORTANT: A response like "I see you're from California with robotics" would NOT be contamination even if those match another student — only flag if it explicitly mentions ATTRIBUTES THAT AREN'T THE CORRECT ONES for {student['name']}.

Respond ONLY with JSON: {{"personalization":4,"accuracy":3,"hallucination":5,"consistency":4,"contamination":false,"contamination_source":"none","leaked_attributes":[],"preceding_student_match":false}}"""

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
    result["stat_conflicts"] = detect_stat_conflicts(student, response)
    return result


def detect_stat_conflicts(student: dict, response: str) -> list:
    """Regex-based detection of exact stat values from other students."""
    conflicts = []
    for other in ALL_PROFILES:
        if other["name"] == student["name"]:
            continue
        if str(other["gpa"]) in response and str(other["gpa"]) != str(student["gpa"]):
            conflicts.append({"attribute": "gpa", "source_student": other["name"], "value": other["gpa"]})
        if str(other["sat"]) in response and str(other["sat"]) != str(student["sat"]):
            conflicts.append({"attribute": "sat", "source_student": other["name"], "value": other["sat"]})
        if other["name"] in response:
            conflicts.append({"attribute": "name", "source_student": other["name"], "value": other["name"]})
        for ec in other["ecs"]:
            if ec in response and ec not in student["ecs"]:
                conflicts.append({"attribute": "ec", "source_student": other["name"], "value": ec})
    return conflicts


def main():
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"N_STUDENTS: {N_STUDENTS}, N_ROUNDS: {N_ROUNDS}")
    print(f"{'='*60}\n")

    students = generate_students(N_STUDENTS)
    print(f"Generated {len(students)} students (alternating H/L)")
    for s in students:
        print(f"  {s['id']}: {s['name']:20} sim={s['similarity']}  "
              f"GPA={s['gpa']} SAT={s['sat']} major={s['major']:12} state={s['state']}")
    print()

    print("Running SHARED agent (within-subjects: vague r0, self_ref r1, specific r2)...")
    results = run_shared_agent(students)

    # ── Aggregate metrics ──────────────────────────────────────────────
    def avg(metric):
        return round(sum(r[metric] for r in results) / len(results), 2)

    def contamination_rate(subset=None):
        r = results if subset is None else subset
        if not r:
            return 0.0
        return round(sum(1 for x in r if x["contamination"]) / len(r), 2)

    def contamination_rate_round(subset, round_num):
        count = 0
        for r in subset:
            rd = next((x for x in r["rounds"] if x["round"] == round_num), None)
            if rd and rd["contamination"]:
                count += 1
        return round(count / len(subset), 3) if subset else 0.0

    metrics = ["personalization", "accuracy", "hallucination", "consistency"]

    print(f"\n{'─'*60}")
    print(f"RESULTS — {EXPERIMENT_NAME}")
    print(f"{'─'*60}")
    print(f"{'Metric':<20} {'All':>8} {'High-sim':>10} {'Low-sim':>10}")
    print(f"{'─'*50}")
    for m in metrics:
        high = sum(r[m] for r in results if r["similarity"] == "high") / max(1, len([r for r in results if r["similarity"] == "high"]))
        low = sum(r[m] for r in results if r["similarity"] == "low") / max(1, len([r for r in results if r["similarity"] == "low"]))
        print(f"{m:<20} {avg(m):>8.2f} {high:>10.2f} {low:>10.2f}")
    print(f"{'─'*50}")
    print(f"{'contamination_rate':<20} {contamination_rate():>8.3f} "
          f"{contamination_rate([r for r in results if r['similarity']=='high']):>10.3f} "
          f"{contamination_rate([r for r in results if r['similarity']=='low']):>10.3f}")

    # ── Probe-type (within-subjects) breakdown ────────────────────────
    print(f"\n{'─'*60}")
    print(f"CONTAMINATION BY PROBE TYPE (within-subjects, N=24 each)")
    print(f"{'─'*60}")
    for round_num, pt in [(0, "vague"), (1, "self_ref"), (2, "specific")]:
        r0_subset = [r for r in results]
        rate = contamination_rate_round(r0_subset, round_num)
        high_rate = contamination_rate_round([r for r in results if r["similarity"] == "high"], round_num)
        low_rate = contamination_rate_round([r for r in results if r["similarity"] == "low"], round_num)
        print(f"  {pt:<12} N=24   rate={rate:.3f}   "
              f"high={high_rate:.3f}   low={low_rate:.3f}")

    # ── Preceding-student effect ──────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"CONTAMINATION BY PRECEDING STUDENT SIMILARITY")
    print(f"{'─'*60}")
    for pred_sim in ["high", "low"]:
        subset = [r for r in results if r.get("preceding_student_similarity") == pred_sim and r["student_id"] != "S001"]
        r1_contam = []
        for r in subset:
            rd = next((x for x in r["rounds"] if x["round"] == 1), None)
            if rd and rd["contamination"]:
                r1_contam.append(r)
        rate = len(r1_contam) / len(subset) if subset else 0.0
        print(f"  preceded_by_{pred_sim:<4} N={len(subset):2d}   rate={rate:.3f}")

    # ── Contamination events detail ───────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"CONTAMINATION EVENTS DETAIL")
    print(f"{'─'*60}")
    for r in results:
        if r["contamination_events"]:
            for e in r["contamination_events"]:
                pred = r.get("preceding_student_name", "?")
                pred_sim = r.get("preceding_student_similarity", "?")
                pm = e.get("preceding_student_match", False)
                print(f"  {r['student_name']:20} sim={r['similarity']} "
                      f"probe={e['probe_type']:<10} "
                      f"rnd={e['round']} src={e['source_student']:20} "
                      f"leaked={e['leaked_attributes']} "
                      f"pred_match={pm} "
                      f"[preceded_by={pred} ({pred_sim})]")

    # ── Summary stats ─────────────────────────────────────────────────
    any_contam = [r for r in results if r["contamination"]]
    if any_contam:
        total_leaks = sum(len(e["leaked_attributes"])
                          for r in any_contam
                          for e in r["contamination_events"])
        total_events = sum(r['contamination_count'] for r in any_contam)
        print(f"\n  Total contaminated students: {len(any_contam)}/{len(results)}")
        print(f"  Mean leaked attributes per event: {total_leaks / max(1, total_events):.2f}")

    # ── Check preceding_student_match stats ───────────────────────────
    pm_events = []
    for r in results:
        for e in r["contamination_events"]:
            if e.get("preceding_student_match"):
                pm_events.append(e)
    if pm_events:
        print(f"  Events where source = preceding student: {len(pm_events)}/{total_events}" if total_events > 0 else "  No preceding_student_match events")
    else:
        print(f"  Events where source = preceding student: 0/{total_events if total_events > 0 else 0}")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "experiment": EXPERIMENT_NAME,
        "config": {"n_students": N_STUDENTS, "n_rounds": N_ROUNDS},
        "students": [{"id": s["id"], "name": s["name"], "similarity": s["similarity"]} for s in students],
        "results": results,
        "summary": {
            "contamination_rate": contamination_rate(),
            "contamination_rate_high_sim": contamination_rate([r for r in results if r["similarity"] == "high"]),
            "contamination_rate_low_sim": contamination_rate([r for r in results if r["similarity"] == "low"]),
            "contamination_by_probe_type": {
                "vague": contamination_rate_round(results, 0),
                "self_ref": contamination_rate_round(results, 1),
                "specific": contamination_rate_round(results, 2),
            },
            **{f"{m}": avg(m) for m in metrics},
        }
    }

    out_file = Path(__file__).parent / f"results_{EXPERIMENT_NAME}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nRaw results saved to: {out_file}")
    print(f"\nSUMMARY JSON:")
    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
