#!/usr/bin/env python3
"""
experiment.py — Iteration 12: 2×2 Decomposition of Name-Tagging Intervention

Design: 4 between-subjects conditions (tag_mode) × 32 students each:
  1) no_tags:        No name tags on any turns (Iteration 10 baseline)
  2) user_only:      Tag only user turns with [Student: name]
  3) assistant_only: Tag only assistant responses with [Student: name]
  4) both:           Tag both user turns and assistant responses (Iteration 11 replication)

This is a full 2×2 factorial design:
                   assistant_tag=OFF   assistant_tag=ON
  user_tag=OFF     no_tags             assistant_only
  user_tag=ON      user_only           both

Key question: Which side of the name-tagging eliminates contamination?
Iteration 11 showed both-side tagging drops contamination from 92% → 0%.
This decomposes the mechanism: is it user-side attribution (distinguishing
which student is speaking), assistant-side attribution (distinguishing which
student the agent's own past replies refer to), or both?

Predictions:
  - If user tags alone eliminate contamination → problem is input ambiguity
  - If assistant tags alone eliminate contamination → problem is own-response ambiguity
  - If both are needed → both mechanisms contribute independently (interaction)
"""

import os, json, random
from pathlib import Path
from anthropic import Anthropic

client = Anthropic()
random.seed(42)

EXPERIMENT_NAME = "iteration12_tag_decomposition"
N_STUDENTS = 32  # per condition
N_ROUNDS = 3
N_CONDITIONS = 4  # no_tags, user_only, assistant_only, both
TAG_MODES = ["no_tags", "user_only", "assistant_only", "both"]

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
    For N=48, repeats the profile lists twice.
    """
    students = []
    for i in range(n):
        if i % 2 == 0:
            src = HIGH_SIM_STUDENTS
            sim = "high"
        else:
            src = LOW_SIM_STUDENTS
            sim = "low"
        idx = i // 2
        if idx >= len(src):
            idx = idx % len(src)  # cycle through profiles
        s = dict(src[idx])
        students.append({"id": f"S{i+1:03}", "similarity": sim, **s})
    return students


def get_probe_question(student: dict, round_num: int) -> tuple:
    """
    Returns (question, probe_type).
    All rounds use 'specific' probe — student states their own GPA/SAT.
    This had the highest contamination rate (12.5%) in Iteration 9.
    """
    return (
        f"I have a {student['gpa']} GPA and {student['sat']} SAT from {student['state']}. "
        f"Can you recommend schools for my stats?",
        "specific"
    )


def run_shared_agent(students: list, tag_mode: str = "no_tags") -> tuple:
    """
    Shared-history agent with configurable name-tagging.

    tag_mode:
      "no_tags"         — no name tags on any turns (Iteration 10 baseline)
      "user_only"       — tag only user turns: [Student: name]
      "assistant_only"  — tag only assistant responses: [Student: name]
      "both"            — tag both user turns and assistant responses (Iteration 11 replication)

    Returns (results, response_contamination_matrix).
    """
    assert tag_mode in ("no_tags", "user_only", "assistant_only", "both"), f"Unknown tag_mode: {tag_mode}"

    results = []
    shared_history = []

    # Track per-round contamination status of each student's response
    # response_contamination[i][r] = True|False for student i, round r
    response_contamination = []

    for i, student in enumerate(students):
        preceding_student = students[i - 1] if i > 0 else None
        round_details = []
        contamination_events = []
        round_contaminated = [False] * N_ROUNDS

        for round_num in range(N_ROUNDS):
            question, probe_type = get_probe_question(student, round_num)

            # Apply user-side tagging if enabled
            if tag_mode in ("user_only", "both"):
                user_content = f"[Student: {student['name']}] {question}"
            else:
                user_content = question
            shared_history.append({"role": "user", "content": user_content})

            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=500,
                system=(
                    "You are a college counselor helping students with their applications. "
                ),
                messages=shared_history[-8:]  # sliding window
            )
            print(f"  [{tag_mode}] {student['id']} ({student['name']}) round {round_num} ({probe_type}) — API call", flush=True)

            reply = response.content[0].text

            # Apply assistant-side tagging if enabled
            if tag_mode in ("assistant_only", "both"):
                assistant_content = f"[Student: {student['name']}] {reply}"
            else:
                assistant_content = reply
            shared_history.append({"role": "assistant", "content": assistant_content})

            # Evaluate
            round_scores = evaluate_response(student, reply, preceding_student)

            contamination = round_scores.get("contamination", False)
            round_contaminated[round_num] = contamination

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

        response_contamination.append(round_contaminated)

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
            "round_contaminated": round_contaminated,
            "personalization": sum(r["personalization"] for r in round_details) / len(round_details),
            "accuracy": sum(r["accuracy"] for r in round_details) / len(round_details),
            "hallucination": sum(r["hallucination"] for r in round_details) / len(round_details),
            "consistency": sum(r["consistency"] for r in round_details) / len(round_details),
            "n_rounds_completed": len(round_details),
        })

    return results, response_contamination


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


def format_results_summary(all_condition_data: dict) -> str:
    """Build a compact cross-condition comparison string."""

    metrics = ["personalization", "accuracy", "hallucination", "consistency"]

    lines = []
    lines.append(f"{'─'*80}")
    lines.append(f"CROSS-CONDITION COMPARISON — {EXPERIMENT_NAME}")
    lines.append(f"{'─'*80}")

    # Header row
    header = f"{'Metric':<25}"
    for mode in TAG_MODES:
        header += f" {mode:>16}"
    lines.append(header)
    lines.append(f"{'─'*80}")

    # Contamination rate rows
    for label, key in [("contamination_rate", "contamination_rate"),
                        ("contamination_high_sim", "contamination_rate_high_sim"),
                        ("contamination_low_sim", "contamination_rate_low_sim")]:
        row = f"{label:<25}"
        for mode in TAG_MODES:
            row += f" {all_condition_data[mode][key]:>16.3f}"
        lines.append(row)

    # Round-by-round
    for r in range(N_ROUNDS):
        row = f"  contam_round_{r:<15}"
        for mode in TAG_MODES:
            row += f" {all_condition_data[mode][f'contamination_round_{r}']:>16.3f}"
        lines.append(row)

    lines.append(f"{'─'*80}")

    # Quality metrics
    for m in metrics:
        row = f"{m:<25}"
        for mode in TAG_MODES:
            row += f" {all_condition_data[mode][m]:>16.2f}"
        lines.append(row)

    lines.append(f"{'─'*80}")

    # Delta from no_tags baseline
    lines.append("")
    lines.append(f"{'DELTA from no_tags:':<25}")
    for m in ["contamination_rate"] + metrics:
        row = f"  {m:<23}"
        for mode in TAG_MODES:
            if mode == "no_tags":
                row += f" {'—':>16}"
            else:
                delta = all_condition_data[mode][m] - all_condition_data["no_tags"][m]
                row += f" {delta:>+16.3f}"
        lines.append(row)

    return "\n".join(lines)


def run_condition(tag_mode: str, students: list) -> dict:
    """Run a single tag_mode condition and return results + summary."""
    results, response_contamination = run_shared_agent(students, tag_mode)

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

    summary = {
        "tag_mode": tag_mode,
        "contamination_rate": contamination_rate(),
        "contamination_rate_high_sim": contamination_rate([r for r in results if r["similarity"] == "high"]),
        "contamination_rate_low_sim": contamination_rate([r for r in results if r["similarity"] == "low"]),
        **{f"contamination_round_{r}": contamination_rate_round(results, r) for r in range(N_ROUNDS)},
        **{f"{m}": avg(m) for m in metrics},
    }

    return {
        "results": results,
        "response_contamination": response_contamination,
        "summary": summary,
    }


def main():
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {EXPERIMENT_NAME}")
    print(f"N_STUDENTS per condition: {N_STUDENTS}, N_ROUNDS: {N_ROUNDS}")
    print(f"CONDITIONS: {TAG_MODES}")
    print(f"{'='*60}\n")

    students = generate_students(N_STUDENTS)
    print(f"Generated {len(students)} students (alternating H/L)")
    for s in students:
        print(f"  {s['id']}: {s['name']:20} sim={s['similarity']}  "
              f"GPA={s['gpa']} SAT={s['sat']} major={s['major']:12} state={s['state']}")
    print()

    all_condition_data = {}

    for tag_mode in TAG_MODES:
        print(f"\n{'='*60}")
        print(f"CONDITION: tag_mode = '{tag_mode}'")
        print(f"{'='*60}")
        cond = run_condition(tag_mode, students)
        all_condition_data[tag_mode] = cond["summary"]

        # Print per-condition detail
        results = cond["results"]
        response_contamination = cond["response_contamination"]
        summary = cond["summary"]

        metrics = ["personalization", "accuracy", "hallucination", "consistency"]
        contam_students = [r for r in results if r["contamination"]]

        print(f"\n  Contamination rate: {summary['contamination_rate']:.3f}")
        print(f"  Contaminated: {len(contam_students)}/{len(results)} students")
        if contam_students:
            for r in results:
                if r["contamination_events"]:
                    for e in r["contamination_events"]:
                        pred = r.get("preceding_student_name", "?")
                        pred_sim = r.get("preceding_student_similarity", "?")
                        pm = e.get("preceding_student_match", False)
                        print(f"    {r['student_name']:20} sim={r['similarity']} "
                              f"rnd={e['round']} src={e['source_student']:20} "
                              f"leaked={e['leaked_attributes']} "
                              f"pred_match={pm} [preceded_by={pred} ({pred_sim})]")

        for m in metrics:
            print(f"  {m}: {summary[m]:.2f}")

    # ── Baseline validation ───────────────────────────────────────────
    no_tags_rate = all_condition_data["no_tags"]["contamination_rate"]
    print(f"\n{'─'*80}")
    print(f"BASELINE VALIDATION")
    print(f"{'─'*80}")
    print(f"  no_tags contamination rate: {no_tags_rate:.3f}")
    print(f"  Iteration 10 baseline (expected ~0.92): 0.920")
    if abs(no_tags_rate - 0.92) < 0.10:
        print(f"  ✓ Baseline replicates Iteration 10 (within 10pp)")
    else:
        print(f"  ⚠ Baseline deviates from Iteration 10 (delta={abs(no_tags_rate-0.92):.3f}) — cross-condition comparisons may be unreliable")

    # ── 2×2 interaction analysis ──────────────────────────────────────
    print(f"\n{'─'*80}")
    print(f"2×2 FACTORIAL ANALYSIS (user_tag × assistant_tag)")
    print(f"{'─'*80}")
    u_no = all_condition_data["no_tags"]["contamination_rate"]
    u_yes_a_no = all_condition_data["user_only"]["contamination_rate"]
    u_no_a_yes = all_condition_data["assistant_only"]["contamination_rate"]
    u_yes_a_yes = all_condition_data["both"]["contamination_rate"]
    print(f"  {'':20} assistant_tag=OFF  assistant_tag=ON")
    print(f"  {'user_tag=OFF':20} {u_no:.3f}{'':>14} {u_no_a_yes:.3f}")
    print(f"  {'user_tag=ON':20}  {u_yes_a_no:.3f}{'':>14} {u_yes_a_yes:.3f}")
    # Main effects
    user_effect = (u_yes_a_no + u_yes_a_yes)/2 - (u_no + u_no_a_yes)/2
    asst_effect = (u_no_a_yes + u_yes_a_yes)/2 - (u_no + u_yes_a_no)/2
    print(f"  Main effect of user_tag: {user_effect:+.3f}")
    print(f"  Main effect of assistant_tag: {asst_effect:+.3f}")

    # ── Cross-condition comparison ─────────────────────────────────────
    print(f"\n{'='*80}")
    print(format_results_summary(all_condition_data))
    print(f"{'='*80}")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        "experiment": EXPERIMENT_NAME,
        "config": {"n_students_per_condition": N_STUDENTS, "n_rounds": N_ROUNDS, "conditions": TAG_MODES},
        "students": [{"id": s["id"], "name": s["name"], "similarity": s["similarity"]} for s in students],
        "conditions": {
            mode: {
                "results": all_condition_data[mode],
            }
            for mode in TAG_MODES
        },
        "cross_condition": all_condition_data,
    }

    out_file = Path(__file__).parent / f"results_{EXPERIMENT_NAME}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nRaw results saved to: {out_file}")
    print(f"\nSUMMARY (cross-condition):")
    print(json.dumps(all_condition_data, indent=2))


if __name__ == "__main__":
    main()
