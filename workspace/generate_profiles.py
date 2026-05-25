#!/usr/bin/env python3
"""
Generate 50 diverse synthetic student profiles for the experiment.
Profiles vary across: GPA, demographics, intended major, state of residence, ECs, goals.
Output: JSON file with 50 profiles.
"""

import json
import random
import hashlib

random.seed(42)

# --- Dimensions of variation ---

first_names = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn",
    "Maya", "Ethan", "Sophia", "Liam", "Olivia", "Noah", "Emma", "Oliver",
    "Aisha", "Carlos", "Yuki", "Priya", "Kwame", "Mei", "Diego", "Fatima",
    "Raj", "Suki", "Hiro", "Amara", "Leo", "Zara", "Felix", "Lena",
    "Marcus", "Sofia", "Kenji", "Anya", "Tomas", "Leila", "Omar", "Ingrid",
    "Darius", "Noa", "Elena", "Santi", "Hana", "Ravi", "Tara", "Arjun",
    "Mira", "Kofi"
]

last_names = [
    "Smith", "Chen", "Rodriguez", "Kim", "Patel", "Johnson", "Lee", "Garcia",
    "Wang", "Brown", "Jones", "Miller", "Davis", "Martinez", "Hernandez",
    "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
    "Jackson", "Martin", "Thompson", "White", "Nguyen", "Harris", "Clark",
    "Lewis", "Robinson", "Walker", "Young", "Allen", "King", "Wright",
    "Scott", "Torres", "Hill", "Green", "Adams", "Baker", "Nelson", "Carter",
    "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell"
]

demographics = [
    "Asian", "Black/African American", "Hispanic/Latino", "White",
    "Mixed/Two or More Races", "Middle Eastern", "Native American",
    "Pacific Islander", "South Asian", "Southeast Asian"
]

states = [
    "CA", "TX", "NY", "FL", "IL", "WA", "MA", "MI", "NC", "GA",
    "PA", "OH", "VA", "AZ", "CO", "MN", "WI", "MD", "IN", "OR",
    "NV", "UT", "CT", "MO", "AL", "LA", "KY", "SC", "OK", "IA",
    "KS", "AR", "MS", "NM", "NE", "WV", "ID", "HI", "NH", "ME",
    "MT", "RI", "DE", "SD", "ND", "AK", "VT", "WY", "DC", "PR"
]

# CA students are more common (CA context)
state_weights = [0.20 if s == "CA" else 0.016 for s in states]

intended_majors = [
    "Computer Science", "Mechanical Engineering", "Biology", "Psychology",
    "Economics", "Political Science", "English Literature", "Art History",
    "Electrical Engineering", "Chemical Engineering", "Mathematics", "Physics",
    "Philosophy", "Sociology", "Environmental Science", "Business",
    "Neuroscience", "Architecture", "Nursing", "Pre-Medicine",
    "Data Science", "Film Studies", "International Relations", "Music",
    "Journalism", "Education", "Anthropology", "Linguistics",
    "Astronomy", "Marine Biology"
]

gpa_ranges = [
    (4.0, 4.0),   # perfect
    (3.9, 4.0),   # excellent
    (3.7, 3.9),   # very strong
    (3.5, 3.7),   # strong
    (3.2, 3.5),   # good
    (2.8, 3.2),   # average
    (2.5, 2.8),   # below average
    (2.0, 2.5)    # low
]

# Weight toward higher GPAs in college-bound pool
gpa_weights = [0.05, 0.15, 0.25, 0.25, 0.15, 0.08, 0.05, 0.02]

income_brackets = ["<30k", "30-60k", "60-100k", "100-200k", "200k+"]
income_weights = [0.12, 0.23, 0.30, 0.25, 0.10]

# Extracurricular types
ec_types = [
    "Student Government", "Debate Club", "Science Olympiad", "Varsity Soccer",
    "Varsity Basketball", "Varsity Tennis", "School Newspaper", "Yearbook",
    "Robotics Club", "Coding Club", "Volunteer Tutoring", "Hospital Volunteer",
    "Part-time Job", "Family Responsibilities", "Music (Band/Orchestra)",
    "Theatre/Drama", "Art Club", "Model UN", "Chess Club", "Math Club",
    "National Honor Society", "Peer Counseling", "Environmental Club",
    "Language Club", "Film Club", "Debate Team Captain", "Student Council President",
    "Founder of Nonprofit", "Research Internship", "Summer Camp Counselor"
]

num_ec_options = [3, 4, 5, 6, 7]

# Essay themes
essay_themes = [
    "Overcoming personal challenge", "Cultural identity exploration",
    "Family influence on goals", "Community service impact",
    "Intellectual curiosity story", "Leadership experience",
    "Creative passion narrative", "Academic journey",
    "Immigrant family story", "Building something from nothing",
    "Learning from failure", "Mentorship experience",
    "Travel and perspective shift", "Athletic discipline and life lessons",
    "Artistic expression as growth"
]

# Generate test score ranges (SAT equivalent 400-1600)
# Correlated with GPA
def generate_sat_score(gpa):
    base = int(gpa * 400 - 200)  # rough mapping
    noise = random.gauss(0, 50)
    return max(400, min(1600, int(base + noise)))

def generate_profile(i):
    first = first_names[i]
    last = last_names[i]
    demo = random.choice(demographics)
    state = random.choices(states, weights=state_weights, k=1)[0]
    major = random.choice(intended_majors)
    
    # GPA
    gpa_range = random.choices(gpa_ranges, weights=gpa_weights, k=1)[0]
    gpa = round(random.uniform(*gpa_range), 2)
    
    income = random.choices(income_brackets, weights=income_weights, k=1)[0]
    
    # Number of ECs
    num_ec = random.choice(num_ec_options)
    ecs = random.sample(ec_types, min(num_ec, len(ec_types)))
    
    # EC hours per week
    ec_hours = random.randint(1, 25)
    
    # First-gen college?
    first_gen = random.random() < 0.35
    
    # Test scores
    if random.random() < 0.75:  # 75% have test scores
        sat = generate_sat_score(gpa)
        act = round(sat / 36, 0)
    else:
        sat = None
        act = None
    
    # Essay theme
    essay_theme = random.choice(essay_themes)
    
    # Goals
    goals_templates = [
        f"I want to study {major} and eventually work in a field that combines technical skills with social impact.",
        f"After college, I plan to pursue a graduate degree in {major} and become a researcher.",
        f"My goal is to start my own company in the {major} space after gaining industry experience.",
        f"I hope to use my degree in {major} to give back to my community and help others.",
        f"I am passionate about {major} and want to explore its intersection with public policy.",
        f"I aim to become a leader in {major} and mentor the next generation of students.",
        f"I want to use {major} to solve real-world problems, especially in underserved communities.",
        f"My dream is to combine {major} with my interest in creative arts and media.",
        f"I plan to pursue a career in {major} while staying involved in community organizing.",
        f"I want to study {major} abroad and bring international perspectives back to my work."
    ]
    goal = random.choice(goals_templates)
    
    student_id = hashlib.md5(f"{first}{last}{i}".encode()).hexdigest()[:8]
    
    return {
        "id": student_id,
        "first_name": first,
        "last_name": last,
        "age": 17,
        "demographics": demo,
        "state": state,
        "intended_major": major,
        "gpa": gpa,
        "sat": sat,
        "act": act,
        "income_bracket": income,
        "first_generation": first_gen,
        "extracurriculars": ecs,
        "ec_hours_per_week": ec_hours,
        "essay_theme": essay_theme,
        "goal": goal,
        "num_ecs": len(ecs),
        "student_type": "synthetic"
    }

profiles = [generate_profile(i) for i in range(50)]

# Add some paired similar profiles for consistency testing
# Profiles that are very similar pair-wise (same GPA range, same major, similar ECs)
similar_pairs = []
for i in range(0, 50, 2):
    if i + 1 < 50:
        # Make profile i+1 similar to profile i
        p = profiles[i]
        profiles[i+1]["intended_major"] = p["intended_major"]
        profiles[i+1]["gpa"] = p["gpa"] + random.choice([-0.05, -0.03, 0, 0.03, 0.05])
        profiles[i+1]["gpa"] = round(max(2.0, min(4.0, profiles[i+1]["gpa"])), 2)
        profiles[i+1]["state"] = p["state"]
        profiles[i+1]["goal"] = p["goal"]  # Same goal
        similar_pairs.append((i, i+1))

with open("/research/workspace/synthetic_student_profiles.json", "w") as f:
    json.dump(profiles, f, indent=2)

# Also write compact summary for analysis
summary = []
for p in profiles:
    summary.append({
        "id": p["id"],
        "gpa": p["gpa"],
        "major": p["intended_major"],
        "state": p["state"],
        "sat": p["sat"],
        "num_ecs": p["num_ecs"],
        "first_gen": p["first_generation"],
        "income": p["income_bracket"],
        "demo": p["demographics"]
    })

with open("/research/workspace/synthetic_profiles_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

with open("/research/workspace/similar_pairs.json", "w") as f:
    json.dump(similar_pairs, f, indent=2)

print(f"Generated {len(profiles)} profiles")
print(f"Created {len(similar_pairs)} similar pairs for consistency testing")
print(f"GPA range: {min(p['gpa'] for p in profiles):.2f} - {max(p['gpa'] for p in profiles):.2f}")
print(f"Majors: {len(set(p['intended_major'] for p in profiles))} unique")
print(f"States: {len(set(p['state'] for p in profiles))} unique")
