from typing import Dict, List


FIRST_VISIT_PROFILE = {
    "name": "first_visit_cf_team",
    "patient_questions": [
        "What tests will I need at my first CF team visit?",
        "How will this treatment plan affect my daily routine and work/school?",
        "What symptoms should trigger urgent contact with the CF team?",
        "How can I manage treatment burden and stay adherent at home?",
    ],
    "role_needs": {
        "Patient 🧑‍🦽": [
            {"need": "symptom relief and quality of life", "keywords": ["symptom", "quality of life", "fatigue", "burden"]},
            {"need": "clear daily plan and adherence support", "keywords": ["adherence", "routine", "daily plan", "support"]},
            {"need": "cost/access and practical barriers", "keywords": ["cost", "access", "insurance", "transport"]},
        ],
        "GP 👨‍⚕️": [
            {"need": "clear referral threshold and safety-net", "keywords": ["referral", "safety-net", "red flag", "urgent"]},
            {"need": "continuity and follow-up ownership", "keywords": ["follow-up", "continuity", "monitoring"]},
            {"need": "medication reconciliation and comorbidity", "keywords": ["medication", "interaction", "comorbidity"]},
        ],
        "CF Specialist 🩺": [
            {"need": "guideline-concordant stratification", "keywords": ["guideline", "risk", "stratification", "lung function"]},
            {"need": "multidisciplinary pathway", "keywords": ["multidisciplinary", "team", "physio", "dietitian"]},
            {"need": "exacerbation prevention plan", "keywords": ["exacerbation", "prevention", "airway clearance", "infection"]},
        ],
    },
}


SCENARIO_PROFILES: Dict[str, Dict] = {
    "first_visit": FIRST_VISIT_PROFILE,
}


def get_scenario_profile(name: str) -> Dict:
    return SCENARIO_PROFILES.get(name, FIRST_VISIT_PROFILE)
