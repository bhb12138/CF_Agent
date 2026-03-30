from typing import Dict, List


FIRST_VISIT_PROFILE = {
    "name": "first_visit_cf_team",
    "patient_questions": [
        "What tests will I need at my first CF team visit?",
        "How will this treatment plan affect my daily routine and work/school?",
        "What symptoms should trigger urgent contact with the CF team?",
        "How can I manage treatment burden and stay adherent at home?",
        "What financial support is available for my medications and travel?",
        "How will CF affect my mental health and what support is available?",
    ],
    "role_needs": {

        # ── Patient needs ─────────────────────────────────────────────────────
        # Keywords are specific enough to require deliberate mention,
        # not just incidental use of common words.
        "Patient 🧑‍🦽": [
            {
                "need": "symptom relief and physical quality of life",
                "keywords": [
                    "breathlessness", "chest tightness", "lung function",
                    "pulmonary symptom", "physical quality of life", "FEV1",
                ],
            },
            {
                "need": "daily treatment burden and fatigue management",
                "keywords": [
                    "nebulizer", "airway clearance", "treatment schedule",
                    "twice daily", "fatigue management", "vest therapy",
                    "morning routine", "time-consuming",
                ],
            },
            {
                "need": "cost access and financial barriers",
                "keywords": [
                    "Trikafta cost", "prescription cost", "insurance coverage",
                    "transport cost", "financial assistance", "disability benefit",
                    "CF Trust grant", "afford treatment",
                ],
            },
            {
                "need": "mental health and psychological support",
                "keywords": [
                    "anxiety", "depression", "psychological support",
                    "mental health referral", "CBT", "counselling",
                    "emotional burden", "burnout", "PHQ-9", "GAD-7",
                ],
            },
            {
                "need": "social life and work or school impact",
                "keywords": [
                    "work impact", "school impact", "social life",
                    "sick leave", "employer", "university accommodation",
                    "isolation", "peer support", "dating", "fertility",
                ],
            },
        ],

        # ── GP needs ──────────────────────────────────────────────────────────
        "GP 👨‍⚕️": [
            {
                "need": "clear referral threshold and written safety-net",
                "keywords": [
                    "referral threshold", "written action plan", "A&E criteria",
                    "when to escalate", "red flag symptom", "urgent referral",
                    "deterioration plan", "safety-net criteria",
                ],
            },
            {
                "need": "shared care agreement and follow-up ownership",
                "keywords": [
                    "shared care agreement", "written agreement", "follow-up ownership",
                    "GP responsibility", "care coordination protocol",
                    "complementary review", "avoid duplication",
                ],
            },
            {
                "need": "medication reconciliation and drug interaction review",
                "keywords": [
                    "medication reconciliation", "drug interaction", "CYP3A4",
                    "PERT dose", "enzyme dose", "prescription review",
                    "polypharmacy", "drug reconciliation",
                ],
            },
            {
                "need": "comorbidity management protocol",
                "keywords": [
                    "CFRD", "CF-related diabetes", "osteoporosis", "DEXA",
                    "liver disease", "CF arthropathy", "comorbidity protocol",
                    "annual review", "bone density",
                ],
            },
            {
                "need": "mental health screening in primary care",
                "keywords": [
                    "PHQ-9", "GAD-7", "depression screening", "anxiety screening",
                    "mental health screening", "psychological assessment",
                    "SSRI", "psychosocial screening",
                ],
            },
        ],

        # ── CF Specialist needs ───────────────────────────────────────────────
        "CF Specialist 🩺": [
            {
                "need": "CFTR modulator eligibility and optimisation",
                "keywords": [
                    "CFTR modulator", "Trikafta", "elexacaftor", "ivacaftor",
                    "modulator eligibility", "genotype eligibility",
                    "F508del", "ETI therapy", "modulator optimisation",
                ],
            },
            {
                "need": "lung function monitoring and risk stratification",
                "keywords": [
                    "FEV1 decline", "spirometry frequency", "lung function monitoring",
                    "risk stratification", "FEV1 percent predicted",
                    "pulmonary function test", "6-minute walk", "CT chest",
                ],
            },
            {
                "need": "multidisciplinary team pathway",
                "keywords": [
                    "MDT meeting", "physiotherapist referral", "dietitian referral",
                    "psychologist referral", "CF nurse specialist",
                    "multidisciplinary review", "MDT coordination",
                ],
            },
            {
                "need": "exacerbation prevention and infection surveillance",
                "keywords": [
                    "exacerbation prevention", "sputum culture", "Pseudomonas",
                    "infection surveillance", "airway clearance protocol",
                    "inhaled antibiotic", "eradication protocol", "microbiology",
                ],
            },
            {
                "need": "nutritional assessment and enzyme optimisation",
                "keywords": [
                    "nutritional assessment", "pancreatic enzyme", "PERT dosing",
                    "caloric intake", "fat-soluble vitamin", "BMI target",
                    "malnutrition risk", "dietitian", "enteral feeding",
                ],
            },
        ],
    },
}


SCENARIO_PROFILES: Dict[str, Dict] = {
    "first_visit": FIRST_VISIT_PROFILE,
}


def get_scenario_profile(name: str) -> Dict:
    return SCENARIO_PROFILES.get(name, FIRST_VISIT_PROFILE)