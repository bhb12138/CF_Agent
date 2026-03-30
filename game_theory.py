from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


# ── Action space ──────────────────────────────────────────────────────────────
# Initial utility values represent prior beliefs before any dialogue takes place.
# They are grounded in clinical evidence and patient preference literature:
#
#   CF Specialist utility: reflects guideline recommendation strength
#     (CFF Patient Registry Annual Report 2022; Elborn, Lancet 2016)
#   GP utility: reflects feasibility and role alignment in shared care
#     (NICE NG78 2017; Peckham et al., Prim Care Respir J 2010)
#   Patient utility: reflects patient-reported outcome priorities
#     (Gee et al., Thorax 2014; Britto et al., Pediatrics 2004;
#      CF Trust "CF and Me" patient survey 2021)
#
# All values are on [0, 1]. These are INITIAL values and are updated
# dynamically each turn via update_action_utility().

ACTION_UTILITY: Dict[str, Dict[str, float]] = {

    # CFTR modulator therapy (e.g. Trikafta/ETI)
    # CF Specialist: transformative FEV1 and QoL gains (Taylor-Cousar et al., NEJM 2019)
    # GP: initiation requires specialist lead; CYP3A4 interactions create prescribing complexity
    # Patient: dramatic symptom improvement but side effects and access/cost barriers exist
    "start_or_optimize_cftr_modulator": {
        "CF Specialist 🩺": 0.95,
        "GP 👨‍⚕️": 0.62,
        "Patient 🧑‍🦽": 0.80,
    },

    # Rapid exacerbation safety net (written action plan, clear escalation thresholds)
    # CF Specialist: exacerbation prevention is central to lung function preservation
    # GP: safety-netting is core GP competency; reduces emergency admissions (NICE NG78)
    # Patient: reassuring but adds cognitive burden to an already demanding regimen
    "rapid_exacerbation_safety_net": {
        "CF Specialist 🩺": 0.86,
        "GP 👨‍⚕️": 0.90,
        "Patient 🧑‍🦽": 0.72,
    },

    # Monthly shared decision-making visit
    # CF Specialist: supports MDT but less central than disease-specific review
    # GP: continuity and relationship-building are core to GP role
    # Patient: patients consistently value regular contact and shared decisions (Gee et al.)
    "monthly_shared_decision_visit": {
        "CF Specialist 🩺": 0.62,
        "GP 👨‍⚕️": 0.80,
        "Patient 🧑‍🦽": 0.88,
    },

    # Adherence support with social worker or CF nurse specialist
    # CF Specialist: adherence directly affects clinical outcomes but not specialist's primary role
    # GP: practical barriers frequently encountered in primary care
    # Patient: treatment burden and access are top patient priorities (CF Trust 2021)
    "adherence_support_with_social_worker": {
        "CF Specialist 🩺": 0.58,
        "GP 👨‍⚕️": 0.72,
        "Patient 🧑‍🦽": 0.93,
    },

    # Home spirometry and symptom diary
    # CF Specialist: remote monitoring enables early exacerbation detection
    # GP: useful for shared monitoring but requires patient training
    # Patient: reduces clinic visits but adds daily burden (Lechtzin et al., Chest 2013)
    "home_spirometry_plus_symptom_diary": {
        "CF Specialist 🩺": 0.74,
        "GP 👨‍⚕️": 0.68,
        "Patient 🧑‍🦽": 0.58,
    },

    # Mental health and fatigue management programme
    # CF Specialist: depression/anxiety prevalence 2-3x general population (Quittner et al., Thorax 2014)
    # GP: mental health screening (PHQ-9/GAD-7) and first-line treatment within GP remit
    # Patient: fatigue and psychological burden are top unmet needs (CF Trust survey 2021)
    "mental_health_and_fatigue_support": {
        "CF Specialist 🩺": 0.55,
        "GP 👨‍⚕️": 0.78,
        "Patient 🧑‍🦽": 0.95,
    },

    # Financial assistance and care access navigation
    # CF Specialist: outside clinical remit but affects adherence and long-term outcomes
    # GP: first point of contact for benefits, transport, and social prescribing
    # Patient: cost and access are major barriers to treatment (Eidt-Koch et al., PharmacoEcon 2009)
    "financial_and_access_support": {
        "CF Specialist 🩺": 0.42,
        "GP 👨‍⚕️": 0.65,
        "Patient 🧑‍🦽": 0.92,
    },

    # Individualised nutrition and enzyme optimisation plan
    # CF Specialist: malnutrition is a leading cause of morbidity; PERT dosing is complex
    # GP: routine weight/BMI monitoring feasible; dietitian referral usually required
    # Patient: eating restrictions and enzyme burden are significant daily concerns
    "nutrition_and_enzyme_optimisation": {
        "CF Specialist 🩺": 0.82,
        "GP 👨‍⚕️": 0.60,
        "Patient 🧑‍🦽": 0.74,
    },

    # CFTR genotype review for modulator eligibility reassessment
    # CF Specialist: evolving modulator approvals mean eligibility changes over time
    # GP: genotype interpretation requires specialist knowledge; low GP utility
    # Patient: hope for access to new therapies is a significant motivator
    "cftr_genotype_eligibility_review": {
        "CF Specialist 🩺": 0.88,
        "GP 👨‍⚕️": 0.38,
        "Patient 🧑‍🦽": 0.76,
    },

    # Structured airway clearance and physiotherapy optimisation
    # CF Specialist: ACT is a cornerstone of CF management (McIlwaine et al., Cochrane 2014)
    # GP: can reinforce adherence but technique guidance requires physiotherapist
    # Patient: time-consuming; patients recognise importance but burden is real
    "airway_clearance_physiotherapy_optimisation": {
        "CF Specialist 🩺": 0.84,
        "GP 👨‍⚕️": 0.55,
        "Patient 🧑‍🦽": 0.65,
    },
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class BalancedAction:
    action: str
    weighted_payoff: float   # efficiency: weighted sum of utilities
    nash_product: float      # fairness: Nash bargaining product
    min_role_utility: float  # equity: protect the worst-off role


# ── Core game-theory functions ────────────────────────────────────────────────

def compute_balanced_actions(
    role_weights: Dict[str, float],
    action_utility: Dict[str, Dict[str, float]] = ACTION_UTILITY,
    disagreement_point: float = 0.4,
    top_n: int = 2,
) -> List[BalancedAction]:
    """
    Nash-bargaining inspired multi-objective scoring:
      1. weighted_payoff  = efficiency (weighted sum of role-weighted utilities)
      2. nash_product     = fairness (product of surplus over disagreement point)
      3. min_role_utility = equity (protect the worst-off role)
    Lexicographic sort: efficiency > fairness > equity.
    """
    scored: List[BalancedAction] = []
    for action, utilities in action_utility.items():
        weighted = 0.0
        nash_prod = 1.0
        minima = 1.0
        for role, w in role_weights.items():
            u = float(utilities.get(role, 0.0))
            weighted += w * u
            nash_prod *= max(u - disagreement_point, 1e-6)
            minima = min(minima, u)
        scored.append(BalancedAction(
            action=action,
            weighted_payoff=weighted,
            nash_product=nash_prod,
            min_role_utility=minima,
        ))
    scored.sort(
        key=lambda x: (x.weighted_payoff, x.nash_product, x.min_role_utility),
        reverse=True,
    )
    return scored[:top_n]


def normalize_role_weights(
    role_weights: Dict[str, float],
    floor: float = 0.05,
) -> Dict[str, float]:
    """
    Normalize into a probability simplex with a small floor
    so no role is ever completely ignored.
    """
    clipped = {k: max(float(v), floor) for k, v in role_weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        n = max(len(role_weights), 1)
        return {k: 1.0 / n for k in role_weights}
    return {k: v / total for k, v in clipped.items()}


def unmet_need_pressure(
    coverage: Dict[str, Dict[str, bool]],
) -> Dict[str, float]:
    """
    Convert unmet-need ratio into bargaining pressure per role.
    Higher unmet ratio => higher influence weight in Nash computation.
    """
    pressure = {}
    for role, need_map in coverage.items():
        total = max(len(need_map), 1)
        unmet = sum(1 for ok in need_map.values() if not ok)
        pressure[role] = 1.0 + unmet / total
    return pressure


def update_need_coverage(
    text: str,
    role_needs: Dict[str, List[Dict]],
    coverage: Dict[str, Dict[str, bool]],
) -> None:
    """
    Keyword-scan the agent's response and mark needs as covered.
    Once covered, a need stays covered.
    """
    lower = (text or "").lower()
    for role, needs in role_needs.items():
        for item in needs:
            need_name = item["need"]
            if coverage[role].get(need_name):
                continue
            if any(kw.lower() in lower for kw in item.get("keywords", [])):
                coverage[role][need_name] = True


# ── Dynamic utility update ────────────────────────────────────────────────────

# Maps each action to keywords that signal support or relevance in dialogue.
ACTION_SIGNAL_MAP: Dict[str, List[str]] = {
    "start_or_optimize_cftr_modulator":
        ["modulator", "trikafta", "cftr", "ivacaftor", "elexacaftor", "genotype", "eligib"],
    "rapid_exacerbation_safety_net":
        ["exacerbation", "hospitaliz", "escalat", "urgent", "safety net", "red flag", "deteriorat"],
    "monthly_shared_decision_visit":
        ["shared decision", "monthly visit", "coordination", "communication", "joint review"],
    "adherence_support_with_social_worker":
        ["adherence", "social worker", "support worker", "access", "practical barrier"],
    "home_spirometry_plus_symptom_diary":
        ["spirometry", "symptom diary", "home monitor", "peak flow", "self-monitor"],
    "mental_health_and_fatigue_support":
        ["fatigue", "exhausted", "mental health", "anxiety", "depress", "psycholog", "tired", "burnout"],
    "financial_and_access_support":
        ["financial", "cost", "insurance", "afford", "transport", "access barrier", "benefit", "grant"],
    "nutrition_and_enzyme_optimisation":
        ["nutrition", "enzyme", "pert", "calori", "weight", "dietitian", "malnutrition", "vitamin"],
    "cftr_genotype_eligibility_review":
        ["genotype", "mutation", "eligib", "f508del", "class ", "variant", "genetic screen"],
    "airway_clearance_physiotherapy_optimisation":
        ["airway clearance", "physiotherap", "chest physio", "acbt", "nebuli", "pep device", "vest therapy"],
}

# Keywords that signal the speaker is opposing or dismissing an action.
OPPOSE_SIGNALS: List[str] = [
    "doesn't address", "does not address", "misses the mark", "not enough",
    "incomplete", "ignores", "fails to", "overlooked", "inadequate",
    "not feasible", "unrealistic", "won't work", "disagree", "missing",
]


def update_action_utility(
    response: str,
    speaker: str,
    action_utility: Dict[str, Dict[str, float]],
    delta: float = 0.12,
) -> Dict[str, Dict[str, float]]:
    """
    Dynamically update utility values based on dialogue content.

    Mechanism:
      - Speaker mentions keywords associated with an action AND no opposition
        signals present → utility for that speaker increases by delta.
      - Speaker mentions keywords WITH opposition signals
        → utility decreases by delta.
      - All values clamped to [0.10, 1.00].

    This closes the feedback loop between dialogue and game state:
      dialogue content → utility update → Nash equilibrium shift → new agenda
    """
    updated = copy.deepcopy(action_utility)
    lower = response.lower()
    is_opposing = any(s in lower for s in OPPOSE_SIGNALS)

    for action, keywords in ACTION_SIGNAL_MAP.items():
        if any(kw in lower for kw in keywords):
            current = updated[action].get(speaker, 0.5)
            if is_opposing:
                updated[action][speaker] = max(current - delta, 0.10)
            else:
                updated[action][speaker] = min(current + delta, 1.00)

    return updated


# ── Agenda generation (game-theory → dialogue prompt) ────────────────────────

def build_round_agenda(
    role: str,
    plans: List[BalancedAction],
    role_influence: Dict[str, float],
) -> str:
    """
    Translate the Nash-balanced plan into a role-specific agenda prompt.

    - top_action:    Nash-optimal action for this round (drives the topic)
    - second_action: runner-up (provides a contrasting option to consider)
    - urgency:       derived from role_influence; higher unmet needs = stronger voice
    """
    top_action = plans[0].action.replace("_", " ") if plans else "coordinate care"
    second_action = plans[1].action.replace("_", " ") if len(plans) > 1 else ""

    influence = role_influence.get(role, 0.33)
    urgency = (
        "strongly advocate for your position on"
        if influence > 0.4
        else "share your perspective on"
    )

    agendas = {
        "CF Specialist 🩺": (
            f"The care team's current priority is: '{top_action}'. "
            f"From clinical guidelines, {urgency} this. "
            + (f"Consider also whether '{second_action}' should be addressed." if second_action else "")
        ),
        "GP 👨‍⚕️": (
            f"The care team's current priority is: '{top_action}'. "
            f"From a primary care standpoint, {urgency} this. "
            f"Be specific about what you can deliver and where you need specialist input."
        ),
        "Patient 🧑‍🦽": (
            f"The care team is focusing on: '{top_action}'. "
            f"From your daily lived experience with CF, {urgency} this. "
            f"Say whether it addresses what actually matters to you, and why."
        ),
    }
    return agendas.get(role, f"Discuss the care team's current priority: '{top_action}'.")


# ── Logging helper ────────────────────────────────────────────────────────────

def render_balanced_plan(candidate_actions: Iterable[BalancedAction]) -> str:
    """Human-readable summary of Nash-balanced actions for logging."""
    parts = [
        f"{i}) {a.action} "
        f"(payoff={a.weighted_payoff:.3f}, nash={a.nash_product:.4f}, min_role={a.min_role_utility:.2f})"
        for i, a in enumerate(candidate_actions, 1)
    ]
    return "Balanced plan: " + "; ".join(parts) if parts else "Balanced plan: none."