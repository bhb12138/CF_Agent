from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


# A compact action space for CF care-plan simulation.
# Each action is mapped to how much it helps each role's objective (0~1).
ACTION_UTILITY: Dict[str, Dict[str, float]] = {
    "start_or_optimize_cfr_modulator": {
        "CF Specialist 🩺": 0.95,
        "GP 👨‍⚕️": 0.65,
        "Patient 🧑‍🦽": 0.78,
    },
    "home_spirometry_plus_symptom_diary": {
        "CF Specialist 🩺": 0.72,
        "GP 👨‍⚕️": 0.70,
        "Patient 🧑‍🦽": 0.60,
    },
    "monthly_shared_decision_visit": {
        "CF Specialist 🩺": 0.64,
        "GP 👨‍⚕️": 0.82,
        "Patient 🧑‍🦽": 0.88,
    },
    "adherence_support_with_social_worker": {
        "CF Specialist 🩺": 0.60,
        "GP 👨‍⚕️": 0.74,
        "Patient 🧑‍🦽": 0.92,
    },
    "rapid_exacerbation_safety_net": {
        "CF Specialist 🩺": 0.86,
        "GP 👨‍⚕️": 0.90,
        "Patient 🧑‍🦽": 0.75,
    },
}


@dataclass
class BalancedAction:
    action: str
    weighted_payoff: float
    nash_product: float
    min_role_utility: float


def build_information_focus(
    role: str,
    query: str,
    rag_snippets: Sequence[str],
    last_round_text: str,
    unmet_needs: Sequence[str] | None = None,
    patient_questions: Sequence[str] | None = None,
    top_k: int = 2,
) -> str:
    """
    Build a short "information collection focus" so each agent quickly asks/fills key gaps.
    This functions as a cheap signaling mechanism in repeated game rounds.
    """
    top_snippets = [s.strip().replace("\n", " ")[:180] for s in rag_snippets if s.strip()][:top_k]
    snippet_block = " | ".join(top_snippets) if top_snippets else "No external evidence retrieved yet."
    previous = last_round_text.strip().replace("\n", " ")[:160] if last_round_text.strip() else "No previous round signal."
    unmet_block = ", ".join(unmet_needs[:2]) if unmet_needs else "No explicit unmet need detected."
    question_block = " | ".join(patient_questions[:2]) if patient_questions else "No patient-first-visit questions provided."
    return (
        f"Coordination signal for {role}: Prioritize unresolved facts, risks, and trade-offs from the topic '{query}'. "
        f"Previous signal: {previous}. High-value evidence hints: {snippet_block}. "
        f"Unmet needs to cover now: {unmet_block}. "
        f"If relevant, answer first-visit concerns: {question_block}. "
        "Respond by clarifying one uncertain point and one actionable next step."
    )


def compute_balanced_actions(
    role_weights: Dict[str, float],
    action_utility: Dict[str, Dict[str, float]] = ACTION_UTILITY,
    disagreement_point: float = 0.4,
    top_n: int = 2,
) -> List[BalancedAction]:
    """
    Multi-objective scoring with a Nash-bargaining flavor:
    1) weighted sum = efficiency
    2) Nash product over (u_i - disagreement) = fairness pressure
    3) minimum utility = avoid sacrificing one side too much
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
        scored.append(
            BalancedAction(
                action=action,
                weighted_payoff=weighted,
                nash_product=nash_prod,
                min_role_utility=minima,
            )
        )

    # Lexicographic sort: maximize fairness-adjusted welfare then protect the worst-off role.
    scored.sort(key=lambda x: (x.weighted_payoff, x.nash_product, x.min_role_utility), reverse=True)
    return scored[:top_n]


def render_balanced_plan(candidate_actions: Iterable[BalancedAction]) -> str:
    parts = []
    for idx, item in enumerate(candidate_actions, start=1):
        parts.append(
            f"{idx}) {item.action} "
            f"(weighted={item.weighted_payoff:.3f}, nash={item.nash_product:.4f}, min_role={item.min_role_utility:.2f})"
        )
    return "Balanced plan candidates: " + "; ".join(parts) if parts else "Balanced plan candidates: none."


def normalize_role_weights(role_weights: Dict[str, float], floor: float = 0.05) -> Dict[str, float]:
    """
    Normalize role influence weights into a probability simplex.
    Keeps a small floor to avoid any role being completely ignored.
    """
    clipped = {k: max(float(v), floor) for k, v in role_weights.items()}
    total = sum(clipped.values())
    if total <= 0:
        n = max(len(role_weights), 1)
        return {k: 1.0 / n for k in role_weights}
    return {k: v / total for k, v in clipped.items()}


def update_need_coverage(text: str, role_needs: Dict[str, List[Dict]], coverage: Dict[str, Dict[str, bool]]) -> None:
    lower = (text or "").lower()
    for role, needs in role_needs.items():
        for item in needs:
            need_name = item["need"]
            keywords = item.get("keywords", [])
            if coverage[role].get(need_name):
                continue
            if any(kw.lower() in lower for kw in keywords):
                coverage[role][need_name] = True


def unmet_need_pressure(coverage: Dict[str, Dict[str, bool]]) -> Dict[str, float]:
    """
    Convert uncovered-need ratio to role weights for game-theory aggregation.
    Higher unmet ratio => higher bargaining pressure.
    """
    pressure = {}
    for role, need_map in coverage.items():
        total = max(len(need_map), 1)
        unmet = sum(1 for ok in need_map.values() if not ok)
        pressure[role] = 1.0 + unmet / total
    return pressure
