import agent_cf_specialist
import agent_gp
import agent_patient

from adaptive_weight import scheduler
from game_theory import (
    build_information_focus,
    compute_balanced_actions,
    normalize_role_weights,
    render_balanced_plan,
    unmet_need_pressure,
    update_need_coverage,
)
from rag_llm import rag_search
from evaluator import is_responsive, is_rebuttal, non_repetition, evidence_usage, stance_shift
from scenario_presets import get_scenario_profile


def run_dialogue(
        query: str,
        use_R: bool,
        rule_mode: str,
        adaptive_weight: bool,
        rounds: int,
        alpha: float,
        use_game_theory: bool = True,
        scenario_name: str = "first_visit",
):
    agents = [
        {"name": "CF Specialist 🩺", "module": agent_cf_specialist, "rag_name": "CFSpecialistAgent"},
        {"name": "GP 👨‍⚕️", "module": agent_gp, "rag_name": "GPAgent"},
        {"name": "Patient 🧑‍🦽", "module": agent_patient, "rag_name": "PatientAgent"}
    ]

    scenario = get_scenario_profile(scenario_name)
    patient_questions = scenario.get("patient_questions", [])
    role_needs = scenario.get("role_needs", {})

    print(f"\n--- Running dialogue ---")
    dialogue_history = [
        f"Initial topic:: {query}",
        f"Patient first-visit concerns:: {' | '.join(patient_questions)}"
    ]

    w_pool = {a["name"]: [1.0, 1.0, 1.0] for a in agents}
    # Game-theory state: needs coverage and bargaining pressure (decoupled from QTMD prompt weights).
    needs_coverage = {
        role: {item["need"]: False for item in needs}
        for role, needs in role_needs.items()
    }
    role_influence = normalize_role_weights(unmet_need_pressure(needs_coverage)) if needs_coverage else {
        a["name"]: 1.0 / len(agents) for a in agents
    }

    last_self = {a["name"]: "" for a in agents}
    utter_history = []

    num_agents = len(agents)

    results = []

    for r in range(rounds):
        print(f"\n========== round {r}  ==========")
        for agent_info in agents:
            name = agent_info["name"]
            module = agent_info["module"]
            rag_name = agent_info["rag_name"]

            print(f"\nNow it's {name} speaking...")
            # history_text = "\n".join(dialogue_history)
            if r == 0:
                history_text = query
                use_R_this = False
                prev_round_text = ""
            else:
                history_text = "\n".join(dialogue_history[-3:])
                use_R_this = use_R
                idx_prev_round_last = r * num_agents - 1  # 上一轮最后一条发言在 utter_history 里的索引
                prev_round_text = utter_history[idx_prev_round_last] if idx_prev_round_last < len(utter_history) else ""

            wT, wM, wD = w_pool[name]

            # prev_text = utter_history[-1] if utter_history else ""  # 上一位“纯文本”发言
            prev_self = last_self[name]

            retrieved = rag_search(history_text, agent=rag_name)
            rag_sents = [robj["content"] for robj in retrieved]

            info_focus = ""
            if use_game_theory and r > 0:
                unmet_needs = [
                    need for need, done in needs_coverage.get(name, {}).items() if not done
                ]
                info_focus = build_information_focus(
                    role=name,
                    query=query,
                    rag_snippets=rag_sents,
                    last_round_text=prev_round_text,
                    unmet_needs=unmet_needs,
                    patient_questions=patient_questions,
                )

            response = module.invoke(
                history=history_text,
                info_focus=info_focus,
                round_num=r,
                query=query,
                # ===== 默认使用 T/M/D，权重1 =====
                use_T=True, use_M=False, use_D=False,
                wT=wT, wM=wM, wD=wD,
                use_R=use_R_this,
                rule_mode=rule_mode,
                max_sentences=3
            )

            print(f"{name}: {response}")

            dialogue_history.append(f"{name}: {response}")
            utter_history.append(response)
            last_self[name] = response

            resp_score = is_responsive(response, prev_round_text)
            reb_score = is_rebuttal(response, prev_round_text)
            nr_score = non_repetition(response, prev_self)
            evi_score = evidence_usage(response, rag_sents)

            # 新增：立场变化
            stance_score = stance_shift(response, module.MY_TASK)

            print(
                f"[metrics] responsive={resp_score}, rebuttal={reb_score}, non_repetition={nr_score:.2f}, evidence={evi_score}, stance_score={stance_score:.2f}")

            if adaptive_weight:
                new_wT, new_wM, new_wD = scheduler(
                    round_num=r,
                    last_response=response,
                    responsive=resp_score,
                    rag_sentences=rag_sents,
                    wT=wT,
                    wM=wM,
                    wD=wD,
                    alpha=alpha
                )
                w_pool[name] = [new_wT, new_wM, new_wD]

            balanced_plan_text = ""
            if use_game_theory:
                update_need_coverage(response, role_needs, needs_coverage)
                role_influence = normalize_role_weights(unmet_need_pressure(needs_coverage))
                role_weights = role_influence
                plans = compute_balanced_actions(role_weights=role_weights, top_n=2)
                balanced_plan_text = render_balanced_plan(plans)

            results.append({
                "round": r,
                "agent": name,
                "response": response,
                "info_focus": info_focus,
                "balanced_plan": balanced_plan_text,
                "role_influence": dict(role_influence),
                "metrics": {
                    "responsive": resp_score,
                    "rebuttal": reb_score,
                    "non_repetition": nr_score,
                    "evidence_usage": evi_score,
                    "stance_shift": stance_score,
                },
                "weights": {"wT": wT, "wM": wM, "wD": wD},
            })

    return  results
