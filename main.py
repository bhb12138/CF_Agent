import argparse
import csv
import json
import os

from dialogue_runner import run_dialogue

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str,
                   default="How should a care plan be coordinated between a cystic fibrosis patient, GP, and CF specialist?")

    p.add_argument("--use_R", type=int, default=1)
    p.add_argument("--rule_mode", type=str, default="light")
    p.add_argument("--adaptive", type=int, default=1)
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--use_game_theory", type=int, default=1)
    p.add_argument("--scenario", type=str, default="first_visit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    results = list(run_dialogue(
        query=args.query,
        use_R=bool(args.use_R),
        rule_mode=args.rule_mode,
        adaptive_weight=bool(args.adaptive),
        rounds=args.rounds,
        alpha=args.alpha,
        use_game_theory=bool(args.use_game_theory),
        scenario_name=args.scenario,
    ))

    os.makedirs(args.log_dir, exist_ok=True)
    metrics_path = os.path.join(args.log_dir, "metrics.csv")
    dialogues_path = os.path.join(args.log_dir, "dialogues.jsonl")
    weights_path = os.path.join(args.log_dir, "weights.csv")

    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "round",
                "agent",
                "responsive",
                "rebuttal",
                "non_repetition",
                "evidence_usage",
                "stance_shift",
            ]
        )
        for row in results:
            m = row["metrics"]
            writer.writerow(
                [
                    row["round"],
                    row["agent"],
                    m["responsive"],
                    m["rebuttal"],
                    f"{m['non_repetition']:.4f}",
                    m["evidence_usage"],
                    f"{m['stance_shift']:.4f}",
                ]
            )

    with open(weights_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "agent", "wT", "wM", "wD"])
        for row in results:
            w = row["weights"]
            writer.writerow([row["round"], row["agent"], w["wT"], w["wM"], w["wD"]])

    with open(dialogues_path, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved weights to: {weights_path}")
    print(f"Saved dialogues to: {dialogues_path}")
