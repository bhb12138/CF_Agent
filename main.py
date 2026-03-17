import argparse, os, csv, json, time, uuid
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

    ))

    os.makedirs(args.log_dir, exist_ok=True)
    metrics_path = os.path.join(args.log_dir, "metrics.csv")
    dialogues_path = os.path.join(args.log_dir, "dialogues.jsonl")
    weights_path = os.path.join(args.log_dir, "weights.csv")

