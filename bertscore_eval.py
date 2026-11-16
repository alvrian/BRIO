"""
BERTScore for the saved summaries (semantic similarity to the reference summary).

ROUGE counts shared words. BERTScore compares the contextual embeddings of the
words instead, so a summary that says the same thing with different words still
gets credit. It is computed against the reference summary, like ROUGE.

Run generate_outputs.py first. This script reads the folder it wrote
(sample.target and <name>.out) and writes into <data_dir>/bertscore/:
  bertscore.csv           precision, recall and F1 for every article and system
  bertscore_summary.txt   mean scores per system (next to ROUGE) and the paired comparison

Example:
  python bertscore_eval.py --data_dir /content/drive/MyDrive/BRIO/eval_outputs_full \
      --systems baseline brio --pair baseline brio

By default it uses multilingual BERT (bert-base-multilingual-cased), the model
the bert-score package picks for Indonesian. To use IndoBERT instead:
  --model_type indobenchmark/indobert-base-p1 --num_layers 9
"""
import argparse
import json
import math
import os
import random


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").strip() for line in f]


def compute_bertscore(candidates, references, args):
    """Returns three lists (precision, recall, F1), one value per summary, and the config hash."""
    from bert_score import score
    (P, R, F), config_hash = score(
        candidates, references,
        lang="id",
        model_type=args.model_type,     # None = the package default for Indonesian
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        verbose=True,
        return_hash=True,
    )
    return P.tolist(), R.tolist(), F.tolist(), config_hash


def sign_test_p(wins, losses):
    """Two-sided sign test: chance of a split at least this uneven if both systems were equal."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def bootstrap_ci(diffs, seed, rounds=2000):
    """95% confidence interval of the mean difference, by resampling the articles."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="the --out_dir used with generate_outputs.py")
    parser.add_argument("--systems", nargs="+", required=True, help="names used with generate_outputs.py --name")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                        help="two systems to compare on the same articles (can be given more than once)")
    parser.add_argument("--model_type", default=None, help="HuggingFace model for the embeddings (default: multilingual BERT)")
    parser.add_argument("--num_layers", type=int, default=None, help="needed only with a custom --model_type")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd

    ids = [int(x) for x in read_lines(os.path.join(args.data_dir, "sample_ids.txt")) if x]
    # the model outputs are lowercased, so lowercase the references too
    references = [t.lower() for t in read_lines(os.path.join(args.data_dir, "sample.target"))]

    out_dir = os.path.join(args.data_dir, "bertscore")
    os.makedirs(out_dir, exist_ok=True)

    tables = []
    config_hash = ""
    for name in args.systems:
        candidates = read_lines(os.path.join(args.data_dir, name + ".out"))
        if len(candidates) != len(ids):
            raise ValueError("%s.out has %d lines, expected %d (generation not finished?)" % (name, len(candidates), len(ids)))
        print("Scoring %s (%d summaries)" % (name, len(candidates)))
        P, R, F, config_hash = compute_bertscore(candidates, references, args)
        tables.append(pd.DataFrame({"id": ids, "system": name, "precision": P, "recall": R, "f1": F}))

    scores = pd.concat(tables, ignore_index=True)
    scores.to_csv(os.path.join(out_dir, "bertscore.csv"), index=False)

    # mean BERTScore per system, with the ROUGE saved by generate_outputs.py next to it
    summary = scores.groupby("system")[["precision", "recall", "f1"]].mean()
    summary.columns = ["bertscore_P", "bertscore_R", "bertscore_F1"]
    for name in args.systems:
        rouge_path = os.path.join(args.data_dir, name + ".rouge.json")
        if os.path.exists(rouge_path):
            with open(rouge_path) as f:
                rouge = json.load(f)
            for key in ["rouge1", "rouge2", "rougeL"]:
                summary.loc[name, key] = rouge[key]
    summary = summary.loc[args.systems].round(4)

    lines = ["BERTScore config: %s" % config_hash,
             "Articles: %d" % len(ids), "",
             summary.to_string()]

    # paired comparison: both systems are scored on the same articles
    wide = scores.pivot(index="id", columns="system", values="f1")
    for (a, b) in args.pair:
        diff = (wide[b] - wide[a]).tolist()
        higher = sum(1 for d in diff if d > 0)
        lower = sum(1 for d in diff if d < 0)
        low, high = bootstrap_ci(diff, args.seed)
        lines.append("")
        lines.append("BERTScore F1, %s minus %s: mean diff %+.4f, 95%% CI [%+.4f, %+.4f]"
                     % (b, a, sum(diff) / len(diff), low, high))
        lines.append("  %s higher on %d articles, lower on %d, sign test p = %.4g"
                     % (b, higher, lower, sign_test_p(higher, lower)))

    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(out_dir, "bertscore_summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
