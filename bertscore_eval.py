"""
ROUGE and BERTScore for the saved summaries, on one or more test partitions.

ROUGE counts shared words. BERTScore compares the contextual embeddings of the
words instead, so a summary that says the same thing with different words still
gets credit. Both are computed against the reference summary.

Run generate_outputs.py first, once per partition, each into its own folder
(sample_ids.txt, sample.target and <name>.out). Give each folder with --data_dir,
optionally labelled:

  python bertscore_eval.py \
      --data_dir canonical=/content/drive/MyDrive/BRIO/eval_outputs_full \
      --data_dir xtreme=/content/drive/MyDrive/BRIO/eval_outputs_xtreme \
      --systems baseline brio_1_1_seed1 brio_1_1_seed2 \
      --pair baseline brio_1_1_seed1 --pair baseline brio_1_1_seed2

Written into each <data_dir>/bertscore/ (<model> = mbert by default, or the --model_type name):
  rouge_scores.csv           ROUGE-1/2/L F1 per article and system
  bertscore_<model>.csv      BERTScore P/R/F1 per article and system
  summary_<model>.txt        the tables and comparisons for this partition only
Written into --report_dir (default: the folder above the first --data_dir):
  report_<model>.txt         all partitions: per system, per setting (mean +- SD over seeds), comparisons
  results_<model>.csv        one row per partition and system
  results_<model>_by_setting.csv   one row per partition and setting (seeds averaged)

Scores already saved in those csv files are reused, so a rerun only scores the
systems that are new (use --recompute to score everything again).

ROUGE is computed as in compare_and_contrast.py / generate_outputs.py: lowercased,
NLTK word and sentence tokenization, compare-mt's ROUGE without stemming.
BERTScore uses multilingual BERT by default (the bert-score package's choice for
Indonesian). For IndoBERT: --model_type indobenchmark/indobert-base-p1 --num_layers 9
"""
import argparse
import math
import os
import re

import numpy as np
import pandas as pd

ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]
BERT_KEYS = ["bertscore_P", "bertscore_R", "bertscore_F1"]


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").strip() for line in f]


def model_slug(args):
    return "mbert" if not args.model_type else args.model_type.rstrip("/").split("/")[-1]


def split_setting(name):
    """'brio_1_1_seed2_12cand' -> ('brio_1_1_12cand', 2); a name without _seedN is its own setting."""
    m = re.search(r"_seed(\d+)", name)
    if not m:
        return name, None
    return name[:m.start()] + name[m.end():], int(m.group(1))


# ------------------------------------------------------------------ scoring


def compute_rouge(candidates, references):
    """ROUGE-1/2/L F1 per summary, same preprocessing as compare_and_contrast.compute_rouge_scores."""
    from compare_mt.rouge.rouge_scorer import RougeScorer
    from nltk import sent_tokenize, word_tokenize
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=False)

    def prep(x):
        return "\n".join(sent_tokenize(" ".join(word_tokenize(x.strip().lower()))))

    out = {k: [] for k in ROUGE_KEYS}
    for hyp, ref in zip(candidates, references):
        s = scorer.score(prep(ref), prep(hyp))
        out["rouge1"].append(s["rouge1"].fmeasure)
        out["rouge2"].append(s["rouge2"].fmeasure)
        out["rougeL"].append(s["rougeLsum"].fmeasure)
    return out


def compute_bertscore(candidates, references, args):
    """BERTScore P/R/F1 per summary and the bert-score config hash."""
    from bert_score import score
    (P, R, F), config_hash = score(
        candidates, references,
        lang="id",
        model_type=args.model_type,     # None = the package default for Indonesian
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        verbose=False,
        return_hash=True,
    )
    return {"bertscore_P": P.tolist(), "bertscore_R": R.tolist(), "bertscore_F1": F.tolist()}, config_hash


def load_or_score(csv_path, keys, ids, systems, outputs, score_fn, recompute):
    """Per-article scores for every system: reuse what csv_path holds, score the rest, save."""
    saved = None
    if os.path.exists(csv_path) and not recompute:
        saved = pd.read_csv(csv_path)
        # keep only systems scored on exactly these articles
        ok = [s for s, g in saved.groupby("system") if len(g) == len(ids) and set(g["id"]) == set(ids)]
        saved = saved[saved["system"].isin(ok)]
    done = set(saved["system"]) if saved is not None else set()
    tables = [saved] if saved is not None and len(saved) else []
    extra = None
    for name in systems:
        if name in done:
            continue
        print("  scoring %s" % name, flush=True)
        values, extra = score_fn(outputs[name])
        t = pd.DataFrame({"id": ids, "system": name})
        for k in keys:
            t[k] = values[k]
        tables.append(t)
    scores = pd.concat(tables, ignore_index=True)
    scores.to_csv(csv_path, index=False)
    reused = [s for s in systems if s in done]
    if reused:
        print("  reused saved scores for %d system(s) from %s" % (len(reused), csv_path))
    return scores[scores["system"].isin(systems)], extra


# ------------------------------------------------------------------ statistics


def sign_test_p(wins, losses):
    """Two-sided exact sign test, computed in log space so it works for thousands of articles."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    top = max(logs)
    return min(1.0, 2.0 * math.exp(top) * sum(math.exp(x - top) for x in logs))


def bootstrap_ci(diffs, seed, rounds=2000, chunk=200):
    """95% confidence interval of the mean difference, by resampling the articles."""
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs, dtype=float)
    means = []
    for start in range(0, rounds, chunk):
        idx = rng.integers(0, len(d), size=(min(chunk, rounds - start), len(d)))
        means.append(d[idx].mean(axis=1))
    means = np.sort(np.concatenate(means))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def p_text(p):
    return "p < 1e-16" if p < 1e-16 else "p = %.4g" % p


# ------------------------------------------------------------------ one partition


def evaluate_partition(label, data_dir, args, slug):
    print("\n===== %s: %s" % (label, data_dir), flush=True)
    ids = [int(x) for x in read_lines(os.path.join(data_dir, "sample_ids.txt")) if x]
    references = [t.lower() for t in read_lines(os.path.join(data_dir, "sample.target"))]
    if len(references) != len(ids):
        raise ValueError("%s: sample.target has %d lines, sample_ids.txt %d" % (data_dir, len(references), len(ids)))

    outputs, missing = {}, []
    for name in args.systems:
        path = os.path.join(data_dir, name + ".out")
        if not os.path.exists(path):
            missing.append(name + " (no file)")
            continue
        lines = read_lines(path)
        if len(lines) != len(ids):
            missing.append("%s (%d of %d summaries)" % (name, len(lines), len(ids)))
            continue
        outputs[name] = lines
    if missing:
        msg = "%s: not complete for %s" % (label, ", ".join(missing))
        if not args.skip_missing:
            raise ValueError(msg + " (finish generation first, or use --skip_missing)")
        print("  WARNING: skipped " + ", ".join(missing))
    systems = [s for s in args.systems if s in outputs]

    out_dir = os.path.join(data_dir, "bertscore")
    os.makedirs(out_dir, exist_ok=True)
    print("  ROUGE", flush=True)
    rouge, _ = load_or_score(os.path.join(out_dir, "rouge_scores.csv"), ROUGE_KEYS, ids, systems, outputs,
                             lambda c: (compute_rouge(c, references), None), args.recompute)
    print("  BERTScore (%s)" % slug, flush=True)
    bert, config_hash = load_or_score(os.path.join(out_dir, "bertscore_%s.csv" % slug), BERT_KEYS, ids, systems,
                                      outputs, lambda c: compute_bertscore(c, references, args), args.recompute)
    if config_hash:
        with open(os.path.join(out_dir, "bertscore_%s_config.txt" % slug), "w", encoding="utf-8") as f:
            f.write(config_hash + "\n")
    config_path = os.path.join(out_dir, "bertscore_%s_config.txt" % slug)
    config_hash = open(config_path, encoding="utf-8").read().strip() if os.path.exists(config_path) else "unknown"

    per_article = rouge.merge(bert, on=["id", "system"])
    words = pd.DataFrame({"system": systems,
                          "words": [float(np.mean([len(s.split()) for s in outputs[n]])) for n in systems]})
    table = per_article.groupby("system")[ROUGE_KEYS + BERT_KEYS].mean().reset_index().merge(words, on="system")
    table["partition"] = label
    table["articles"] = len(ids)
    table["setting"], table["seed"] = zip(*[split_setting(n) for n in table["system"]])
    table = table.set_index("system").loc[systems].reset_index()

    # paired comparisons: both systems scored on the same articles
    comparisons = []
    for a, b in args.pair:
        if a not in systems or b not in systems:
            continue
        row = {"partition": label, "a": a, "b": b}
        for k in ["rouge1", "rouge2", "rougeL", "bertscore_F1"]:
            wide = per_article.pivot(index="id", columns="system", values=k)
            diff = (wide[b] - wide[a]).to_numpy()
            low, high = bootstrap_ci(diff, args.seed)
            higher, lower = int((diff > 0).sum()), int((diff < 0).sum())
            row[k] = (float(diff.mean()), float(low), float(high), higher, lower, sign_test_p(higher, lower))
        comparisons.append(row)
    return table, comparisons, config_hash


# ------------------------------------------------------------------ report


def fmt_table(df, cols, digits=4):
    show = df[cols].copy()
    for c in cols:
        if c in ROUGE_KEYS + BERT_KEYS:
            show[c] = show[c].map(lambda v: "%.*f" % (digits, v))
        elif c == "words":
            show[c] = show[c].map(lambda v: "%.1f" % v)
    return show.to_string(index=False)


def by_setting(table):
    rows = []
    for (part, setting), g in table.groupby(["partition", "setting"], sort=False):
        row = {"partition": part, "setting": setting, "seeds": len(g)}
        for k in ROUGE_KEYS + BERT_KEYS + ["words"]:
            row[k + "_mean"] = g[k].mean()
            row[k + "_sd"] = g[k].std(ddof=1) if len(g) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def fmt_setting(s):
    lines = []
    for _, r in s.iterrows():
        cells = []
        for k in ROUGE_KEYS + BERT_KEYS:
            m, sd = r[k + "_mean"], r[k + "_sd"]
            cells.append("%.4f" % m + ("" if math.isnan(sd) else " +- %.4f" % sd))
        w, wsd = r["words_mean"], r["words_sd"]
        cells.append("%.1f" % w + ("" if math.isnan(wsd) else " +- %.1f" % wsd))
        lines.append([r["setting"], str(r["seeds"])] + cells)
    head = ["setting", "seeds"] + ROUGE_KEYS + BERT_KEYS + ["words"]
    widths = [max(len(head[i]), *(len(l[i]) for l in lines)) for i in range(len(head))]
    out = ["  ".join(h.rjust(w) for h, w in zip(head, widths))]
    out += ["  ".join(c.rjust(w) for c, w in zip(l, widths)) for l in lines]
    return "\n".join(out)


def fmt_comparisons(comps):
    lines = []
    for c in comps:
        lines.append("%s minus %s:" % (c["b"], c["a"]))
        for k in ["rouge1", "rouge2", "rougeL", "bertscore_F1"]:
            mean, low, high, hi, lo, p = c[k]
            lines.append("  %-13s %+.4f  95%% CI [%+.4f, %+.4f]  higher on %d, lower on %d articles, sign test %s"
                         % (k, mean, low, high, hi, lo, p_text(p)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", action="append", required=True,
                        help="folder written by generate_outputs.py, as label=path or path; give once per partition")
    parser.add_argument("--systems", nargs="+", required=True, help="names used with generate_outputs.py --name")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                        help="two systems to compare on the same articles (can be given more than once)")
    parser.add_argument("--model_type", default=None, help="HuggingFace model for the embeddings (default: multilingual BERT)")
    parser.add_argument("--num_layers", type=int, default=None, help="needed only with a custom --model_type")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_dir", default=None, help="where the combined report goes (default: parent of the first --data_dir)")
    parser.add_argument("--recompute", action="store_true", help="score every system again instead of reusing saved scores")
    parser.add_argument("--skip_missing", action="store_true", help="leave out systems whose summaries are missing or incomplete")
    args = parser.parse_args()

    slug = model_slug(args)
    partitions = []
    for d in args.data_dir:
        if "=" in d and not os.path.exists(d):
            label, path = d.split("=", 1)
        else:
            label, path = os.path.basename(os.path.normpath(d)), d
        partitions.append((label, path))

    tables, comps, hashes = [], [], {}
    for label, path in partitions:
        t, c, h = evaluate_partition(label, path, args, slug)
        tables.append(t)
        comps.extend(c)
        hashes[label] = h
        # per-partition summary next to the scores
        text = "\n".join(["BERTScore config: %s" % h, "Partition: %s, %d articles" % (label, t["articles"].iloc[0]), "",
                          fmt_table(t, ["system"] + ROUGE_KEYS + BERT_KEYS + ["words"]), "",
                          fmt_comparisons([x for x in c])])
        with open(os.path.join(path, "bertscore", "summary_%s.txt" % slug), "w", encoding="utf-8") as f:
            f.write(text + "\n")

    table = pd.concat(tables, ignore_index=True)
    settings = by_setting(table)

    report_dir = args.report_dir or os.path.dirname(os.path.normpath(partitions[0][1]))
    os.makedirs(report_dir, exist_ok=True)
    cols = ["partition", "system", "setting", "seed", "articles"] + ROUGE_KEYS + BERT_KEYS + ["words"]
    table[cols].to_csv(os.path.join(report_dir, "results_%s.csv" % slug), index=False)
    settings.to_csv(os.path.join(report_dir, "results_%s_by_setting.csv" % slug), index=False)

    out = ["ROUGE (F1) and BERTScore, embedding model: %s" % slug]
    for label, _ in partitions:
        t = table[table["partition"] == label]
        out += ["", "=" * 100, "Partition: %s (%d articles)   BERTScore config: %s" % (label, t["articles"].iloc[0], hashes[label]),
                "=" * 100, "", "Per system:", fmt_table(t, ["system"] + ROUGE_KEYS + BERT_KEYS + ["words"]),
                "", "Per setting (mean +- SD over seeds):", fmt_setting(settings[settings["partition"] == label])]
        pc = [c for c in comps if c["partition"] == label]
        if pc:
            out += ["", "Paired comparisons:", fmt_comparisons(pc)]
    out += ["", "BERTScore values are only comparable within one embedding model."]
    text = "\n".join(out)
    print("\n" + text)
    path = os.path.join(report_dir, "report_%s.txt" % slug)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\nSaved: %s, results_%s.csv, results_%s_by_setting.csv in %s" % (os.path.basename(path), slug, slug, report_dir))


if __name__ == "__main__":
    main()
