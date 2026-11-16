"""
Check a candidate file for gibberish candidates.

A candidate that shares no word with the reference summary has ROUGE = 0.
This script counts those candidates per article and per candidate position,
so you can see whether a candidate file is healthy before preprocessing and training.

Example (32 candidates per article):
  python check_candidates.py --out_file /content/drive/MyDrive/BRIO/liputan6_candidates_32/canonical/val.out \
      --tgt_file ./liputan6_converted/canonical/val.target --num_cands 32

If the .out file covers only the first N articles (a test run with --num_samples), the first N references are used.
"""
import argparse
from collections import Counter

from compare_mt.rouge.rouge_scorer import RougeScorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_file", required=True, help="candidate file written by gen_candidate.py")
    parser.add_argument("--tgt_file", required=True, help="reference summaries of the same split")
    parser.add_argument("--num_cands", type=int, required=True, help="candidates per article in the file")
    parser.add_argument("--max_zero_rouge", type=int, default=1, help="an article is flagged when it has more than this many ROUGE = 0 candidates")
    parser.add_argument("--show", type=int, default=3, help="number of flagged articles to print")
    args = parser.parse_args()

    # same scorer settings as preprocess.py; ROUGE-1 = 0 means ROUGE-2 and ROUGE-L are 0 too
    scorer = RougeScorer(['rouge1'], use_stemmer=True)

    with open(args.out_file, encoding="utf-8") as f:
        cands = [line.rstrip("\n").lower() for line in f]
    with open(args.tgt_file, encoding="utf-8") as f:
        refs = [line.strip().lower() for line in f]

    n_articles = len(cands) // args.num_cands
    if len(cands) % args.num_cands != 0:
        print("WARNING: %d lines is not a multiple of %d, the last incomplete group is ignored" % (len(cands), args.num_cands))
    if n_articles > len(refs):
        raise ValueError("the candidate file has %d articles but the reference file has only %d lines" % (n_articles, len(refs)))

    zero_by_position = Counter()   # which candidate position is gibberish
    zero_per_article = Counter()   # how many gibberish candidates an article has
    unique_total = 0               # distinct candidates per article (duplicates carry no ranking signal)
    flagged = []
    for i in range(n_articles):
        group = cands[i * args.num_cands:(i + 1) * args.num_cands]
        zero_positions = [j for j, c in enumerate(group) if scorer.score(refs[i], c)['rouge1'].fmeasure == 0]
        for j in zero_positions:
            zero_by_position[j] += 1
        zero_per_article[len(zero_positions)] += 1
        unique_total += len(set(group))
        if len(zero_positions) > args.max_zero_rouge:
            flagged.append((i, zero_positions))

    total = n_articles * args.num_cands
    n_zero = sum(zero_by_position.values())
    print("File: %s" % args.out_file)
    print("Articles: %d, candidates per article: %d" % (n_articles, args.num_cands))
    print("Candidates with ROUGE = 0: %d of %d (%.2f%%)" % (n_zero, total, 100.0 * n_zero / max(total, 1)))
    print("Articles with more than %d such candidates: %d (%.2f%%)"
          % (args.max_zero_rouge, len(flagged), 100.0 * len(flagged) / max(n_articles, 1)))
    print("Distinct candidates per article (average): %.1f of %d" % (unique_total / max(n_articles, 1), args.num_cands))
    print("Articles by number of ROUGE = 0 candidates:", sorted(zero_per_article.items()))
    print("ROUGE = 0 candidates by position (0 = first beam group):", sorted(zero_by_position.items()))
    for i, positions in flagged[:args.show]:
        print("\nArticle %d, ROUGE = 0 at positions %s" % (i, positions))
        print("  reference:", refs[i][:150])
        print("  candidate:", cands[i * args.num_cands + positions[0]][:150])


if __name__ == "__main__":
    main()
