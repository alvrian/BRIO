"""
Join candidate files that were generated in parts (gen_candidate.py --start_index / --end_index)
into one file, after checking that every part is complete.

Each part is given as  start:end:path  (end exclusive), in any order. The parts must cover
0 to --num_articles without gaps or overlaps, and every part must hold (end - start) * num_cands lines.

Example (32 candidates, train split split between Colab and a local PC):
  python merge_candidates.py --num_cands 32 --num_articles 193883 \
      --part 0:100000:train.out \
      --part 100000:193883:train.100000-193883.out \
      --out train_merged.out
"""
import argparse
import os


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", action="append", required=True, help="start:end:path (end exclusive)")
    parser.add_argument("--num_cands", type=int, required=True)
    parser.add_argument("--num_articles", type=int, required=True, help="number of articles in the split")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    parts = []
    for p in args.part:
        start, end, path = p.split(":", 2)
        parts.append((int(start), int(end), path))
    parts.sort()

    # the parts must cover 0 .. num_articles exactly
    expected_start = 0
    for start, end, path in parts:
        if start != expected_start:
            raise SystemExit("gap or overlap: a part starts at %d, expected %d (%s)" % (start, expected_start, path))
        expected_start = end
    if expected_start != args.num_articles:
        raise SystemExit("the parts end at %d, but the split has %d articles" % (expected_start, args.num_articles))

    # every part must be complete
    problems = []
    for start, end, path in parts:
        if not os.path.exists(path):
            problems.append("missing file: %s" % path)
            continue
        n = count_lines(path)
        need = (end - start) * args.num_cands
        print("%-60s articles %7d-%7d: %d of %d lines" % (path, start, end - 1, n, need))
        if n != need:
            problems.append("%s has %d lines, expected %d (not finished?)" % (path, n, need))
    if problems:
        raise SystemExit("not merged:\n  " + "\n  ".join(problems))

    with open(args.out, "w", encoding="utf-8", newline="\n") as fout:
        for start, end, path in parts:
            with open(path, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line.rstrip("\r\n") + "\n")
    total = count_lines(args.out)
    print("Merged %d lines (%d articles x %d candidates) into %s" % (total, args.num_articles, args.num_cands, args.out))


if __name__ == "__main__":
    main()
