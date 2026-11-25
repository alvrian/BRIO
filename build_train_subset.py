"""
Build a training subset from candidate files that cover only some ranges of the train split
(for example when generation was stopped early to train for a fixed number of steps).

Each part is given as  start:end:path  (end exclusive): the candidate file of articles start to end-1,
as written by gen_candidate.py. The parts may leave gaps, but must not overlap. For every part the
script takes the matching lines of <split>.source and <split>.target, so the three output files stay
aligned article by article, and writes them to --out_dir:

  <split>.source, <split>.target, <split>.out   the subset, in the order of the parts
  <split>.subset_ranges.json                    the original article ranges, for the paper

Example (32 candidates; Colab did articles 0-34,700, the local PC 132,584-193,882):
  python build_train_subset.py --src_dir ./liputan6_converted/canonical --split train --num_cands 32 \
      --part 0:34701:train.out --part 132584:193883:train.132584-193883.out \
      --out_dir ./liputan6_converted_6000/canonical --expected_articles 96000
"""
import argparse
import json
import os


def read_lines(path):
    # same line splitting as gen_candidate.py and preprocess.py (text mode, universal newlines)
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="folder with the full <split>.source and <split>.target")
    parser.add_argument("--split", default="train")
    parser.add_argument("--part", action="append", required=True, help="start:end:path (end exclusive)")
    parser.add_argument("--num_cands", type=int, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--expected_articles", type=int, default=None, help="stop with an error if the subset has a different size")
    parser.add_argument("--allow_extra", action="store_true",
                        help="a part file may hold more articles than its range; only the first (end - start) are used")
    parser.add_argument("--steps_batch", type=int, default=16, help="articles per optimizer step, only for the printed step count")
    args = parser.parse_args()

    sources = read_lines(os.path.join(args.src_dir, args.split + ".source"))
    targets = read_lines(os.path.join(args.src_dir, args.split + ".target"))
    if len(sources) != len(targets):
        raise SystemExit("%s.source has %d lines but %s.target has %d" % (args.split, len(sources), args.split, len(targets)))
    n_total = len(sources)

    parts = []
    for p in args.part:
        start, end, path = p.split(":", 2)
        parts.append((int(start), int(end), path))
    parts.sort()

    # ranges: inside the split, not empty, no overlaps
    prev_end = 0
    for start, end, path in parts:
        if not 0 <= start < end <= n_total:
            raise SystemExit("invalid range %d-%d for a split of %d articles (%s)" % (start, end, n_total, path))
        if start < prev_end:
            raise SystemExit("parts overlap: %s starts at %d, the previous part ends at %d" % (path, start, prev_end))
        prev_end = end

    # every part must hold exactly (end - start) complete groups (or more, with --allow_extra)
    problems, cand_parts = [], []
    for start, end, path in parts:
        if not os.path.exists(path):
            problems.append("missing file: %s" % path)
            continue
        lines = read_lines(path)
        need = (end - start) * args.num_cands
        print("%-70s articles %7d-%7d: %d lines, need %d" % (path, start, end - 1, len(lines), need))
        if len(lines) < need:
            problems.append("%s has %d lines, needs %d (not finished?)" % (path, len(lines), need))
        elif len(lines) > need and not args.allow_extra:
            problems.append("%s has %d lines, more than the %d of its range (wrong range? use --allow_extra to keep only the first %d articles)"
                            % (path, len(lines), need, end - start))
        if any(not l.strip() for l in lines[:need]):
            problems.append("%s has empty candidate lines" % path)
        cand_parts.append(lines[:need])
    if problems:
        raise SystemExit("subset not written:\n  " + "\n  ".join(problems))

    n_subset = sum(end - start for start, end, _ in parts)
    if args.expected_articles is not None and n_subset != args.expected_articles:
        raise SystemExit("the parts hold %d articles, expected %d" % (n_subset, args.expected_articles))

    os.makedirs(args.out_dir, exist_ok=True)
    out = {ext: open(os.path.join(args.out_dir, "%s.%s" % (args.split, ext)), "w", encoding="utf-8", newline="\n")
           for ext in ["source", "target", "out"]}
    try:
        for (start, end, _), cands in zip(parts, cand_parts):
            for i in range(start, end):
                out["source"].write(sources[i] + "\n")
                out["target"].write(targets[i] + "\n")
            for line in cands:
                out["out"].write(line + "\n")
    finally:
        for f in out.values():
            f.close()

    ranges = {"split": args.split, "num_cands": args.num_cands, "articles": n_subset, "split_articles": n_total,
              "ranges": [{"start": s, "end_exclusive": e, "articles": e - s, "file": os.path.basename(p)} for s, e, p in parts]}
    with open(os.path.join(args.out_dir, args.split + ".subset_ranges.json"), "w", encoding="utf-8") as f:
        json.dump(ranges, f, indent=1)

    # final alignment check on what was written
    counts = {ext: len(read_lines(os.path.join(args.out_dir, "%s.%s" % (args.split, ext)))) for ext in ["source", "target", "out"]}
    if not counts["source"] == counts["target"] == n_subset or counts["out"] != n_subset * args.num_cands:
        raise SystemExit("written files do not line up: %s" % counts)
    print("Wrote %d articles (%d candidate lines) to %s" % (n_subset, counts["out"], args.out_dir))
    print("Ranges: " + ", ".join("%d-%d" % (s, e - 1) for s, e, _ in parts))
    print("One epoch at %d articles per step: %d full optimizer steps (%d articles left over)"
          % (args.steps_batch, n_subset // args.steps_batch, n_subset % args.steps_batch))


if __name__ == "__main__":
    main()
