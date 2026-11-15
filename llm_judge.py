"""
LLM-as-a-judge evaluation of the saved summaries, using the Gemini API.

Run generate_outputs.py first. This script reads the folder it wrote
(sample.source, sample.target, <name>.out) and asks Gemini to:

  1. SCORE every summary on relevance, consistency, fluency and coherence (1-5),
     reading the article, and list the statements the article does not support.
  2. COMPARE two systems head to head on the same article (blind, random A/B order).

Everything is saved after each answer in <data_dir>/judge/, so when the daily
quota runs out you just run the same command again the next day and it continues.

Files written to <data_dir>/judge/:
  scores.jsonl / pairwise.jsonl   raw answers (also the resume cache, do not delete)
  blocked.jsonl                   articles refused by Gemini's content filter (excluded for all systems)
  scores.csv / pairwise.csv       the same as tables
  summary.txt                     mean scores per system and the win counts
  human_eval_sheet.csv            blind sheet for human raters (with --human_n)
  human_eval_key.csv              which system each row of the sheet is (keep hidden from raters)

Example:
  export GEMINI_API_KEY=...
  python llm_judge.py --data_dir /content/drive/MyDrive/BRIO_eval_outputs \
      --systems baseline brio --pair baseline brio --include_reference

Requests needed = articles x systems (scoring) + articles x pairs (comparison).
100 articles, baseline + brio + reference, one pair = 400 requests.
Use --dry_run to see the number of requests and the prompt without calling the API.
"""
import argparse
import csv
import json
import math
import os
import random
import re
import time

import requests

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
CRITERIA = ["relevance", "consistency", "fluency", "coherence"]

SCORE_PROMPT = """You are an expert evaluator of Indonesian news summaries.

You will be given one news article in Indonesian and one summary of that article.
Rate the summary on four criteria. Each score is an integer from 1 (worst) to 5 (best).

Criteria:
- relevance: the summary contains the important information of the article and only that. Lower the score for missing key points and for unimportant or repeated details.
- consistency: every statement in the summary is supported by the article. Lower the score for anything that contradicts the article or cannot be found in it (wrong names, numbers, places, who did what to whom, invented details). 5 = no unsupported statement, 3 = one unsupported detail that is not the main point, 1 = the main statement is wrong or most of the summary is unsupported.
- fluency: each sentence is grammatical, natural Indonesian.
- coherence: the sentences fit together as one well-organised summary.

Rules:
- The summary is lowercased and has spaces around punctuation because of preprocessing. Do not lower any score for casing or spacing.
- Judge only against the article. Do not use outside knowledge.
- The criteria are independent: a fluent summary can still be inconsistent.

Evaluation steps:
1. Read the article and identify its main points.
2. Check each statement of the summary against the article and collect the unsupported ones.
3. Give the four scores.

Article:
<article>
{article}
</article>

Summary:
<summary>
{summary}
</summary>

Answer with JSON only, in this exact form:
{{"unsupported_claims": ["..."], "reasoning": "two or three sentences", "relevance": 1-5, "consistency": 1-5, "fluency": 1-5, "coherence": 1-5}}
Use an empty list for unsupported_claims when there are none."""

PAIR_PROMPT = """You are an expert evaluator of Indonesian news summaries.

You will be given one news article in Indonesian and two summaries of it, A and B.
Decide which summary is better overall. Weigh the criteria in this order:
1. consistency: statements must be supported by the article (most important),
2. relevance: the important information is covered, without unimportant details,
3. fluency and coherence.

Rules:
- Both summaries are lowercased and have spaces around punctuation because of preprocessing. Ignore casing and spacing.
- Judge only against the article. Do not use outside knowledge.
- Answer "tie" only when there is no meaningful difference in quality.

Article:
<article>
{article}
</article>

Summary A:
<summary_a>
{summary_a}
</summary_a>

Summary B:
<summary_b>
{summary_b}
</summary_b>

Answer with JSON only, in this exact form:
{{"reasoning": "two or three sentences", "winner": "A" or "B" or "tie"}}"""


class StopRun(Exception):
    """Raised when the request budget or the API quota is used up."""


class BadAnswer(Exception):
    """Raised when the model's answer cannot be used."""


class Blocked(Exception):
    """Raised when Gemini's content filter refuses the article. Retrying does not help."""


REQUESTS_MADE = 0


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").strip() for line in f]


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_gemini(prompt, args):
    """Send one prompt and return the text of the answer. Waits and retries on rate limits."""
    global REQUESTS_MADE
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    headers = {"x-goog-api-key": args.api_key, "Content-Type": "application/json"}
    waits = [30, 60, 120]
    for attempt in range(len(waits) + 1):
        if REQUESTS_MADE >= args.max_requests:
            raise StopRun("request budget of %d reached" % args.max_requests)
        REQUESTS_MADE += 1
        try:
            r = requests.post(API_URL.format(model=args.model), headers=headers, json=body, timeout=120)
            status, text = r.status_code, r.text
        except requests.RequestException as e:
            status, text = 0, str(e)

        if status == 200:
            time.sleep(args.sleep)
            try:
                data = r.json()
            except ValueError:
                raise BadAnswer("response is not JSON: %s" % text[:200])
            # the content filter refused the prompt (for example a crime news article)
            reason = (data.get("promptFeedback") or {}).get("blockReason")
            if reason:
                raise Blocked("prompt blocked: %s" % reason)
            candidate = (data.get("candidates") or [{}])[0]
            try:
                return candidate["content"]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                finish = candidate.get("finishReason", "")
                if finish in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION", "SPII"):
                    raise Blocked("answer blocked: %s" % finish)
                raise BadAnswer("no text in the response: %s" % text[:200])
        if status in (0, 429, 500, 502, 503, 504) and attempt < len(waits):
            print("  HTTP %s, waiting %ds then retrying" % (status, waits[attempt]))
            time.sleep(waits[attempt])
            continue
        if status == 429:
            raise StopRun("rate limit still hit after retries (daily quota is probably used up)")
        raise RuntimeError("Gemini API error %s: %s" % (status, text[:500]))


def parse_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise BadAnswer("not JSON: %s" % text[:200])
        try:
            return json.loads(match.group(0))
        except ValueError:
            raise BadAnswer("not JSON: %s" % text[:200])


def ask(prompt, args, check):
    """Call the model, parse and check the answer. One extra try if the answer is unusable."""
    last_error = None
    for _ in range(2):
        try:
            answer = parse_json(call_gemini(prompt, args))
            if not isinstance(answer, dict):
                raise BadAnswer("answer is not a JSON object")
            return check(answer)
        except BadAnswer as e:
            last_error = e
            print("  unusable answer (%s), trying once more" % e)
    raise last_error


def check_scores(answer):
    out = {}
    for c in CRITERIA:
        try:
            value = int(answer[c])
        except (KeyError, TypeError, ValueError):
            raise BadAnswer("missing score for %s" % c)
        if value < 1 or value > 5:
            raise BadAnswer("%s score out of range: %s" % (c, value))
        out[c] = value
    claims = answer.get("unsupported_claims") or []
    if not isinstance(claims, list):
        claims = [str(claims)]
    out["unsupported_claims"] = [str(x) for x in claims]
    out["reasoning"] = str(answer.get("reasoning", ""))
    return out


def check_winner(answer):
    winner = str(answer.get("winner", "")).strip().lower()
    if winner not in ("a", "b", "tie"):
        raise BadAnswer("winner must be A, B or tie, got %r" % winner)
    return {"winner": winner, "reasoning": str(answer.get("reasoning", ""))}


def load_data(args):
    """Read the folder written by generate_outputs.py into one list of dicts (one per article)."""
    ids = [int(x) for x in read_lines(os.path.join(args.data_dir, "sample_ids.txt")) if x]
    sources = read_lines(os.path.join(args.data_dir, "sample.source"))
    targets = read_lines(os.path.join(args.data_dir, "sample.target"))
    if not (len(ids) == len(sources) == len(targets)):
        raise ValueError("sample_ids.txt, sample.source and sample.target have different lengths")

    summaries = {}
    for name in args.systems:
        lines = read_lines(os.path.join(args.data_dir, name + ".out"))
        if len(lines) != len(ids):
            raise ValueError("%s.out has %d lines, expected %d (generation not finished?)" % (name, len(lines), len(ids)))
        summaries[name] = lines
    if args.include_reference:
        # lowercase the reference so it looks like the model outputs
        summaries["reference"] = [t.lower() for t in targets]

    rows = []
    for k in range(len(ids)):
        rows.append({"id": ids[k], "article": sources[k],
                     "summaries": dict((name, summaries[name][k]) for name in summaries)})
    if args.limit > 0:
        rows = rows[:args.limit]
    return rows, list(summaries.keys())


def judge_article(args, row, systems, pairs, scores, pairwise, scores_path, pairwise_path):
    """All requests for one article: score each system, then the head-to-head comparisons."""
    for system in systems:
        key = (row["id"], system)
        if key in scores:
            continue
        summary = row["summaries"][system]
        # identical summary from another system on the same article: reuse its scores
        same = [scores[(row["id"], s)] for s in systems
                if (row["id"], s) in scores and scores[(row["id"], s)]["summary"] == summary]
        try:
            if same:
                result = dict((c, same[0][c]) for c in CRITERIA + ["unsupported_claims", "reasoning"])
            else:
                result = ask(SCORE_PROMPT.format(article=row["article"], summary=summary), args, check_scores)
        except BadAnswer as e:
            print("  skipped article %d / %s: %s (will be retried on the next run)" % (row["id"], system, e))
            continue
        record = {"id": row["id"], "system": system, "summary": summary, "judge_model": args.model}
        record.update(result)
        scores[key] = record
        append_jsonl(scores_path, record)

    for (a, b) in pairs:
        key = (row["id"], a, b)
        if key in pairwise:
            continue
        sum_a, sum_b = row["summaries"][a], row["summaries"][b]
        record = {"id": row["id"], "system_1": a, "system_2": b, "judge_model": args.model}
        if sum_a == sum_b:
            record.update({"winner": "tie", "shown_first": "", "reasoning": "identical summaries, no request sent"})
        else:
            # random order per article so the judge cannot learn which system is A
            swap = random.Random(args.seed * 1000003 + row["id"]).random() < 0.5
            first, second = (b, a) if swap else (a, b)
            prompt = PAIR_PROMPT.format(article=row["article"], summary_a=row["summaries"][first],
                                        summary_b=row["summaries"][second])
            try:
                result = ask(prompt, args, check_winner)
            except BadAnswer as e:
                print("  skipped comparison on article %d: %s (will be retried on the next run)" % (row["id"], e))
                continue
            winner = {"a": first, "b": second, "tie": "tie"}[result["winner"]]
            record.update({"winner": winner, "shown_first": first, "reasoning": result["reasoning"]})
        pairwise[key] = record
        append_jsonl(pairwise_path, record)


def run_judging(args, rows, systems, pairs, scores_path, pairwise_path, blocked_path):
    scores = dict(((r["id"], r["system"]), r) for r in load_jsonl(scores_path))
    pairwise = dict(((r["id"], r["system_1"], r["system_2"]), r) for r in load_jsonl(pairwise_path))

    # articles that Gemini's content filter refused on an earlier run are not sent again
    blocked = dict((r["id"], r) for r in load_jsonl(blocked_path))
    rows = [row for row in rows if row["id"] not in blocked]
    if blocked:
        print("Skipping %d article(s) blocked by Gemini's content filter: %s" % (len(blocked), sorted(blocked)))

    todo_scores = sum(1 for row in rows for s in systems if (row["id"], s) not in scores)
    todo_pairs = sum(1 for row in rows for (a, b) in pairs if (row["id"], a, b) not in pairwise)
    print("Articles: %d | systems: %s | pairs: %s" % (len(rows), systems, pairs))
    print("Still to do: %d scoring + %d comparison = at most %d requests (budget this run: %d)"
          % (todo_scores, todo_pairs, todo_scores + todo_pairs, args.max_requests))
    if args.dry_run:
        row = rows[0]
        print("\n----- example scoring prompt -----\n")
        print(SCORE_PROMPT.format(article=row["article"], summary=row["summaries"][systems[0]]))
        return

    try:
        for n, row in enumerate(rows):
            try:
                judge_article(args, row, systems, pairs, scores, pairwise, scores_path, pairwise_path)
            except Blocked as e:
                # the filter reacts to the article itself, so drop the whole article for every system
                print("  article %d blocked by Gemini's content filter (%s), excluded from the evaluation" % (row["id"], e))
                append_jsonl(blocked_path, {"id": row["id"], "reason": str(e), "judge_model": args.model})
            if (n + 1) % 10 == 0:
                print("%d / %d articles done, %d requests used" % (n + 1, len(rows), REQUESTS_MADE))
        print("All done. Requests used in this run: %d" % REQUESTS_MADE)
    except StopRun as e:
        print("\nStopped: %s. Progress is saved, run the same command again later to continue." % e)


def sign_test_p(wins, losses):
    """Two-sided sign test: chance of a split at least this uneven if both systems were equal."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, p)


def summarise(args, systems, pairs, out_dir, scores_path, pairwise_path, blocked_path):
    import pandas as pd

    lines = []
    # blocked articles are left out for every system, so all systems are compared on the same articles
    blocked_ids = set(r["id"] for r in load_jsonl(blocked_path))
    if blocked_ids:
        lines.append("Excluded: %d article(s) refused by the judge's content filter, ids %s\n"
                     % (len(blocked_ids), sorted(blocked_ids)))
    score_rows = [r for r in load_jsonl(scores_path) if r["id"] not in blocked_ids]
    if score_rows:
        scores = pd.DataFrame(score_rows)
        scores["n_unsupported"] = scores["unsupported_claims"].apply(len)
        scores["unsupported_claims"] = scores["unsupported_claims"].apply(lambda x: " | ".join(x))
        scores.to_csv(os.path.join(out_dir, "scores.csv"), index=False, encoding="utf-8-sig")

        # mean score per system for each criterion
        table = scores.groupby("system")[CRITERIA].mean().round(3)
        table["n"] = scores.groupby("system")["id"].count()
        table["pct_with_unsupported_claim"] = (scores.groupby("system")["n_unsupported"]
                                               .apply(lambda x: (x > 0).mean() * 100).round(1))
        lines.append("Mean judge scores (1-5), judge model: %s" % args.model)
        lines.append(table.to_string())

        # paired difference on the articles that both systems have scores for
        for (a, b) in pairs:
            wide = scores[scores["system"].isin([a, b])].pivot(index="id", columns="system", values=CRITERIA).dropna()
            if len(wide) == 0:
                continue
            lines.append("\nPaired difference %s minus %s (n=%d articles)" % (b, a, len(wide)))
            for c in CRITERIA:
                diff = wide[c][b] - wide[c][a]
                text = "  %-12s mean diff %+.3f" % (c, diff.mean())
                try:
                    from scipy.stats import wilcoxon
                    if (diff != 0).any():
                        text += "   Wilcoxon p = %.4f" % wilcoxon(wide[c][b], wide[c][a]).pvalue
                except ImportError:
                    pass
                lines.append(text)

    pair_rows = [r for r in load_jsonl(pairwise_path) if r["id"] not in blocked_ids]
    if pair_rows:
        pairwise = pd.DataFrame(pair_rows)
        pairwise.to_csv(os.path.join(out_dir, "pairwise.csv"), index=False, encoding="utf-8-sig")
        for (a, b) in pairs:
            part = pairwise[(pairwise["system_1"] == a) & (pairwise["system_2"] == b)]
            if len(part) == 0:
                continue
            wins_a = int((part["winner"] == a).sum())
            wins_b = int((part["winner"] == b).sum())
            ties = int((part["winner"] == "tie").sum())
            lines.append("\nHead to head (n=%d): %s wins %d, %s wins %d, ties %d   sign test p = %.4f"
                         % (len(part), a, wins_a, b, wins_b, ties, sign_test_p(wins_a, wins_b)))

    if lines:
        text = "\n".join(lines)
        print("\n" + text)
        with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(text + "\n")


def write_human_sheet(args, rows, systems, out_dir):
    """Blind rating sheet for human raters, on a random subset of the same articles."""
    rng = random.Random(args.seed)
    picked = rng.sample(rows, min(args.human_n, len(rows)))
    sheet, key = [], []
    for row in picked:
        order = list(systems)
        rng.shuffle(order)
        for system in order:
            item_id = "H%04d" % (len(sheet) + 1)
            sheet.append([item_id, row["article"], row["summaries"][system], "", "", "", ""])
            key.append([item_id, row["id"], system])
    with open(os.path.join(out_dir, "human_eval_sheet.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "article", "summary"] + CRITERIA)
        writer.writerows(sheet)
    with open(os.path.join(out_dir, "human_eval_key.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["item_id", "id", "system"])
        writer.writerows(key)
    print("Human rating sheet: %d rows -> %s" % (len(sheet), os.path.join(out_dir, "human_eval_sheet.csv")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="the --out_dir used with generate_outputs.py")
    parser.add_argument("--systems", nargs="+", required=True, help="names used with generate_outputs.py --name")
    parser.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                        help="two systems to compare head to head (can be given more than once)")
    parser.add_argument("--include_reference", action="store_true", help="also score the gold reference summaries")
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--max_requests", type=int, default=450, help="stop after this many API requests in one run")
    parser.add_argument("--sleep", type=float, default=6.5, help="seconds between requests (free tier per-minute limit)")
    parser.add_argument("--limit", type=int, default=0, help="only use the first N articles (for a quick test)")
    parser.add_argument("--human_n", type=int, default=0, help="also write a blind human rating sheet for N articles")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="show request count and an example prompt, no API call")
    parser.add_argument("--summary_only", action="store_true", help="only rebuild the tables from saved answers")
    args = parser.parse_args()

    rows, systems = load_data(args)
    pairs = [tuple(p) for p in args.pair]
    for (a, b) in pairs:
        if a not in systems or b not in systems:
            parser.error("--pair %s %s: both names must be in --systems" % (a, b))

    out_dir = os.path.join(args.data_dir, "judge")
    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.jsonl")
    pairwise_path = os.path.join(out_dir, "pairwise.jsonl")
    blocked_path = os.path.join(out_dir, "blocked.jsonl")

    if args.human_n > 0:
        write_human_sheet(args, rows, systems, out_dir)

    if not args.summary_only:
        args.api_key = os.environ.get("GEMINI_API_KEY", "")
        if not args.api_key and not args.dry_run:
            parser.error("set the GEMINI_API_KEY environment variable first")
        run_judging(args, rows, systems, pairs, scores_path, pairwise_path, blocked_path)

    if not args.dry_run:
        summarise(args, systems, pairs, out_dir, scores_path, pairwise_path, blocked_path)


if __name__ == "__main__":
    main()
