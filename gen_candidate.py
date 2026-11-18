# pyrefly: ignore [missing-import]
from transformers import BartForConditionalGeneration, BartTokenizer, PegasusTokenizer, PegasusForConditionalGeneration, AutoModelForSeq2SeqLM
from indobenchmark import IndoNLGTokenizer
import torch
import sys
import argparse
from typing import List
import os
import shutil
import json
import time
import tempfile
import contextlib

def generate_summaries_cnndm(args):
    device = f"cuda:{args.gpuid}"
    mname = "facebook/bart-large-cnn"
    model = BartForConditionalGeneration.from_pretrained(mname).to(device)
    model.eval()
    tokenizer = BartTokenizer.from_pretrained(mname)
    max_length = 140
    min_length = 55
    count = 1
    bsz = 8
    with open(args.src_dir) as source, open(args.tgt_dir, 'w') as fout:
        sline = source.readline().strip().lower()
        slines = [sline]
        for sline in source:
            if count % 100 == 0:
                print(count, flush=True)
            if count % bsz == 0:
                with torch.no_grad():
                    dct = tokenizer.batch_encode_plus(slines, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    summaries = model.generate(
                        input_ids=dct["input_ids"].to(device),
                        attention_mask=dct["attention_mask"].to(device),
                        num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
                        max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                        min_length=min_length + 1,  # +1 from original because we start at step=1
                        no_repeat_ngram_size=3,
                        length_penalty=2.0,
                        early_stopping=True,
                    )
                    dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for hypothesis in dec:
                    hypothesis = hypothesis.replace("\n", " ")
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []
            sline = sline.strip().lower()
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
            count += 1
        if slines != []:
            with torch.no_grad():
                dct = tokenizer.batch_encode_plus(slines, max_length=1024, return_tensors="pt", pad_to_max_length=True, truncation=True)
                summaries = model.generate(
                    input_ids=dct["input_ids"].to(device),
                    attention_mask=dct["attention_mask"].to(device),
                    num_return_sequences=16, num_beam_groups=16, diversity_penalty=1.0, num_beams=16,
                    max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                    min_length=min_length + 1,  # +1 from original because we start at step=1
                    no_repeat_ngram_size=3,
                    length_penalty=2.0,
                    early_stopping=True,
                )
                dec = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
            for hypothesis in dec:
                    hypothesis = hypothesis.replace("\n", " ")
                    fout.write(hypothesis + '\n')
                    fout.flush()


def generate_summaries_xsum(args):
    device = f"cuda:{args.gpuid}"
    mname = "google/pegasus-xsum"
    model = PegasusForConditionalGeneration.from_pretrained(mname).to(device)
    model.eval()
    tok = PegasusTokenizer.from_pretrained(mname)
    count = 1
    bsz = 2
    with open(args.src_dir) as source, open(args.tgt_dir, 'w') as fout:
        sline = source.readline().strip()
        slines = [sline]
        for (i, sline) in enumerate(source):
            if count % 10 == 0:
                print(count)
            if count % bsz == 0:
                with torch.no_grad():
                    batch = tok.prepare_seq2seq_batch(src_texts=slines, return_tensors="pt").to(device)
                    gen = model.generate(**batch, num_return_sequences=128, num_beam_groups=16, diversity_penalty=0.1, num_beams=128, length_penalty=0.6)
                    dec: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
                dec = [dec[i] for i in range(len(dec)) if i % 8 == 0]
                for hypothesis in dec:
                    fout.write(hypothesis + '\n')
                    fout.flush()
                slines = []
            sline = sline.strip()
            if len(sline) == 0:
                sline = " "
            slines.append(sline)
            count += 1
        if slines != []:
            with torch.no_grad():
                batch = tok.prepare_seq2seq_batch(src_texts=slines, return_tensors="pt").to(device)
                gen = model.generate(**batch, num_return_sequences=128, num_beam_groups=16, diversity_penalty=0.1, num_beams=128, length_penalty=0.6)
                dec: List[str] = tok.batch_decode(gen, skip_special_tokens=True)
            dec = [dec[i] for i in range(len(dec)) if i % 8 == 0]
            for hypothesis in dec:
                    fout.write(hypothesis + '\n')
                    fout.flush()


_original_pad = IndoNLGTokenizer.pad
def _patched_pad(self, *args, **kwargs):
    kwargs.pop("padding_side", None)
    return _original_pad(self, *args, **kwargs)
IndoNLGTokenizer.pad = _patched_pad

def _patched_save_vocabulary(self, save_directory, filename_prefix=None):
    if not os.path.isdir(save_directory):
        raise ValueError(f"Vocabulary path ({save_directory}) should be a directory")
    out_vocab_file = os.path.join(
        save_directory,
        (filename_prefix + "-" if filename_prefix else "") + "sentencepiece.bpe.model"
    )
    if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
        shutil.copyfile(self.vocab_file, out_vocab_file)
    return (out_vocab_file,)
IndoNLGTokenizer.save_vocabulary = _patched_save_vocabulary

_original_decode = IndoNLGTokenizer.decode
def _patched_decode(self, *args, **kwargs):
    kwargs.pop("clean_up_tokenization_spaces", None)
    return _original_decode(self, *args, **kwargs)
IndoNLGTokenizer.decode = _patched_decode

LANG_ID = 40002  # [indonesian]
# Working copy of the output file on fast local disk; the result is copied to --tgt_dir regularly.
# On Colab this is /content/..., elsewhere (for example a Windows PC) the system temp folder.
LOCAL_TMP_DIR = "/content/_gen_candidate_tmp" if os.path.isdir("/content") else os.path.join(tempfile.gettempdir(), "_gen_candidate_tmp")

# When an article has too many candidates with ROUGE = 0, its candidates are generated again
# with the diversity penalty multiplied by these factors, one after another.
RETRY_PENALTY_FACTORS = [0.5, 0.25, 0.1]

_rouge1_scorer = None


def _count_zero_rouge(cands, reference):
    """Number of candidates with ROUGE = 0 against the reference summary.
    A candidate that shares no word with the reference has ROUGE-1 = 0, and then
    ROUGE-2 and ROUGE-L are 0 too, so checking ROUGE-1 is enough.
    Same scorer settings as preprocess.py."""
    global _rouge1_scorer
    if _rouge1_scorer is None:
        from compare_mt.rouge.rouge_scorer import RougeScorer
        _rouge1_scorer = RougeScorer(['rouge1'], use_stemmer=True)
    return sum(1 for c in cands if _rouge1_scorer.score(reference, c)['rouge1'].fmeasure == 0)


def _build_batch(slines, tokenizer, max_src_len=1024):
    """Builds <s> X </s> [indonesian] formatted, padded input_ids + attention_mask for a batch."""
    pad_id = tokenizer.pad_token_id
    batch_ids = []
    for s in slines:
        ids = tokenizer.encode(s, max_length=max_src_len - 3, truncation=True, add_special_tokens=False)
        ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id, LANG_ID]
        batch_ids.append(ids)

    max_len = max(len(ids) for ids in batch_ids)
    input_ids = torch.tensor([ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids])
    attention_mask = torch.tensor([[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids])
    return input_ids, attention_mask


def _generate_candidates(slines, tokenizer, model, device, max_length, min_length, num_cands, diversity_penalty, use_cache=False, bf16=False):
    """Diverse beam search. Returns one list of num_cands candidate summaries per source line."""
    # The decoder cache is off by default. With transformers 4.46.3 cached generation for this model fails
    # with a CUDA "device-side assert" (index out of range while the cache positions are updated), the same
    # MBart cache problem that main.py and compare_and_contrast.py avoid with use_cache=False.
    extra = {} if use_cache else {"use_cache": False}
    # bf16: run the model in bfloat16 (faster on recent GPUs), as main.py does for its evaluation
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if bf16 else contextlib.nullcontext()
    with torch.no_grad(), amp:
        input_ids, attention_mask = _build_batch(slines, tokenizer, max_src_len=1024)
        summaries = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            # IndoBART decoder format is [indonesian] <s> Y </s>: start with the language token and
            # force <s> as the next token, the same way main.py and compare_and_contrast.py generate.
            decoder_start_token_id=LANG_ID,
            forced_bos_token_id=model.config.bos_token_id,
            num_return_sequences=num_cands, num_beam_groups=num_cands, diversity_penalty=diversity_penalty, num_beams=num_cands,
            max_length=max_length + 2,
            min_length=min_length + 1,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            **extra,
        )
        dec = [tokenizer.decode(g.tolist(), skip_special_tokens=True).replace("\n", " ") for g in summaries]
    return [dec[i * num_cands:(i + 1) * num_cands] for i in range(len(slines))]


def _regenerate_bad_articles(slines, refs, cands, generate_fn, base_penalty, max_zero_rouge, first_index):
    """Checks every article of a batch. If more than max_zero_rouge of its candidates have ROUGE = 0
    (gibberish), the candidates of that article are generated again, alone, with a lower diversity penalty.
    The attempt with the fewest ROUGE = 0 candidates is kept.
    generate_fn(list_of_source_lines, diversity_penalty) returns the candidates.
    Returns a list of log records, one per article that needed a retry."""
    records = []
    for k in range(len(slines)):
        zeros = _count_zero_rouge(cands[k], refs[k])
        if zeros <= max_zero_rouge:
            continue
        first_zeros = zeros
        used_penalty = base_penalty
        attempts = 0
        for factor in RETRY_PENALTY_FACTORS:
            penalty = base_penalty * factor
            attempts += 1
            new_cands = generate_fn([slines[k]], penalty)[0]
            new_zeros = _count_zero_rouge(new_cands, refs[k])
            if new_zeros < zeros:
                cands[k], zeros, used_penalty = new_cands, new_zeros, penalty
            if zeros <= max_zero_rouge:
                break
        records.append({
            "article": first_index + k,
            "zero_rouge_first": first_zeros,
            "zero_rouge_final": zeros,
            "attempts": attempts,
            "diversity_penalty_used": used_penalty,
            "fixed": zeros <= max_zero_rouge,
        })
    return records


def _sync_to_drive(local_path, drive_path):
    shutil.copyfile(local_path, drive_path)


def generate_summaries_liputan6(args):
    """
    generate candidate summaries for Liputan6 dataset, resuming from a partially-written
    output file if one already exists (e.g. after a Colab session timeout).
    """
    device = f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu"
    mname = args.model_dir
    num_cands = args.num_cands

    uses_drive = any(str(p).startswith("/content/drive") for p in (mname, args.tgt_dir, args.src_dir))
    if uses_drive and not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive isn't mounted. Run `from google.colab import drive; drive.mount('/content/drive')` "
            "in a notebook cell before running this script."
        )
    if not os.path.isdir(mname):
        raise RuntimeError(f"Checkpoint directory not found: {mname}")

    # reference summaries, needed for the ROUGE = 0 check
    check_rouge = not args.no_rouge_check
    ref_path = args.ref_dir if args.ref_dir else args.src_dir.replace(".source", ".target")
    if check_rouge and not os.path.exists(ref_path):
        raise RuntimeError(f"Reference file not found: {ref_path} (give --ref_dir, or --no_rouge_check to skip the check)")

    model = AutoModelForSeq2SeqLM.from_pretrained(mname).to(device)
    model.config.decoder_start_token_id = LANG_ID
    model.eval()

    tokenizer = IndoNLGTokenizer.from_pretrained(mname)

    max_length = 100
    min_length = 20
    bsz = args.batch_size

    tmp_dir = args.tmp_dir or LOCAL_TMP_DIR
    os.makedirs(tmp_dir, exist_ok=True)
    local_tgt_dir = os.path.join(tmp_dir, os.path.basename(args.tgt_dir))
    if os.path.abspath(local_tgt_dir) == os.path.abspath(args.tgt_dir):
        raise RuntimeError("--tmp_dir must be a different folder from the folder of --tgt_dir")
    log_path = args.tgt_dir + ".regen_log.jsonl"

    # --- resume support: the Drive copy is the only data that reliably survived a disconnect ---
    if os.path.exists(args.tgt_dir) and os.path.getsize(args.tgt_dir) > 0:
        print("Found existing output at --tgt_dir — copying it to the working folder to resume from.")
        shutil.copyfile(args.tgt_dir, local_tgt_dir)
    elif os.path.exists(local_tgt_dir):
        # a local temp file with the same name left by another run (other split folder or settings)
        os.remove(local_tgt_dir)

    already_done_articles = 0
    if os.path.exists(local_tgt_dir) and os.path.getsize(local_tgt_dir) > 0:
        with open(local_tgt_dir, 'r', encoding='utf-8') as f:
            existing_lines = f.readlines()
        # a last line without a newline was cut off in the middle (for example during a copy to Drive)
        if existing_lines and not existing_lines[-1].endswith("\n"):
            existing_lines = existing_lines[:-1]
        complete_groups = len(existing_lines) // num_cands
        keep_lines = complete_groups * num_cands
        if keep_lines != len(existing_lines):
            print(f"Trimming {len(existing_lines) - keep_lines} incomplete lines from a partial last group.")
            with open(local_tgt_dir, 'w', encoding='utf-8', newline='\n') as f:
                f.writelines(existing_lines[:keep_lines])
        already_done_articles = complete_groups
        print(f"Resuming: {already_done_articles} articles already have candidates — skipping them.")

    with open(args.src_dir, encoding='utf-8') as source:
        all_source_lines = [line.strip().lower() for line in source]
    all_ref_lines = None
    if check_rouge:
        with open(ref_path, encoding='utf-8') as ref:
            all_ref_lines = [line.strip().lower() for line in ref]
        if len(all_ref_lines) != len(all_source_lines):
            raise RuntimeError(f"{args.src_dir} has {len(all_source_lines)} lines but {ref_path} has {len(all_ref_lines)}")

    # --start_index / --end_index: generate only this range of articles (a shard), so that the
    # split can be divided between machines; shards are joined afterwards with merge_candidates.py
    start = args.start_index
    end = args.end_index if args.end_index is not None else len(all_source_lines)
    if not 0 <= start < end <= len(all_source_lines):
        raise RuntimeError(f"invalid range {start}-{end} for {len(all_source_lines)} articles")
    if start > 0 or end < len(all_source_lines):
        all_source_lines = all_source_lines[start:end]
        if all_ref_lines is not None:
            all_ref_lines = all_ref_lines[start:end]
        print(f"Generating articles {start} to {end - 1} only ({end - start} articles).")

    if args.num_samples is not None:
        all_source_lines = all_source_lines[:args.num_samples]
        print(f"Limiting to first {args.num_samples} source lines (--num_samples).")

    remaining_lines = all_source_lines[already_done_articles:]
    if not remaining_lines:
        print("Nothing left to do — all source lines already processed.")
        _sync_to_drive(local_tgt_dir, args.tgt_dir)
        return True

    remaining_lines = [(s if len(s) > 0 else " ") for s in remaining_lines]

    def generate_fn(lines, penalty):
        # If the GPU runs out of memory, the batch is split in half and tried again,
        # so one batch of long articles does not stop the whole run.
        try:
            return _generate_candidates(lines, tokenizer, model, device, max_length, min_length, num_cands, penalty,
                                        use_cache=args.use_cache, bf16=args.bf16)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(lines) == 1:
                raise RuntimeError("Out of GPU memory even with one article per batch.")
            half = len(lines) // 2
            print(f"  out of GPU memory with {len(lines)} articles, splitting the batch", flush=True)
            return generate_fn(lines[:half], penalty) + generate_fn(lines[half:], penalty)

    print(f"Generating {num_cands} candidates per article, diversity penalty {args.diversity_penalty}, batch size {bsz}"
          f"{', bfloat16' if args.bf16 else ''}{', decoder cache on' if args.use_cache else ''}.")
    fout = open(local_tgt_dir, 'a', encoding='utf-8', newline='\n')  # append locally; '\n' line endings on every OS
    start_time = time.time()
    batch_count = 0
    retried = 0
    not_fixed = 0
    try:
        for i in range(0, len(remaining_lines), bsz):
            batch = remaining_lines[i:i + bsz]
            processed_so_far = already_done_articles + i
            if processed_so_far % 100 < bsz:
                print(processed_so_far, flush=True)
            cands = generate_fn(batch, args.diversity_penalty)

            if check_rouge:
                refs = all_ref_lines[processed_so_far:processed_so_far + len(batch)]
                records = _regenerate_bad_articles(batch, refs, cands, generate_fn, args.diversity_penalty,
                                                   args.max_zero_rouge, start + processed_so_far)
                if records:
                    retried += len(records)
                    not_fixed += sum(1 for r in records if not r["fixed"])
                    with open(log_path, "a", encoding="utf-8", newline="\n") as flog:
                        for r in records:
                            flog.write(json.dumps(r) + "\n")

            for article_cands in cands:
                for hypothesis in article_cands:
                    fout.write(hypothesis + '\n')
            fout.flush()
            os.fsync(fout.fileno())  # force the OS to commit local writes to disk immediately

            batch_count += 1
            if batch_count % 25 == 0 or batch_count == 3:
                done_now = min(i + bsz, len(remaining_lines))
                per_article = (time.time() - start_time) / done_now
                hours_left = per_article * (len(remaining_lines) - done_now) / 3600
                peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                print(f"  {per_article:.2f} s per article, about {hours_left:.1f} h left for this split, "
                      f"peak GPU memory {peak_gb:.1f} GB", flush=True)
            if batch_count % args.sync_every == 0:
                _sync_to_drive(local_tgt_dir, args.tgt_dir)
                print(f"  copied to --tgt_dir at batch {batch_count}")
    finally:
        fout.close()
        _sync_to_drive(local_tgt_dir, args.tgt_dir)  # final sync, whether we finished or an exception hit
        print("Final copy to --tgt_dir complete.")
        if check_rouge:
            print(f"ROUGE = 0 check: {retried} articles regenerated in this run, {not_fixed} still above the limit "
                  f"(details in {log_path}).")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--gpuid", type=int, default=0, help="gpu id")
    parser.add_argument("--src_dir", type=str, help="source file")
    parser.add_argument("--tgt_dir", type=str, help="target file")
    parser.add_argument("--dataset", type=str, default="cnndm", help="dataset")
    parser.add_argument("--num_samples", type=int, default=None, help="limit to the first N source lines (for testing)")
    # the options below are used for liputan6 only
    parser.add_argument("--num_cands", type=int, default=16, help="number of candidate summaries per article")
    parser.add_argument("--diversity_penalty", type=float, default=0.5, help="diversity penalty of the diverse beam search")
    parser.add_argument("--batch_size", type=int, default=16, help="articles per batch (use a smaller value with more candidates)")
    parser.add_argument("--model_dir", type=str, default="/content/drive/MyDrive/indobart-liputan6-finetuned",
                        help="folder of the fine-tuned IndoBART used to generate the candidates")
    parser.add_argument("--ref_dir", type=str, default=None,
                        help="reference summary file for the ROUGE = 0 check (default: --src_dir with .source replaced by .target)")
    parser.add_argument("--max_zero_rouge", type=int, default=1,
                        help="regenerate an article when more than this many of its candidates have ROUGE = 0")
    parser.add_argument("--no_rouge_check", action="store_true", help="skip the ROUGE = 0 check and the regeneration")
    parser.add_argument("--sync_every", type=int, default=100, help="copy the output file to --tgt_dir every N batches")
    parser.add_argument("--start_index", type=int, default=0, help="first article of the range to generate (0-based)")
    parser.add_argument("--end_index", type=int, default=None, help="end of the range, exclusive (default: last article)")
    parser.add_argument("--tmp_dir", type=str, default=None, help="folder for the working copy (default: /content/... on Colab, else the system temp folder)")
    parser.add_argument("--bf16", action="store_true", help="generate in bfloat16 (faster on GPUs that support it)")
    parser.add_argument("--use_cache", action="store_true",
                        help="turn the decoder cache on (faster, but fails with a CUDA device-side assert on transformers 4.46.3)")
    args = parser.parse_args()

    if args.src_dir is None or args.tgt_dir is None:
        print("Please provide --src_dir and --tgt_dir")
        sys.exit(1)
    if not os.path.exists(args.src_dir):
        print(f"Error: Source file '{args.src_dir}' does not exist.")
        sys.exit(1)

    src_name = os.path.basename(args.src_dir).split('.')[0]
    tgt_name = os.path.basename(args.tgt_dir).split('.')[0]
    if src_name != tgt_name:
        print(f"Error: Source file '{args.src_dir}' and target file '{args.tgt_dir}' must have the same split prefix (e.g. both 'train').")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.tgt_dir), exist_ok=True)

    if args.dataset == "cnndm":
        generate_summaries_cnndm(args)
    elif args.dataset == "xsum":
        generate_summaries_xsum(args)
    elif args.dataset == "liputan6":
        generate_summaries_liputan6(args)


# command examples:
# python gen_candidate.py --src_dir ./liputan6_converted/canonical/train.source \
#   --tgt_dir /content/drive/MyDrive/liputan6_candidates/train.out --dataset liputan6

# 32 candidates with the local fine-tuned model:
# python gen_candidate.py --src_dir ./liputan6_converted/canonical/train.source \
#   --tgt_dir /content/drive/MyDrive/BRIO/liputan6_candidates_32/canonical/train.out --dataset liputan6 \
#   --num_cands 32 --batch_size 8 --model_dir ./indobart-liputan6-finetuned

# test on a small slice first:
# python gen_candidate.py --src_dir ./liputan6_converted/canonical/val.source \
#   --tgt_dir /content/drive/MyDrive/liputan6_candidates/val_test.out --dataset liputan6 --num_samples 50
