# pyrefly: ignore [missing-import]
from transformers import BartForConditionalGeneration, BartTokenizer, PegasusTokenizer, PegasusForConditionalGeneration, AutoModelForSeq2SeqLM
from indobenchmark import IndoNLGTokenizer
import torch
import sys
import argparse
from typing import List
import os
import shutil

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
NUM_CANDIDATES = 16
LOCAL_TMP_DIR = "/content/_gen_candidate_tmp"  # fast local disk, no network dependency
SYNC_EVERY_N_BATCHES = 100  # copy local file -> Drive every N batches 


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


def _generate_and_write(slines, tokenizer, model, device, fout, max_length, min_length):
    with torch.no_grad():
        input_ids, attention_mask = _build_batch(slines, tokenizer, max_src_len=1024)
        summaries = model.generate(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            num_return_sequences=NUM_CANDIDATES, num_beam_groups=NUM_CANDIDATES, diversity_penalty=0.5, num_beams=NUM_CANDIDATES,
            max_length=max_length + 2,
            min_length=min_length + 1,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
        )
        dec = [tokenizer.decode(g.tolist(), skip_special_tokens=True) for g in summaries]
    for hypothesis in dec:
        hypothesis = hypothesis.replace("\n", " ")
        fout.write(hypothesis + '\n')
    fout.flush()
    os.fsync(fout.fileno())  # force the OS to commit local writes to disk immediately


def _sync_to_drive(local_path, drive_path):
    shutil.copyfile(local_path, drive_path)


def generate_summaries_liputan6(args):
    """
    generate candidate summaries for Liputan6 dataset, resuming from a partially-written
    output file if one already exists (e.g. after a Colab session timeout).
    """
    device = f"cuda:{args.gpuid}" if torch.cuda.is_available() else "cpu"
    mname = "/content/drive/MyDrive/indobart-liputan6-finetuned"

    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive isn't mounted. Run `from google.colab import drive; drive.mount('/content/drive')` "
            "in a notebook cell before running this script."
        )
    if not os.path.isdir(mname):
        raise RuntimeError(f"Checkpoint directory not found: {mname}")

    model = AutoModelForSeq2SeqLM.from_pretrained(mname).to(device)
    model.config.decoder_start_token_id = LANG_ID
    model.eval()

    tokenizer = IndoNLGTokenizer.from_pretrained(mname)

    max_length = 100
    min_length = 20
    bsz = 16

    os.makedirs(LOCAL_TMP_DIR, exist_ok=True)
    local_tgt_dir = os.path.join(LOCAL_TMP_DIR, os.path.basename(args.tgt_dir))

    # --- resume support: the Drive copy is the only data that reliably survived a disconnect ---
    if os.path.exists(args.tgt_dir) and os.path.getsize(args.tgt_dir) > 0:
        print("Found existing synced output on Drive — copying it locally to resume from.")
        shutil.copyfile(args.tgt_dir, local_tgt_dir)

    already_done_articles = 0
    if os.path.exists(local_tgt_dir) and os.path.getsize(local_tgt_dir) > 0:
        with open(local_tgt_dir, 'r') as f:
            existing_lines = f.readlines()
        complete_groups = len(existing_lines) // NUM_CANDIDATES
        keep_lines = complete_groups * NUM_CANDIDATES
        if keep_lines != len(existing_lines):
            print(f"Trimming {len(existing_lines) - keep_lines} incomplete lines from a partial last group.")
            with open(local_tgt_dir, 'w') as f:
                f.writelines(existing_lines[:keep_lines])
        already_done_articles = complete_groups
        print(f"Resuming: {already_done_articles} articles already have candidates — skipping them.")

    with open(args.src_dir) as source:
        all_source_lines = [line.strip().lower() for line in source]

    if args.num_samples is not None:
        all_source_lines = all_source_lines[:args.num_samples]
        print(f"Limiting to first {args.num_samples} source lines (--num_samples).")

    remaining_lines = all_source_lines[already_done_articles:]
    if not remaining_lines:
        print("Nothing left to do — all source lines already processed.")
        _sync_to_drive(local_tgt_dir, args.tgt_dir)
        return True

    remaining_lines = [(s if len(s) > 0 else " ") for s in remaining_lines]

    fout = open(local_tgt_dir, 'a')  # append locally
    batch_count = 0
    try:
        for i in range(0, len(remaining_lines), bsz):
            batch = remaining_lines[i:i + bsz]
            processed_so_far = already_done_articles + i
            if processed_so_far % 100 < bsz:
                print(processed_so_far, flush=True)
            _generate_and_write(batch, tokenizer, model, device, fout, max_length, min_length)
            batch_count += 1
            if batch_count % SYNC_EVERY_N_BATCHES == 0:
                fout.flush()
                os.fsync(fout.fileno())
                _sync_to_drive(local_tgt_dir, args.tgt_dir)
                print(f"  synced to Drive at batch {batch_count}")
    finally:
        fout.close()
        _sync_to_drive(local_tgt_dir, args.tgt_dir)  # final sync, whether we finished or an exception hit
        print("Final sync to Drive complete.")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--gpuid", type=int, default=0, help="gpu id")
    parser.add_argument("--src_dir", type=str, help="source file")
    parser.add_argument("--tgt_dir", type=str, help="target file")
    parser.add_argument("--dataset", type=str, default="cnndm", help="dataset")
    parser.add_argument("--num_samples", type=int, default=None, help="limit to the first N source lines (for testing)")
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

# test on a small slice first:
# python gen_candidate.py --src_dir ./liputan6_converted/canonical/val.source \
#   --tgt_dir /content/drive/MyDrive/liputan6_candidates/val_test.out --dataset liputan6 --num_samples 50