import torch
from transformers import AutoModelForSeq2SeqLM
from indobenchmark import IndoNLGTokenizer
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize, word_tokenize
import nltk
from tqdm import tqdm
import argparse
import os
import shutil

# Required compatibility patches
_original_pad = IndoNLGTokenizer.pad
def _patched_pad(self, *args, **kwargs):
    kwargs.pop("padding_side", None)
    return _original_pad(self, *args, **kwargs)
IndoNLGTokenizer.pad = _patched_pad

_original_decode = IndoNLGTokenizer.decode
def _patched_decode(self, *args, **kwargs):
    kwargs.pop("clean_up_tokenization_spaces", None)
    return _original_decode(self, *args, **kwargs)
IndoNLGTokenizer.decode = _patched_decode

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

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

LANG_ID = 40002  # [indonesian]

# Disabled stemmer to match main.py configuration for Liputan6
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=False)

def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip().lower() for l in f.readlines()]

def build_input_ids(text, tokenizer, max_len):
    ids = tokenizer.encode(text, max_length=max_len - 3, truncation=True, add_special_tokens=False)
    return [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id, LANG_ID]

def batchify(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]

def compute_rouge_scores(predictions, references):
    r1_sum = r2_sum = rl_sum = 0.0
    n = len(predictions)
    for pred, ref in zip(predictions, references):
        # Added word_tokenize to match main.py preprocessing
        pred_joined = "\n".join(sent_tokenize(" ".join(word_tokenize(pred.strip()))))
        ref_joined = "\n".join(sent_tokenize(" ".join(word_tokenize(ref.strip()))))
        score = all_scorer.score(ref_joined, pred_joined)
        r1_sum += score["rouge1"].fmeasure
        r2_sum += score["rouge2"].fmeasure
        rl_sum += score["rougeLsum"].fmeasure
    r1, r2, rl = r1_sum / n, r2_sum / n, rl_sum / n
    average = 1 - (r1 * r2 + rl) / 3  # Match eval_fn ranking metric logic in main.py
    return {"rouge1": r1, "rouge2": r2, "rougeL": rl, "loss_metric": average}

def evaluate_model(model_dir, base_model, src_file, tgt_file, batch_size=8, max_src_len=1024, max_gen_len=140, min_gen_len=55):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating model from {model_dir} on {src_file} using device {device}...")
    
    is_brio_bin = os.path.isfile(model_dir) and model_dir.endswith('.bin')

    if is_brio_bin:
        print("Detected BRIO .bin checkpoint. Loading base model and applying state dict...")
        tokenizer = IndoNLGTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        
        # IndoBART size-mismatch guard
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            
        # Extract only the HF base model weights from the BRIO wrapper class
        state_dict = torch.load(model_dir, map_location=device)
        hf_state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(hf_state_dict, strict=False)
    else:
        print("Detected HuggingFace directory structure. Loading standard model...")
        tokenizer = IndoNLGTokenizer.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        
    model = model.to(device)
    model.config.decoder_start_token_id = LANG_ID
    model.eval()

    sources = load_lines(src_file)
    references = load_lines(tgt_file)
    pad_id = tokenizer.pad_token_id
    predictions = []

    for batch_src in tqdm(list(batchify(sources, batch_size))):
        batch_ids = [build_input_ids(s, tokenizer, max_src_len) for s in batch_src]
        max_len = max(len(x) for x in batch_ids)
        input_ids = torch.tensor([ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids]).to(device)
        attention_mask = torch.tensor([[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids]).to(device)

        with torch.no_grad():
            # Parameters updated to exactly match main.py generation mode
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_len + 2,
                min_length=min_gen_len + 1,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        predictions.extend(tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip() for ids in gen_ids)

    results = compute_rouge_scores(predictions, references)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the HF directory or BRIO .bin file")
    parser.add_argument("--base_model", type=str, default="indobenchmark/indobart-v2", help="Base model for tokenizer when loading .bin files")
    parser.add_argument("--partition", type=str, choices=["canonical", "xtreme"], default="canonical", help="Liputan6 partition")
    parser.add_argument("--split", type=str, default="test", help="Which split to evaluate")
    parser.add_argument("--src_file", type=str, default=None, help="Override explicit path to source file")
    parser.add_argument("--tgt_file", type=str, default=None, help="Override explicit path to target file")

    args = parser.parse_args()

    src_file = args.src_file or f"liputan6_converted/{args.partition}/{args.split}.source"
    tgt_file = args.tgt_file or f"liputan6_converted/{args.partition}/{args.split}.target"

    results = evaluate_model(args.model_dir, args.base_model, src_file, tgt_file)
    print(f"\nPartition: {args.partition} | Split: {args.split}")
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    print(f"Main.py Loss Metric: {results['loss_metric']:.4f}")