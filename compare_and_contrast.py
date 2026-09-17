import torch
from transformers import AutoModelForSeq2SeqLM
from indobenchmark import IndoNLGTokenizer
from compare_mt.rouge.rouge_scorer import RougeScorer
from nltk import sent_tokenize
import nltk
from tqdm import tqdm
import argparse
import os
import shutil

# --- required compatibility patches ---
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
all_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)


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
    """Matches preprocess.py's scoring: sent_tokenize + '\n'.join, compare_mt's RougeScorer with a stemmer."""
    r1_sum = r2_sum = rl_sum = 0.0
    n = len(predictions)
    for pred, ref in zip(predictions, references):
        pred_joined = "\n".join(sent_tokenize(pred))
        ref_joined = "\n".join(sent_tokenize(ref))
        score = all_scorer.score(ref_joined, pred_joined)
        r1_sum += score["rouge1"].fmeasure
        r2_sum += score["rouge2"].fmeasure
        rl_sum += score["rougeLsum"].fmeasure
    r1, r2, rl = r1_sum / n, r2_sum / n, rl_sum / n
    average = (r1 + r2 + rl) / 3  # same formula as preprocess.py's non-xsum branch
    return {"rouge1": r1, "rouge2": r2, "rougeL": rl, "average": average}


def evaluate_model(model_dir, src_file, tgt_file, batch_size=8, max_src_len=1024, max_gen_len=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating model from {model_dir} on {src_file} and {tgt_file} using device {device}...")
    
    tokenizer = IndoNLGTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
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
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_gen_len,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        predictions.extend(tokenizer.decode(ids.tolist(), skip_special_tokens=True).strip() for ids in gen_ids)

    results = compute_rouge_scores(predictions, references)
    return results, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="indobart-liputan6-finetuned", help="Path to the model directory")
    parser.add_argument("--partition", type=str, choices=["canonical", "xtreme"], default="canonical", help="Which Liputan6 partition to evaluate on")
    parser.add_argument("--split", type=str, default="test", help="Which split to evaluate (typically 'val' or 'test')")
    parser.add_argument("--src_file", type=str, default=None, help="Override: explicit path to the source file") #liputan6_converted/{args.partition}/{args.split}.source
    parser.add_argument("--tgt_file", type=str, default=None, help="Override: explicit path to the target file")

    args = parser.parse_args()

    src_file = args.src_file or f"liputan6_converted/{args.partition}/{args.split}.source"
    tgt_file = args.tgt_file or f"liputan6_converted/{args.partition}/{args.split}.target"

    results, preds = evaluate_model(args.model_dir, src_file, tgt_file)
    print(f"Partition: {args.partition} | Split: {args.split}")
    print(f"ROUGE-1: {results['rouge1']:.4f}")
    print(f"ROUGE-2: {results['rouge2']:.4f}")
    print(f"ROUGE-L: {results['rougeL']:.4f}")
    print(f"Average: {results['average']:.4f}")
    
#command line example:
# python compare_and_contrast.py --model_dir indobart-liputan6-finetuned --partition canonical 
# python compare_and_contrast.py --model_dir indobart-liputan6-finetuned --partition xtreme