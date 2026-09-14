import torch
from transformers import AutoModelForSeq2SeqLM
from indobenchmark import IndoNLGTokenizer
import evaluate
from tqdm import tqdm

LANG_ID = 40002  # [indonesian]

def load_lines(path):
    with open(path, encoding="utf-8") as f:
        return [l.strip().lower() for l in f.readlines()]

def build_input_ids(text, tokenizer, max_len):
    ids = tokenizer.encode(text, max_length=max_len - 3, truncation=True, add_special_tokens=False)
    return [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id, LANG_ID]

def batchify(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]

def evaluate_model(model_dir, src_file, tgt_file, batch_size=8, max_src_len=1024, max_gen_len=100):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = IndoNLGTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
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

    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=predictions, references=references), predictions

if __name__ == "__main__":
    results, preds = evaluate_model(
        "./indobart-liputan6-finetuned",  # point this at each checkpoint's output_dir
        "liputan6_converted/canonical/test.source",
        "liputan6_converted/canonical/test.target",
    )
    print(results)