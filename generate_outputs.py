"""
Generate summaries from a saved checkpoint and SAVE them to a file.

Why this exists: main.py / compare_and_contrast.py print ROUGE but the generated
summaries are lost when the Colab session ends. This script writes them to a
folder (put it on Google Drive) so they can be inspected or judged later.

It works with both kinds of checkpoint:
  - a BRIO checkpoint file      e.g. .../master_1_0/model_generation.bin
  - a HuggingFace model folder  e.g. /content/drive/MyDrive/indobart-liputan6-finetuned
  - a HuggingFace Hub name      e.g. indobenchmark/indobart-v2 (the pretrained model, not fine-tuned)

Files written into --out_dir:
  sample_ids.txt    line numbers (0-based) of the test articles that were picked
  sample.source     the picked articles          (same order as sample_ids.txt)
  sample.target     their reference summaries    (same order)
  <name>.out        the summaries from this model (same order)
  <name>.rouge.json ROUGE of this model on the picked articles

The first run picks the articles and saves sample_ids.txt. Every later run with
the same --out_dir reuses that file, so all models are compared on the same articles.

Example (run once per model, same --out_dir):
  python generate_outputs.py --name baseline \
      --model_path /content/drive/MyDrive/indobart-liputan6-finetuned \
      --src_file liputan6_converted/canonical/test.source \
      --tgt_file liputan6_converted/canonical/test.target \
      --out_dir /content/drive/MyDrive/BRIO_eval_outputs

  python generate_outputs.py --name brio \
      --model_path /content/drive/MyDrive/<checkpoint folder>/master_1_0/model_generation.bin \
      --src_file ... --tgt_file ... --out_dir /content/drive/MyDrive/BRIO_eval_outputs

Use --sample_n 0 to generate for the whole test set instead of a sample.
If the session disconnects, run the same command again: it continues where it stopped.
"""
import argparse
import json
import os
import random


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n").strip() for line in f]


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))


def prepare_sample(args):
    """Pick the test articles once, save them, and reuse the same ones on later runs."""
    os.makedirs(args.out_dir, exist_ok=True)
    ids_path = os.path.join(args.out_dir, "sample_ids.txt")

    all_sources = read_lines(args.src_file)
    all_targets = read_lines(args.tgt_file)
    if len(all_sources) != len(all_targets):
        raise ValueError("source has %d lines but target has %d" % (len(all_sources), len(all_targets)))

    if os.path.exists(ids_path):
        ids = [int(x) for x in read_lines(ids_path) if x]
        print("Reusing %d article ids from %s" % (len(ids), ids_path))
        if max(ids) >= len(all_sources):
            raise ValueError("sample_ids.txt does not match this source file (id out of range)")
    else:
        n_total = len(all_sources)
        if args.sample_n <= 0 or args.sample_n >= n_total:
            ids = list(range(n_total))
        else:
            ids = sorted(random.Random(args.seed).sample(range(n_total), args.sample_n))
        write_lines(ids_path, [str(i) for i in ids])
        print("Picked %d of %d articles (seed %d) -> %s" % (len(ids), n_total, args.seed, ids_path))

    sources = [all_sources[i] for i in ids]
    targets = [all_targets[i] for i in ids]
    write_lines(os.path.join(args.out_dir, "sample.source"), sources)
    write_lines(os.path.join(args.out_dir, "sample.target"), targets)
    return ids, sources, targets


def load_model(model_path, base_model, device):
    """Same loading logic as compare_and_contrast.py (BRIO .bin file or HuggingFace folder)."""
    import torch
    from transformers import AutoModelForSeq2SeqLM
    import compare_and_contrast  # noqa: F401  (importing it applies the IndoNLGTokenizer patches)
    from indobenchmark import IndoNLGTokenizer

    if os.path.isfile(model_path):
        print("BRIO checkpoint file: loading %s and applying the saved weights" % base_model)
        tokenizer = IndoNLGTokenizer.from_pretrained(base_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
        state_dict = torch.load(model_path, map_location="cpu")
        # the BRIO wrapper stores the HuggingFace model under "model.", strip that prefix
        hf_state_dict = {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
        if len(hf_state_dict) == 0:
            raise RuntimeError("No 'model.*' keys in %s, this does not look like a BRIO checkpoint" % model_path)
        missing, unexpected = model.load_state_dict(hf_state_dict, strict=False)
        print("Loaded %d tensors, missing: %d, unexpected: %d" % (len(hf_state_dict), len(missing), len(unexpected)))
        if missing:
            print("WARNING: these weights were NOT loaded from the checkpoint:", missing[:10])
    elif os.path.isdir(model_path):
        print("HuggingFace folder: loading %s" % model_path)
        tokenizer = IndoNLGTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    elif model_path.endswith(".bin") or model_path.startswith(("/", "./", "../")):
        raise FileNotFoundError("model_path not found: %s" % model_path)
    else:
        # not a local path: treat it as a HuggingFace Hub name, e.g. indobenchmark/indobart-v2
        print("HuggingFace Hub model: loading %s" % model_path)
        tokenizer = IndoNLGTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    model = model.to(device)
    model.eval()
    return tokenizer, model


def generate(args, sources, out_path):
    """Generate one summary per article and append it to out_path (resumable)."""
    import time
    import torch
    import compare_and_contrast as cc

    done = len(read_lines(out_path)) if os.path.exists(out_path) else 0
    if done >= len(sources):
        print("%s already has all %d summaries, nothing to generate" % (out_path, done))
        return
    if done > 0:
        print("Resuming: %d of %d summaries already saved" % (done, len(sources)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, model = load_model(args.model_path, args.base_model, device)
    pad_id = tokenizer.pad_token_id

    print("Model loaded on %s, generating %d summaries" % (device, len(sources) - done), flush=True)
    start_time = time.time()
    with open(out_path, "a", encoding="utf-8") as fout:
        for batch_number, start in enumerate(range(done, len(sources), args.batch_size)):
            # the models were trained on lowercased text, so lowercase the article first
            batch_src = [s.lower() for s in sources[start:start + args.batch_size]]
            batch_ids = [cc.build_input_ids(s, tokenizer, args.max_src_len) for s in batch_src]
            max_len = max(len(x) for x in batch_ids)
            input_ids = torch.tensor([x + [pad_id] * (max_len - len(x)) for x in batch_ids]).to(device)
            attention_mask = torch.tensor([[1] * len(x) + [0] * (max_len - len(x)) for x in batch_ids]).to(device)

            # same generation settings as main.py / compare_and_contrast.py for liputan6
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=cc.LANG_ID,
                    forced_bos_token_id=model.config.bos_token_id,
                    max_length=args.max_gen_len + 2,
                    min_length=args.min_gen_len + 1,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                    use_cache=False,
                )
            texts = [tokenizer.decode(g.tolist(), skip_special_tokens=True).strip().replace("\n", " ") for g in gen_ids]
            fout.write("".join(t + "\n" for t in texts))
            fout.flush()

            # progress line (plain print, so it always shows up in the Colab output)
            if batch_number < 2 or (batch_number + 1) % 20 == 0:
                finished = min(start + args.batch_size, len(sources))
                per_article = (time.time() - start_time) / (finished - done)
                hours_left = per_article * (len(sources) - finished) / 3600
                print("%d / %d summaries, %.2f s per article, about %.1f h left"
                      % (finished, len(sources), per_article, hours_left), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="short name of this model, e.g. baseline or brio")
    parser.add_argument("--model_path", default=None, help="BRIO .bin file or HuggingFace model folder")
    parser.add_argument("--existing_out", default=None,
                        help="instead of generating, take the picked lines from a full test.out you already have")
    parser.add_argument("--base_model", default="indobenchmark/indobart-v2", help="architecture used for .bin checkpoints")
    parser.add_argument("--src_file", default="liputan6_converted/canonical/test.source")
    parser.add_argument("--tgt_file", default="liputan6_converted/canonical/test.target")
    parser.add_argument("--out_dir", required=True, help="folder for the outputs (use a Google Drive path)")
    parser.add_argument("--sample_n", type=int, default=100, help="number of test articles, 0 = whole test set")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_src_len", type=int, default=1024)
    parser.add_argument("--max_gen_len", type=int, default=100)
    parser.add_argument("--min_gen_len", type=int, default=20)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=4)
    args = parser.parse_args()

    ids, sources, targets = prepare_sample(args)
    out_path = os.path.join(args.out_dir, args.name + ".out")

    if args.existing_out:
        full = read_lines(args.existing_out)
        if max(ids) >= len(full):
            raise ValueError("%s has only %d lines, it is not a full test output" % (args.existing_out, len(full)))
        write_lines(out_path, [full[i] for i in ids])
    elif args.model_path:
        generate(args, sources, out_path)
    else:
        parser.error("give --model_path (to generate) or --existing_out (to reuse a saved test.out)")

    predictions = read_lines(out_path)
    print("Saved %d summaries to %s" % (len(predictions), out_path))

    # ROUGE on the picked articles, computed the same way as compare_and_contrast.py
    try:
        import compare_and_contrast as cc
        rouge = cc.compute_rouge_scores(predictions, [t.lower() for t in targets])
        with open(os.path.join(args.out_dir, args.name + ".rouge.json"), "w") as f:
            json.dump(rouge, f, indent=2)
        print("ROUGE-1 %.4f | ROUGE-2 %.4f | ROUGE-L %.4f  (n=%d)"
              % (rouge["rouge1"], rouge["rouge2"], rouge["rougeL"], len(predictions)))
    except ImportError as e:
        print("ROUGE skipped (missing package: %s)" % e)


if __name__ == "__main__":
    main()
