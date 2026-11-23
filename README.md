# BRIO for Indonesian Summarization (IndoBART-v2 on Liputan6)

This repository adapts **BRIO** (Liu et al., [BRIO: Bringing Order to Abstractive Summarization](https://arxiv.org/abs/2203.16804)) to Indonesian news summarization. The base model is [`indobenchmark/indobart-v2`](https://huggingface.co/indobenchmark/indobart-v2), trained and evaluated on the [Liputan6](https://arxiv.org/abs/2011.00679) dataset (Koto et al., AACL 2020).

It is the code for a master's thesis at Binus University. The code is a fork of the [original BRIO implementation](https://github.com/yixinL7/BRIO); the CNN/DM and XSum code paths are kept, but this README covers the Liputan6 / IndoBART setup only.

> **Original BRIO README:** the setup, preprocessing, training and evaluation instructions for the original CNN/DailyMail, XSum and NYT experiments, together with the authors' results and checkpoints, are kept in [README_original_BRIO.md](README_original_BRIO.md).

## Contents

- [Overview](#overview)
- [Repository layout](#repository-layout)
- [Setup (Google Colab)](#setup-google-colab)
- [Pipeline](#pipeline)
  - [1. Convert Liputan6](#1-convert-liputan6)
  - [2. Fine-tune the MLE baseline](#2-fine-tune-the-mle-baseline)
  - [3. Generate candidate summaries](#3-generate-candidate-summaries)
  - [4. Preprocess into the BRIO format](#4-preprocess-into-the-brio-format)
  - [5. Train BRIO](#5-train-brio)
  - [6. Evaluate](#6-evaluate)
- [Notebooks](#notebooks)
- [Known issues and notes](#known-issues-and-notes)
- [References](#references)

## Overview

Training has two stages:

1. **MLE baseline.** IndoBART-v2 is fine-tuned on Liputan6 with the usual cross-entropy loss (`finetune_indobart.py`). This model is the comparison baseline.
2. **BRIO.** The baseline generates several candidate summaries per article with diverse beam search. The candidates are scored with ROUGE against the reference, and BRIO continues training the baseline with a contrastive ranking loss (candidates with higher ROUGE should get higher model probability) plus the MLE loss.

The total loss is `mle_weight × MLE loss + rank_weight × ranking loss`. Experiments vary the loss weights, the number of candidates (`max_num`) and the random seed.

Evaluation uses ROUGE, BERTScore, and an LLM judge (Gemini) for consistency, relevance, fluency and coherence, with a blind sheet for human raters.

<div align="center">
 <img src="model.png" width="550" alt="BRIO overview" />
</div>

## Repository layout

| File | Purpose |
|---|---|
| `convert_liputan6_brio.py` | Converts the Liputan6 json files into `{split}.source` / `{split}.target` text files (one article per line) |
| `finetune_indobart.py` | Fine-tunes IndoBART-v2 on Liputan6 (MLE baseline) |
| `gen_candidate.py` | Generates candidate summaries with diverse beam search; regenerates articles with too many ROUGE = 0 candidates |
| `check_candidates.py` | Reports ROUGE = 0 candidates per article and per position, and the number of distinct candidates |
| `preprocess.py` | Scores the candidates with ROUGE and writes one json file per article in the BRIO format |
| `main.py` | BRIO training and evaluation |
| `config.py` | Default hyper-parameters (`liputan6_setting` for this project) |
| `model.py`, `data_utils.py`, `label_smoothing_loss.py`, `utils.py` | Model wrapper, data loader, losses, logging |
| `compare_and_contrast.py` | Loads a BRIO `.bin` checkpoint or a HuggingFace folder and reports ROUGE on a test file |
| `generate_outputs.py` | Generates and **saves** summaries from a checkpoint (resumable), with ROUGE |
| `bertscore_eval.py` | ROUGE and BERTScore for the saved summaries on one or more partitions, with paired comparisons against the baseline |
| `llm_judge.py` | LLM-as-a-judge evaluation with the Gemini API, plus a blind human rating sheet |
| `colab_vsc/*.ipynb` | Colab notebooks that run the steps above (see [Notebooks](#notebooks)) |

## Setup (Google Colab)

All experiments run on Google Colab with a GPU runtime. The code needs **Python 3.8**, provided by a conda environment through `condacolab`:

```python
!pip install -q condacolab
import condacolab
condacolab.install()          # the runtime restarts, this is expected
```

```python
!git clone https://github.com/alvrian/BRIO.git
from google.colab import drive
drive.mount('/content/drive')
%cd BRIO
```

```python
! conda create --name env --file spec-file.txt
! conda run --no-capture-output -n env pip install -r requirements.txt
! conda run -n env python -c "import nltk;nltk.download('punkt')"
! git clone https://github.com/neulab/compare-mt.git
! cd compare-mt && conda run --no-capture-output -n env pip install -r requirements.txt
! cd compare-mt && conda run -n env python setup.py install
```

Run every script through the env, for example `!conda run --no-capture-output -n env python main.py ...`.

Main package versions (`requirements.txt`): `transformers==4.46.3`, `torch==2.4.1`, `indobenchmark-toolkit==0.1.7`.

### Google Drive layout

Colab sessions are temporary, so inputs and outputs are kept on Google Drive:

```
MyDrive/BRIO/
├── indobart-liputan6-finetuned.zip      # MLE baseline, unzipped into the repo root
├── Dataset/
│   ├── liputan6_converted.zip           # .source / .target files
│   ├── liputan6_preprocessed.zip        # BRIO json files, 16 candidates
│   └── liputan6_32.zip                  # BRIO json files, 32 candidates
├── liputan6_candidates/canonical/       # 16-candidate .out files, test.source / test.target
├── liputan6_candidates_32/canonical/    # 32-candidate .out files and regeneration logs
├── checkpoints/                         # BRIO checkpoints, one folder per run
├── eval_outputs/                        # 100-article sample: summaries, judge and BERTScore results
└── eval_outputs_full/                   # whole test set: summaries and BERTScore results
```

Local working folders inside the repo: `./cache` (checkpoints of the current session), `./result` (evaluation outputs) and `./liputan6/canonical/{train,val,test}` (BRIO json files).

## Pipeline

### 1. Convert Liputan6

Liputan6 has to be requested from its authors. It comes in two partitions; this project uses **canonical** (train 193,883, val 10,972, test 10,972 articles).

```console
python convert_liputan6_brio.py --input_dir dataset/liputan6_data --output_dir liputan6_converted --partition canonical
```

This writes `liputan6_converted/canonical/{train,val,test}.{source,target}`. All later steps lowercase the text.

### 2. Fine-tune the MLE baseline

```console
python finetune_indobart.py
```

The paths are set in the script: data from `liputan6_converted/canonical`, output to `/content/drive/MyDrive/indobart-liputan6-finetuned`. The notebooks expect this folder zipped as `MyDrive/BRIO/indobart-liputan6-finetuned.zip` and unzipped into the repo root, where `config.py` (`pretrained = "indobart-liputan6-finetuned"`) loads it.

IndoBART input format: the encoder input is `<s> X </s> [indonesian]` and the decoder starts with `[indonesian] <s>` (language token id 40002).

### 3. Generate candidate summaries

```console
python gen_candidate.py --dataset liputan6 \
    --src_dir ./liputan6_converted/canonical/train.source \
    --tgt_dir /content/drive/MyDrive/BRIO/liputan6_candidates_32/canonical/train.out \
    --model_dir ./indobart-liputan6-finetuned \
    --num_cands 32 --diversity_penalty 0.5 --batch_size 4 --sync_every 500
```

| Option | Meaning |
|---|---|
| `--num_cands` | candidates per article (default 16) |
| `--diversity_penalty` | diverse beam search penalty (default 0.5) |
| `--batch_size` | articles per batch. Larger is not faster here (see [Known issues](#known-issues-and-notes)) |
| `--model_dir` | the fine-tuned baseline that generates the candidates |
| `--max_zero_rouge` | regenerate an article when more than this many candidates have ROUGE = 0 (default 1) |
| `--no_rouge_check` | skip the check and the regeneration |
| `--num_samples` | only the first N articles, for tests |
| `--sync_every` | copy the output to `--tgt_dir` every N batches |
| `--bf16` | generate in bfloat16 |
| `--use_cache` | turn the decoder cache on (crashes with transformers 4.46.3, see below) |

The output has `num_cands` lines per article, in article order. The script writes to a local file and copies it to Drive regularly. **It resumes automatically**: run the same command again after a disconnect and it continues from the last complete article saved on Drive. Keep `--num_cands` the same when resuming, because resume counts articles in groups of that size.

**ROUGE = 0 check.** After each batch, an article with more than `--max_zero_rouge` candidates at ROUGE = 0 against the reference is generated again with the diversity penalty multiplied by 0.5, 0.25 and 0.1, and the attempt with the fewest ROUGE = 0 candidates is kept. Each case is logged to `<tgt_dir>.regen_log.jsonl`. Note that ROUGE = 0 is not always gibberish: some valid summaries share no word with the reference.

Check a candidate file before preprocessing:

```console
python check_candidates.py --out_file .../val.out --tgt_file ./liputan6_converted/canonical/val.target --num_cands 32
```

### 4. Preprocess into the BRIO format

`preprocess.py` expects `.tokenized` copies of every file. The Stanford PTB tokenizer is not used for Indonesian, so the copies are identical to the originals:

```console
cp test.source test.source.tokenized   # same for .target and .out, for each split
python preprocess.py --src_dir ./liputan6_converted/canonical --tgt_dir ./liputan6/canonical \
    --split train --cand_num 32 --dataset liputan6 -l
```

`--cand_num` must equal the number of candidates per article in the `.out` file, otherwise candidates are assigned to the wrong articles without an error. For evaluation, also copy `test.source` and `test.target` into `./liputan6/canonical/`.

### 5. Train BRIO

```console
python main.py --cuda --gpuid 0 --config liputan6 -l \
    --seed 70905 --max_num 32 --mle_weight 1 --rank_weight 10 \
    --checkpoint_save_dir /content/drive/MyDrive/BRIO/checkpoints/BRIO_Liputan6_checkpoint_1-10_32_seed70905
```

These options override `config.py` only when they are given:

| Option | Default in `liputan6_setting` | Meaning |
|---|---|---|
| `--seed` | 970903 | random seed |
| `--max_num` | 16 | candidates used per article (the first N in generation order) |
| `--mle_weight` | 1 | weight of the MLE loss |
| `--rank_weight` | 1 | weight of the ranking loss |
| `--batch_size` | 2 | articles per GPU step |
| `--accumulate_step` | 8 | gradient accumulation steps |

Other defaults: 1 epoch, max learning rate 0.0007, 1,200 warm-up steps, validation every 2,000 steps, generation length 20 to 100 tokens, 4 beams, length penalty 1.0.

`--checkpoint_save_dir` is required for Liputan6. Checkpoints are copied to `<checkpoint_save_dir>/<step>/` at each validation and to `<checkpoint_save_dir>/master_<mle_weight>_<rank_weight>/` at the end of each epoch:

- `model_generation.bin`: best checkpoint for generation (use this for summaries)
- `model_ranking.bin`: best checkpoint for candidate ranking
- `model_cur.bin`, `optimizer.bin`, `train_state.json`: latest state, used to resume
- `config.txt`, `log.txt`: the full run settings and the training log

Use a different `--checkpoint_save_dir` for every run, so runs do not overwrite each other. With 32 candidates, if training runs out of GPU memory, use `--batch_size 1 --accumulate_step 16`.

### 6. Evaluate

**ROUGE with `main.py`.** Copy the checkpoint folder into `./cache` first; `--model_pt` is relative to `./cache`:

```console
python main.py --cuda --gpuid 0 --config liputan6 -e --model_pt <folder>/model_generation.bin -g -r
```

`-g` evaluates the model as a generator (summaries in `./result/<folder>/test.out`), `-r` as a reranker of the candidates. Pass the same `--max_num` as in training. For Liputan6, ROUGE is computed with compare-mt's Python ROUGE without stemming.

**Saved summaries, from Drive directly.** `generate_outputs.py` reads a BRIO `.bin` file or a HuggingFace folder, saves the summaries, resumes after a disconnect, and writes ROUGE to `<name>.rouge.json`. The first run picks the articles (`--sample_n`, seed 42) and saves `sample_ids.txt`, so every model is evaluated on the same articles. `--sample_n 0` uses the whole test set.

```console
python generate_outputs.py --name baseline --model_path ./indobart-liputan6-finetuned --out_dir <dir> --sample_n 0
python generate_outputs.py --name brio_1_10_seed1 --model_path <checkpoint>/master_1_10/model_generation.bin --out_dir <dir> --sample_n 0
```

**ROUGE and BERTScore, on one or both partitions.** `bertscore_eval.py` scores every system per article with ROUGE-1/2/L (same settings as above) and BERTScore (multilingual BERT by default; `--model_type indobenchmark/indobert-base-p1 --num_layers 9` for IndoBERT). Give one `--data_dir` per partition, each a folder written by `generate_outputs.py`. Saved scores are reused, so a rerun only scores new systems.

```console
python bertscore_eval.py --data_dir canonical=<canonical dir> --data_dir xtreme=<xtreme dir> --report_dir <dir> \
    --systems baseline brio_1_1_seed1 brio_1_1_seed2 --pair baseline brio_1_1_seed1 --pair baseline brio_1_1_seed2
```

It writes `report_<model>.txt` (per partition: a table per system, mean ± SD over seeds per setting, and for each `--pair` the difference in ROUGE and BERTScore F1 with a bootstrap 95% confidence interval and a sign test), plus `results_<model>.csv` and `results_<model>_by_setting.csv`. Systems named `<setting>_seed<N>` are grouped by setting. Per-article scores are kept in each folder's `bertscore/`.

**LLM judge (Gemini).** Scores relevance, consistency, fluency and coherence (1 to 5), lists unsupported statements, and runs a blind head-to-head comparison. Needs `GEMINI_API_KEY` in the environment. It stops at `--max_requests` (default 450, for the free tier) and resumes on the next run. Articles refused by Gemini's content filter are recorded in `judge/blocked.jsonl` and excluded for every system.

```console
python llm_judge.py --data_dir <dir> --systems baseline brio --pair baseline brio --include_reference --human_n 30
```

Results are written to `<dir>/judge/`: `summary.txt`, `scores.csv`, `pairwise.csv`, and with `--human_n` a blind `human_eval_sheet.csv` plus its key `human_eval_key.csv`. The protocol follows G-Eval in spirit (criteria, evaluation steps, form-filling scores) but uses hand-written evaluation steps and integer scores instead of probability-weighted scores.

## Notebooks

| Notebook | Use |
|---|---|
| `colab_vsc/BRIO_indobart_liputan6.ipynb` | Original end-to-end pipeline for 16 candidates |
| `colab_vsc/BRIO_32_candidates.ipynb` | 32 candidates: penalty test, generation, preprocessing, training per seed |
| `colab_vsc/BRIO_bertscore.ipynb` | Saves summaries for several checkpoints and computes BERTScore |
| `colab_vsc/BRIO_llm_judge.ipynb` | Saves summaries for a 100-article sample and runs the Gemini judge |

After `git pull`, reopen a notebook from GitHub to get its latest version: pulling updates the scripts on the Colab disk but not a notebook already open in the browser.

## Known issues and notes

- **Decoder cache crash.** With transformers 4.46.3, IndoBART generation with the decoder cache on fails with a CUDA device-side assert (an index out of range in the decoder position embeddings). All scripts therefore generate with `use_cache=False`. This is correct but slow: 32-candidate generation runs at about 3.5 s per article, which is roughly 190 hours for the train split.
- **Larger generation batches are slower.** Without the cache, a batch of 16 ran at about 4.7 s per article against 3.5 s for a batch of 4, so keep `--batch_size 4` for 32 candidates.
- **Old 16-candidate files.** A 16-candidate test file generated with an earlier version of `gen_candidate.py` (cache on, no forced `<s>`) has gibberish at candidate positions 14 and 15 for about 91% of articles. Check older candidate files with `check_candidates.py` before using them. The current version produced no ROUGE = 0 candidates in a 48-article test.
- **Stemming mismatch between candidate ranking and evaluation.** `preprocess.py` scores the candidates with ROUGE using `use_stemmer=True`, which applies the English Porter stemmer. These scores decide the order of the candidates that BRIO learns to rank. `main.py` (and `compare_and_contrast.py`, `generate_outputs.py`) evaluate Liputan6 with `use_stemmer=False`. The Porter stemmer is designed for English, so on Indonesian text it mostly strips English-like word endings and does not reduce Indonesian affixes (for example *me-*, *-kan*, *-nya*); its effect on the scores is expected to be small, but it means training ranks candidates with a slightly different ROUGE than the one reported. `gen_candidate.py` and `check_candidates.py` also use the stemmer, to stay consistent with `preprocess.py`. The 16- and 32-candidate datasets were built this way, so this applies to all reported runs.
- **IndoNLGTokenizer.** The scripts patch `pad`, `decode` and `save_vocabulary` of `IndoNLGTokenizer` for compatibility with transformers 4.46.3.
- **Colab limits.** Sessions end after at most 24 hours. Every long step (candidate generation, `generate_outputs.py`, `llm_judge.py`) resumes from Drive; check the file counts before restarting.

## References

- Yixin Liu, Pengfei Liu, Dragomir Radev, Graham Neubig. *BRIO: Bringing Order to Abstractive Summarization*. ACL 2022. [arXiv:2203.16804](https://arxiv.org/abs/2203.16804) · [original code](https://github.com/yixinL7/BRIO) · [original README](README_original_BRIO.md)
- Fajri Koto, Jey Han Lau, Timothy Baldwin. *Liputan6: A Large-scale Indonesian Dataset for Text Summarization*. AACL-IJCNLP 2020. [arXiv:2011.00679](https://arxiv.org/abs/2011.00679)
- Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, Yoav Artzi. *BERTScore: Evaluating Text Generation with BERT*. ICLR 2020. [bert-score package](https://github.com/Tiiiger/bert_score)
- Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu. *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*. [arXiv:2303.16634](https://arxiv.org/abs/2303.16634)
