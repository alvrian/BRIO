import json

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import os
import random
# pyrefly: ignore [missing-import]
from compare_mt.rouge.rouge_scorer import RougeScorer
# pyrefly: ignore [missing-import]
from transformers import BartTokenizer, PegasusTokenizer
from utils import Recorder
from data_utils import to_cuda, collate_mp_brio, BrioDataset, LANG_ID
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import partial
from model import RankingLoss, BRIO
import logging
from label_smoothing_loss import label_smoothing_loss
# pyrefly: ignore [missing-import]
from nltk import sent_tokenize, word_tokenize
from config import cnndm_setting, xsum_setting, liputan6_setting
# pyrefly: ignore [missing-import]
from indobenchmark import IndoNLGTokenizer
from tqdm import tqdm
import shutil


logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_fast").setLevel(logging.ERROR)


def base_setting(args):
    args.batch_size = getattr(args, 'batch_size', 1) # batch size on one gpu, one step
    args.epoch = getattr(args, 'epoch', 100) 
    args.report_freq = getattr(args, "report_freq", 100) # report frequency
    args.accumulate_step = getattr(args, "accumulate_step", 32) # accumulate gradients steps
    args.margin = getattr(args, "margin", 0.001) # margin for ranking loss on candidate summaries
    args.gold_margin = getattr(args, "gold_margin", 0) # margin for ranking loss on gold summaries
    args.gold_weight = getattr(args, "gold_weight", 0) # weight for ranking loss on gold summaries
    args.mle_weight = getattr(args, "mle_weight", 1) # weight for mle loss on gold summaries
    args.rank_weight = getattr(args, "rank_weight", 1) # weight for ranking loss on candidate summaries
    args.model_type = getattr(args, "model_type", "facebook/bart-large-cnn") # model type
    args.warmup_steps = getattr(args, "warmup_steps", 10000) # warmup steps
    args.normalize = getattr(args, "normalize", True) # normalize predicited likelihood
    args.grad_norm = getattr(args, "grad_norm", 0) # gradient norm
    args.seed = getattr(args, "seed", 970903) # random seed
    args.no_gold = getattr(args, "no_gold", False) # whether to use gold summaries
    args.pretrained = getattr(args, "pretrained", None) # pretrained model path
    args.max_lr = getattr(args, "max_lr", 2e-3) # max learning rate (* 1e-2)
    args.scale = getattr(args, "scale", 1) # scale of ranking loss
    args.score_mode = getattr(args, "score_mode", "log") # use log-likelihood for ranking loss
    args.datatype = getattr(args, "datatype", "diverse") # data type
    args.dataset = getattr(args, "dataset", "cnndm") # dataset
    args.max_len = getattr(args, "max_len", 120) # max length of summary
    args.max_num = getattr(args, "max_num", 16) # max number of candidate summaries
    args.smooth = getattr(args, "smooth", 0.1) # label smoothing
    args.total_len = getattr(args, "total_len", 1024) # total length of source article
    args.length_penalty = getattr(args, "length_penalty", 2.0) # length penalty
    args.do_sample = getattr(args, "do_sample", True) # whether to generaet summaries during evaluation
    args.gen_max_len = getattr(args, "gen_max_len", 140) # max length of generated summaries
    args.gen_min_len = getattr(args, "gen_min_len", 55) # min length of generated summaries
    args.is_pegasus = getattr(args, "is_pegasus", False) # whether to use Pegasus as the baseline model
    args.adding = getattr(args, "adding", 0) # used for numerical stability
    args.eval_interval = getattr(args, "eval_interval", 1000) # evaluation intervals
    args.num_beams = getattr(args, "num_beams", 4) # number of beams for beam search
    args.dataset_folder = getattr(args, "dataset_folder", None) #non default path to dataset directory

def _build_indonlg_batch(slines, tok, max_src_len):
    """Builds <s> X </s> [indonesian] formatted, padded input_ids + attention_mask."""
    pad_id = tok.pad_token_id
    batch_ids = []
    for s in slines:
        ids = tok.encode(s, max_length=max_src_len - 3, truncation=True, add_special_tokens=False)
        ids = [tok.bos_token_id] + ids + [tok.eos_token_id, LANG_ID]
        batch_ids.append(ids)
    max_len = max(len(ids) for ids in batch_ids)
    input_ids = torch.tensor([ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids])
    attention_mask = torch.tensor([[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids])
    return input_ids, attention_mask
    
def evaluation(args):
    # load data
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    elif args.config == "liputan6":
        liputan6_setting(args)
    else:
        base_setting(args)
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    elif args.config == "liputan6":
        tok = IndoNLGTokenizer.from_pretrained(args.model_type)
        print("change data type to canonical")
        args.datatype = "canonical"
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)

    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True)
    test_set = BrioDataset(f"./{args.dataset}/{args.datatype}/test", args.model_type, is_test=True, max_len=512,
                is_sorted=False, max_num=args.max_num, is_untok=True, total_len=args.total_len,
                is_pegasus=args.is_pegasus, is_indonlg=(args.dataset == "liputan6"))
    batch_size = 8
    optimal_workers = min(6, os.cpu_count())
    dataloader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=optimal_workers,
        collate_fn=collate_fn, pin_memory=True,
        persistent_workers=optimal_workers > 0,
    )
    # build models

    model_path = args.pretrained if args.pretrained is not None else args.model_type
    if args.pretrained is not None:
        print(f"loading pretrained model from {args.pretrained}")
    model = BRIO(model_path, tok.pad_token_id, args.is_pegasus)
    if args.cuda:
        model = model.cuda()

    model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=f'cuda:{args.gpuid[0]}'))
    device = f'cuda:{args.gpuid[0]}'
    model.eval()

    model_name = args.model_pt.split("/")[0]

    def mkdir(path):
        if not os.path.exists(path):
            os.mkdir(path)

    print(model_name)
    root_dir = "./result/%s" % model_name
    mkdir(root_dir)
    use_stemmer = False if args.config == "liputan6" else True
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=use_stemmer)

    autocast_ctx = lambda: torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda" if args.cuda else "cpu")

    if args.do_reranking:
        # evaluate the model as a scorer
        mkdir("./result/%s/reference_ranking" % model_name)
        mkdir("./result/%s/candidate_ranking" % model_name)
        rouge1, rouge2, rougeLsum = 0, 0, 0
        cnt = 0
        model.scoring_mode()
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(test_set) // batch_size):
                if args.cuda:
                    to_cuda(batch, args.gpuid[0])
                samples = batch["data"]
                with autocast_ctx():
                    output = model(
                        batch["src_input_ids"],
                        batch["candidate_ids"],
                        args.normalize,
                        args.score_mode,
                        args.length_penalty,
                        adding=args.adding
                    )
                similarity = output['score']
                similarity = similarity.cpu().float().numpy()  # bf16 can't go straight to numpy
                max_ids = similarity.argmax(1)
                for j in range(similarity.shape[0]):
                    sample = samples[j]
                    sents = sample["candidates"][max_ids[j]][0]
                    score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                    rouge1 += score["rouge1"].fmeasure
                    rouge2 += score["rouge2"].fmeasure
                    rougeLsum += score["rougeLsum"].fmeasure
                    with open("./result/%s/candidate_ranking/%d.dec" % (model_name, cnt), "w") as f:
                        for s in sents:
                            print(s, file=f)
                    with open("./result/%s/reference_ranking/%d.ref" % (model_name, cnt), "w") as f:
                        for s in sample["abstract"]:
                            print(s, file=f)
                    cnt += 1
        rouge1 = rouge1 / cnt
        rouge2 = rouge2 / cnt
        rougeLsum = rougeLsum / cnt
        print("ranking rouge1: %.6f, rouge2: %.6f, rougeL: %.6f" % (rouge1, rouge2, rougeLsum))

    if args.do_generation:
        # evaluate the model as a generator
        rouge1, rouge2, rougeLsum = 0, 0, 0
        tokenizer = tok
        count = 1
        bsz = 8
        model.generation_mode()
        total_num = len(os.listdir(f"./{args.dataset}/{args.datatype}/test"))
        with open(f'./{args.dataset}/{args.datatype}/test.source') as source, open(os.path.join(root_dir, "test.out"), 'w') as fout:
            sline = source.readline().strip()
            slines = [sline]
            for sline in tqdm(source, total=total_num):
                if count % bsz == 0:
                    with torch.no_grad():
                        if args.config == "liputan6":
                            input_ids, attention_mask = _build_indonlg_batch(slines, tokenizer, args.total_len)
                            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                        else:
                            dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                            input_ids, attention_mask = dct["input_ids"].to(device), dct["attention_mask"].to(device)
                        gen_max_len = min(args.gen_max_len + 2, 1023)

                        with autocast_ctx():
                            summaries = model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_length=gen_max_len,
                                min_length=args.gen_min_len + 1,
                                no_repeat_ngram_size=3,
                                num_beams=args.num_beams,
                                length_penalty=args.length_penalty,
                                early_stopping=True,
                            )
                        dec = [tokenizer.decode(g.tolist(), skip_special_tokens=True) for g in summaries]
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
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
                    if args.config == "liputan6":
                        input_ids, attention_mask = _build_indonlg_batch(slines, tokenizer, args.total_len)
                        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
                    else:
                        dct = tokenizer.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                        input_ids, attention_mask = dct["input_ids"].to(device), dct["attention_mask"].to(device)
                    gen_max_len = min(args.gen_max_len + 2, 1023)

                    with autocast_ctx():
                        summaries = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=gen_max_len,
                            min_length=args.gen_min_len + 1,
                            no_repeat_ngram_size=3,
                            num_beams=args.num_beams,
                            length_penalty=args.length_penalty,
                            early_stopping=True,
                        )
                    dec = [tokenizer.decode(g.tolist(), skip_special_tokens=True) for g in summaries]
                    for hypothesis in dec:
                        hypothesis = hypothesis.replace("\n", " ")
                        fout.write(hypothesis + '\n')
                        fout.flush()
        # calculate rouge score
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))

        if args.dataset == "liputan6":
            target_dir_temp = f'./{args.dataset}/{args.datatype}/test.target'

        with open(os.path.join(root_dir, "test.out")) as fout, open(target_dir_temp) as target:
            for (hyp, ref) in zip(fout, target):
                hyp = hyp.strip()
                ref = ref.strip()
                hyp = process(hyp)
                ref = process(ref)
                score = rouge_scorer.score("\n".join(ref), "\n".join(hyp))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
            rouge1 = rouge1 / total_num
            rouge2 = rouge2 / total_num
            rougeLsum = rougeLsum / total_num
            print("evaluation rouge1: %.6f, rouge2: %.6f, rougeL: %.6f" % (rouge1, rouge2, rougeLsum))

def test(dataloader, gen_dataloader, model, args, tok, gpuid, do_sample=False):
    model.eval()
    if args.cuda:
        device = f"cuda:{gpuid}"
    else:
        device = "cpu"
    if len(args.gpuid) > 1:
        _model = model.module
    else:
        _model = model
    cnt = 0
    use_stemmer = False if args.config == "liputan6" else True
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=use_stemmer)
    rouge1, rouge2, rougeLsum = 0, 0, 0
    mle_loss = 0
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    _model.scoring_mode()
    
    with torch.no_grad():
        # scoring
        for (i, batch) in enumerate(dataloader):
            if args.cuda:
                to_cuda(batch, device)
            samples = batch["data"]
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda" if args.cuda else "cpu"):
                output = model(
                        batch["src_input_ids"], 
                        batch["candidate_ids"], 
                        args.normalize, 
                        args.score_mode, 
                        args.length_penalty, 
                        adding=args.adding
                    )
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity * args.scale
            gold_similarity = gold_similarity * args.scale
            similarity = similarity.cpu().float().numpy()
            
            probs = output["probs"]  # [bz, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right
            
            batch_mle_loss = mle_fn(probs.transpose(1, 2), gold)
            mle_loss += batch_mle_loss.item()
            
            if i % 1000 == 0:
                print(f"test similarity: {similarity[0]}")
            max_ids = similarity.argmax(1)
            for j in range(similarity.shape[0]):
                cnt += 1
                sample = samples[j]
                sents = sample["candidates"][max_ids[j]][0]
                score = rouge_scorer.score("\n".join(sample["abstract"]), "\n".join(sents))
                rouge1 += score["rouge1"].fmeasure
                rouge2 += score["rouge2"].fmeasure
                rougeLsum += score["rougeLsum"].fmeasure
            del output, probs, similarity, gold_similarity, batch_mle_loss
    rouge1 = rouge1 / cnt
    rouge2 = rouge2 / cnt
    rougeLsum = rougeLsum / cnt
    mle_loss = mle_loss / cnt

    if len(args.gpuid) > 1:
        # Convert floats back to tensors for distributed reduction
        metrics_tensor = torch.FloatTensor([rouge1, rouge2, rougeLsum, mle_loss]).to(device)
        dist.all_reduce(metrics_tensor, op=dist.reduce_op.SUM)
        metrics_tensor = metrics_tensor / len(args.gpuid)
        rouge1 = metrics_tensor[0].item()
        rouge2 = metrics_tensor[1].item()
        rougeLsum = metrics_tensor[2].item()
        mle_loss = metrics_tensor[3].item()

    cnt = 0
    sample_rouge1, sample_rouge2, sample_rougeLsum = 0, 0, 0
    if do_sample:
        # generation
        _model.generation_mode()
        def process(x):
            return sent_tokenize(" ".join(word_tokenize(x.strip())))
        with torch.no_grad():
            for (i, batch) in enumerate(gen_dataloader):
                if args.cuda:
                    to_cuda(batch, device)
                samples = batch["data"]
                slines = [" ".join(x["article_untok"]) for x in samples]

                if args.config == "liputan6":
                    input_ids, attention_mask = _build_indonlg_batch(slines, tok, args.total_len)
                else:
                    dct = tok.batch_encode_plus(slines, max_length=args.total_len, return_tensors="pt", pad_to_max_length=True, truncation=True)
                    input_ids = dct["input_ids"]
                    attention_mask = dct["attention_mask"]  
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                    
                max_gen_len = min(args.gen_max_len + 2, 1023)
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda" if args.cuda else "cpu"):
                    summaries = _model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_gen_len,
                        min_length=args.gen_min_len + 1,
                        no_repeat_ngram_size=3,
                        num_beams=args.num_beams,
                        num_return_sequences=1,
                        length_penalty=args.length_penalty,
                        early_stopping=True,
                        num_beam_groups=1,
                        # use_cache=False
                    )
                dec = [tok.decode(g.tolist() if args.config == "liputan6" else g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summaries]
                for (hypothesis, x) in zip(dec, samples):
                    hypothesis = hypothesis.replace("\n", " ")
                    ref = " ".join(x["abstract_untok"])
                    x = process(ref)
                    y = process(hypothesis)
                    score = rouge_scorer.score("\n".join(x), "\n".join(y))
                    sample_rouge1 += score["rouge1"].fmeasure
                    sample_rouge2 += score["rouge2"].fmeasure
                    sample_rougeLsum += score["rougeLsum"].fmeasure
                    cnt += 1
                del summaries, input_ids, attention_mask
        _model.scoring_mode()
        sample_rouge1 = sample_rouge1 / cnt
        sample_rouge2 = sample_rouge2 / cnt
        sample_rougeLsum = sample_rougeLsum / cnt
        if len(args.gpuid) > 1:
            sample_metrics_tensor = torch.FloatTensor([sample_rouge1, sample_rouge2, sample_rougeLsum]).to(device)
            dist.all_reduce(sample_metrics_tensor, op=dist.reduce_op.SUM)
            sample_metrics_tensor = sample_metrics_tensor / len(args.gpuid)
            sample_rouge1 = sample_metrics_tensor[0].item()
            sample_rouge2 = sample_metrics_tensor[1].item()
            sample_rougeLsum = sample_metrics_tensor[2].item()
            # sample_rouge1 = torch.FloatTensor([sample_rouge1]).to(device)
            # dist.all_reduce(sample_rouge1, op=dist.reduce_op.SUM)
            # sample_rouge1 = sample_rouge1.item() / len(args.gpuid)
            # sample_rouge2 = torch.FloatTensor([sample_rouge2]).to(device)
            # dist.all_reduce(sample_rouge2, op=dist.reduce_op.SUM)
            # sample_rouge2 = sample_rouge2.item() / len(args.gpuid)
            # sample_rougeLsum = torch.FloatTensor([sample_rougeLsum]).to(device)
            # dist.all_reduce(sample_rougeLsum, op=dist.reduce_op.SUM)
            # sample_rougeLsum = sample_rougeLsum.item() / len(args.gpuid)
    model.train()
    return {
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeLsum": rougeLsum,
        "sample_rouge1": sample_rouge1,
        "sample_rouge2": sample_rouge2,
        "sample_rougeLsum": sample_rougeLsum,
        "mle_loss": mle_loss
        } 

def _sync_to_drive(local_path, drive_path):
    shutil.copyfile(local_path, drive_path)
    print('Synced %s to %s' % (local_path, drive_path)  )
    
def run(rank, args):
    if args.config == "cnndm":
        cnndm_setting(args)
    elif args.config == "xsum":
        xsum_setting(args)
    elif args.config == "liputan6":
        liputan6_setting(args)
    else:
        base_setting(args)
    # task initialization
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    gpuid = args.gpuid[rank] if args.cuda else "cpu"
    is_master = rank == 0
    is_mp = len(args.gpuid) > 1
    world_size = len(args.gpuid)
    if is_master:
        id = len(os.listdir("./cache"))
        recorder = Recorder(id, args.log)
    # build dataloader
    if args.is_pegasus:
        tok = PegasusTokenizer.from_pretrained(args.model_type)
    elif args.config == "liputan6":
        tok = IndoNLGTokenizer.from_pretrained(args.model_type)
    else:
        tok = BartTokenizer.from_pretrained(args.model_type)
    collate_fn = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=False)
    collate_fn_val = partial(collate_mp_brio, pad_token_id=tok.pad_token_id, is_test=True)
    #change non standard dataset directory if specified (e.g. for subsets of the dataset)
    if args.dataset_folder is not None:
        train_path = os.path.join(args.dataset_folder, args.datatype, "train")
        val_path = os.path.join(args.dataset_folder, args.datatype, "val")
    else:
        train_path = f"./{args.dataset}/{args.datatype}/train"
        val_path = f"./{args.dataset}/{args.datatype}/val"
        
    #train_set = BrioDataset(train_path, args.model_type, max_len=args.max_len, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    #val_set = BrioDataset(val_path, args.model_type, is_test=True, max_len=512, is_sorted=False, max_num=args.max_num, total_len=args.total_len, is_pegasus=args.is_pegasus)
    train_set = BrioDataset(
        train_path, args.model_type,
        max_len=args.max_len, max_num=args.max_num, total_len=args.total_len,
        is_pegasus=args.is_pegasus, is_indonlg=(args.dataset == "liputan6")
    )
    val_set = BrioDataset(
        val_path, args.model_type, is_test=True, max_len=512, is_sorted=False,
        max_num=args.max_num, total_len=args.total_len,
        is_pegasus=args.is_pegasus, is_indonlg=(args.dataset == "liputan6")
    )
    
    print(f'done extracting data for {args.dataset} for BRIO')
    #is_mp -> multi gpu
    if is_mp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
    	 train_set, num_replicas=world_size, rank=rank, shuffle=True)
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=collate_fn, sampler=train_sampler)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
    	 val_set, num_replicas=world_size, rank=rank)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate_fn_val, sampler=val_sampler)
        val_gen_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=6, collate_fn=collate_fn_val, sampler=val_sampler)
    else:
        dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=6, collate_fn=collate_fn, pin_memory=True)
        val_dataloader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=6, collate_fn=collate_fn_val, pin_memory=True)
        val_gen_dataloader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=6, collate_fn=collate_fn_val, pin_memory=True)
    # build models
    print(f'start building model')
    model_path = args.pretrained if args.pretrained is not None else args.model_type
    
    model = BRIO(model_path, tok.pad_token_id, is_pegasus=args.is_pegasus)

    # --- IndoBART size-mismatch guards ---
    if args.dataset == "liputan6" and len(args.model_pt) == 0:
        tok_vocab_size = len(tok)
        model_vocab_size = model.model.config.vocab_size
        if tok_vocab_size != model_vocab_size:
            print(f"[size-mismatch] tokenizer vocab={tok_vocab_size}, "
                  f"model vocab={model_vocab_size} → resizing embeddings")
            model.model.resize_token_embeddings(tok_vocab_size)
        max_pos = model.model.config.max_position_embeddings
        if args.total_len > max_pos:
            print(f"[size-mismatch] total_len={args.total_len} > "
                  f"max_position_embeddings={max_pos} → capping to {max_pos}")
            args.total_len = max_pos

    start_epoch = 0
    resume_all_step_cnt = 0
    resume_min_ranking_loss = 100
    resume_min_mle_loss = 1e5

    if len(args.model_pt) > 0:
        map_loc = f'cuda:{gpuid}' if args.cuda else 'cpu'
        ckpt_dir = os.path.join("./cache", os.path.dirname(args.model_pt))
        # model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=map_loc, weight_only=True))
        model.load_state_dict(torch.load(os.path.join("./cache", args.model_pt), map_location=map_loc, weights_only=True))

        state_path = os.path.join(ckpt_dir, "train_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                saved_state = json.load(f)
            start_epoch = saved_state["epoch"]
            resume_all_step_cnt = saved_state["all_step_cnt"]
            resume_min_ranking_loss = saved_state["minimum_ranking_loss"]
            resume_min_mle_loss = saved_state["minimum_mle_loss"]
            print(f"Resuming from epoch {start_epoch}, step {resume_all_step_cnt}")
        else:
            print(f"WARNING: no train_state.json found in {ckpt_dir} — resuming weights only, "
                  f"step count/LR schedule/epoch will restart from scratch.")

        optimizer_path = os.path.join(ckpt_dir, "optimizer.bin")
        if os.path.exists(optimizer_path):
            s_optimizer_state = torch.load(optimizer_path, map_location=map_loc, weights_only=True)
        else:
            s_optimizer_state = None
            print(f"WARNING: no optimizer.bin found in {ckpt_dir} — optimizer state will restart fresh.")
    if args.cuda:
        if is_mp:
            # Using DDP
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            model = nn.parallel.DistributedDataParallel(model.to(gpuid), [gpuid], find_unused_parameters=False)
        else:
            model = model.cuda()
    model.train()
    
    # set the model to scoring mode
    if is_mp:
        model.module.scoring_mode()
    else:
        model.scoring_mode()
    if args.smooth > 0:
        mle_fn = label_smoothing_loss(ignore_index=tok.pad_token_id, epsilon=args.smooth)
    else:
        mle_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)
    # s_optimizer = optim.Adam(model.parameters())
    # [OPTIMIZATION] Use fused Adam optimizer if on PyTorch 2.0+ for faster weight updates
    use_fused = True if hasattr(optim.Adam, "fused") and args.cuda else False
    s_optimizer = optim.Adam(model.parameters(), fused=use_fused)
    if len(args.model_pt) > 0 and s_optimizer_state is not None:
        s_optimizer.load_state_dict(s_optimizer_state)
    if is_master:
        recorder.write_config(args, [model], __file__)
        
    # minimum_ranking_loss = 100
    # minimum_mle_loss = 1e5
    # all_step_cnt = 0
    
    minimum_ranking_loss = resume_min_ranking_loss if len(args.model_pt) > 0 else 100
    minimum_mle_loss = resume_min_mle_loss if len(args.model_pt) > 0 else 1e5
    all_step_cnt = resume_all_step_cnt if len(args.model_pt) > 0 else 0
    if is_mp:
        if is_master:
            id = torch.FloatTensor([id]).to(gpuid)
        else:
            id = torch.zeros(1).to(gpuid)
        dist.all_reduce(id, op=dist.reduce_op.SUM)
        id = int(id.item())
    # define evaluation function
    if args.dataset == "xsum":
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - 2 * rouge1 * rouge2 / (rouge1 + rouge2)
    else:
        def eval_fn(rouge1, rouge2, rougeLsum):
            return 1 - (rouge1 * rouge2 + rougeLsum) / 3
    # start training
    for epoch in range(start_epoch, args.epoch):
        s_optimizer.zero_grad()
        avg_ranking_loss = 0
        avg_mle_loss = 0
        step_cnt = 0
        epoch_step = 0
        avg_loss = 0
        batches_to_skip = 0
        last_best_avg_mle_loss = 99
        
        if epoch == start_epoch and len(args.model_pt) > 0:
            steps_per_epoch = len(dataloader) // args.accumulate_step
            steps_in_current_epoch = resume_all_step_cnt % steps_per_epoch
            batches_to_skip = steps_in_current_epoch * args.accumulate_step
            epoch_step = steps_in_current_epoch  # Restore the logging counter
            if is_master:
                print(f"Fast-forwarding: Skipping first {batches_to_skip} batches to resume at step {resume_all_step_cnt}")
                
        for (i, batch) in enumerate(dataloader):
            if epoch == start_epoch and i < batches_to_skip:
                continue
            if args.cuda:
                to_cuda(batch, gpuid)
            step_cnt += 1
            # forward pass
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                output = model(
                    batch["src_input_ids"], 
                    batch["candidate_ids"], 
                    args.normalize, 
                    args.score_mode, 
                    args.length_penalty, 
                    adding=args.adding
                )
            similarity, gold_similarity = output['score'], output['summary_score']
            similarity = similarity * args.scale
            gold_similarity = gold_similarity * args.scale
            
            ranking_loss = RankingLoss(similarity, gold_similarity, args.margin, args.gold_margin, args.gold_weight)
            #probs = output["probs"]  # [bz, seq_len, word_num]
            probs = output["probs"][:, :-1]  # truncate last token
            gold = batch["candidate_ids"][:, 0, 1:]  # shift right
            
            mle_loss = mle_fn(probs.transpose(1, 2), gold)
            
            loss = args.rank_weight * ranking_loss + args.mle_weight * mle_loss
            loss = loss / args.accumulate_step
            
            avg_loss += loss.item()
            avg_mle_loss += mle_loss.item() / args.accumulate_step
            avg_ranking_loss += ranking_loss.item() / args.accumulate_step
            loss.backward()
            
            if step_cnt == args.accumulate_step:
                # updating
                if args.grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
                step_cnt = 0
                epoch_step += 1
                all_step_cnt += 1
                # adjust learning rate
                lr = args.max_lr * min(all_step_cnt ** (-0.5), all_step_cnt * (args.warmup_steps ** (-1.5)))
                for param_group in s_optimizer.param_groups:
                    param_group['lr'] = lr
                s_optimizer.step()
                s_optimizer.zero_grad()
            if epoch_step % args.report_freq == 0 and step_cnt == 0 and is_master:
                # report stats
                print("id: %d"%id)
                print(f"similarity: {similarity[:, :10]}")
                avg_loss_final = avg_loss / args.report_freq
                avg_ranking_loss_final = avg_ranking_loss / args.report_freq
                avg_mle_loss_final = avg_mle_loss / args.report_freq
                
                if not args.no_gold:
                    print(f"gold similarity: {gold_similarity}")
                recorder.print("epoch: %d, batch: %d, avg loss: %.6f, avg ranking loss: %.6f, avg mle loss: %.6f"
                %(epoch+1, epoch_step, avg_loss_final, avg_ranking_loss_final, avg_mle_loss_final))
                recorder.print(f"learning rate: {lr:.6f}")
                recorder.plot("loss", {"loss": avg_loss_final}, all_step_cnt)
                recorder.plot("mle_loss", {"loss": avg_mle_loss_final}, all_step_cnt)
                recorder.plot("ranking_loss", {"loss": avg_ranking_loss_final}, all_step_cnt)
                recorder.print()
                if(avg_mle_loss_final < last_best_avg_mle_loss and args.dataset == "liputan6"):
                    recorder.print(f"mle loss decreased from {last_best_avg_mle_loss:.6f} to {avg_mle_loss_final:.6f}")
                    if is_mp:
                        recorder.save(model.module, "model_gen_temp.bin")
                        try:
                            _sync_to_drive(os.path.join(recorder.dir, "model_gen_temp.bin"), os.path.join(args.checkpoint_save_dir, "model_gen_temp.bin"))
                        except Exception as e:
                            print(f"WARNING: failed to sync model_gen_temp.bin to Google Drive: {e}")
                    else:
                        recorder.save(model, "model_gen_temp.bin")
                        try:
                            _sync_to_drive(os.path.join(recorder.dir, "model_gen_temp.bin"), os.path.join(args.checkpoint_save_dir, "model_gen_temp.bin"))
                        except Exception as e:
                            print(f"WARNING: failed to sync model_gen_temp.bin to Google Drive: {e}")
                    last_best_avg_mle_loss = avg_mle_loss_final
                avg_mle_loss, avg_ranking_loss, avg_loss = 0, 0, 0
            del similarity, gold_similarity, loss, mle_loss, ranking_loss, output, probs

            if all_step_cnt % args.eval_interval == 0 and all_step_cnt != 0 and step_cnt == 0:
                result = test(val_dataloader, val_gen_dataloader, model, args, tok, gpuid, args.do_sample)
                loss = eval_fn(result["rouge1"], result["rouge2"], result["rougeLsum"])
                if loss < minimum_ranking_loss and is_master:
                    minimum_ranking_loss = loss
                    if is_mp:
                        recorder.save(model.module, "model_ranking.bin")
                    else:
                        recorder.save(model, "model_ranking.bin")
                    recorder.print("best ranking loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                if is_master:
                    recorder.print("val ranking loss: %.6f"%(loss))
                    recorder.print("val ranking rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                    %(result["rouge1"], result["rouge2"], result["rougeLsum"]))
                # evaluate the model as a generator
                if args.do_sample:
                    mle_loss = eval_fn(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"])
                else:
                    mle_loss = result["mle_loss"]
                if mle_loss < minimum_mle_loss and is_master:
                    minimum_mle_loss = mle_loss
                    if is_mp:
                        recorder.save(model.module, "model_generation.bin")
                    else:
                        recorder.save(model, "model_generation.bin")
                    recorder.print("best generation loss - epoch: %d, batch: %d"%(epoch, i / args.accumulate_step))
                if is_master:
                    recorder.print("val generation loss: %.6f"%(mle_loss))
                    if args.do_sample:
                        recorder.print("val generation rouge1: %.6f, rouge2: %.6f, rougeLsum: %.6f"
                        %(result["sample_rouge1"], result["sample_rouge2"], result["sample_rougeLsum"]))
                # save current model
                if is_master:
                    if is_mp:
                        recorder.save(model.module, "model_cur.bin")
                    else:
                        recorder.save(model, "model_cur.bin")
                    recorder.save(s_optimizer, "optimizer.bin")
                    train_state = {
                        "epoch": epoch,
                        "all_step_cnt": all_step_cnt,
                        "minimum_ranking_loss": minimum_ranking_loss,
                        "minimum_mle_loss": minimum_mle_loss,
                    }
                    with open(os.path.join(recorder.dir, "train_state.json"), "w") as f:
                        json.dump(train_state, f)
                    # --- sync checkpoints to Google Drive every eval_interval steps ---
                    if args.dataset == "liputan6" and args.checkpoint_save_dir is not None:
                        drive_dir = f"{args.checkpoint_save_dir}/{all_step_cnt}" #hardcoded for now, please change later
                        os.makedirs(drive_dir, exist_ok=True)
                        for ckpt_name in ["model_ranking.bin", "model_generation.bin", "model_cur.bin", "optimizer.bin", 'train_state.json']:
                            src = os.path.join(recorder.dir, ckpt_name)
                            if os.path.exists(src):
                                _sync_to_drive(src, os.path.join(drive_dir, ckpt_name))
                            else:
                                print(f"WARNING: {src} does not exist, skipping sync to Google Drive.")
                        recorder.print(f"[step {all_step_cnt}] checkpoints synced to {drive_dir}")
                        
        # --- sync checkpoints to Google Drive after each epoch (liputan6 only)
        # please change later
        if args.dataset == "liputan6" and args.checkpoint_save_dir is not None and is_master:
            drive_dir = f"{args.checkpoint_save_dir}/master_{args.mle_weight}_{args.rank_weight}" #hardcoded for now, please change later
            os.makedirs(drive_dir, exist_ok=True)
            print("Syncing checkpoints to Google Drive...")
            for ckpt_name in ["model_ranking.bin", "model_generation.bin", "model_cur.bin", "optimizer.bin", "train_state.json"]:
                src = os.path.join(recorder.dir, ckpt_name)
                if os.path.exists(src):
                    _sync_to_drive(src, os.path.join(drive_dir, ckpt_name))
            recorder.print(f"[epoch {epoch+1}] checkpoints synced to {drive_dir}")
            print("sync complete.")

def main(args):
    # set env
    if(args.checkpoint_save_dir is None and args.dataset == "liputan6"):
        print("WARNING: --checkpoint_save_dir is not set. Checkpoints will not be synced to Google Drive.")
        return
    else:
        os.makedirs(args.checkpoint_save_dir, exist_ok=True)
        
    if len(args.gpuid) > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = f'{args.port}'
        mp.spawn(run, args=(args,), nprocs=len(args.gpuid), join=True)
    else:
        run(0, args)

if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--gpuid", nargs='+', type=int, default=[0], help="gpu ids")
    parser.add_argument("-e", "--evaluate", action="store_true", help="evaluate model")
    parser.add_argument("-r", "--do_reranking", action="store_true", help="do reranking evaluation")
    parser.add_argument("-g", "--do_generation", action="store_true", help="do generation evaluation")
    parser.add_argument("-l", "--log", action="store_true", help="logging")
    parser.add_argument("-p", "--port", type=int, default=12355, help="port")
    parser.add_argument("--model_pt", default="", type=str, help="model path")
    parser.add_argument("--config", default="", type=str, help="config path")
    parser.add_argument("--dataset_folder", default=None, type=str, help="custom path to dataset folder")
    parser.add_argument("--checkpoint_save_dir", default=None, type=str, help="custom path to save checkpoints (for Google Drive sync)")
    args = parser.parse_args()
    
    print("start...")
    if args.cuda is False:
        if args.evaluate:
            evaluation(args)
        else:
            main(args)
    else:
        torch.set_float32_matmul_precision('high')
        if args.evaluate:
            with torch.cuda.device(args.gpuid[0]):
                evaluation(args)
        elif len(args.gpuid) == 1:
            with torch.cuda.device(args.gpuid[0]):
                main(args)
        else:
            main(args)


#command to run the code with liputan6 dataset and indobart-v2 model:
# python main.py --cuda --gpuid 0 --model_pt indobart-v2/modela_generation.bin --config liputan6 --do_generation --do_reranking --dataset_folder /path/to/liputan6/dataset

# !conda run --no-capture-output -n env python main.py \
#     --cuda --gpuid 0 --config liputan6 -l \
#     --model_pt <recorder-folder-name>/model_cur.bin