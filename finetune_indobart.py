import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
import os
import shutil
import glob

from indobenchmark import IndoNLGTokenizer
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


LANG_ID = 40002  # [indonesian]

class Liputan6Dataset(Dataset):
    def __init__(self, source_file, target_file, tokenizer, max_src_len=1024, max_tgt_len=100):
        with open(source_file, 'r', encoding='utf-8') as fs, open(target_file, 'r', encoding='utf-8') as ft:
            self.sources = [line.strip().lower() for line in fs.readlines()]
            self.targets = [line.strip().lower() for line in ft.readlines()]

        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        src = self.sources[idx]
        tgt = self.targets[idx]

        src_ids = self.tokenizer.encode(src, max_length=self.max_src_len - 3, truncation=True, add_special_tokens=False)
        src_ids = [self.tokenizer.bos_token_id] + src_ids + [self.tokenizer.eos_token_id, LANG_ID]

        tgt_ids = self.tokenizer.encode(tgt, max_length=self.max_tgt_len - 3, truncation=True, add_special_tokens=False)
        label_ids = [self.tokenizer.bos_token_id] + tgt_ids + [self.tokenizer.eos_token_id, LANG_ID]

        pad_id = self.tokenizer.pad_token_id

        src_pad_len = self.max_src_len - len(src_ids)
        input_ids = src_ids + [pad_id] * src_pad_len
        attention_mask = [1] * len(src_ids) + [0] * src_pad_len

        tgt_pad_len = self.max_tgt_len - len(label_ids)
        label_ids = label_ids + [pad_id] * tgt_pad_len
        labels = [(l if l != pad_id else -100) for l in label_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


def main():
    model_name = "indobenchmark/indobart-v2"
    output_dir = "/content/drive/MyDrive/indobart-liputan6-finetuned"

    if not os.path.isdir("/content/drive/MyDrive"):
        raise RuntimeError(
            "Google Drive isn't mounted. Run `from google.colab import drive; drive.mount('/content/drive')` "
            "in a notebook cell before running this script."
        )

    tokenizer = IndoNLGTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    train_dataset = Liputan6Dataset(
        "liputan6_converted/canonical/train.source",
        "liputan6_converted/canonical/train.target",
        tokenizer
    )
    val_dataset = Liputan6Dataset(
        "liputan6_converted/canonical/val.source",
        "liputan6_converted/canonical/val.target",
        tokenizer
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        weight_decay=0.01,
        num_train_epochs=5,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )

    existing_checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    resume = True if existing_checkpoints else None

    try:
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        print("Interrupted — saving current state...")
    finally:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()