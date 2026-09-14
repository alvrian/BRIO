import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from indobenchmark import IndoNLGTokenizer
_original_pad = IndoNLGTokenizer.pad
def _patched_pad(self, *args, **kwargs):
    kwargs.pop("padding_side", None)
    return _original_pad(self, *args, **kwargs)
IndoNLGTokenizer.pad = _patched_pad

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
        
        # Use encode directly to bypass the tokenizer wrapper's internal padding call
        src_ids = self.tokenizer.encode(src, max_length=self.max_src_len, truncation=True)
        tgt_ids = self.tokenizer.encode(tgt, max_length=self.max_tgt_len, truncation=True)
        
        pad_id = self.tokenizer.pad_token_id
        
        # 1. Manually pad the source sequence and build the attention mask
        src_pad_len = self.max_src_len - len(src_ids)
        input_ids = src_ids + [pad_id] * src_pad_len
        attention_mask = [1] * len(src_ids) + [0] * src_pad_len
        
        # 2. Manually pad the target sequence
        tgt_pad_len = self.max_tgt_len - len(tgt_ids)
        label_ids = tgt_ids + [pad_id] * tgt_pad_len
        
        # 3. Replace padding token IDs with -100 so they are ignored in the loss function
        labels = [
            (l if l != pad_id else -100) for l in label_ids
        ]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

def main():
    model_name = "indobenchmark/indobart-v2"
    output_dir = "./indobart-liputan6-finetuned"
    
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
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=2,
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
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    main()