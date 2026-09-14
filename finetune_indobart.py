import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from indobenchmark import IndoNLGTokenizer

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
        
        # Tokenize source and target
        model_inputs = self.tokenizer(src, max_length=self.max_src_len, padding="max_length", truncation=True)
        labels = self.tokenizer(tgt, max_length=self.max_tgt_len, padding="max_length", truncation=True)
        
        # Assign labels to the model inputs
        model_inputs["labels"] = labels["input_ids"]
        
        # Replace padding token ids in labels with -100 to ignore them in loss computation
        model_inputs["labels"] = [
            (l if l != self.tokenizer.pad_token_id else -100) for l in model_inputs["labels"]
        ]
        
        return {key: torch.tensor(val) for key, val in model_inputs.items()}

def main():
    model_name = "indobenchmark/indobart-v2"
    output_dir = "./indobart-liputan6-finetuned"
    
    tokenizer = IndoNLGTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    # Update paths to match your converted dataset directory
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
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
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
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")

if __name__ == "__main__":
    main()