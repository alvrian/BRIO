import argparse
import os
import nltk
from nltk.tokenize import word_tokenize

def tokenize_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Skipping: {input_path} (File not found)")
        return
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
                continue

            tokens = word_tokenize(line.lower())
            fout.write(" ".join(tokens) + "\n")
            
    print(f"Created tokenized file: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Tokenize Liputan6 evaluation files using NLTK")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing test.source, test.target, and test.out")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"], help="Which splits to tokenize")
    args = parser.parse_args()

    # Download required NLTK tokenizer resources silently
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    for split in args.splits:
        for suffix in ["source", "target", "out"]:
            src_path = os.path.join(args.dir, f"{split}.{suffix}")
            tgt_path = os.path.join(args.dir, f"{split}.{suffix}.tokenize")
            tokenize_file(src_path, tgt_path)

if __name__ == "__main__":
    main()
    
#command examples
# python tokenize_files.py --dir ./liputan6/diverse/ --splits train val test