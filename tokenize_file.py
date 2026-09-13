import argparse
import os
import nltk
from nltk.tokenize import word_tokenize

def tokenize_file(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Skipping: {input_path} (File not found)")
        return
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                fout.write("\n")
                continue
            
            # Lowercase and separate punctuation with spaces
            tokens = word_tokenize(line.lower())
            fout.write(" ".join(tokens) + "\n")
            
    print(f"Created tokenized file: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Tokenize Liputan6 evaluation files using NLTK")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing test.source, test.target, and test.out")
    args = parser.parse_args()

    # Download required NLTK tokenizer resources silently
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    file_pairs = [
        ("test.source", "test.source.tokenize"),
        ("test.target", "test.target.tokenize"),
        ("test.out", "test.out.tokenize")
    ]

    for src_name, tgt_name in file_pairs:
        src_path = os.path.join(args.dir, src_name)
        tgt_path = os.path.join(args.dir, tgt_name)
        tokenize_file(src_path, tgt_path)

if __name__ == "__main__":
    main()
    
#command examples
# python tokenize_eval_files.py --dir ./liputan6/diverse/