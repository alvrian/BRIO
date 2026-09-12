import json
import os
import argparse

def process_liputan6(input_path, output_dir, split_name):
    os.makedirs(output_dir, exist_ok=True)
    
    source_file = os.path.join(output_dir, f"{split_name}.source")
    target_file = os.path.join(output_dir, f"{split_name}.target")
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(source_file, 'w', encoding='utf-8') as fs, \
         open(target_file, 'w', encoding='utf-8') as ft:
        
        # Determine if it's a single json object or multiple lines
        content = fin.read().strip()
        lines = content.split('\n')
        
        for line in lines:
            if not line.strip():
                continue
            data = json.loads(line)
            
            # clean_article and clean_summary are lists of list of tokens
            article_tokens = [token for sentence in data.get('clean_article', []) for token in sentence]
            summary_tokens = [token for sentence in data.get('clean_summary', []) for token in sentence]
            
            # join by space
            article_text = " ".join(article_tokens).replace('\n', ' ')
            summary_text = " ".join(summary_tokens).replace('\n', ' ')
            
            fs.write(article_text + '\n')
            ft.write(summary_text + '\n')
            
    print(f"Processed {len(lines)} examples into {output_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to liputan6 JSON file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory mapping for BRIO")
    parser.add_argument("--split", type=str, default="test", help="train/val/test")
    args = parser.parse_args()
    
    process_liputan6(args.input, args.output_dir, args.split)

if __name__ == "__main__":
    main()
