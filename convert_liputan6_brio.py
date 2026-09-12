import json
import os
import argparse
import glob

def process_partition(input_dir, output_dir, partition):
    partition_path = os.path.join(input_dir, partition)
    if not os.path.isdir(partition_path):
        print(f"Directory {partition_path} does not exist. Skipping.")
        return

    for split in ['train', 'dev', 'test']:
        split_path = os.path.join(partition_path, split)
        if not os.path.isdir(split_path):
            continue
            
        split_name = "val" if split == "dev" else split
        
        output_split_dir = os.path.join(output_dir, partition)
        os.makedirs(output_split_dir, exist_ok=True)
        
        source_file = os.path.join(output_split_dir, f"{split_name}.source")
        target_file = os.path.join(output_split_dir, f"{split_name}.target")
        
        json_files = glob.glob(os.path.join(split_path, "*.json"))
        
        with open(source_file, 'w', encoding='utf-8') as fs, \
             open(target_file, 'w', encoding='utf-8') as ft:
             
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as fin:
                    data = json.load(fin)
                    
                    article_tokens = [token for sentence in data.get('clean_article', []) for token in sentence]
                    summary_tokens = [token for sentence in data.get('clean_summary', []) for token in sentence]
                    
                    article_text = " ".join(article_tokens).replace('\n', ' ')
                    summary_text = " ".join(summary_tokens).replace('\n', ' ')
                    
                    if article_text.strip() and summary_text.strip():
                        fs.write(article_text + '\n')
                        ft.write(summary_text + '\n')
            
        print(f"Processed {len(json_files)} examples for {partition}/{split} into {output_split_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to liputan6 dataset root containing canonical/xtreme folders")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory mapping for BRIO")
    args = parser.parse_args()
    
    process_partition(args.input_dir, args.output_dir, 'canonical')
    process_partition(args.input_dir, args.output_dir, 'xtreme')

if __name__ == "__main__":
    main()
