import os
from datasets import Dataset

def load_wic_partition(data_path, gold_path=None):
    """
    Parses a single WiC dataset partition (train, dev, or test).
    
    Args:
        data_path: Path to the .data.txt file
        gold_path: Path to the .gold.txt file (optional for test set if labels not provided)
        
    Returns:
        A list of dictionaries representing the parsed examples.
    """
    examples = []
    
    with open(data_path, 'r', encoding='utf-8') as f_data:
        data_lines = f_data.read().strip().split('\n')
        
    labels = None
    if gold_path and os.path.exists(gold_path):
        with open(gold_path, 'r', encoding='utf-8') as f_gold:
            labels = f_gold.read().strip().split('\n')
            
    for i, line in enumerate(data_lines):
        parts = line.split('\t')
        if len(parts) >= 5:
            word = parts[0]
            pos = parts[1]
            indices = parts[2] # Format: idx1-idx2
            sentence1 = parts[3]
            sentence2 = parts[4]
            
            # Extract indices for both sentences
            idx1, idx2 = map(int, indices.split('-'))
            
            example = {
                'id': i,
                'target_word': word,
                'pos': pos,
                'idx1': idx1,
                'idx2': idx2,
                'sentence1': sentence1,
                'sentence2': sentence2,
            }
            
            if labels and i < len(labels):
                example['label'] = 1 if labels[i] == 'T' else 0
                
            examples.append(example)
            
    return examples

def get_wic_hf_dataset(data_dir):
    """
    Loads all partitions from the data_dir into a Hugging Face DatasetDict mapping.
    
    Args:
        data_dir: Path to the base WiC_dataset directory
        
    Returns:
        dict of HuggingFace Dataset objects (train, dev, test)
    """
    partitions = ['train', 'dev', 'test']
    hf_datasets = {}
    
    for split in partitions:
        split_dir = os.path.join(data_dir, split)
        data_file = os.path.join(split_dir, f'{split}.data.txt')
        gold_file = os.path.join(split_dir, f'{split}.gold.txt')
        
        if os.path.exists(data_file):
            examples = load_wic_partition(data_file, gold_file)
            if examples:
                hf_datasets[split] = Dataset.from_list(examples)
                
    return hf_datasets
