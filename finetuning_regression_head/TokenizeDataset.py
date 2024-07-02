import re
import random
from huggingface_hub import login
import torch



access_token = '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'
login(token=access_token)

class TokenizeDataset():
    
    def __init__(self, tokenizer) :        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = tokenizer
        
    
    def tokenize_data(self, dataset):
        masked_data = self.randomly_mask_dataset(dataset)
        tokenize_data = masked_data.map(self.tokenize_function, batched=True, remove_columns=["text", "true_value"])
        tokenize_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return tokenize_data       
            
            
    def mask_output(self, example):
        pattern = r'(?i)output\s*=\s*(-?\d+(\.\d+)?)'
        match = re.search(pattern, example['text'])
        if match:
            true_value = float(match.group(1))
            example['true_value'] = true_value        
        masked_text = re.sub(pattern, 'Output= [MASK]', example['text'])
        example['text'] = masked_text
        return example


    def randomly_mask_dataset(self, dataset, mask_probability=1.0):
        total_examples = len(dataset)
        num_examples_to_mask = int(total_examples * mask_probability)
        indices_to_mask = random.sample(range(total_examples), num_examples_to_mask)
        masked_dataset = dataset.map(lambda example, idx: self.mask_output(example) if idx in indices_to_mask else example, 
                                    with_indices=True)
        return masked_dataset
    

    def tokenize_function(self, samples):
        tokenized_samples = self.tokenizer(samples['text'], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        tokenized_samples.to(self.device)
        
        if 'true_value' in samples:
            true_values = samples['true_value']
            true_values = [v if v is not None else 0.0 for v in true_values]
            true_values = torch.tensor(true_values, dtype=torch.float).to(self.device)
            tokenized_samples['labels'] = true_values
        
        return tokenized_samples
    
    
