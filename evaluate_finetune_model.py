import pickle
from LlamaRegression import LlamaForMaskedLLM
from transformers import LlamaTokenizer
from configurations import _set_huggingface_config
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from huggingface_hub import login
import numpy as np
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
access_token = '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'
login(token=access_token)
hugging_config = _set_huggingface_config()
dataset_path = hugging_config["tuning_dataset"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def load_test_data():
    with open('tokenize_test.pkl', 'rb') as f:
        tokenize_test = pickle.load(f)
    return tokenize_test


model = LlamaForMaskedLLM.from_pretrained(hugging_config["save_model"])
tokenizer = LlamaTokenizer.from_pretrained(hugging_config["save_model"])

tokenize_test = load_test_data()
test_dataloader = DataLoader(tokenize_test, batch_size=5)
mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

model.eval()
results= []
mse_loss_fn = nn.MSELoss(reduction="mean")

for batch in test_dataloader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
   
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, regression_scores = output[0], output[1]
        
        for i in range(input_ids.size(0)):
            input_text = tokenizer.decode(input_ids[i], skip_special_tokens=True)
            masked_indices = torch.where(input_ids[i] == mask_token_id)[0]
            
            predicted_values = []
            true_values = []
            for idx in masked_indices:
                predicted_value = regression_scores[i, idx].item()  # Directly use the regression score
                true_value = labels[i].item()
                true_values.append(true_value)
                predicted_values.append(predicted_value)
            
            if predicted_values and true_values:
                predicted_values_tensor = torch.tensor(predicted_values, dtype=torch.float, device=device)
                true_values_tensor = torch.tensor(true_values, dtype=torch.float, device=device)
                mse_value = mse_loss_fn(predicted_values_tensor, true_values_tensor).item()
            else:
                mse_value = float('nan') 
                
            result = {
                'input_text': input_text,
                'masked_indices': masked_indices.tolist(),
                'predicted_values': predicted_values,
                'true_values': true_values,
                "mse": mse_value
            }
            results.append(result)
            
for result in results:
    print(f"Input text: {result['input_text']}")
    print(f"Masked indices: {result['masked_indices']}")
    print(f"Predicted values: {result['predicted_values']}")
    print(f"MSE: {result['mse']}")
    print(f"True values: {result['true_values']}")
    mean_mse = np.mean(result['mse'])
    print(f'Mean mse i s{mean_mse}')  