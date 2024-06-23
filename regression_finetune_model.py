from datasets import Dataset
from transformers import Trainer, TrainingArguments, LlamaTokenizer, BitsAndBytesConfig
from LlamaRegression import LlamaForMaskedLLM
from huggingface_hub import login
import random
import torch
from configurations import _set_huggingface_config
import re
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import csv
import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
access_token = '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'
login(token=access_token)
hugging_config = _set_huggingface_config()
dataset_path = hugging_config["tuning_dataset"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


def get_dataset(file_path):
    text = []
    with open (file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for sample in reader:
            text.append(sample[0])
    return text
    
    
def mask_output(example):
    pattern = r'(?i)output\s*=\s*(-?\d+(\.\d+)?)'
    match = re.search(pattern, example['text'])
    if match:
        true_value = float(match.group(1))
        example['true_value'] = true_value        
    masked_text = re.sub(pattern, 'Output= [MASK]', example['text'])
    example['text'] = masked_text
    return example


def randomly_mask_dataset(dataset, mask_probability=0.15):
    total_examples = len(dataset)
    num_examples_to_mask = int(total_examples * mask_probability)
    indices_to_mask = random.sample(range(total_examples), num_examples_to_mask)
    masked_dataset = dataset.map(lambda example, idx: mask_output(example) if idx in indices_to_mask else example, 
                                 with_indices=True)
    return masked_dataset


def tokenize_function(samples):
    tokenized_samples = tokenizer(samples['text'], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    tokenized_samples.to(device)
    
    if 'true_value' in samples:
        true_values = samples['true_value']
        true_values = [v if v is not None else 0.0 for v in true_values]
        true_values = torch.tensor(true_values, dtype=torch.float).to(device)
        tokenized_samples['labels'] = true_values
    
    return tokenized_samples


def save_test_data(tokenize_test):
    with open('tokenize_test.pkl', 'wb') as f:
        pickle.dump(tokenize_test, f)


# Get model, tokenizer, bnb
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",  token=access_token)
tokenizer.pad_token = tokenizer.eos_token

if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})   
  
    
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) #not sure if 16 or 32
model = LlamaForMaskedLLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token, tokenizer= tokenizer, quantization_config=bnb_config)
model.resize_token_embeddings(len(tokenizer))
model.enable_gradient_checkpointing()
model.supports_gradient_checkpointing= True
model = prepare_model_for_kbit_training(model) # preprocess the quantized model for training

# Lora Config
config =  LoraConfig( r=16,
    lora_alpha=8, #try next 16 the 32 this is for stabalizing training
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none")

model = get_peft_model(model, peft_config=config)

# Get the data for function 1
text = get_dataset(dataset_path)
data = {'text': text}
data["true_value"] = [0.0 for _ in range(len(data["text"]))]
dataset = Dataset.from_dict(data) #dataset = ({features: ['text'], num_rows:25})
dataset = dataset.train_test_split(test_size=0.3) #dataset = ({train:Dataset({}), test:Dataset})
masked_train_dataset = randomly_mask_dataset(dataset['train'])
masked_test_dataset = randomly_mask_dataset(dataset['test'])

# Tokenize the datasets
tokenize_test = masked_test_dataset.map(tokenize_function, batched=True, remove_columns=["text","true_value"])
tokenize_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenize_train = masked_train_dataset.map(tokenize_function, batched=True, remove_columns=["text","true_value"])
tokenize_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
save_test_data(tokenize_test=tokenize_test)

training_args = TrainingArguments(
    output_dir= hugging_config["output_dir"],
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir=hugging_config["logging_dir"],
    fp16=False,
    optim="paged_adamw_8bit",
    learning_rate=2e-5,
    weight_decay=0.01
    
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenize_train,
    eval_dataset=tokenize_test,
    tokenizer=tokenizer    
)

# Train the model
trainer.train()
model.eval()
    
model.save_pretrained(hugging_config["save_model"])
tokenizer.save_pretrained(hugging_config["save_model"])
print("Model and tokenizer saved successfully.")