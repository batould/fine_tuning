from datasets import Dataset
from transformers import Trainer, TrainingArguments, LlamaTokenizer, BitsAndBytesConfig
import torch.nn as nn
from LlamaRegression import LlamaForMaskedLLM
from huggingface_hub import login
from torch.utils.data import DataLoader
import torch
from configurations import _set_huggingface_config
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import os
import csv
import numpy as np
from TokenizeDataset import TokenizeDataset
import hashlib

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
access_token = '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'
login(token=access_token)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

hugging_config = _set_huggingface_config()
dataset_path = hugging_config["tuning_dataset"]


def get_dataset(file_path):
    text = []
    with open (file_path, 'r', newline='') as file:
        reader = csv.reader(file)
        for sample in reader:
            text.append(sample[0])
    return text


def check_prediction(trainer, test_data):
    predictions = trainer.predict(test_data).predictions
    print(f'Prediction size is {predictions[0].shape}')
    targets = np.reshape(np.array(test_data["labels"]), (-1, 1))
    print(f'targets shape {targets.shape}')
    mse = nn.MSELoss()(torch.tensor(predictions[0]), torch.tensor(targets))
    return mse   


tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",  token=access_token)
tokenizer.pad_token = tokenizer.eos_token

if tokenizer.mask_token is None: tokenizer.add_special_tokens({'mask_token': '[MASK]'})   
     
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) 

model = LlamaForMaskedLLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=access_token, tokenizer=tokenizer, quantization_config=bnb_config)
model.resize_token_embeddings(len(tokenizer))
model.enable_gradient_checkpointing()
model.supports_gradient_checkpointing= True
model = prepare_model_for_kbit_training(model) # preprocess the quantized model for training

config =  LoraConfig( r=16,
    lora_alpha=8, #try next 16 the 32 this is for stabalizing training
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none")

model = get_peft_model(model, peft_config=config)
mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

# Data - Function 1
tokenize_pipeline = TokenizeDataset(tokenizer)
text = get_dataset(dataset_path)
data = {'text': text}
data["true_value"] = [0.0 for _ in range(len(data["text"]))]

dataset = Dataset.from_dict(data) #dataset = ({features: ['text'], num_rows:25})
dataset = dataset.train_test_split(test_size=0.3) #dataset = ({train:Dataset({}), test:Dataset})
tokenize_train = tokenize_pipeline.tokenize_data(dataset['train'])
tokenize_validate = tokenize_pipeline.tokenize_data(dataset['test'])

training_args = TrainingArguments(
    output_dir= hugging_config["output_dir"],
    overwrite_output_dir=True,
    num_train_epochs=2,
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
    eval_dataset=tokenize_validate,
    tokenizer=tokenizer    
)

prediction = check_prediction(trainer, tokenize_train)
print(f'old mse is: {prediction}')

# Train the model
trainer.train()

trainer.save_model(hugging_config["save_model"])
tokenizer.save_pretrained(hugging_config["save_model"])
print("Model and tokenizer saved successfully.")

model_config_str = str(model.config)
model_hash = hashlib.md5(model_config_str.encode()).hexdigest()
with open(os.path.join(hugging_config["save_model"], "model_hash.txt"), "w") as f:
    f.write(model_hash)

print(f"Model configuration hash: {model_hash}")