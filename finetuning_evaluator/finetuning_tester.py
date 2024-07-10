"""This module contains methods to train an LLM from Huggingface.
"""
import os
import pickle
import numpy
import torch
import sklearn
import datasets
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

class FinetunedModel():
    """Allows finetuning of an LLM from Huggingface.

    The Api of this class is formed by its central method "run_test()".

    Attributes:
        _config: Dictionary containing configuration variables 
    """
    def __init__(self, config):
        """Initializes the instance with configuration variables.
        Args:
            config:     Dictionary with configuration variables.
        """
        self._config = config


    def run_test(self):
    
        tokenizer = self._load_tokenizer()
        collator  = self._load_collator(tokenizer)
        test_data       = self._load_data(self._config["test_input_file"])
        test_data       = self._tokenize_data(tokenizer, test_data)
        self._check_context_length(test_data)

        model  = self._load_model()
        scaler = self._load_scaler()
        
        lora_model= self.load_peft_model(model)
        lora_model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lora_model.to(device)
        
        trained_state_dict_path = os.path.join(self._config["output_dir"], "trained_model_state_dict.pth")
        trained_state_dict = torch.load(trained_state_dict_path, map_location=device)

        # Load model state dict and ensure they are on the same device
        loaded_state_dict = lora_model.state_dict()
        
        for key in trained_state_dict.keys():
            if key in loaded_state_dict:
                trained_state_dict[key] = trained_state_dict[key].to(device)

        # Apply loaded state dict to the model
        lora_model.load_state_dict(trained_state_dict, strict=False)
        
        # Function to compare state dictionaries with a tolerance
        def compare_state_dicts(dict1, dict2, tolerance=1e-5):
            for key in dict1:
                if key not in dict2:
                    print(f"Key {key} missing in second state dict")
                    return False
                if not torch.allclose(dict1[key], dict2[key], atol=tolerance):
                    print(f"Mismatch found in layer: {key}")
                    return False
            return True

        # Verify state dictionary matches with a tolerance
        if compare_state_dicts(trained_state_dict, lora_model.state_dict()):
            print("State dictionaries match. Model loaded correctly.")
        else:
            print("State dictionaries do not match.")



        training_args = transformers.TrainingArguments(
            output_dir                  = self._config["output_dir"],
            evaluation_strategy         = "epoch",
            per_device_train_batch_size = 4,
            per_device_eval_batch_size  = 32,
            num_train_epochs            = 1,
            logging_dir                 = self._config["logging_dir"],
            logging_strategy            = "epoch",
            save_strategy               = "epoch",
            save_total_limit            = 1,
            remove_unused_columns       = True,
            report_to                   = "none"
        )

        trainer = transformers.Trainer(
            model           = lora_model,
            args            = training_args,
            tokenizer       = tokenizer,
            data_collator   = collator,
        )
    
        prediction_trained = self._check_prediction(trainer, scaler, test_data)
        print( "Prediction of test data after training:\n")
        print(f"Error MSE:\n{      prediction_trained['MSE']}")
        


    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self._config["output_model"])
        
        if self._config["checkpoint"] == "meta-llama/Llama-2-7b-hf":
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        elif self._config["checkpoint"] ==  "meta-llama/Meta-Llama-3-8B":
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
            
        return tokenizer


    def _load_collator(self, tokenizer):
        return transformers.DataCollatorWithPadding(tokenizer)


    def _load_data(self, file_name):
        in_path = os.path.join(self._config["input_dir"], f"{file_name}.pickle")
        with open(in_path, mode="rb") as in_file:
            prompts_and_targets = pickle.load(in_file)
        return datasets.Dataset.from_list(prompts_and_targets)


    def _tokenize_data(self, tokenizer, data):
        tokenize_function = lambda data : tokenizer(
                text               = data["text"],
                padding            = False,
                truncation         = False,
                add_special_tokens = True,
        )
        return data.map(tokenize_function, batched=True)


    def _check_context_length(self, data):
        context_length = self._select_context_length()
        for prompt_ids in data["input_ids"]:
            if len(prompt_ids) > context_length:
                raise ValueError("Prompt too long for context window!")


    def _select_context_length(self):
        if self._config["checkpoint"] == "meta-llama/Llama-2-7b-hf":
            return 4096
        elif self._config["checkpoint"] == "meta-llama/Meta-Llama-3-8B" :
            return 7096

        raise ValueError("No context length for this checkpoint set!")


    def _load_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path = self._config["output_model"],
            cache_dir                     = self._config["cache_dir"],
            num_labels                    = 1,
            problem_type                  = "regression",
        )
        if self._config["checkpoint"] == "meta-llama/Llama-2-7b-hf":
            model.config.pad_token_id = model.config.eos_token_id
        elif self._config["checkpoint"] == "meta-llama/Meta-Llama-3-8B":
            model.config.pad_token_id = model.config.eos_token_id 

        return model
    
    
    def load_peft_model(self, base_model):
        peft_model = PeftModel.from_pretrained(base_model, self._config["output_model"])
        return peft_model


    def _load_scaler(self):
        if self._config["evaluate_metric_rescaled"]:
            in_path = os.path.join(self._config["input_dir"], "scaler.pickle")
            with open(in_path, "rb") as in_file:
                return pickle.load(in_file)
        else:
            return None


    def _check_prediction(self, trainer, scaler, data):
        predictions = trainer.predict(data.remove_columns(["text", "label"])).predictions
        targets     = numpy.reshape(numpy.array(data["label"]), (-1,1))
        return {
            "predictions": predictions,
            "targets":     targets,
            "MSE":         self._compute_metrics_mse(scaler,(predictions, targets))["mse"]
        }

    
    def _compute_metrics_mse(self, scaler, scaled_predictions_and_labels):
        scaled_predictions, scaled_labels = scaled_predictions_and_labels

        if self._config["evaluate_metric_rescaled"]:
            return {"mse": sklearn.metrics.mean_squared_error(
                y_pred = scaler.inverse_transform(scaled_predictions),
                y_true = scaler.inverse_transform(scaled_labels),

            )}
        return {"mse": sklearn.metrics.mean_squared_error(
            y_pred = scaled_predictions,
            y_true = scaled_labels,
        )}
        