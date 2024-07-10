"""This module contains methods to train an LLM from Huggingface.

The methods are organized in the class FinetuningTrainer.

Typical usage example:

    ft = finetuning_trainer.FinetuningTrainer(config)
"""
import os
import pickle
import numpy
import peft
import torch
import sklearn
import datasets
import transformers


class FinetuningTrainer():
    """Allows finetuning of an LLM from Huggingface.

    The Api of this class is formed by its central method "run_finetuning()".

    Attributes:
        _config: Dictionary containing configuration variables 
    """
    def __init__(self, config):
        """Initializes the instance with configuration variables.
        Args:
            config:     Dictionary with configuration variables.
        """
        self._config = config


    def run_finetuning(self):
        """Runs the finetuning of the LLM from Huggingface

        In this method, data, targets, model, tokenizer and collator are loaded. The training is 
        performed with the Huggingface Trainer class. Before and after the training the predictions
        of the model for the test data are checked, to get an impression of the effect of the 
        finetuning. The outcomes of these predictions are printed to the console.
        """
        tokenizer = self._load_tokenizer()
        collator  = self._load_collator(tokenizer)

        training_data   = self._load_data(self._config["training_input_file"])
        validation_data = self._load_data(self._config["validation_input_file"])
        test_data       = self._load_data(self._config["test_input_file"])
        training_data   = self._tokenize_data(tokenizer, training_data)
        validation_data = self._tokenize_data(tokenizer, validation_data)
        test_data       = self._tokenize_data(tokenizer, test_data)
        self._check_context_length(training_data)
        self._check_context_length(validation_data)
        self._check_context_length(test_data)

        model  = self._load_model()
        scaler = self._load_scaler()

        lora_args = peft.LoraConfig(
            r              = 16,
            lora_alpha     = 16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
            lora_dropout   = 0.1,
            bias           = "none",
        )

        lora_model = peft.get_peft_model(model, lora_args)

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
            report_to                   = "none",
            #load_best_model_at_end      = True,
            # metric_for_best_model       = "eval_mae",
            # greater_is_better           = False,
        )

        trainer = transformers.Trainer(
            model           = lora_model,
            args            = training_args,
            train_dataset   = training_data,
            eval_dataset    = validation_data,
            compute_metrics = lambda preds_labels : self._compute_metrics_mae(scaler, preds_labels),
            tokenizer       = tokenizer,
            data_collator   = collator,
        )

        prediction = self._check_prediction(trainer, scaler, test_data)
        print( "Prediction of test data before training:\n")
        print(f"Error:\n{      prediction['MAE']}")
        print(f"Error:\n{      prediction['MSE']}")

        trainer.train()
        
        #lora_model.merge_adapter()
        trainer.save_model(self._config["output_model"])  # Save core model
        tokenizer.save_pretrained(self._config["output_model"])  # Save tokenizer
        lora_model.save_pretrained(self._config["output_model"])  # Save LoRA model
        torch.save(lora_model.state_dict(), f"{self._config['output_dir']}/trained_model_state_dict.pth")


        prediction = self._check_prediction(trainer, scaler, test_data)
        print( "Prediction of test data after training:\n")
        print(f"Error:\n{      prediction['MAE']}")
        print(f"Error:\n{      prediction['MSE']}")
        
       
        

    def _load_tokenizer(self):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path = self._config["checkpoint"],
            cache_dir                     = self._config["cache_dir"],
            use_fast                      = False,
        )

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
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path = self._config["checkpoint"],
            cache_dir                     = self._config["cache_dir"],
            num_labels                    = 1,
            problem_type                  = "regression",
        )

        if self._config["checkpoint"] == "meta-llama/Llama-2-7b-hf":
            model.config.pad_token_id = model.config.eos_token_id
        elif self._config["checkpoint"] == "meta-llama/Meta-Llama-3-8B":
            model.config.pad_token_id = model.config.eos_token_id 

        return model


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
            "MAE":         self._compute_metrics_mae(scaler, (predictions, targets))["mae"],
            "MSE":         self._compute_metrics_mse(scaler,(predictions, targets))["mse"]
        }


    def _compute_metrics_mae(self, scaler, scaled_predictions_and_labels):
        scaled_predictions, scaled_labels = scaled_predictions_and_labels

        if self._config["evaluate_metric_rescaled"]:
            return {"mae": sklearn.metrics.mean_absolute_error(
                y_pred = scaler.inverse_transform(scaled_predictions),
                y_true = scaler.inverse_transform(scaled_labels),

            )}
        return {"mae": sklearn.metrics.mean_absolute_error(
            y_pred = scaled_predictions,
            y_true = scaled_labels,
        )}
    
    
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
        