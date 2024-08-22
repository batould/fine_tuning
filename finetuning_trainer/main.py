"""Main module.

The module contains the main function.
"""
import os

import azure_finetuning_trainer
import hugging_finetuning_trainer


def main():
    """Main function.

    Creates instance of class FinetuningTrainer and runs its central method.
    """
    general_config = _set_general_config()

    if general_config["platform"] == "huggingface":
        huggingface_config = _set_huggingface_config()
        ft = hugging_finetuning_trainer.HuggingfaceFinetuningTrainer(general_config,
                                                                         huggingface_config)
        ft.run_finetuning()

    elif general_config["platform"] == "azure":
        azure_config = _set_azure_config()
        at = azure_finetuning_trainer.AzureFinetuningTrainer(general_config, azure_config)
        prompt_ids = at.upload_prompts()
        at.run_finetuning(prompt_ids[0], prompt_ids[1])
        # at.deploy_finetuned_model(job_id="")
        # at.withdraw_finetuned_model()
        at.clear_prompts()

    else:
        raise ValueError(f"Platform {general_config['platform']} is unvalid!")


def _set_general_config():
    cwd = os.getcwd()
    return {
        "input_dir":                os.path.join(cwd, "finetuning_preprocessor", "finetune_data", "output"),
        "output_dir":               os.path.join(cwd, "finetuning_preprocessor", "finetune_data", "input"),
        "training_input_file":      "training_prompts_and_targets",
        "validation_input_file":    "validation_prompts_and_targets",
        "test_input_file":          "test_prompts_and_targets",
        "evaluation_metric":        "mae",
        "evaluate_metric_rescaled": True,
        "platform":                 "azure",
    }


def _set_huggingface_config():
    return {
        "checkpoint":  "meta-llama/Llama-2-7b-hf",
        "logging_dir": os.path.join("..", "..", "log"),
        "cache_dir":   os.path.join("..", "..", "..", "huggingface_cache")
    }


def _set_azure_config():
    return {
        "model":                    "gpt-35-turbo-0613",
        "deployment_name":          "custom_name",
        "batch_size":               16,
        "learning_rate_multiplier": 0.2,
        "epochs":                   1,
    }


if __name__ == "__main__":
    main()