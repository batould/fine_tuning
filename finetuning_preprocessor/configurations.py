import os

CWD = os.path.join("/", "fibus", "fs2","0c","cbd8159","fine_tune", "fine_tune_sequence", "src")
OUT_DIR = os.path.join('/','work', 'kwm', 'cbd8159', 'output_sequence_model')


def _set_config_generate(problem, model): 
    file_name = {}
    model_name = {}
    config = {
       "input_dir":               os.path.join("prompt_generator", "input"),
        "output_dir":              os.path.join("prompt_generator", "output"),
        "data_columns":            ["x", "y"],
        "prompt_file_name":        "prompt",
        "test_samples_per_batch":  5,
        
    }
    if problem == 2:
        file_name = {
            "training_data_file_name": "unscaled_training_samples_f2",
            "test_data_file_name":     "unscaled_test_samples_f2",
        }
    else: 
        file_name = {
            "training_data_file_name": "unscaled_training_samples_f3",
            "test_data_file_name":     "unscaled_test_samples_f3",
        }
    if model == 'Llama':
        model_name = {"test_prompts" :           "prompts_Llama.json"}  
    else: 
        model_name = {"test_prompts" :           "prompts_gpt.json"}   
             
    config = {**config, **file_name, **model_name}
    return config


def _set_config_comunicator():
        
    communicator_config = {

        "input_dir":         os.path.join("prompt_generator", "output"),
        "output_dir":        os.path.join("prompt_communicator", "output"),
        "prompt_file_name":  "prompt",
        "output_file_name":  "output",
        "device":            "gpu",
        "platform":          "huggingface", #huggingface #azure
        "max_output_tokens":  3000,
        "temperature":        0.5 # [0.5,1,1.5, 2], default = 1.0, the higher, the more creative
    }
    return communicator_config


def _set_post_process_config(problem):
    config = {
        "test_data_input_dir":       os.path.join("prompt_generator", "input"),
        "prompt_response_input_dir": os.path.join("prompt_communicator","output"),
        "output_dir":                os.path.join("output"),
        "output_files_dir":          os.path.join("output", "output_evaluation"),
        "prompt_response_file_name": "output",
        "output_file_name":          "info"
    }
    if problem == 2:
        file_name = {
            "test_data_file_name":     "unscaled_test_samples_f2",
        }
    else: 
        file_name = {
            "test_data_file_name":     "unscaled_test_samples_f3",
        }
        
    config = {**config, **file_name}
    return config


def _set_azure_config():
    azure_config = {
        "model":                   "gpt-4o", #"gpt-3.5-turbo-16k-0613", #gpt-4-32k-0613 # gpt-4o
        "api_version":             "2024-05-01-preview",  #for 3.5 and 4.0 - 2024-02-15-preview #gpt-4o: 2024-05-13
        "chat_completion_choices": 1,
        "enforce_json":            False
    }
    return azure_config


def _set_huggingface_config():
    huggingface_config = {
        "checkpoint": "meta-llama/Meta-Llama-3-8B-Instruct" ,# "mistralai/Mistral-7B-Instruct-v0.2" #"meta-llama/Llama-2-7b-chat-hf"#,
        "cache_dir" : os.path.join('/','work', 'kwm', 'cbd8159', 'cache_huggingface'),
        "output_dir": os.path.join("output"),
        "logging_dir": os.path.join("output", "logging"),
        "save_model": os.path.join("output", "trained_models"),
        "tokenizer_path": os.path.join("fine_tune", "src", "tokenizer"),
        "tuning_dataset": os.path.join("generate_data", "dataset.csv")
    }
    return huggingface_config


def _set_config_finetune_data_preprocessor():
    WORK_DIR = os.path.join(os.getcwd(), "finetuning_preprocessor", "data_input", "mesh")
    configuration = {
        "input_dir" : os.path.join(os.getcwd(),  "finetuning_preprocessor", "data_input", "mesh", "input"),
        "output_dir": os.path.join(os.getcwd(),  "finetuning_preprocessor", "data_input", "mesh", "output"),
        "evaluate_dir": os.path.join(os.getcwd(), "evaluate_model"), 
        "training_data_file_name":   "unscaled_training_data",
        "validation_data_file_name": "unscaled_validation_data",
        "test_data_file_name":       "unscaled_test_data",
        "few_shot_data_file_name":   "few_shot_data",
        "few_shot":                   False,
        "scale":                     False,
        "input_features":            55,
        "output_features":           2,
        "samples_per_batch":         5,
    }
    return configuration


def _set_config_raw_data_preprocess():
    cwd = os.getcwd()
    config = {
        "input_dir":                 os.path.join(cwd, "finetuning_preprocessor", "data_input", "concrete_data", "input"),
        "output_dir":                os.path.join(cwd, "finetuning_preprocessor", "data_input", "concrete_data", "output"),
        "info_file_name":            "info",
        "data_file_name":            "training_data.csv",
        "output_data_file_name":     "data",
        "input_already_split":       False,          # If only a subset of the possible splits
        "split_data_file_names":                     # training/validation/test) is provided,
            {"training":   "training_data.csv",      # reduce dictionary to according keys.
             "validation": "validation_data.csv",
             "test":       "test_data.csv"
            }
    }
    return config
    
    
def _set_config_trainer():
    configuration = {
        "checkpoint":               "meta-llama/Llama-2-7b-hf", #"meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"
        "input_dir":                os.path.join(OUT_DIR, "preprocessor_output"),
        "output_dir":               os.path.join(OUT_DIR, "output"),
        "output_model":             os.path.join(OUT_DIR, "output", "output_model"),        
        "logging_dir":              os.path.join(OUT_DIR, "logging"),
        "cache_dir":                os.path.join(OUT_DIR, "huggingface_cache"),
        "training_input_file":      "training_prompts_and_targets",
        "validation_input_file":    "validation_prompts_and_targets",
        "test_input_file":          "test_prompts_and_targets",
        "evaluation_metric":        "mae",
        "evaluate_metric_rescaled": True,
        "checkpoint_dir":           os.path.join(OUT_DIR, "output", "checkpoint-1250")

    }
    return configuration

def _set_azure_config_trainer():
    config = {
        "model":                    "gpt-35-turbo-0613",
        "deployment_name":          "batoul_concrete",
        "batch_size":               10,
        "learning_rate_multiplier": 0.2,
        "epochs":                   1,
    }
    return config

def _set_general_config_trainer():
    configuration = {
        "input_dir":                os.path.join(OUT_DIR, "preprocessor_output"),
        "output_dir":               os.path.join(OUT_DIR, "output"),
        "training_input_file":      "training_prompts_and_targets",
        "validation_input_file":    "validation_prompts_and_targets",
        "test_input_file":          "test_prompts_and_targets",
        "evaluation_metric":        "mae",
        "evaluate_metric_rescaled": True,
        "platform":                 "azure"
    }
    return configuration


def _set_hugging_face_config_trainer():
    configuration = {
        "checkpoint":               "meta-llama/Llama-2-7b-hf", #"meta-llama/Llama-2-7b-hf", "meta-llama/Meta-Llama-3-8B"
        "output_model":             os.path.join(OUT_DIR, "output", "output_model"),        
        "logging_dir":              os.path.join(OUT_DIR, "logging"),
        "cache_dir":                os.path.join(OUT_DIR, "huggingface_cache"),        
        "checkpoint_dir":           os.path.join(OUT_DIR, "output", "checkpoint-1250")
    }
    return configuration