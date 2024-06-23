import os

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