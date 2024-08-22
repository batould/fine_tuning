"""This module contains methods to finetune an LLM from Azure.

The methods are organized in the class AzureFinetuningTrainer.

Typical usage example:

    Upload prompts previously generated with finetuning_preprocessor and run the finetuning:
        azure_config = _set_azure_config()
        at = azure_finetuning_trainer.AzureFinetuningTrainer(general_config, azure_config)
        prompt_ids = at.upload_prompts()
        at.run_finetuning(prompt_ids[0], prompt_ids[1])
    
    After the finetuning has finished, the model need to be deployed to be accessible. The function
    deploy_finetuned_model needs the job id that has been printed to the console during the
    finetuning as input argument. Remember that deployed models are billed hourly:
        at.deploy_finetuned_model(job_id="ftjob-e3642aaf159a4ccab6daaac39cfe1f33")
    
    When the testing of the model is finished, the model has to be withdrawn to stop being billed:
        at.withdraw_finetuned_model()
        
    Finally, it makes sense to clear the uploaded prompts:
        at.clear_prompts()
"""
import os
import json
import time
import shutil
import pickle
import openai
import dotenv
import requests
import tiktoken
import azure.cli.core


class AzureFinetuningTrainer():
    """Allows finetuning of an LLM from Azure.
    
    The Api of this class is formed by its central methods "upload_prompts()", "run_finetuning()",
    "deploy_finetuned_model()", "withdraw_finetuned_model()" and "clear_prompts()".
    The implementation of these methods follows the official Azure tutorial:
    # https://learn.microsoft.com/en-us/azure/ai-services/openai/tutorials/fine-tune
    
    Attributes:
        _general_config: Dictionary containing general configuration variables.
        _azure_config:   Dictionary containing azure-specific configuration variables.
        _client:         Instance of the AzureOpenAI class from the openai module.
    """
    def __init__(self, general_config, azure_config):
        """Initializes the instance with configuration variables.
        
        Args:
            general_config: Dictionary with general configuration variables.
            azure_config:   Dictionary with azure-specific configuration variables.
        """
        self._general_config = general_config
        self._azure_config   = azure_config
        self._client         = self._setup_client()


    def upload_prompts(self):
        """Loads prompts for training and validation, checks whether they fit into the context
           length, and uploads them to Azure.
           
        Returns:
            Tuple of two strings: The id of the uploaded training prompts and the id of the
            uploaded validation prompts.
        """
        training_prompts   = self._load_prompts(self._general_config[  "training_input_file"])
        validation_prompts = self._load_prompts(self._general_config["validation_input_file"])
        self._check_context_length(  training_prompts)
        self._check_context_length(validation_prompts)
        training_prompts_id   = self._upload_file(  training_prompts,   "training_set.jsonl")
        validation_prompts_id = self._upload_file(validation_prompts, "validation_set.jsonl")
        return (training_prompts_id, validation_prompts_id)


    def clear_prompts(self, file_id = None):
        """Deletes either the file specified with file_id or all uploaded files.
        
        Args:
            file_id: Optional. If set, id of file to be deleted. Otherwise all files that have been
                               uploaded are deleted.    
        """
        if file_id is not None:
            self._client.files.delete(file_id)
        else:
            for file in self._client.files.list().data:
                self._client.files.delete(file.id)


    def run_finetuning(self, training_prompts_id, validation_prompts_id):
        """Starts the fine-tuning process and tracks its progress.
        
        Args:
            training_prompts_id:   Id of the previously uploaded training prompts.
            validation_prompts_id: Id of the previously uploaded validation prompts.
            
        Returns:
            Id of the finetuning-job.
        """
        job_id = self._start_finetuning(training_prompts_id, validation_prompts_id)
        self._track_finetuning(job_id)
        return job_id


    def deploy_finetuned_model(self, job_id):
        """Deploys the fine-tuned model to make it accessible with the Prompt Communicator.
           Attention: A deployed model is billed hourly, so remember to withdraw it when it is not
           needed anymore.
           
        Args:
            job_id: Id of the job that has been printed to the console during the fine-tuning.
        """
        self._deploy_model(job_id)


    def withdraw_finetuned_model(self):
        """Withdraws the fine-tuned model to stop being billed for it. A withdrawn model is not
           deleted. It can be re-deployed at any time.
        """
        self._withdraw_model()


    def _setup_client(self):
        env = self._load_env()
        return openai.AzureOpenAI(
            azure_endpoint   = env["openai_endpoint"],
            api_version      = env["api_version"],
            api_key          = env["openai_token"],
        )


    def _load_prompts(self, file_name):
        in_path = os.path.join(self._general_config["input_dir"], f"{file_name}.pickle")
        with open(in_path, mode="rb") as in_file:
            prompts_and_targets = pickle.load(in_file)
        return prompts_and_targets # prompts_and_targets[0] is the list of prompts


    def _check_context_length(self, prompts):
        characteristics = self._select_tokenization_characteristics()

        for prompt in prompts:
            length = 0
            for message in prompt["messages"]:
                length += characteristics["tokens_per_message"]
                for key, value in message.items():
                    length += len(characteristics["encoding"].encode(value))
                    if key == "name":
                        length += characteristics["tokens_per_name"]
            length += 3

            if length > characteristics["context_length"]:
                raise ValueError("Message is too long, it would exceed the context window!")


    def _select_tokenization_characteristics(self):
        if self._azure_config["model"] == "gpt-35-turbo-0613":
            return{
                "context_length":     4096, 
                "encoding":           tiktoken.encoding_for_model(self._azure_config["model"]),
                "tokens_per_message": 3,
                "tokens_per_name":    1
            }
        if self._azure_config["model"] == "gpt-4-0613":
            return{
                "context_length":     8192, 
                "encoding":           tiktoken.encoding_for_model(self._azure_config["model"]),
                "tokens_per_message": 3,
                "tokens_per_name":    1
            }

        raise ValueError(f"Encoding characteristics for model {self._azure_config['model']} unset")


    def _upload_file(self, prompts, temp_file_name):
        os.mkdir(os.path.join(self._general_config["output_dir"], "temp"))
        temp_file_name = os.path.join(self._general_config["output_dir"], "temp", temp_file_name)

        with open(temp_file_name, "w") as temp_file:
            for prompt in prompts:
                json.dump(prompt, temp_file)
                temp_file.write("\n")

        with open(temp_file_name, "rb") as temp_file:
            response = self._client.files.create(
                file    = temp_file,  # Argument has to be a file object, which requires creating a
                purpose = "fine-tune" # temporary file containing the prompts.
            )

        shutil.rmtree(os.path.join(self._general_config["output_dir"], "temp"))
        self._track_file_upload(response.id, temp_file_name)
        return response.id


    def _track_file_upload(self, file_id, file_name):
        response   = self._client.files.retrieve(file_id)
        while response.status in ["pending", "running"]:
            time.sleep(1) # Poll every second
            response = self._client.files.retrieve(file_id)
        file_name = file_name.split("\\")[-1]
        print(f"Upload of file {file_name} with id {file_id} finished with status: "
              f"{response.status}")


    def _start_finetuning(self, training_prompts_id, validation_prompts_id):
        response = self._client.fine_tuning.jobs.create(
            training_file   = training_prompts_id,
            validation_file = validation_prompts_id,
            model           = self._azure_config["model"],
            hyperparameters = {
                "batch_size":               self._azure_config["batch_size"], 
                "learning_rate_multiplier": self._azure_config["learning_rate_multiplier"], 
                "n_epochs":                 self._azure_config["epochs"],
                "seed":                     42
            }
        )
        return response.id


    def _track_finetuning(self, job_id):
        print("\n############################")
        print("### Fine-tuning started! ###")
        print("############################")
        print(f"Job ID: {job_id}\n")
        response   = self._client.fine_tuning.jobs.retrieve(job_id)
        start_time = time.time()
        while response.status not in ["succeeded", "failed"]:
            time.sleep(60) # Poll every 60 seconds
            response = self._client.fine_tuning.jobs.retrieve(job_id)
            print(f"Elapsed time: {int((time.time() - start_time) // 60)} minute(s)")
            print(f"Status: {response.status}")
        print("#############################")
        print("### Fine-tuning finished! ###")
        print("#############################")
        print(f"Status: {response.status}")


    def _deploy_model(self, job_id):
        env = self._load_env()
        response = requests.put(
            url = (
                f"https://management.azure.com/subscriptions/{env['subscription_id']}/"
                f"resourceGroups/{env['ressource_group']}/providers/Microsoft.CognitiveServices/"
                f"accounts/{env['ressource_name']}/deployments/"
                f"{self._azure_config['deployment_name']}"
            ),
            params = {
                "api-version": env["api_version"],
            },
            headers = {
                "Authorization": f"Bearer {self._generate_azure_token()}",
                "Content-Type":   "application/json"
            },
            data = json.dumps({
                "sku": {
                    "name":     "standard", 
                    "capacity": 1
                },
                "properties": {
                    "model": {
                        "format":  "OpenAI",
                        "name":    f"{self._azure_config['model']}.{job_id.replace('job', '')}",
                        "version": "1"
                    }
                }
            })
        )
        if response.status_code == 200:
            print(f"Deployment of model {self._azure_config['deployment_name']} with id {job_id} "
                  f"succeeded!")
        elif response.status_code == 201:
            print(f"Deployment of model {self._azure_config['deployment_name']} with id {job_id} "
                  f"successfully created!")
        else:
            raise ValueError(f"Deployment of model {self._azure_config['deployment_name']} with "
                             f"id {job_id} failed with code {response.status_code} and reason "
                             f"'{response.reason}'")


    def _withdraw_model(self):
        env = self._load_env()
        response = requests.delete(
            url = (
                f"https://management.azure.com/subscriptions/{env['subscription_id']}/"
                f"resourceGroups/{env['ressource_group']}/providers/Microsoft.CognitiveServices/"
                f"accounts/{env['ressource_name']}/deployments/"
                f"{self._azure_config['deployment_name']}"
            ),
            params = {
                "api-version": env["api_version"],
            },
            headers = {
                "Authorization": f"Bearer {self._generate_azure_token()}",
                "Content-Type":   "application/json"
            }
        )
        if response.status_code == 200:
            print(f"Deployment of model {self._azure_config['deployment_name']} successfully "
                   "deleted!")
        elif response.status_code == 204:
            print(f"Deployment of model {self._azure_config['deployment_name']} had already been "
                   "deleted!")
        else:
            raise ValueError(f"Deleting deployment of model "
                             f"{self._azure_config['deployment_name']} failed with code "
                             f"{response.status_code} and reason '{response.reason}'. Take care "
                             f"of this! Deployed models are billed hourly!")


    def _load_env(self):
        dotenv.load_dotenv()
        return {
            "openai_token":    os.getenv("OPENAI_TOKEN"),
            "subscription_id": os.getenv("SUBSCRIPTION_ID"),
            "ressource_group": os.getenv("RESSOURCE_GROUP"),
            "ressource_name":  os.getenv("RESSOURCE_NAME"),
            "api_version":     os.getenv("API_VERSION"),
            "openai_endpoint": os.getenv("OPENAI_ENDPOINT")
        }


    def _generate_azure_token(self):
        cli = azure.cli.core.get_default_cli()
        print("\n### Start generating azure token ###\n")
        cli.invoke(["account", "get-access-token"])
        print("\n### Finished generating azure token ###\n")
        return cli.result.result["accessToken"]
