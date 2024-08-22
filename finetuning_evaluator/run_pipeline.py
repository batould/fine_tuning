from configurations import _set_hugging_face_config_trainer, _set_general_config_trainer, _set_azure_config_trainer, _set_config_finetune_model_preprocessor
import finetuning_preprocessor
from huggingface_trainer import HuggingFaceFinetuningTrainer
from testing_finetuned_model import FinetunedModel
from azure_training import AzureFinetuningTrainer

    
def fine_tune_preprocess(preprocessor_config):
    fp = finetuning_preprocessor.FinetuningPreprocessor(preprocessor_config)
    print("Loaded preprocessor")
    fp.prepare_prompts()


def test(trainer_config):
    ftest = FinetunedModel(trainer_config)
    ftest.run_test() 
    print(f'Done testing')
    
if __name__ == "__main__":
 
    general_config = _set_general_config_trainer()
    preprocessor_config = _set_config_finetune_model_preprocessor()
    fine_tune_preprocess(preprocessor_config=preprocessor_config)

    if general_config["platform"] == "huggingface":
        hugging_face_config = _set_hugging_face_config_trainer()
        ft = HuggingFaceFinetuningTrainer(general_config, hugging_face_config)
        ft.run_finetuning()
        
    elif general_config["platform"] == "azure":
        azure_config = _set_azure_config_trainer()
        at = AzureFinetuningTrainer(general_config, azure_config)
        prompt_ids = at.upload_prompts()
        at.run_finetuning(prompt_ids[0], prompt_ids[1])
        # at.deploy_finetuned_model(job_id="")
        # at.withdraw_finetuned_model()
        at.clear_prompts()
        
    else:
        raise ValueError(f"Platform {general_config['platform']} is unvalid!")
    
    #test(trainer_config=trainer_config)