import os
from configurations import _set_config_trainer, _set_config_preprocessor, _set_config_causal_trainer
import finetuning_preprocessor
from finetuning_trainer import FinetuningTrainer
from finetuning_trainer_causal import FintuneTrainer_Causal
from testing_finetuned_model import FinetunedModel



def preprocess(preprocessor_config):
    fp = finetuning_preprocessor.FinetuningPreprocessor(preprocessor_config)
    print("Loaded preprocessor")
    fp.prepare_prompts()

def train(trainer_config):
    ft = FinetuningTrainer(trainer_config)
    ft.run_finetuning()
    print(f'Trainer is done training adn testing')

def test(trainer_config):
    ftest = FinetunedModel(trainer_config)
    ftest.run_test() 
    print(f'Done testing')
    
if __name__ == "__main__":
 
    trainer_config = _set_config_trainer()
    preprocessor_config = _set_config_preprocessor()
    preprocess(preprocessor_config=preprocessor_config)
    #train(trainer_config=trainer_config)
    test(trainer_config=trainer_config)