from finetuning_preprocessor import FinetuningPreprocessor
from configurations import _set_config_finetune_data_preprocessor #_set_config_raw_data_preprocess
from ucimlrepo import fetch_ucirepo 
#import data_preprocessor


#data_preprocess = _set_config_raw_data_preprocess()

#dp = data_preprocessor.DataPreprocessor(data_preprocess)
#dp.run_raw_preprocess()

finetune_preprocess = _set_config_finetune_data_preprocessor()
fp = FinetuningPreprocessor(finetune_preprocess)
fp.prepare_prompts()