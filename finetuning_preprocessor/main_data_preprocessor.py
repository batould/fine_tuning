"""Main module.

The module contains the main function.
"""
import os
import data_preprocessor

# Todo: Löschen von Samples einarbeiten. Vlt. über train_test_split, damit man das stratified machen kann. Dann schmeißt man einfach den test-teil weg und erklärt den train-teil zum neuen Datensatz.

def main():
    """Main function.

    Creates and instance of the DataPreprocessor class and uses it. Parameters may be set in
    _set_config.
    """
    config = _set_preprocess_data_config()

    dp = data_preprocessor.DataPreprocessor(config)
    dp.run_raw_preprocess()


def _set_preprocess_data_config():
    cwd = os.getcwd()
    config = {
        "input_dir":                 os.path.join(cwd, "finetuning_preprocessor",  "finetune_data", "input", "f1-500train,50test,50 validate"),
        "output_dir": os.path.join(os.getcwd(),  "finetuning_preprocessor", "finetune_data", 'output'),
        "info_file_name":            "info",
        "data_file_name":            "unscaled_training_data.csv",
        "output_data_file_name":     "data",
        "input_already_split":       False,          # If only a subset of the possible splits
        "split_data_file_names":                     # training/validation/test) is provided,
            {"training":   "training_data.csv",      # reduce dictionary to according keys.
             "validation": "validation_data.csv",
             "test":       "test_data.csv"
            }
    }
    return config


if __name__ == "__main__":
    main()
