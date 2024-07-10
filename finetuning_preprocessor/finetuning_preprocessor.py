import os
import copy
import pickle
import numpy
import pandas
import sklearn.preprocessing


class FinetuningPreprocessor():

    def __init__(self, config):
        self._config = config


    def prepare_prompts(self):
        input_dir = self._config["input_dir"]
        training_data   = self._load_data(self._config["training_data_file_name"], input_dir)
        validation_data = self._load_data(self._config["validation_data_file_name"], input_dir)
        
        unscaled_training_targets   = self._extract_targets(training_data)
        unscaled_validation_targets = self._extract_targets(validation_data)
        
        scaler                    = self._fit_scaler(unscaled_training_targets)
        #scaled_training_targets   = self._scale_targets(scaler, unscaled_training_targets)
        #scaled_validation_targets = self._scale_targets(scaler, unscaled_validation_targets)

        training_prompts   = self._define_prompts(training_data)
        validation_prompts = self._define_prompts(validation_data)
        
        if self._config["few_shot"]:
            input_dir = self._config["evaluate_dir"]
            test_data       = self._load_data(self._config["test_data_file_name"], input_dir)
            few_shot_data       = self._load_data(self._config["few_shot_data_file_name"], input_dir)
            unscaled_test_targets       = self._extract_targets(test_data)
            few_shot_prompts = self._define_few_shot_prompts(few_shot_data)
            test_prompts       = self._define_test_prompts(few_shot_prompts, test_data)
            
        else:
            input_dir = self._config["input_dir"]
            test_data       = self._load_data(self._config["test_data_file_name"], input_dir)
            unscaled_test_targets       = self._extract_targets(test_data)
            test_prompts = self._define_prompts(test_data)
            
        #scaled_test_targets       = self._scale_targets(scaler, unscaled_test_targets)
        training_prompts_and_scaled_targets   = self._combine_prompts_and_targets(training_prompts,   unscaled_training_targets)
        validation_prompts_and_scaled_targets = self._combine_prompts_and_targets(validation_prompts, unscaled_validation_targets)
        test_prompts_and_scaled_targets       = self._combine_prompts_and_targets(test_prompts,       unscaled_test_targets)

        self._clear_output_dir()
        self._save_scaler(scaler)
        self._save_prompts_and_targets_binary(   "training",   training_prompts_and_scaled_targets)
        self._save_prompts_and_targets_binary(   "validation", validation_prompts_and_scaled_targets)
        self._save_prompts_and_targets_binary(   "test",       test_prompts_and_scaled_targets)
        self._save_prompts_and_targets_textually("training",   training_prompts_and_scaled_targets,   unscaled_training_targets)
        self._save_prompts_and_targets_textually("validation", validation_prompts_and_scaled_targets, unscaled_validation_targets)
        self._save_prompts_and_targets_textually("test",       test_prompts_and_scaled_targets,      unscaled_test_targets)


    def _load_data(self, data_file_name, input_dir):
        return pandas.read_csv(
            filepath_or_buffer = os.path.join(input_dir, f"{data_file_name}.csv"),
            index_col          = 0,
        )


    def _extract_targets(self, data):
        targets = []

        for _, target_column in data.iloc[:, -self._config["output_features"]:].items():
            targets.append(target_column.to_list())

        if len(targets) == 1:
            return targets[0]

        return targets # Caution! Targets with two or more features are returned as list of lists!


    def _fit_scaler(self, unscaled_targets):
        unscaled_targets_as_array = numpy.reshape(numpy.array(unscaled_targets), (-1,1))
        return sklearn.preprocessing.StandardScaler().fit(unscaled_targets_as_array)


    def _scale_targets(self, scaler, unscaled_targets):
        unscaled_targets_as_array = numpy.reshape(numpy.array(unscaled_targets), (-1,1))
        scaled_targets_as_array = scaler.transform(copy.deepcopy(unscaled_targets_as_array))
        return scaled_targets_as_array[:,0].tolist()


    def _define_prompts(self, data):
        prompts = []

        for input, _ in data.iterrows():
            
            prompt = "The given examples are samples from a mathematical function mapping input values (x) to output values (y)."
            prompt += f"Input = {input}, Output = , "
            prompt += "The task is to learn the underlying pattern between the input and output values, and infer the mathematical function that maps the input values to the output values."           
            prompts.append(prompt)

        return prompts
    
    
    def _define_few_shot_prompts(self, samples):
        few_shot_prompt = f' --- EXAMPLES ---'
        for input, output in samples.iterrows():
            few_shot_prompt += f"Input = {input}, Output = {output['output']}, "
        return few_shot_prompt
    
    
    def _define_test_prompts(self, few_shot_prompt, test_data):
        test_prompts = self._define_prompts(test_data)
        final_test_prompt = []
        for test_prompt in  test_prompts:
            final_test_prompt.append([few_shot_prompt,test_prompt])
        return final_test_prompt
        


    def _combine_prompts_and_targets(self, prompts, targets):
        return [{"text": prompt, "label": target} for prompt, target in zip(prompts, targets)]


    def _clear_output_dir(self):
        for file in os.listdir(self._config["output_dir"]):
            os.remove(os.path.join(self._config["output_dir"], file))


    def _save_scaler(self, scaler):
        out_path = os.path.join(
            self._config["output_dir"],
            "scaler.pickle"
        )
        with open(out_path, "wb") as out_file:
            pickle.dump(
                obj      = scaler,
                file     = out_file,
                protocol = pickle.HIGHEST_PROTOCOL
            )


    def _save_prompts_and_targets_binary(self, file_name_prefix, prompts_and_scaled_targets):
        out_path = os.path.join(
            self._config["output_dir"],
            f"{file_name_prefix}_prompts_and_targets.pickle"
        )
        with open(out_path, mode="wb") as out_file:
            pickle.dump(
                obj      = prompts_and_scaled_targets,
                file     = out_file,
                protocol = pickle.HIGHEST_PROTOCOL
            )


    def _save_prompts_and_targets_textually(self, file_name_prefix, prompts_and_scaled_targets, unscaled_targets):
        out_path = os.path.join(
            self._config["output_dir"],
            f"{file_name_prefix}_prompts_and_targets.txt"
        )
        with open(out_path, mode="w") as out_file:
            for prompt_and_scaled_target, unscaled_target in zip(prompts_and_scaled_targets, unscaled_targets):
                out_file.write(f"{prompt_and_scaled_target['text']} -> "
                               f"Scaled: {  round(prompt_and_scaled_target['label'], 3)} / "
                               f"Unscaled: {round(unscaled_target,                   3)}\n")
