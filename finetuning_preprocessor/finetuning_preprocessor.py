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
        scaled_training_targets   = self._scale_targets(scaler, unscaled_training_targets)
        scaled_validation_targets = self._scale_targets(scaler, unscaled_validation_targets)
  
        training_prompts   = self._define_mesh_prompts(training_data)
        validation_prompts = self._define_mesh_prompts(validation_data)
        
        if self._config["few_shot"]:
            input_dir = self._config["evaluate_dir"]
            test_data       = self._load_data(self._config["test_data_file_name"], input_dir)
            #few_shot_data       = self._load_data(self._config["few_shot_data_file_name"], input_dir)
            unscaled_test_targets       = self._extract_targets(test_data)
            #few_shot_prompts = self._define_few_shot_prompts(few_shot_data)
            test_prompts       = self._define_test_prompts(test_data)
            
        else:
            input_dir = self._config["input_dir"]
            test_data       = self._load_data(self._config["test_data_file_name"], input_dir)
            unscaled_test_targets       = self._extract_targets(test_data)
            test_prompts = self._define_mesh_prompts(test_data)
            
        scaled_test_targets       = self._scale_targets(scaler, unscaled_test_targets)
        training_prompts_and_scaled_targets   = self._combine_prompts_and_targets(training_prompts,   scaled_training_targets)
        validation_prompts_and_scaled_targets = self._combine_prompts_and_targets(validation_prompts, scaled_validation_targets)
        test_prompts_and_scaled_targets       = self._combine_prompts_and_targets(test_prompts,       scaled_test_targets)

        self._clear_output_dir()
        self._save_scaler(scaler)
        self._save_prompts_and_targets_binary(   "training",   training_prompts_and_scaled_targets)
        self._save_prompts_and_targets_binary(   "validation", validation_prompts_and_scaled_targets)
        self._save_prompts_and_targets_binary(   "test",       test_prompts_and_scaled_targets)
        self._save_list_prompts_and_list_targets_textually("training",   training_prompts_and_scaled_targets,   unscaled_training_targets)
        self._save_list_prompts_and_list_targets_textually("validation", validation_prompts_and_scaled_targets, unscaled_validation_targets)
        self._save_list_prompts_and_list_targets_textually("test",       test_prompts_and_scaled_targets,      unscaled_test_targets)


    def _load_data(self, data_file_name, input_dir):
        #columns_head = ["Cement", "Blast Furnace", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age", "Concrete Compressive Strength"]
        #columns_head = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
        #columns_head = ["input", "output"]
        return pandas.read_csv(
            filepath_or_buffer = os.path.join(input_dir, f"{data_file_name}.csv"),
            delimiter=',',
            #skiprows=1,
            #names=columns_head   
        )


    def _extract_targets(self, data):
        targets = []

        for _, target_column in data.iloc[:, -self._config["output_features"]:].items():
            targets.append(target_column.to_list())

        if len(targets) == 1:
            return targets[0]

        return targets # Caution! Targets with two or more features are returned as list of lists!


    def _fit_scaler(self, unscaled_targets):
        n_labels = len(unscaled_targets)
        unscaled_targets_as_array = numpy.reshape(numpy.array(unscaled_targets), (-1,n_labels))
        return sklearn.preprocessing.StandardScaler().fit(unscaled_targets_as_array)


    def _scale_targets(self, scaler, unscaled_targets):
        n_labels = len(unscaled_targets)
        unscaled_targets_as_array = numpy.reshape(numpy.array(unscaled_targets), (-1,n_labels))
        scaled_targets_as_array = scaler.transform(copy.deepcopy(unscaled_targets_as_array))
        scaled_targets_as_list = []
        
        for label in range(n_labels):
            scaled_targets_as_list.append(scaled_targets_as_array[:,label].tolist())
            
        return scaled_targets_as_list
    


    def _define_science_prompts(self,data):
        prompts = []
        data = data.to_dict("records")

        for i in range(len(data)):
            #prompt = "The task is to predict the concrete compressive strength in MPa given the following input variables"
            #prompt += f'\n Input Variables: \n -Cement: {data[i]["Cement"]} kg/m³, \n- Blast Furnace Slag: {data[i]["Blast Furnace"]}kg/m³\n- Fly Ash: {data[i]["Fly Ash"]} kg/m³\n- Water: {data[i]["Water"]} kg/m³\n- Superplasticizer: {data[i]["Superplasticizer"]} kg/m³\n- Coarse Aggregate: {data[i]["Coarse Aggregate"]} kg/m³\n- Fine Aggregate: {data[i]["Fine Aggregate"]} kg/m³\n- Age: {data[i]["Age"]}days'
            #prompt += f'\n- Predicted Concrete Compressive Strength:\n'
            #prompts.append(prompt)
            prompt = "The task is to predict the number of rings given the following information: \n"
            prompt += f'-Sex={data[i]["Sex"]} \n- Length={data[i]["Length"]} mm \n- Diamerter={data[i]["Diameter"]} mm \n- Height={data[i]["Height"]} mm \n- Whole weight={data[i]["Whole weight"]} grams \n-Shucked weight={data[i]["Shucked weight"]} grams \n- Viscera weight={data[i]["Viscera weight"]} grams \n- Shell weight={data[i]["Shell weight"]} grams \n'
            prompt += f'Number of rings =\n '
            prompts.append(prompt)

        return prompts
        
    def _define_function_prompts(self, data):
        prompts = []
        data = data.to_dict("records")

        for i in range(len(data)):
            prompt = "The given examples are samples from a mathematical function mapping input values (x) to output values (y).\n"
            prompt += f'Input={data[i]["input"]}, Output= '
            prompt += "\nThe task is to learn the underlying pattern between the input and output values, and infer the mathematical function that maps the input values to the output values.\n"  
            prompts.append(prompt)
        return prompts
    
    
    def _define_mesh_prompts(self, data):
        prompts = []
        data = data.to_dict("records")
        parameters = {
                "material_property": {
                    "log_beta": "Logarithm of the ratio between Lamé constants lambda and mu"
                },
                "external_forces": {
                    "f_x": "Areal load in the x direction",
                    "f_y": "Areal load in the y direction"
                },
                "geometry": {
                    "p_x_i": "X coordinate of the start point of edge i",
                    "p_y_i": "Y coordinate of the start point of edge i",
                    "x_x": "X coordinate of a point in the polygon",
                    "x_y": "Y coordinate of a point in the polygon"
                },
                "boundary_conditions": {
                    "delta_i": "Flag for boundary condition type of edge i (0 for Neumann, 1 for Dirichlet)",
                    "v_x_i": "X component of boundary condition for edge i",
                    "v_y_i": "Y component of boundary condition for edge i"
                },
                "prediction_error": {
                    "log_c": "Logarithm of the parameter for bias of prediction error line in log-log plot",
                    "m": "Slope of the prediction error line in log-log plot"
                }
            }
        
        for i in range(len(data)):
            data_element = f"\n--- GLOBAL PARAMETERS ---\n"
            
            for key, value in data[i].items():
                
                if key in ["log(c)", "m"]:
                    continue
                data_element += f'{key} = {value}'
            data_element += "\nLog(c)= ..., m=...\n"
            
            prompt = "The given examples describe the relationship between the global parameters and the prediction error parameters (log(c), and m)."
            prompt += f"{data_element}"      
            prompt += "\n The task is to predict the parameters log(c) and m based on the given global parameters to determine the finite element mesh size."
            prompts.append(prompt)
        return prompts
    
            
    
    def _define_few_shot_prompts(self, samples):
        few_shot_prompt = f' --- EXAMPLES ---'
        for input, output in samples.iterrows():
            few_shot_prompt += f"Input = {input}, Output = {output['output']}, "
        return few_shot_prompt
    
    
    def _define_test_prompts(self, test_data, few_shot_prompt=None):
        test_prompts = self._define_prompts(test_data)
        if few_shot_prompt:
            final_test_prompt = []
            for test_prompt in  test_prompts:
                final_test_prompt.append([few_shot_prompt,test_prompt])
            return final_test_prompt
        else: 
            return test_prompts
        

    def _combine_prompts_and_targets(self, prompts, targets):
        transposed_targets = list(zip(*targets))
        combined = []
        for prompt, target_values in zip(prompts, transposed_targets):
            # If there's only one target value, don't wrap it in a tuple
            if len(target_values) == 1:
                combined.append({"text": prompt, "label": target_values[0]})
            else:
                combined.append({"text": prompt, "label": tuple(target_values)})
        
        return combined
            


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
                               f"Scaled: {  round(prompt_and_scaled_target['label'], 3)} \n "
                               f"Unscaled: {round(unscaled_target,                   3)}\n")
                

    def _save_list_prompts_and_list_targets_textually(self, file_name_prefix, prompts_and_scaled_targets, unscaled_targets):
        out_path = os.path.join(
            self._config["output_dir"],
            f"{file_name_prefix}_prompts_and_targets.txt"
        )
        with open(out_path, mode="w") as out_file:
            transposed_targets = list(zip(*unscaled_targets))
            for prompt_and_scaled_target, unscaled_target in zip(prompts_and_scaled_targets, transposed_targets):
                    out_file.write(f"{prompt_and_scaled_target['text']} -> "
                                f"Scaled: \n{prompt_and_scaled_target['label']} \n "
                                f"Unscaled: {unscaled_target}\n")
