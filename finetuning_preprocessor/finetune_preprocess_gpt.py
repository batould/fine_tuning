import os
import copy
import pickle
import numpy
import pandas
import sklearn.preprocessing
import json

class GPTFinetuningPreprocessor():

    def __init__(self, config):
        self._config = config
        if self._config["scale"]:
            self._scaler = sklearn.preprocessing.StandardScaler()


    def prepare_prompts(self):
        input_dir = self._config["input_dir"]
        training_data   = self._load_data(self._config["training_data_file_name"], input_dir)
        validation_data = self._load_data(self._config["validation_data_file_name"], input_dir)
        test_data       = self._load_data(self._config["test_data_file_name"], input_dir)

        if self._config["scale"]:
            training_data = self._scale_data(training_data, training_data, ["input", "output"])
            test_data = self._scale_data(training_data, test_data, ["input", "output"])
            validation_data = self._scale_data(training_data, validation_data, ["input", "output"])
         
        training_prompts   = self._define_gpt_prompts(training_data)
        validation_prompts = self._define_gpt_prompts(validation_data)
        test_prompts = []
        test_data_batches = self._split_batches(test_data)
        for test_df_batch in test_data_batches:
            test_prompt = self.define_batch_test_prompt(test_df_batch)
            test_prompts.append(test_prompt)
        
        system_and_test_prompts = [[self.define_system_prompt()] + [test_prompt] for test_prompt in test_prompts]
            

        self._clear_output_dir()
        self._save_prompts_and_targets_binary(   "training",   training_prompts)
        self._save_prompts_and_targets_binary(   "validation", validation_prompts)
        self._save_prompts_and_targets_binary(   "test",       system_and_test_prompts)
        self._save_prompts_and_targets_textually("training",   training_prompts)
        self._save_prompts_and_targets_textually("validation", validation_prompts)
        self._save_prompts_and_targets_textually("test",       system_and_test_prompts)


    def _load_data(self, data_file_name, input_dir):
        #columns_head_construct = ["Cement", "Blast Furnace", "Fly Ash", "Water", "Superplasticizer", "Coarse Aggregate", "Fine Aggregate", "Age", "Concrete Compressive Strength"]
        columns_head_abalone = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
        #columns_head_function = ["input", "output"]
        return pandas.read_csv(
            filepath_or_buffer = os.path.join(input_dir, f"{data_file_name}.csv"),
            delimiter=',',
            skiprows=1,
            names=columns_head_abalone   
        )

    
    def _scale_data(self, training_data, data_to_scale, columns):
        for column in columns:
            scalar = self._scaler.fit(training_data[column])
            data_to_scale[column] = scalar.transform(data_to_scale[column])
        return [scalar, data_to_scale]
    

    def _export_scaler(self):
        with open(os.path.join(self._config["output_dir"], "scaler.pickle"), "wb") as out_file:
            pickle.dump(obj=self._scaler, file=out_file, protocol=pickle.HIGHEST_PROTOCOL)

    
    def _define_gpt_prompts(self,data):
        all_prompts = []
        system_prompt = self.define_system_prompt()
        data = data.to_dict("records")
        
        for i in range(len(data)):
            prompt = {}
            prompt_message = []
            #user_message = {"role": "user", "content": f'\n Input = {data[i]["input"]}'}
            user_message = {"role": "user", "content": f'\n Measurements: \n -Cement: {data[i]["Cement"]} kg/m³, \n- Blast Furnace Slag: {data[i]["Blast Furnace"]}kg/m³\n- Fly Ash: {data[i]["Fly Ash"]} kg/m³\n- Water: {data[i]["Water"]} kg/m³\n- Superplasticizer: {data[i]["Superplasticizer"]} kg/m³\n- Coarse Aggregate: {data[i]["Coarse Aggregate"]} kg/m³\n- Fine Aggregate: {data[i]["Fine Aggregate"]} kg/m³\n- Age: {data[i]["Age"]}days'}
            #user_message = {"role": "user", "content": f'-Sex={data[i]["Sex"]} \n- Length={data[i]["Length"]} mm \n- Diamerter={data[i]["Diameter"]} mm \n- Height={data[i]["Height"]} mm \n- Whole weight={data[i]["Whole weight"]} grams \n-Shucked weight={data[i]["Shucked weight"]} grams \n- Viscera weight={data[i]["Viscera weight"]} grams \n- Shell weight={data[i]["Shell weight"]} grams \n'}
            #assistant_message = {"role": "assistant", "content":f'\n Number of Rings={data[i]["Rings"]}'}
            assistant_message = {"role": "assistant", "content":f'\n Concrete Compressive Strength={data[i]["Concrete Compressive Strength"]}'}
            prompt_message.append(system_prompt)
            prompt_message.append(user_message)
            prompt_message.append(assistant_message)
            prompt["messages"] = prompt_message
            all_prompts.append(prompt)
        return all_prompts
            
     
    def define_system_prompt(self):
        role = "system"
        #system_content_function = "Calculate the function values of the following inputs.\n Learn the underlying pattern between the input and output values, and infer the mathematical function that maps the input values to the output values."
        system_content = "Calculate the concrete compressive strength in MPa given the following information. The concrete compressive strength is a highly nonlinear function."
        #system_content = "The task is to predict the age of abalone from physical measurements. The age of abalone is determined by the number of rings on its shell."
        training_prompt = {"role":    role,
                        "content": system_content}
        return training_prompt
            
    
    def define_batch_test_prompt(self, test_df):
        #initial_test_prompt = "Provide the function output values for the following inputs in the given json format: {\"1\": \"f(x)=y\", \"2\": \f(x)=y\", ...}. Provide only the final values without any introduction. Do not include \"rad\" or \"degree\" in the output."
        #initial_test_prompt = 'Predict the number of rings of the abalone based on the following physical measurements. Provide the output in JSON format: {\"1\": \"<predicted value>\", \"2\": \<predicted value>\", ...}. Provide only the final values without any introduction.'
        initial_test_prompt = 'Predict the concrete compressive strenght based on the following measurements. Provide the output in JSON format: {\"1\": <predicted value>, \"2\": <predicted value>, ..."}. Provide only the final values without any introduction.'
        data_test_prompt    = self._define_data_test_prompt(test_df)
        test_prompt = {"role":    "user",
                       "content": initial_test_prompt + data_test_prompt}
        return test_prompt
    
    
    def _define_data_test_prompt(self, test_df):
        samples_dict = {}

        # Helper function to format each row
        def _sample_to_dict(row):
            #samples_function = f"f({row['input']}) = ..."
            
            samples_concrete = (
            f"\n Measurements: \n"
            f"- Cement: {row['Cement']} kg/m³, \n"
            f"- Blast Furnace Slag: {row['Blast Furnace Slag']} kg/m³\n"
            f"- Fly Ash: {row['Fly Ash']} kg/m³\n"
            f"- Water: {row['Water']} kg/m³\n"
            f"- Superplasticizer: {row['Superplasticizer']} kg/m³\n"
            f"- Coarse Aggregate: {row['Coarse Aggregate']} kg/m³\n"
            f"- Fine Aggregate: {row['Fine Aggregate']} kg/m³\n"
            f"- Age: {row['Age']} days\n"
            "Predicted Concrete Compressive Strength = ..."
        )
            
            
            samples_dict[row.name] = samples_concrete

        test_df.apply(_sample_to_dict, axis=1)

        return json.dumps(samples_dict)
    
    
    def _split_batches(self, df): 
        n = self._config["samples_per_batch"]
        total_samples = df.shape[0]
        df_batches = []
        
        for idx in range(0, total_samples, n):
            batch = df.iloc[idx:idx+n]
            df_batches.append(batch)
            
        return df_batches
        

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


    def _save_prompts_and_targets_textually(self, file_name_prefix, prompts_and_scaled_targets):
        out_path = os.path.join(
            self._config["output_dir"],
            f"{file_name_prefix}_prompts_and_targets.txt"
        )
        with open(out_path, mode="w") as out_file:
            for prompt_and_scaled_target in prompts_and_scaled_targets:
                out_file.write(f"{prompt_and_scaled_target}")