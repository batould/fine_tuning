import os 
import numpy as np 
import math 
import pandas as pd
import csv


class DummyRegressor():
    '''
        Class to generate output and input data in csv file 
        CSV file then saved in data directory 
        data-> raw_data -> csv file
    '''
    def __init__(self,  size_test, size_train, size_validate, k):
        '''
            k: defines the constant
            size_train: size of training data
            size_test: size of test data
        '''
        self.k = k 
        self.size_train = size_train 
        self.size_test = size_test
        self.size_validate = size_validate
        self.total_samples = size_test + size_train + size_validate
    
    
    def set_directory (self, destination):
        current_directory = os.getcwd()
        data_folder = current_directory + f'{destination}'
        if os.path.exists(data_folder):
            print('true')
        else:
            os.mkdir(data_folder)        
        return data_folder
    
    
    def generate_data(self):
        input = np.random.randint(-100,100, (self.total_samples))
        output = np.sin(input) + self.k* np.cos(input**2) # function1
        #output = np.sin(input**2) + self.k * np.cos(input**2) + np.cbrt(np.pi**self.k) # function2 
     
        return input,output
    
    
    def split_data(self):
        [input,output] = self.generate_data()
        train_data = np.stack([input[0:self.size_train], output[0:self.size_train]]).reshape(2,-1)
        test_data = np.stack([input[self.size_train:self.size_test+self.size_train], output[self.size_train:self.size_test+self.size_train]]).reshape(2,-1)
        validate_data = np.stack([input[self.size_test+self.size_train: ], output[self.size_test+self.size_train:]]).reshape(2,-1)
        return [train_data, test_data, validate_data]
        
    
    def save_output(self, directory, data, file_name):
        fields = ['input','output']
        dataframe = pd.DataFrame(data.T, columns = fields)
        dataframe.to_csv(directory + f'/{file_name}', index = False)
        
        
    def generate_output(self, input_dir):
        train_data, test_data, validate_data = self.split_data()
        #directory_input = _set_config_generate(function, model)["input_dir"]
        #regressor.save_output(directory_input, train_data, _set_config_generate(function, model)['training_data_file_name']+'.csv')
        #regressor.save_output(directory_input, test_data, _set_config_generate(function, model)['test_data_file_name']+'.csv')
        self.save_output(input_dir, train_data, 'few_shot_data.csv' )
        self.save_output(input_dir, test_data, 'test_data.csv' )
        #self.save_output(input_dir, validate_data, 'validation_data.csv' )
        #self.save_output(input_dir, validate_data, 'training_data.csv' )
        
    def generate_dic(self):
        [input,output] = self.generate_data()
        text = []
        for input, output in zip(input,output):
            text.append(f'Input = {input} Output = {output}')
        return text
    
    
    def save_data(self):
        config = os.getcwd()      
        file_name = os.path.join(config, "generate_data", "test_dataset.csv") 
        text = self.generate_dic()
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            for example in text:
                writer.writerow([example])
                  
    
if __name__ == "__main__":
    k = 3
    size_train = 7
    size_test = 10
    size_validate = 0
    function = 1
    input_dir = os.path.join(os.getcwd(), "finetuning_preprocessor", "input")
    regressor = DummyRegressor(k = k, size_train = size_train, size_test = size_test, size_validate=size_validate)
    #text = regressor.save_data()
    regressor.generate_output(input_dir=input_dir)
    "meta-llama/Meta-Llama-3-8B"