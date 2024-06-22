import os 
import numpy as np 
import math 
import pandas as pd
import csv
#from configurations import _set_config_generate


class DummyRegressor():
    '''
        Class to generate output and input data in csv file 
        CSV file then saved in data directory 
        data-> raw_data -> csv file
    '''
    def __init__(self,  size_test, size_train, k):
        '''
            k: defines the constant
            size_train: size of training data
            size_test: size of test data
        '''
        self.k = k 
        self.size_train = size_train 
        self.size_test = size_test
    
    
    def set_directory (self, destination):
        current_directory = os.getcwd()
        data_folder = current_directory + f'{destination}'
        if os.path.exists(data_folder):
            print('true')
        else:
            os.mkdir(data_folder)        
        return data_folder
    
    
    def generate_data(self):
        input = np.random.randint(-100,100, (self.size_test + self.size_train))
        #output = ((self.k * input)**2 + np.sqrt(self.k/self.split_rate) + (self.size*(input)**3))** (1/self.size)
        output = np.sin(input) + self.k* np.cos(input**2)
        #output = np.sin(input**2) + self.k * np.cos(input**2) + np.cbrt(np.pi**self.k)
     
        return input,output
    
    
    def split_data(self):
        [input,output] = self.generate_data()
        train_data = np.stack([input[0:self.size_train], output[0:self.size_train]]).reshape(2,-1)
        test_data = np.stack([input[self.size_train:], output[self.size_train:]]).reshape(2,-1)
        return [train_data,test_data]
        
    
    def save_output(self, directory, data, file_name):
        fields = ['input','output']
        dataframe = pd.DataFrame(data.T, columns = fields)
        dataframe.to_csv(directory + f'/{file_name}', index = False)
        
        
    def generate_output(self, model, function):
        train_data,test_data = self.split_data()
        #directory_input = _set_config_generate(function, model)["input_dir"]
        #regressor.save_output(directory_input, train_data, _set_config_generate(function, model)['training_data_file_name']+'.csv')
        #regressor.save_output(directory_input, test_data, _set_config_generate(function, model)['test_data_file_name']+'.csv')
    
    def generate_dic(self):
        [input,output] = self.generate_data()
        text = []
        for input, output in zip(input,output):
            text.append(f'Input = {input} Output = {output}')
        return text
                  
    
if __name__ == "__main__":
    k = 3
    size_train = 7
    size_test = 10
    model  = 'llama'
    function = 2
    regressor = DummyRegressor(k = k, size_train = size_train, size_test = size_test)
    text = regressor.generate_dic()
    
    