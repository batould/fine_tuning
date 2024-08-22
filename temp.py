import pandas as pd

path = r'C:\Users\batou\OneDrive\Desktop\Master Thesis\LLM\FineTuning\finetuning_preprocessor\data_input\mesh\train2k,test100,validate100\unscaled_validation_data.csv'
df = pd.read_csv(path, delimiter=',')
print(df.shape)