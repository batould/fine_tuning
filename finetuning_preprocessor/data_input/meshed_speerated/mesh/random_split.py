import pandas as pd


n_samples = 1000
data_dir = "all_data/all_unscaled_training_data.csv"
df = pd.read_csv(data_dir, delimiter=',')
header = df.head(1)
# Sample 14 rows randomly from the remaining rows
sampled_rows = df.iloc[1:].sample(n=n_samples, random_state=1)

# Concatenate the header and the sampled rows
result = pd.concat([header, sampled_rows])

# Save the resulting DataFrame to a new CSV file (optional)
result.to_csv('unscaled_validation_data.csv', index=False)
