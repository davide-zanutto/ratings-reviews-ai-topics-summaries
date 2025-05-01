import pandas as pd

# Load the CSV data
df = pd.read_csv('csv/GroundTruthProdArea20k.csv')
total_rows = len(df)

train_size = 0.9
train_output = 'csv/train.csv'
test_output = 'csv/test.csv'

# Shuffle the dataset
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Determine split index
split_index = int(total_rows * train_size)

# Split into train and test sets
train_df = df_shuffled.iloc[:split_index]
test_df = df_shuffled.iloc[split_index:]

# Save to CSV files
train_df.to_csv(train_output, index=False)
test_df.to_csv(test_output, index=False)

# Summary
print(f"Total rows: {total_rows}")
print(f"Training rows ({train_size*100}%): {len(train_df)} -> {train_output}")
print(f"Testing rows ({(1-train_size)*100}%): {len(test_df)} -> {test_output}")
