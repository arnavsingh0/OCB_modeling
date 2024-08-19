import pandas as pd

# Specify the file paths
txt_file_path = 'AE_OCB2.txt'
csv_file_path = 'AE_OCB2.csv'

# Read the .txt file
# Assuming the file is space-delimited; change the delimiter if needed
df = pd.read_csv(txt_file_path, delimiter =',')

# Display the first few rows of the dataframe
print(df.head())

newheaders = ['AE','OCB']
df.columns = newheaders
# Write the dataframe to a .csv file
df.to_csv(csv_file_path, index=False)
print("Headers", df.columns.tolist())
print(f"File has been converted to {csv_file_path}")
