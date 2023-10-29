from CanDataLoader import CanDataLoader
from dotenv import load_dotenv
import os

# Replace 'your_data_dir_path_here' with the actual path to your data directory
load_dotenv()
data_path = os.getenv('DATA_PATH')
data_loader = CanDataLoader(data_path)

# Get the processed ambient data as a dictionary of DataFrames:
ambient_data_dict = data_loader.get_ambient_data()

# Get the processed attack data as a dictionary of DataFrames:
attack_data_dict = data_loader.get_attack_data()

# Print out the data:
# For ambient data:
for key, df in ambient_data_dict.items():
    print(f"Ambient Data for {key}:")
    print(df.head())  # Print the first 5 rows of each DataFrame

# For attack data:
for key, df in attack_data_dict.items():
    print(f"Attack Data for {key}:")
    print(df.head())  # Print the first 5 rows of each DataFrame