from json import load
from math import floor
from helpers import *
import os
import pandas as pd
import requests
import zipfile
import os
import shutil
import numpy as np
from tqdm import tqdm
# import polars as pl
import gc

from TorchLoader import CANDataset

DATASET_DOWNLOAD_LINK = "https://road.nyc3.digitaloceanspaces.com/road.zip"


import numpy as np
# from tensorflow.keras.utils import Sequence

# class PreparedCANDataLoaderPolars(Sequence):
#     def __init__(self, can_data, config, batch_size=32):
#         self.num_rows = 0
#         for df in can_data:
#             self.num_rows += df.height - df.filter(pl.col("aid").is_null()).height
#         self.can_data = can_data
#         self.config = config
#         self.batch_size = batch_size
#         self.num_samples = len(can_data)
#         self.current_df = 0
#         self.completed_index = 0

#     def __len__(self):
#         return self.num_rows // self.batch_size

#     def __getitem__(self, idx):
#         current_df = self.can_data[self.current_df]
#         processed_batch = self.extract_features(current_df)

#         while len(processed_batch) < self.batch_size:
#             self.current_df += 1
#             if self.current_df >= len(self.can_data):
#                 return None  # or an appropriate handling for the end of the dataset
#             self.completed_index = 0
#             current_df = self.can_data[self.current_df]
#             processed_batch = self.extract_features(current_df)
        
#         return processed_batch

#     def extract_features(self, df):
#         processed_batch = []

#         feature_len = 1
#         for _, feature_config in self.config.items():
#             feature_len += feature_config['records_back']

#         while len(processed_batch) < self.batch_size:
#             index = self.completed_index
#             self.completed_index += 1
#             row = df[index]

#             features = [row['aid']]
#             for feature_name, feature_config in self.config.items():
#                 if feature_config['specific_to_can_id']:
#                     # Filter the DataFrame by the CAN ID and then get the last N records
#                     history = df.filter(pl.col('aid') == row['aid']).limit(index)
#                     records = history.tail(feature_config['records_back'])
#                 else:
#                     # Get the last N records regardless of CAN ID
#                     records = df.slice(max(0, index - feature_config['records_back']), feature_config['records_back'])

#                 if records.height < feature_config['records_back']:
#                     # Handle padding here if necessary
#                     continue

#                 feature_values = records[feature_name].to_list()
#                 features.extend(feature_values)

#             if len(features) != feature_len:
#                 continue
#             processed_batch.append(features)
        
#         return processed_batch







# class PreparedCANDataLoader(Sequence):
#     def __init__(self, can_data, config, batch_size=1):
#         self.num_rows = 0
#         for df in can_data:
#             self.num_rows += len(df)
#             df.dropna()

#         self.can_data = can_data
#         self.config = config
#         self.batch_size = batch_size
#         self.num_samples = len(can_data)
#         self.current_df = 0
#         self.completed_index = 0
#         self.features_len = 1 # one because we are keeping the CANID to start it
#         for _, feature_config in self.config.items():
#             self.features_len += feature_config['records_back']


#     def __len__(self):
#         # Return the number of batches per epoch
#         # return (self.num_samples + self.batch_size - 1) // self.batch_size
#         return self.num_rows // self.batch_size

#     def __getitem__(self, idx):
#         current_df = self.can_data[self.current_df]
#         processed_batch = self.extract_features(current_df)

#         while len(processed_batch) < self.batch_size:
#             self.current_df += 1
#             if self.current_df >= len(self.can_data):
#                 raise StopIteration
            
#             self.completed_index = 0
#             current_df = self.can_data[self.current_df]
#             processed_batch = self.extract_features(current_df)

#         return processed_batch,processed_batch
    
#     def extract_features(self, df):

#         processed_batch = []

#         while len(processed_batch) < self.batch_size:

#             if self.completed_index >= len(df):
#                 return processed_batch
            
#             index = self.completed_index
#             self.completed_index += 1
#             row = df.iloc[index]
#             features = [row['aid']]
#             # features = []

#             for feature_name, feature_config in self.config.items():
#                 # If the feature requires looking back at specific CAN IDs
#                 if feature_config['specific_to_can_id']:
#                     # Filter the DataFrame by the CAN ID and then get the last N records
#                     history = df[df['aid'] == row['aid']].iloc[:index]
#                     records = history.tail(feature_config['records_back'])
#                 else:
#                     # Get the last N records regardless of CAN ID
#                     records = df.iloc[max(0, index - feature_config['records_back']):index]

#                 # If there aren't enough records, handle this case, e.g., by padding
#                 if len(records) < feature_config['records_back']:
#                     # Handle padding here if necessary
#                     continue

#                 feature_values = records[feature_name].tolist()
#                 features.extend(np.array(feature_values))

#             if len(features) != self.features_len:
#                 continue
#             # Add the extracted features for this row to the processed batch
#             processed_batch.append(features)
        
#         return processed_batch
    













        # you'll have to get the current rows can id
        # loop through each config and try to find N records back
        # if you cant go to next iteration through can 
        # if you get n records back 


    # def extract_features(self, df):
    #     # Initialize a list to hold the processed feature data for the batch
    #     processed_batch = []

    #     # Iterate over each row in the batch DataFrame
    #     for index, row in df.iterrows():
    #         # Initialize a list to hold the features for this row
    #         features = []

    #         # Iterate over each feature to be extracted based on the config
    #         for feature_name, feature_config in self.config.items():
    #             # If the feature requires looking back at specific CAN IDs
    #             if feature_config['specific_to_can_id']:
    #                 # Filter the DataFrame by the CAN ID and then get the last N records
    #                 history = self.can_data[self.can_data['aid'] == row['aid']].iloc[self.completed_index:index]
    #                 records = history.tail(feature_config['records_back'])
    #             else:
    #                 # Get the last N records regardless of CAN ID
    #                 records = self.can_data.iloc[max(0, index - feature_config['records_back']):index]

    #             # If there aren't enough records, handle this case, e.g., by padding
    #             if len(records) < feature_config['records_back']:
    #                 # Handle padding here if necessary
    #                 pass

    #             # Extract the feature from the records and append to the features list
    #             # For example, you might want to take the 'delta_time_last_msg' from each record
    #             feature_values = records[feature_name].tolist()
    #             features.extend(feature_values)

    #         # Add the extracted features for this row to the processed batch
    #         processed_batch.append(features)

    #     # Convert the processed batch to a NumPy array and return it
    #     return np.array(processed_batch)

    # def on_epoch_end(self):
    #     # If you want to shuffle the data at the end of each epoch, implement it here
    #     self.can_data_iter = iter(self.can_data)  # Reset the iterator for the next epoch



def payload_matches(payload, injection_data_str):
    for i, value in enumerate(payload):
        if injection_data_str[i] == "X":
            continue
        else:
            if value != injection_data_str[i]:
                return False
    return True

# def prep_data_helper(df, config):
#     X = []
#     for index, row in df.iterrows():
#         data_point = []
#         valid_data_point = True
#         for col, col_config in config.items():
#             if col_config["specific_to_can_id"]:
#                 # Extract past records for the same CAN ID
#                 same_id_rows = df[(df["aid"] == row["aid"]) & (df.index < index)]
#                 if len(same_id_rows) >= col_config["records_back"]:
#                     values = same_id_rows[col].iloc[-col_config["records_back"]:].values
#                 else:
#                     valid_data_point = False
#                     break
#             else:
#                 # Extract past records irrespective of CAN ID
#                 prev_rows = df[df.index < index]
#                 if len(prev_rows) >= col_config["records_back"]:
#                     values = prev_rows[col].iloc[-col_config["records_back"]:].values
#                 else:
#                     valid_data_point = False
#                     break
#             data_point.extend(values)
#         if valid_data_point:
#             X.append(data_point)

# def process_data_for_config_key(df, aid, config_key, config_value):
#     """
#     Helper function to process data for a specific config key.
#     """

#     if config_value['specific_to_can_id']:
#         relevant_rows = df[df['aid'] == aid].tail(config_value['records_back'])
#     else:
#         relevant_rows = df.tail(config_value['records_back'])

#     if len(relevant_rows) < config_value['records_back']:
#         return None

#     return relevant_rows[config_key].tolist()

# def data_preparation_helper(df, config):
#     """
#     Process the dataframe based on the given configuration.
#     """
#     processed_data = []

#     # Iterate over each row in the dataframe
#     for index, row in tqdm(df.iterrows(), total=df.shape[0]):
#         aid = row['aid']  # Extract the 'aid' from the current row
#         data_row = [aid]

#         skip_row = False

#         # Process each key in the config
#         for config_key, config_value in config.items():
#             processed_values = process_data_for_config_key(df[:index], aid, config_key, config_value)

#             if processed_values is None:
#                 skip_row = True
#                 break

#             data_row.extend(processed_values)

#         # If there weren't enough records back for any config key, skip the row
#         if skip_row:
#             continue

#         # Append the processed row to the result
#         processed_data.append(data_row)

#     return processed_data

# def process_data_for_config_key(df, aid, config_key, config_value):
#     """
#     Helper function to process data for a specific config key.
#     """
#     if config_value['specific_to_can_id']:
#         relevant_df = df[df['aid'] == aid]
#     else:
#         relevant_df = df

#     start_idx = max(0, len(relevant_df) - config_value['records_back'])
#     relevant_rows = relevant_df.iloc[start_idx:]

#     if relevant_rows.shape[0] < config_value['records_back']:
#         return None

#     return relevant_rows[config_key].tolist()

# def data_preparation_helper(df, config):
#     """
#     Process the dataframe based on the given configuration.
#     """
#     processed_data = []

#     # Iterate over each index in the dataframe
#     for index in tqdm(range(df.shape[0])):
#         aid = df.iloc[index]['aid']  # Extract the 'aid' from the current row
#         data_row = [aid]

#         skip_row = False

#         # Process each key in the config
#         for config_key, config_value in config.items():
#             processed_values = process_data_for_config_key(df.iloc[:index+1], aid, config_key, config_value)

#             if processed_values is None:
#                 skip_row = True
#                 break

#             data_row.extend(processed_values)

#         # If there weren't enough records back for any config key, skip the row
#         if skip_row:
#             continue

#         # Append the processed row to the result
#         processed_data.append(data_row)

#     return processed_data

# def process_data_for_config_key_optimized(df, config_key, config_value):
#     """
#     Optimized helper function to process data for a specific config key.
#     """
#     if config_value['specific_to_can_id']:
#         # Use groupby with rolling to get the last N records for each 'aid'
#         grouped = df.groupby('aid')[config_key].rolling(config_value['records_back'], min_periods=1).apply(list, raw=True).reset_index()
#         last_N_records = grouped.set_index('level_1')[config_key]
#     else:
#         last_N_records = df[config_key].rolling(config_value['records_back'], min_periods=1).apply(list, raw=True)
    
#     # Filter out rows that don't have the exact N records
#     last_N_records = last_N_records[last_N_records.apply(len) == config_value['records_back']]
    
#     return last_N_records

# def data_preparation_helper_optimized(df, config):
#     """
#     Optimized function to process the dataframe based on the given configuration.
#     """
#     result_df = df[['aid']].copy()
    
#     for config_key, config_value in config.items():
#         result_df[config_key] = process_data_for_config_key_optimized(df, config_key, config_value)
    
#     # Drop rows with NaN (where we couldn't get N records)
#     result_df = result_df.dropna()
    
#     return result_df


# import polars as pl


# def data_preparation_helper_optimized(df, config):
#     processed_data = []

#     # Ensure the DataFrame is a Polars DataFrame
#     if not isinstance(df, pl.DataFrame):
#         df = pl.DataFrame(df)

#     # Process each column based on the configuration
#     for column, column_config in config.items():
#         records_back = column_config['records_back']
#         specific_to_can_id = column_config['specific_to_can_id']

#         if specific_to_can_id:
#             # Create shifted columns based on the records_back value
#             for shift_val in range(1, records_back + 1):
#                 shifted_column = pl.col(column).shift(shift_val).alias(f"{column}_shifted_{shift_val}")
#                 df = df.with_columns(shifted_column)
                
#             # Instead of aggregating into lists, we'll simply join on 'aid'. The shifted columns are already created in the DataFrame.
#             # There's no need for the df_grouped and join in this context.
#         else:
#             for i in range(1, records_back + 1):
#                 shifted_col = pl.col(column).shift(i).alias(f"{column}_shifted_{i}")
#                 df = df.with_columns(shifted_col)

#     # Filter out rows where any of the columns (except 'aid') have null values
#     df = df.drop_nulls()

#     # Convert the Polars DataFrame to a list of lists
#     processed_data = df.to_pandas().values.tolist()

#     return processed_data




class Logger():
    def __init__(self, log_verbosity=0):
        self.log_verbosity = log_verbosity

    def __call__(self, msg, level=1):
        if level <= 0:
            raise "Level must be greater than zero"
        if level <= self.log_verbosity:
            padding = "  "*(level-1) 
            print(padding + msg) 

# Don't worry about this
# This wonderful mess is how we can easily 
# iterate through the ambient and attack datas 
# through an attribute of CanDataLoader
class CanData():
    def __init__(self, dfs):
        self.length = len(dfs)
        for key, item in dfs.items():
            setattr(self, key, item)
        self._keys = list(dfs.keys())

    def __iter__(self):
        return self.CanDataIterator(self)

    class CanDataIterator():
        def __init__(self, can_data):
            self._can_data = can_data
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self._index < len(self._can_data._keys):
                key = self._can_data._keys[self._index]
                self._index += 1
                return getattr(self._can_data, key)
            else:
                raise StopIteration
            

        
class CanDataLoader():
    def __init__(self, data_dir_path, log_verbosity=0):
        self.data_dir_path = data_dir_path
        self.log = Logger(log_verbosity)

        if not os.path.exists(data_dir_path):
            raise FileNotFoundError(f'Directory {data_dir_path} not found.')

        if self._is_ambient_and_attack_dir():
            self.log('Found ambient and attack directories.')
        else:
            self.log('Downloading data...')
            self._download_data()
        
        self.log('Loading CAN metadata...')
        self._load_metadata()

        if self._is_parquet_files():
            self.log('Parquet files found...')
            if self._is_processed_parquet_files():
                self.log('Found processed parquet files...')
                self.log('Loading processed parquet files...')
                self.processed_ambient_dfs, self.processed_attack_dfs = self._load_processed_parquet_data()
            else:
                self.log('Loading preprocessed parquet files...')
                self.preprocessed_ambient_dfs, self.preprocessed_attack_dfs = self._load_parquet_data()
                self.log('Processing parquet files...')
                self.processed_ambient_dfs, self.processed_attack_dfs = self._process_can_data()
                self.log('Saving processed parquet files...')
                self._save_data(is_processed=True)
                self.log('Done processing CAN data.')

        else:
            self.log('Loading raw can data...')
            self.preprocessed_ambient_dfs, self.preprocessed_attack_dfs = self._load_can_data()
            self.log('Saving parquet files...')
            self._save_data(is_processed=False)
            self.log('Processing raw can data...')
            self.processed_ambient_dfs, self.processed_attack_dfs = self._process_can_data()
            self.log('Done processing CAN data.')
            self.log('Saving processed parquet files...')
            self._save_data(is_processed=True)
            self.log('Done saving processed parquet files.')

        self.log("Loading processing data into 'CanData' structure")
        self.ambient_data = CanData(self.processed_ambient_dfs)
        self.attack_data = CanData(self.processed_attack_dfs)

    def get_ambient_data(self):
        return self.processed_ambient_dfs
    
    def get_attack_data(self):
        return self.processed_attack_dfs
    
    def prepare_data(self, config: dict):
        batch_size = config.pop("batch_size", None)
        if not batch_size:
            raise Exception("Config needs `batch_size`")
        


        ambient_dfs = []
        validate_dfs = []
        for i, df in enumerate(self.ambient_data):
            if i < 3:
                validate_dfs.append(df)
            else:    
                ambient_dfs.append(df)

        ambient_data_loader = CANDataset(ambient_dfs, config, batch_size)
        validate_data_loader = CANDataset(validate_dfs, config, batch_size)

        attack_dfs = []
        for df in self.attack_data:
            attack_dfs.append(df)
        attack_data_loader = CANDataset(attack_dfs, config, batch_size)

        return ambient_data_loader, validate_data_loader, attack_data_loader
    
    def get_unique_can_ids(self):
        df_list = []
        df_list.extend([df for df in self.ambient_data])
        df_list.extend([df for df in self.ambient_data])
        df_concatenated = pd.concat(df_list, ignore_index=True)
        return df_concatenated['aid'].unique()

    
    def _download_data(self):    
        self.log("Downloading the zip file...", 2)
        response = requests.get(DATASET_DOWNLOAD_LINK, stream=True)
        zip_path = os.path.join(self.data_dir_path, "temp.zip")
        with open(zip_path, 'wb') as zip_file:
            for chunk in response.iter_content(chunk_size=8192):
                zip_file.write(chunk)

        self.log("Unzipping the file...", 2)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir_path)

        extracted_dir = os.path.join(self.data_dir_path, zip_ref.namelist()[0].split('/')[0])
        
        self.log("Moving contents to the data directory...", 2)
        for item in os.listdir(extracted_dir):
            shutil.move(os.path.join(extracted_dir, item), self.data_dir_path)
        
        self.log("Deleting the zip and the empty extracted directory...", 2)
        os.remove(zip_path)
        os.rmdir(extracted_dir)
    
    def _is_ambient_and_attack_dir(self):
        return os.path.exists(self.data_dir_path + '/ambient') and os.path.exists(self.data_dir_path + '/attacks')

    def _load_metadata(self):
        ambient_dir = f'{self.data_dir_path}/ambient/' 
        attack_dir = f'{self.data_dir_path}/attacks/'

        
        with open(ambient_dir + "capture_metadata.json") as f:
            self.ambient_metadata = load(f)

        with open(attack_dir + "capture_metadata.json") as f:
            self.attack_metadata = load(f)

        self.ambient_files = [key for key in self.ambient_metadata.keys() if 'masquerade' not in key]
        self.attack_files = [key for key in self.attack_metadata.keys() if 'masquerade' not in key]
    
    def _is_parquet_files(self):

        for key in self.ambient_files:
            ambient_parquet_dir = self.data_dir_path + '/ambient/parquet/' + key + '.parquet'
            self.log(f"Looking for {ambient_parquet_dir}", 2)
            if not os.path.exists(ambient_parquet_dir):
                return False

        for key in self.attack_files:
            attack_parquet_dir = self.data_dir_path + '/attacks/parquet/' + key + '.parquet'
            self.log(f"Looking for {attack_parquet_dir}", 2)
            if not os.path.exists(attack_parquet_dir):
                return False
        return True

    def _is_processed_parquet_files(self):
        for key in self.ambient_files:
            if not os.path.exists(self.data_dir_path + '/ambient/parquet/processed/' + key + '.parquet'):
                self.log(f'File {key} not found in {self.data_dir_path + "ambient/parquet/processed/"}', level=2)
                return False
        for key in self.attack_files:
            if not os.path.exists(self.data_dir_path + '/attacks/parquet/processed/' + key + '.parquet'):
                self.log(f'File {key} not found in {self.data_dir_path + "attacks/parquet/processed/"}', level=2)
                return False
        return True
    
    def _load_processed_parquet_data(self):
        ambient_dir = f'{self.data_dir_path}/ambient/parquet/processed'
        attack_dir = f'{self.data_dir_path}/attacks/parquet/processed'

        ambient_dfs = {}
        for parquet_file in self.ambient_files:
            self.log(f'Loading {parquet_file}...', level=2)
            parquet_filepath = os.path.join(ambient_dir, f'{parquet_file}.parquet')
            df = pd.read_parquet(parquet_filepath)
            ambient_dfs[parquet_file] = df

        attack_dfs = {}
        for parquet_file in self.attack_files:
            self.log(f'Loading {parquet_file}...', level=2)
            parquet_filepath = os.path.join(attack_dir, f'{parquet_file}.parquet')
            df = pd.read_parquet(parquet_filepath)
            attack_dfs[parquet_file] = df

        return ambient_dfs, attack_dfs
    
    def _load_parquet_data(self):
        ambient_dir = f'{self.data_dir_path}/ambient/parquet'
        attack_dir = f'{self.data_dir_path}/attacks/parquet'

        ambient_dfs = {}
        for parquet_file in self.ambient_files:
            self.log(f'Loading {parquet_file}...', level=2)
            parquet_filepath = os.path.join(ambient_dir, f'{parquet_file}.parquet')
            df = pd.read_parquet(parquet_filepath)
            ambient_dfs[parquet_file] = df

        attack_dfs = {}
        for parquet_file in self.attack_files:
            self.log(f'Loading {parquet_file}...', level=2)
            parquet_filepath = os.path.join(attack_dir, f'{parquet_file}.parquet')
            df = pd.read_parquet(parquet_filepath)
            attack_dfs[parquet_file] = df

        return ambient_dfs, attack_dfs
    
    def _load_can_data(self):
        ambient_dir = f'{self.data_dir_path}/ambient'
        attack_dir = f'{self.data_dir_path}/attacks'

        # Extract Ambient Data
        self.log('Extracting ambient data...', level=2)
        ambient_dfs = {}
        for log_file in os.listdir(ambient_dir):
            if log_file.endswith('.log'):
                log_filepath = os.path.join(ambient_dir, log_file)
                self.log(f'Extracting {log_file}...', level=2)
                df = make_can_df(log_filepath)
                ambient_dfs[log_file[:-4]] = df[['time', 'aid', 'data']]

        # Extract Attack Data
        attack_dfs = {}
        self.log('Extracting attack data...', level=2)
        for log_file in os.listdir(attack_dir):
            if log_file.endswith('.log'):
                log_filepath = os.path.join(attack_dir, log_file)
                self.log(f'Extracting {log_file}...', level=2)
                df = make_can_df(log_filepath)
                attack_dfs[log_file[:-4]] = df[['time', 'aid', 'data']]

        return ambient_dfs, attack_dfs

    def _save_data(self, is_processed):
        ambient_dir = f'{self.data_dir_path}/ambient/parquet'
        attack_dir = f'{self.data_dir_path}/attacks/parquet'
        ambient_data = self.preprocessed_ambient_dfs
        attack_data = self.preprocessed_attack_dfs

        if is_processed:
            ambient_data = self.processed_ambient_dfs
            attack_data = self.processed_attack_dfs
            ambient_dir += '/processed'
            attack_dir += '/processed'
    
        # create directories if they don't exist
        if not os.path.exists(ambient_dir):
            self.log(f'Creating {ambient_dir}...', level=3)
            os.makedirs(ambient_dir)
        
        if not os.path.exists(attack_dir):
            self.log(f'Creating {attack_dir}...', level=3)
            os.makedirs(attack_dir)

        for key, ambient_file_df in ambient_data.items():
            self.log(f'Saving {key}...', level=2)
            ambient_parquet_file = os.path.join(ambient_dir, f'{key}.parquet')
            ambient_file_df.to_parquet(ambient_parquet_file, index=False)

        for key, attack_file_df in attack_data.items():
            self.log(f'Saving {key}...', level=2)
            attack_parquet_file = os.path.join(attack_dir, f'{key}.parquet')
            attack_file_df.to_parquet(attack_parquet_file, index=False)

    def _process_can_data(self):

        processed_ambient_dfs= {} 
        for key, ambient_file_df in self.preprocessed_ambient_dfs.items():
            self.log(f'Processing {key}...', level=2)
            self._add_time_diff_per_aid_col(ambient_file_df)
            self._add_time_diff_since_last_msg_col(ambient_file_df)
            processed_ambient_dfs[key] = ambient_file_df

        processed_attack_dfs = {}
        for key, attack_file_df in self.preprocessed_attack_dfs.items():
            self.log(f'Processing {key}...', level=2)
            self._add_time_diff_per_aid_col(attack_file_df)
            self._add_time_diff_since_last_msg_col(attack_file_df)
            processed_attack_dfs[key] = attack_file_df

        self.log('Annotating data...', level=2)
        processed_ambient_dfs, processed_attack_dfs = self._add_actual_attack_col(processed_ambient_dfs, processed_attack_dfs)

        return processed_ambient_dfs, processed_attack_dfs
    
    def _add_time_diff_per_aid_col(self, df, order_by_time=True):
        if order_by_time:
            df.sort_values(by="time", ascending=True, inplace=True)
        
        df["delta_time_last_msg"] = df["time"].diff().fillna(0)
        # No return statement needed

    def _add_time_diff_since_last_msg_col(self, df, order_by_time=True):
        if order_by_time:
            df.sort_values(by="time", ascending=True, inplace=True)
        
        df["last_time"] = df.groupby("aid")["time"].shift()
        df["delta_time_last_same_aid"] = df["time"] - df["last_time"]
        df.drop(columns=["last_time"], inplace=True)
    
    def _add_actual_attack_col(self, processed_ambient_dfs, processed_attack_dfs):
        self.log('Adding actual attack column for ambient data...', level=3)
        for key in self.ambient_files:
            processed_ambient_dfs[key]['actual_attack'] = False

        self.log('Adding actual attack column for attack data...', level=3)
        for key in self.attack_files:
            attack_file_metadata = self.attack_metadata[key]
            self.log(f'Adding actual attack column for {key}...', level=4)
            injection_data_str = attack_file_metadata.get("injection_data_str")
            injection_id = attack_file_metadata.get("injection_id")
            injection_interval = attack_file_metadata.get("injection_interval")

            # print(injection_data_str, injection_id, injection_interval)
            if not (injection_data_str and injection_data_str and injection_data_str):
                self.log(f"Missing metadata for {key}. Skipping...", level=4)
                continue
    
            processed_attack_dfs[key]['actual_attack'] = False
            if injection_id == "XXX":
                for index, row in processed_attack_dfs[key].iterrows():
                    if injection_interval[0] < row['time'] < injection_interval[1] \
                        and payload_matches(row['data'], injection_data_str):
                            processed_attack_dfs[key].at[index, 'actual_attack'] = True
            else:
                for index, row in processed_attack_dfs[key].iterrows():
                    # some of the injection ids are in hex some are in decimal
                    try: injection_id = int(injection_id, 16)
                    except: pass
                    if injection_interval[0] < row['time'] < injection_interval[1] \
                        and row['aid'] == injection_id \
                        and payload_matches(row['data'], injection_data_str):
                                processed_attack_dfs[key].at[index, 'actual_attack'] = True

        return processed_ambient_dfs, processed_attack_dfs