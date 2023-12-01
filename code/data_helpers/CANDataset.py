from .CANDataLoader import CANDataLoader
from copy import deepcopy
from json import load
from helpers import *
import pandas as pd
import requests
import zipfile
import shutil
import os

DATASET_DOWNLOAD_LINK = "https://road.nyc3.digitaloceanspaces.com/road.zip"

class Logger():
    def __init__(self, log_verbosity=0):
        self.log_verbosity = log_verbosity

    def __call__(self, msg, level=1):
        if level <= 0:
            raise "Level must be greater than zero"
        if level <= self.log_verbosity:
            padding = "  "*(level-1) 
            print(padding + msg) 

class CANData():
    def __init__(self, dfs):
        self.length = len(dfs)
        for key, item in dfs.items():
            setattr(self, key, item)
        self._keys = list(dfs.keys())

    def __iter__(self):
        return self.CANDataIterator(self)

    class CANDataIterator():
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
            
class CANDataset():
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

        self.log("Loading processing data into 'CANData' structure")
        self.ambient_data = CANData(self.processed_ambient_dfs)
        self.attack_data = CANData(self.processed_attack_dfs)

    def get_ambient_data(self):
        return self.processed_ambient_dfs
    
    def get_attack_data(self):
        return self.processed_attack_dfs
    
    def get_dataloaders(self, config: dict):
        config = deepcopy(config)
        batch_size = config.pop("batch_size", None) # ensure batch_size is in config
        if not batch_size:
            raise Exception("Config needs `batch_size`")

        num_of_ambient_files_for_validation = 3
        ambient_dfs = []
        validate_dfs = []
        for i, df in enumerate(self.ambient_data):
            if i < num_of_ambient_files_for_validation:
                validate_dfs.append(df)
            else:    
                ambient_dfs.append(df)

        ambient_data_loader = CANDataLoader(ambient_dfs, config, batch_size)
        validate_data_loader = CANDataLoader(validate_dfs, config, batch_size)

        attack_dfs = []
        for df in self.attack_data:
            attack_dfs.append(df)
        attack_data_loader = CANDataLoader(attack_dfs, config, batch_size)

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
                ambient_dfs[log_file[:-4]] = df[['time', 'aid', 'data', 'filename']]

        # Extract Attack Data
        attack_dfs = {}
        self.log('Extracting attack data...', level=2)
        for log_file in os.listdir(attack_dir):
            if log_file.endswith('.log'):
                log_filepath = os.path.join(attack_dir, log_file)
                self.log(f'Extracting {log_file}...', level=2)
                df = make_can_df(log_filepath)
                attack_dfs[log_file[:-4]] = df[['time', 'aid', 'data', 'filename']]

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

    def _standardize_columns(self, df, cols_to_normalize):
        scaler = StandardScaler()
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        return df, scaler.mean_, scaler.scale_

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
            found_attack = False
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
                            found_attack = True
                            processed_attack_dfs[key].at[index, 'actual_attack'] = True
            else:
                for index, row in processed_attack_dfs[key].iterrows():
                    # some of the injection ids are in hex some are in decimal
                    try: injection_id = int(injection_id, 16)
                    except: pass
                    if injection_interval[0] < row['time'] < injection_interval[1] \
                        and row['aid'] == injection_id \
                        and payload_matches(row['data'], injection_data_str):
                                found_attack = True
                                processed_attack_dfs[key].at[index, 'actual_attack'] = True
            
            if found_attack:
                self.log(f'Found attack for {key}...', level=4)

        return processed_ambient_dfs, processed_attack_dfs