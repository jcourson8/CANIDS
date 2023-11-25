from json import load
from dotenv import load_dotenv
from code.data_prep.helpers import *
import os

load_dotenv()

data_path = os.getenv('DATA_PATH')
ambient_dir = f'{data_path}/ambient' 
attack_dir = f'{data_path}/attacks'

ambient_metadata_file = os.path.join(ambient_dir, 'capture_metadata.json')
attack_metadata_file = os.path.join(attack_dir, 'capture_metadata.json')

with open(ambient_metadata_file) as f:
    ambient_metadata = load(f)

with open(attack_metadata_file) as f:
    attack_metadata = load(f)

ambient_keys = [
                "ambient_dyno_drive_benign_anomaly", 
                "ambient_dyno_drive_basic_long",
                "ambient_highway_street_driving_long",
                "ambient_dyno_reverse",
                "ambient_dyno_idle_radio_infotainment",
                "ambient_dyno_drive_radio_infotainment",
                "ambient_dyno_drive_winter",
                "ambient_dyno_exercise_all_bits",
                "ambient_dyno_drive_extended_short",
                "ambient_dyno_drive_basic_short",
                "ambient_dyno_drive_extended_long",
                "ambient_highway_street_driving_diagnostics"
]

attack_keys = [
                "accelerator_attack_reverse_1",
                "accelerator_attack_drive_1",
                "accelerator_attack_drive_2",
                "accelerator_attack_reverse_2",
                "fuzzing_attack_1",
                "fuzzing_attack_2",
                "fuzzing_attack_3",
                "correlated_signal_attack_1",
                "correlated_signal_attack_2",
                "correlated_signal_attack_3",
                "reverse_light_on_attack_1",
                "reverse_light_on_attack_2",
                "reverse_light_on_attack_3",
                "reverse_light_off_attack_1",
                "reverse_light_off_attack_2",
                "reverse_light_off_attack_3",
                "max_speedometer_attack_1",
                "max_speedometer_attack_2",
                "max_speedometer_attack_3",
                "max_engine_coolant_temp_attack",
]

# load parquet files into dataframes
ambient_dfs = {}
for parquet_file in ambient_keys:
    parquet_filepath = os.path.join(ambient_dir, f'{parquet_file}.parquet')
    df = pd.read_parquet(parquet_filepath)
    ambient_dfs[parquet_file] = df

attack_dfs = {}
for parquet_file in attack_keys:
    parquet_filepath = os.path.join(attack_dir, f'{parquet_file}.parquet')
    df = pd.read_parquet(parquet_filepath)
    attack_dfs[parquet_file] = df

from code.data_prep.helpers import add_time_diff_since_last_msg_col

ambient_dfs_with_time_diff = {} 
for key, ambient_file_df in ambient_dfs.items():
    ambient_dfs_with_time_diff[key] = add_time_diff_per_aid_col(ambient_file_df, True)
    ambient_dfs_with_time_diff[key] = add_time_diff_since_last_msg_col(ambient_file_df, True)
    if ambient_metadata[key]['injection_id'] != 'XXX':
        ambient_dfs_with_time_diff[key] = add_actual_attack_col(ambient_file_df, ambient_metadata, True)

attack_dfs_with_time_diff = {}
for key, attack_file_df in attack_dfs.items():
    attack_dfs_with_time_diff[key] = add_time_diff_per_aid_col(attack_file_df, True)
    attack_dfs_with_time_diff[key] = add_time_diff_since_last_msg_col(attack_file_df, True)
    if attack_metadata[key]['injection_id'] != 'XXX':
        attack_dfs_with_time_diff[key] = add_actual_attack_col(attack_file_df, attack_metadata, True)



for df_keys in ambient_dfs_with_time_diff.keys():
    ambient_parquet_file = os.path.join(ambient_dir, f'{df_keys}_with_time_diffs.parquet')
    ambient_dfs[df_keys].to_parquet(ambient_parquet_file, index=False)


for df_keys in attack_dfs_with_time_diff.keys():
    attack_parquet_file = os.path.join(attack_dir, f'{df_keys}_with_time_diffs.parquet')
    attack_dfs[df_keys].to_parquet(attack_parquet_file, index=False)