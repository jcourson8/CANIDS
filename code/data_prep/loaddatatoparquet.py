from json import load
from dotenv import load_dotenv
from helpers import *
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

# Extract Ambient Data
ambient_dfs = {}
for log_file in os.listdir(ambient_dir):
    if log_file.endswith('.log'):
        log_filepath = os.path.join(ambient_dir, log_file)
        df = make_can_df(log_filepath)
        ambient_dfs[log_file[:-4]] = df[['time', 'aid', 'data']]

# Extract Attack Data
attack_dfs = {}
for log_file in os.listdir(attack_dir):
    if log_file.endswith('.log'):
        log_filepath = os.path.join(attack_dir, log_file)
        df = make_can_df(log_filepath)
        attack_dfs[log_file[:-4]] = df[['time', 'aid', 'data']]

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

for df_keys in ambient_dfs.keys():
    ambient_parquet_file = os.path.join(ambient_dir, f'{df_keys}.parquet')
    ambient_dfs[df_keys].to_parquet(ambient_parquet_file, index=False)


for df_keys in attack_dfs.keys():
    attack_parquet_file = os.path.join(attack_dir, f'{df_keys}.parquet')
    attack_dfs[df_keys].to_parquet(attack_parquet_file, index=False)
