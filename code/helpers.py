import os
import pandas as pd
from copy import deepcopy
from data_helpers.CANDataLoader import CANDataLoader

def add_actual_attack_col(df, intervals, aid, payload):
    """
    Adds column to df to indicate which signals were part of attack
    """

    if aid == "XXX":
        df['actual_attack'] = df.time.apply(lambda x: sum(
            x >= intvl[0] and x <= intvl[1] for intvl in intervals) >= 1) & (df.data.str.match(payload))
    else:
        df['actual_attack'] = df.time.apply(lambda x: sum(
            x >= intvl[0] and x <= intvl[1] for intvl in intervals) >= 1) & (df.aid == aid)
    return df


def annotate_attack_data(attack_data, injection_intervals):
    """
    Annotates the attack data based on the injection intervals.
    """
    for index, row in injection_intervals.iterrows():
        aid = row['aid']
        payload = row['payload']
        intervals = [(row['start_time'], row['end_time'])]
        attack_data = add_actual_attack_col(attack_data, intervals, aid, payload)
    return attack_data


def add_time_diff_per_aid_col(df, order_by_time=False):
    """
    Sorts df by aid and time and takes time diff between each successive col and puts in col "time_diffs"
    Then removes first instance of each aids message unless it's the only instance.
    Returns sorted df with new column
    """

    df.sort_values(['aid', 'time'], inplace=True)
    df['time_diffs'] = df['time'].diff()
    # get bool mask of repeated aids
    repeated_aids = df.duplicated('aid', keep=False)
    # get bool mask to filter out first msg of each group unless there is only one.
    mask = (df.aid == df.aid.shift(1)) | ~repeated_aids
    df = df[mask]
    if order_by_time:
        df = df.sort_values('time').reset_index(drop=True)
    return df

def add_time_diff_since_last_msg_col(df, order_by_time=False):
    """
    Sorts df by time and computes the time difference between each message and the 
    previous message irrespective of the aid. Adds this difference in a new column "time_diff_since_last_msg".
    If there's no previous message, the value will be NaN.
    Returns df with the new column.
    """
    
    # Sort the dataframe by time
    df.sort_values('time', inplace=True)
    
    # Compute the time difference from the previous row for all rows
    df['time_diff_since_last_msg'] = df['time'].diff()
    
    # Sort by 'time' if order_by_time is True
    if order_by_time:
        df = df.sort_values('time').reset_index(drop=True)
    
    return df

def make_can_df(log_filepath):
    """
    Puts candump data into a dataframe with columns 'time', 'aid', and 'data'
    """
    can_df = pd.read_fwf(
        log_filepath, delimiter = ' '+ '#' + '('+')',
        skiprows = 1,skipfooter=1,
        usecols = [0,2,3],
        dtype = {0:'float64', 1:str, 2: str},
        names = ['time','aid', 'data'] )

    # print(can_df)

    can_df.aid = can_df.aid.apply(lambda x: int(x, 16))
    # pad with 0s on the left for data with dlc < 8
    can_df.data = can_df.data.apply(lambda x: x.zfill(16))
    can_df.time = can_df.time - can_df.time.min()

    filename = os.path.basename(log_filepath)
    
    # add filename col
    can_df['filename'] = filename

    # print(can_df)
    return can_df[can_df.aid <= 0x700]

def payload_matches(payload, injection_data_str):
    for i, value in enumerate(payload):
        if injection_data_str[i] == "X":
            continue
        else:
            if value != injection_data_str[i]:
                return False
    return True

def calculate_feature_vec_length(config):
    """
    Calculate the feature vector length by summing up the 'records_back' values
    from the configuration.

    :param config: Dictionary containing the configuration
    :return: The calculated feature vector length
    """
    feature_vec_length = 0

    # Iterate through the configuration items
    for key, value in config.items():
        if key == "batch_size":
            continue  # Skip the 'batch_size' key
        feature_vec_length += value.get("records_back", 0)  # Add 'records_back', defaulting to 0 if not found

    return feature_vec_length

def calculate_metrics(results):
    # Initializing the confusion matrix values
    TP, TN, FP, FN = 0, 0, 0, 0

    for pred, actual in results:
        if pred == actual == 1:
            TP += 1
        elif pred == actual == 0:
            TN += 1
        elif pred == 1 and actual == 0:
            FP += 1
        elif pred == 0 and actual == 1:
            FN += 1

    # Calculating accuracy
    total_predictions = len(results)
    accuracy = (TP + TN) / total_predictions if total_predictions > 0 else 0

    # Creating the confusion matrix
    confusion_matrix = [[TP, FP],
                        [FN, TN]]
    
    # Calculating various metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Handling division by zero
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # Handling division by zero
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # Handling division by zero


    return accuracy, precision, recall, f1_score, confusion_matrix

def seperate_attack_loader(attack_loader, config, remove_non_labelled=True):

    config_copy = deepcopy(config)
    batch_size = config_copy.pop("batch_size", None) # ensure batch_size is in config
    if not batch_size:
        raise Exception("Config needs `batch_size`")
    
    attack_loaders = []
    for df in attack_loader.can_data:
        attack_loaders.append(CANDataLoader([df], config_copy, batch_size))

    if remove_non_labelled:
        attack_loader_labelled = []
        for i, loader in enumerate(attack_loaders):
            try:
                loader.can_data[0].actual_attack.sum()
                print(f"Found attack labels in {loader.can_data[0].filename[0]}")
                attack_loader_labelled.append(loader)

            except:
                print(f"No attack labels in {loader.can_data[0].filename[0]}")
                pass

        attack_loaders = attack_loader_labelled

    return attack_loaders
