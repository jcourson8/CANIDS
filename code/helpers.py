import os
import pandas as pd

def add_actual_attack_col(df, intervals, aid, payload):
    """
    Adds column to df to indicate which signals were part of attack
    """

    if aid == "XXX":
        df['actual_attack'] = df.time.apply(lambda x: sum(
            x >= intvl[0] and x <= intvl[1] for intvl in intervals) >= 1) & (df.data == payload)
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
        log_filepath, delimiter=' ' + '#' + '('+')',
        skiprows=1, skipfooter=1,
        usecols=[0, 2, 3],
        dtype={0: 'float64', 1: str, 2: str},
        names=['time', 'aid', 'data'])

    # print(can_df)

    can_df.aid = can_df.aid.apply(lambda x: int(x, 16))
    # pad with 0s on the left for data with dlc < 8
    can_df.data = can_df.data.apply(lambda x: x.zfill(16))
    can_df.time = can_df.time - can_df.time.min()

    # print(can_df)
    return can_df[can_df.aid <= 0x700]