import pandas as pd

def compute_time_diffs(df):
    # Sort the dataframe by 'aid' and 'time'
    df = df.sort_values(['aid', 'time'])
    
    # Group by 'aid' and compute time differences
    df['time_diff'] = df.groupby('aid')['time'].diff().fillna(0)
    return df

# Assuming N is defined
N = 10

def process_key(key, df):
    local_samples = []
    local_labels = []
    
    df = compute_time_diffs(df)
    
    for aid, group in df.groupby('aid'):
        # Create rolling windows of size N+1
        for i in range(len(group) - N):
            window = group.iloc[i:i+N+1]
            
            # The features are the first N time differences
            features = [aid] + window['time_diff'].iloc[:-1].tolist()
            
            # The label is the N+1th time difference
            label = window['time_diff'].iloc[-1]
            
            local_samples.append(features)
            local_labels.append(label)

    return local_samples, local_labels