import torch
from torch.utils.data import Dataset

class CANDataLoader(Dataset):
    def __init__(self, can_data, config, batch_size=1):
        self.can_data = can_data
        self.config = config
        self.batch_size = batch_size
        self.current_df_index = 0
        self.completed_index = 0
        self.features_len = 1  # Starting with CAN ID

        for _, feature_config in self.config.items():
            self.features_len += feature_config['records_back']

        self.num_rows = 0
        for df in can_data:
            self.num_rows += len(df)
            df.dropna()

        self.num_batches = self.num_rows // self.batch_size

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if self.current_df_index >= len(self.can_data):
            raise StopIteration

        # can_data = [df1,df2,df3]
        current_df = self.can_data[self.current_df_index]
        processed_batch = self.extract_features(current_df)

        while len(processed_batch) < self.batch_size:
            self.current_df_index += 1
            if self.current_df_index >= len(self.can_data):
                raise StopIteration

            self.completed_index = 0
            current_df = self.can_data[self.current_df_index]
            processed_batch.extend(self.extract_features(current_df))

        # Extract CAN IDs, features, and actual attack labels
        can_ids = [item[0] for item in processed_batch]
        features = [item[1:-1] for item in processed_batch]  # Exclude the last item (attack label)
        actual_attacks = [item[-1] for item in processed_batch]  # Extract actual attack labels

        # Convert to Tensors
        can_ids = torch.tensor(can_ids, dtype=torch.long)
        features = torch.tensor(features, dtype=torch.float32)
        actual_attacks = torch.tensor(actual_attacks, dtype=torch.float32)  # Assuming binary labels

        return can_ids, features, actual_attacks

    def extract_features(self, df):
        processed_batch = []
        while len(processed_batch) < self.batch_size:
            if self.completed_index >= len(df):
                return processed_batch

            index = self.completed_index
            self.completed_index += 1
            row = df.iloc[index]
            features = [row['aid']]

            for feature_name, feature_config in self.config.items():
                if feature_config['specific_to_can_id']:
                    history = df[df['aid'] == row['aid']].iloc[:index]
                    history = history.tail(feature_config['records_back'])
                else:
                    history = df.iloc[max(0, index - feature_config['records_back']):index]

                if len(history) < feature_config['records_back']:
                    # Handle padding or other strategies here
                    continue

                feature_values = history[feature_name].tolist()
                features.extend(feature_values)

            attack_label = row['actual_attack']  # Assuming 'actual_attack' is the column name
            features.append(attack_label)

            if len(features) != self.features_len + 1: # +1 for attack label
                continue

            processed_batch.append(features)

        return processed_batch
    
    def reset(self):
        self.current_df_index = 0
        self.completed_index = 0

# # Example Usage
# config = {
#     "delta_time_last_msg": {
#         "specific_to_can_id": False,
#         "records_back": 30
#     },
#     "delta_time_last_same_aid": {
#         "specific_to_can_id": True,
#         "records_back": 15
#     },
# }

# # Assuming can_data is a list of DataFrames
# dataset = CANDataset(can_data, config, batch_size=64)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
