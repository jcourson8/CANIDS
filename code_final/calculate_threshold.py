from model.CANnoloAttackDetector import CANnoloAttackDetector
from data_helpers.CANDataset import CANDataset
from helpers import calculate_feature_vec_length, seperate_attack_loader, calculate_metrics
from dotenv import load_dotenv
# from data_helpers.CANDataLoader import CANDataLoader
import os

load_dotenv()
data_path = os.getenv('DATA_PATH')
dataset = CANDataset(data_path, log_verbosity=1)

config = {
    "batch_size": 32,
    "delta_time_last_msg": {
        "specific_to_can_id": False,
        "records_back": 30
    },
    "delta_time_last_same_aid": {
        "specific_to_can_id": True,
        "records_back": 15
    },
}

ambient_loader, validation_loader, attack_loader = dataset.get_dataloaders(config)

unique_can_ids = dataset.get_unique_can_ids()
num_can_ids = len(unique_can_ids)
feature_vec_length = ambient_loader.features_len

# Load model
model_path = 'models/canolo_model_112.pt'
threshold = 9.7e-06

model_config = {
    "embedding_dim": num_can_ids,
    "lstm_units": 128,
    "dense_units": 256,
    "dropout_rate": 0.2,
    "num_embeddings": max(unique_can_ids) + 1, # not sure why + 1 rn but it works
    "feature_vec_length": calculate_feature_vec_length(config)
}

detector = CANnoloAttackDetector(model_path, threshold, model_config)

attack_loaders = seperate_attack_loader(attack_loader, config, remove_non_labelled=True)

detector.determine_threshold(ambient_loader, 90)