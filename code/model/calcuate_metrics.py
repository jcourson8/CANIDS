# %%
import os 
import sys 
parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir + "/code")

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

attack_loaders = seperate_attack_loader(attack_loader, config, remove_non_labelled=True)


# %%
display(attack_loaders)

# %%


# %%

unique_can_ids = dataset.get_unique_can_ids()
num_can_ids = len(unique_can_ids)
feature_vec_length = ambient_loader.features_len

# Load model
model_path = 'models/canolo_model_112.pt'
threshold = 6.3e-06

model_config = {
    "embedding_dim": num_can_ids,
    "lstm_units": 128,
    "dense_units": 256,
    "dropout_rate": 0.2,
    "num_embeddings": max(unique_can_ids) + 1, # not sure why + 1 rn but it works
    "feature_vec_length": calculate_feature_vec_length(config)
}

detector = CANnoloAttackDetector(model_path, threshold, model_config)


#     # make enought detectors the length of attack_loaders * deep copy
#     detectors = [deepcopy(detector) for _ in range(len(attack_loaders))]
#     # make a list of tuples of detectors and loaders
#     detector_loader_pairs = zip(detectors, attack_loaders)
#     # run the detectors in parallel
#     results = executor.map(lambda x: calculate_metrics(x[0].detect_attacks(x[1])), detector_loader_pairs)
#     # convert the results to a dictionary
#     results = dict(zip([loader.can_data[0].filename[0] for loader in attack_loaders], results))



# %%
from concurrent.futures import ProcessPoolExecutor
from detect import detect, detect_wrapper


thresholds = [.000095, .000085, .000075, .000065, .000055]
meta_data = {}
for threshold in thresholds:
    num_attack_files = len(attack_loaders)
    detectors = []

    for _ in range(num_attack_files):
        detector = CANnoloAttackDetector(model_path, threshold, model_config, force_cpu=True)
        detectors.append(detector)
    
    with ProcessPoolExecutor() as executor:
        meta_data[threshold] = list(executor.map(detect_wrapper, zip(attack_loaders, detectors)))

    for loader in attack_loaders:
        loader.reset()

    print(f"Finished testing on threshold: {threshold}")
    print(meta_data[threshold])

print(meta_data)