from tqdm import tqdm
import numpy as np
from .CANnoloAutoencoder import CANnoloAutoencoder
import torch

class CANnoloAttackDetector:
    def __init__(self, model_path, threshold, config, force_cpu=False):

        # Model
        self.model = CANnoloAutoencoder(force_cpu=force_cpu, **config)


        self.loss_fn = torch.nn.BCELoss()
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
        self.threshold = threshold

    def detect_attacks(self, data_loader):
        self.model.eval()  # Ensure the model is in evaluation mode
        results = []
        
        with torch.no_grad():
            for batch in data_loader:
                can_ids, features, actual_attacks = batch
                can_ids, features = can_ids.to(self.model.device), features.to(self.model.device)
                
                reconstructed = self.model(can_ids, features)

                # Compute anomaly scores and predict attacks
                scores = self.compute_anomaly_scores(features, reconstructed)
                predicted_attacks = (scores > self.threshold).int()  # Convert to 0 or 1

                # Store predictions and actual labels
                results.extend(zip(predicted_attacks.tolist(), actual_attacks.tolist()))
        
        data_loader.reset()

        return results

    def get_scores(self, data_loader):
        self.model.eval()  # Ensure the model is in evaluation mode
        scores_and_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                can_ids, features, actual_attacks = batch
                can_ids, features = can_ids.to(self.model.device), features.to(self.model.device)
                
                reconstructed = self.model(can_ids, features)

                # Compute anomaly scores and predict attacks
                scores = self.compute_anomaly_scores(features, reconstructed)
                

                # Store predictions and actual labels
                scores_and_labels.extend(zip(scores, actual_attacks.tolist()))
        
        data_loader.reset()

        return scores_and_labels

    def compute_anomaly_scores(self, original, reconstructed):
        # Compute anomaly scores (e.g., mean squared error) for each instance in the batch
        loss_fn = torch.nn.MSELoss(reduction='none')
        scores = loss_fn(original, reconstructed)
        return scores.mean(dim=1)  # Mean score across features for each instance
    
    def determine_threshold(self, normal_data_loader, percentile=95):
        self.model.eval() 
        all_scores = []

        with torch.no_grad():
            for batch in normal_data_loader:
                can_ids, features, _ = batch  # Assuming normal data does not have actual attacks
                can_ids, features = can_ids.to(self.model.device), features.to(self.model.device)

                reconstructed = self.model(can_ids, features)
                scores = self.compute_anomaly_scores(features, reconstructed)
                all_scores.extend(scores.tolist())
                print(np.percentile(all_scores, percentile), end='\r')

        # Consider using a high percentile as the threshold
        threshold = np.percentile(all_scores, percentile)  # for example, 95th percentile
        normal_data_loader.reset()
        return threshold

