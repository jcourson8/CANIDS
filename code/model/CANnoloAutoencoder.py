import torch.nn as nn
import torch
from math import inf
from tqdm import tqdm
import os
import numpy as np

class CANnoloAutoencoder(nn.Module):
    def __init__(self, embedding_dim, lstm_units, dense_units, dropout_rate, num_embeddings, feature_vec_length, force_cpu=False, **kwargs):
        super(CANnoloAutoencoder, self).__init__()

        self.force_cpu = force_cpu
        self.device = self.get_device()
        self.loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss
        # self.mse_loss = torch.nn.MSELoss()
        
        # Encoder
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder_dense = nn.Linear(embedding_dim+feature_vec_length, dense_units)
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.encoder_lstm = nn.LSTM(input_size=dense_units, hidden_size=lstm_units, num_layers=2, batch_first=True)

        # Decoder
        self.decoder_lstm = nn.LSTM(input_size=lstm_units, hidden_size=lstm_units, num_layers=2, batch_first=True)
        self.decoder_dense = nn.Linear(lstm_units, feature_vec_length)
        self.decoder_output = nn.Sigmoid()  # To reconstruct the original packets
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.to(self.device)

    def forward(self, can_ids, features):
        # Encoding
        embedded_ids = self.embedding(can_ids)
        # You might need to concatenate the embedded IDs with other features
        x = torch.cat([embedded_ids, features], dim=1)
        x = torch.tanh(self.encoder_dense(x))
        x = self.encoder_dropout(x)
        x, _ = self.encoder_lstm(x)

        # Decoding
        x, _ = self.decoder_lstm(x)
        x = self.decoder_dense(x)
        reconstructed = self.decoder_output(x)

        return reconstructed
    
    # def force_cpu(self):
    #     self.force_cpu = True

    def get_device(self):
        if self.force_cpu:
            print("Forcing CPU usage...")
            return torch.device("cpu")
        
        if torch.cuda.is_available():
            print("CUDA is available. Using CUDA...")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("MPS is available. Using MPS...")
            return torch.device("mps")
        else:
            print("Neither CUDA nor MPS is available. Using CPU...")
            return torch.device("cpu")

    def _validate_model(self, validation_loader, num_batches_to_validate=1000):
        self.eval()  # Set the model to evaluation mode
        total_loss = 0
        with torch.no_grad():  # No need to track gradients during validation
            for i, batch in enumerate(validation_loader):
                can_ids, features, _ = batch
                can_ids, features = can_ids.to(self.device), features.to(self.device)
                
                if i == num_batches_to_validate:
                    break
                
                # Forward pass: compute the model output
                reconstructed = self(can_ids, features)

                # Compute the loss
                loss = self.loss_fn(reconstructed, features)  # Ensure correct target is used
                total_loss += loss.item()

        self.train()  # Revert to training mode
        num_processed_batches = validation_loader.batch_size * num_batches_to_validate
        avg_loss = total_loss / num_processed_batches

        return avg_loss

    def train_loop(self, train_loader, validation_loader, training_metadata_file, model_save_directory, num_epochs=inf, psuedo_epoch_size=3000, validation_epoch_size=1000):
        total_train_loss = 0
        pseudo_epoch = 1
        num_processed_batches_in_epoch = train_loader.batch_size * psuedo_epoch_size
        
        # check if training metadata file exists if not create it and add the header
        if not os.path.exists(training_metadata_file):
            with open(training_metadata_file, 'w') as f:
                f.write('\t'.join(['total_batches_processed', 'total_train_loss', 'avg_train_loss', 'validation_loss']) + '\n')

        # ensure model save directory exists
        if not os.path.exists(model_save_directory):
            os.makedirs(model_save_directory)

        self.train()
        for i, batch in enumerate(train_loader):
            can_ids, features, _ = batch
            can_ids, features = can_ids.to(self.device), features.to(self.device)
            print(f"Current Batch: {i}", end="\r")

            # Forward pass: compute the model output
            reconstructed = self(can_ids, features)

            # Compute the loss
            loss = self.loss_fn(reconstructed, features)  # Ensure correct target is used
            total_train_loss += loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()  # Clear existing gradients
            loss.backward()  # Compute gradients
            self.optimizer.step()  # Update weights

            if i % psuedo_epoch_size == 0:
                if i == 0:
                    continue

                # Validate model
                validation_loss = self._validate_model(validation_loader, num_batches_to_validate=validation_epoch_size)
                print(f"Psuedo Epoch {pseudo_epoch}, Validation Loss: {validation_loss}")

                # Show training progress
                avg_train_loss = total_train_loss / num_processed_batches_in_epoch
                print(f"Epoch {pseudo_epoch-1}, Average Training Loss: {avg_train_loss}")
                
                if pseudo_epoch > num_epochs:
                    break
                
                # Save model
                torch.save(self.state_dict(), f'./{model_save_directory}/{pseudo_epoch}.pt')

                # Save metadata
                total_batches_processed = i

                metadata = {
                    "total_batches_processed": total_batches_processed,
                    "total_train_loss": total_train_loss,
                    "avg_train_loss": avg_train_loss,
                    "validation_loss": validation_loss
                }

                with open(training_metadata_file, 'a') as f:
                    f.write('\t'.join(str(metadata[key]) for key in metadata.keys()) + '\n')

                pseudo_epoch += 1
                total_train_loss = 0
            
    # def detect_attacks(self, data_loader, model_path, threshold):
    #     state_dict = torch.load(model_path)
    #     self.load_state_dict(state_dict)
    #     self.threshold = threshold
    #     self.eval()  # Ensure the model is in evaluation mode
    #     results = []
        
    #     with torch.no_grad():
    #         for batch in tqdm(data_loader):
    #             can_ids, features, actual_attacks = batch
                
    #             reconstructed = self(can_ids, features)

    #             # Compute anomaly scores and predict attacks
    #             scores = self._compute_anomaly_scores(features, reconstructed)
    #             predicted_attacks = (scores > self.threshold).int()  # Convert to 0 or 1

    #             # Store predictions and actual labels
    #             results.extend(zip(predicted_attacks.tolist(), actual_attacks.tolist()))
        
    #     data_loader.reset()

    #     return results

    # def _compute_anomaly_scores(self, original, reconstructed):
    #     # Compute anomaly scores (e.g., mean squared error) for each instance in the batch
    #     # loss_fn = torch.nn.MSELoss(reduction='none')
    #     scores = self.loss_fn(original, reconstructed)

    #     return scores.mean(dim=1)  # Mean score across features for each instance
    
    # def _determine_threshold(self, normal_data_loader, percentile=95):
    #     self.eval() 
    #     all_scores = []

    #     with torch.no_grad():
    #         for batch in tqdm(normal_data_loader):
    #             can_ids, features, _ = batch  # Assuming normal data does not have actual attacks
    #             can_ids, features = can_ids.to(self.device), features.to(self.device)
    #             reconstructed = self(can_ids, features)
    #             scores = self._compute_anomaly_scores(features, reconstructed)
    #             all_scores.extend(scores.tolist())

    #     # Consider using a high percentile as the threshold
    #     threshold = np.percentile(all_scores, percentile)  # for example, 95th percentile
    #     normal_data_loader.reset()

    #     return threshold

