#!/usr/bin/env python
# coding: utf-8

# #### Load Data

# In[ ]:


from CanDataset import CanDataset
from dotenv import load_dotenv
import torch
import torch.nn as nn
import os

load_dotenv()
data_path = os.getenv('DATA_PATH')
dataset = CanDataset(data_path, log_verbosity=1)


# #### ML model

# In[ ]:


class CANnoloAutoencoder(nn.Module):
    def __init__(self, embedding_dim, lstm_units, dense_units, dropout_rate, num_embeddings):
        super(CANnoloAutoencoder, self).__init__()

        # Encoder
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.encoder_dense = nn.Linear(embedding_dim+45, dense_units)
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.encoder_lstm = nn.LSTM(input_size=dense_units, hidden_size=lstm_units, num_layers=2, batch_first=True)

        # Decoder
        self.decoder_lstm = nn.LSTM(input_size=lstm_units, hidden_size=lstm_units, num_layers=2, batch_first=True)
        self.decoder_dense = nn.Linear(lstm_units, 45)
        self.decoder_output = nn.Sigmoid()  # To reconstruct the original packets

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


# In[ ]:


# CANID   -
# f1      -        -  reconstructed_f1
# f2      -   -    -  reconstructed_f2
# f3      -        -  reconstructed_f3


# In[ ]:


dataset.attack_data.accelerator_attack_drive_1


# #### Define config
# This is what we feed to the CanDataset object to create a dataloader.

# In[ ]:


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


# use `get_dataloaders` on CanDataset object to get the data loaders

# In[ ]:


ambient_loader, validation_loader, attack_loader = dataset.get_dataloaders(config)


# #### Example Data
# From the config we defined:
#     - Batch size of `32`
#     - Keep track of the current Can ID.
#     - want the last `30` `delta_time_last_msg`
#     - want the last `15` `delta_time_last_same_aid`
# 
# 

# #### Example of 1 input

# In[ ]:


test_batch_can_ids, test_feature_vec = example_data

print(f'Represents Can ID: \n{test_batch_can_ids[0]}\n')
print(f'Represents Feature Vector: \n{test_feature_vec[0]}')


# The `example_data` is a tuple containing a list of 32 (batch_size) Can ID's and the feature vectors defined in the config.
# 
# ([`tensor containing Can ID's`],[`tensor containing features`])

# In[ ]:


unique_can_ids = dataset.get_unique_can_ids()
num_can_ids = len(unique_can_ids)
feature_vec_length = ambient_loader.features_len
print(f"Number of CAN IDs: {num_can_ids}")
print(f"Feature vector length: {feature_vec_length-1}") # minus one because the can id is the first


# In[ ]:


# Hyperparameters
embedding_dim = num_can_ids  # embedding dimension should be equal to the number of CAN IDs
lstm_units = 128 # defined in canolo paper
dense_units = 256 # defined in canolo paper
dropout_rate = 0.2 # defined in canolo paper
num_embeddings = max(unique_can_ids) + 1 # not sure why + 1 rn but it works

# Model
model = CANnoloAutoencoder(embedding_dim, lstm_units, dense_units, dropout_rate, num_embeddings)

# Training parameters
batch_size = ambient_loader.batch_size
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss


# In[ ]:


# Running a forward pass with a batch of data
reconstructed_output = model(test_batch_can_ids, test_feature_vec)

mse_loss = torch.nn.MSELoss()
error = mse_loss(reconstructed_output, test_feature_vec)
print("Reconstruction Error:", error.item())


# #### Defining our loss function and optimizer

# In[ ]:


loss_fn = torch.nn.MSELoss()  # Example loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Example optimizer


# In[ ]:


PSEUDO_EPOCH_SIZE = 3000

def validate_model(model, validation_loader, loss_fn):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    num_batches_to_validate = 1000
    with torch.no_grad():  # No need to track gradients during validation
        for i, batch in enumerate(validation_loader):
            can_ids, features = batch
            
            if i == num_batches_to_validate:
                break
            
            # Forward pass: compute the model output
            reconstructed = model(can_ids, features)
            # Compute the loss
            loss = loss_fn(reconstructed, features)  # Ensure correct target is used
            total_loss += loss.item()

    model.train()  # Revert to training mode
    num_processed_batches = validation_loader.batch_size * num_batches_to_validate
    avg_loss = total_loss / num_processed_batches
    return avg_loss

def train_model(model, train_loader, validation_loader, loss_fn, optimizer, num_epochs, validation_interval):
    total_train_loss = 0
    pseudo_epoch = 1
    num_processed_batches_in_epoch = train_loader.batch_size * PSEUDO_EPOCH_SIZE

    model.train()
    for i, batch in enumerate(train_loader):
        can_ids, features = batch
        print(f"{i}", end="\r")

        # Forward pass: compute the model output
        reconstructed = model(can_ids, features)

        # Compute the loss
        loss = loss_fn(reconstructed, features)  # Ensure correct target is used
        total_train_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear existing gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights

        if i % PSEUDO_EPOCH_SIZE == 0:

            if i == 0:
                continue

            # Validate model
            validation_loss = validate_model(model, validation_loader, loss_fn)
            print(f"Psuedo Epoch {pseudo_epoch}, Validation Loss: {validation_loss}")

            # Show training progress
            avg_train_loss = total_train_loss / num_processed_batches_in_epoch
            print(f"Epoch {pseudo_epoch-1}, Average Training Loss: {avg_train_loss}")
            
            if pseudo_epoch > num_epochs:
                break
            

            # Save model
            torch.save(model.state_dict(), f'./saved_model/canolo_model_{pseudo_epoch}.pt')

            # save metadata
            total_batches_processed = i

            metadata = {
                "total_batches_processed": total_batches_processed,
                "total_train_loss": total_train_loss,
                "avg_train_loss": avg_train_loss,
                "validation_loss": validation_loss
            }

            with open(f'training_metadata.tsv', 'a') as f:
                f.write('\t'.join(str(metadata[key]) for key in metadata.keys()) + '\n')

            pseudo_epoch += 1
            total_train_loss = 0



num_epochs = inf
            
train_model(model, ambient_loader, validation_loader, loss_fn, optimizer, num_epochs, validation_interval)




