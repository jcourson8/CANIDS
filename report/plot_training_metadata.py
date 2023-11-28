
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_and_visualize(file_path):
    data = pd.read_csv(file_path, sep='\\s+')
    plt.style.use('seaborn-darkgrid')

    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Total Batches Processed')
    ax1.set_ylabel('Total Train Loss', color=color)
    ax1.plot(data['total_batches_processed'], data['total_train_loss'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Avg Train Loss & Validation Loss', color=color)
    ax2.plot(data['total_batches_processed'], data['avg_train_loss'], color=color, label='Average Train Loss')
    ax2.plot(data['total_batches_processed'], data['validation_loss'], color='tab:green', label='Validation Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    ax2.legend(loc='upper left')

    plt.title('Training Process Metrics Over Batches')
    plt.show()

def load_and_visualize_remove_outlier(file_path):
    data = pd.read_csv(file_path, sep='\\s+')
    
    q25, q75 = data['total_train_loss'].quantile([0.25, 0.75])
    iqr = q75 - q25
    
    lower_bound = q25 - (1.5 * iqr)
    upper_bound = q75 + (1.5 * iqr)
    
    filtered_data = data[(data['total_train_loss'] >= lower_bound) & 
                         (data['total_train_loss'] <= upper_bound)]
    
    plt.style.use('seaborn-darkgrid')
    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Total Batches Processed')
    ax1.set_ylabel('Total Train Loss', color=color)
    ax1.plot(filtered_data['total_batches_processed'], filtered_data['total_train_loss'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Avg Train Loss & Validation Loss', color=color)
    ax2.plot(filtered_data['total_batches_processed'], filtered_data['avg_train_loss'], color=color, label='Average Train Loss')
    ax2.plot(filtered_data['total_batches_processed'], filtered_data['validation_loss'], color='tab:green', label='Validation Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  
    ax2.legend(loc='upper left')

    plt.title('Training Process Metrics Over Batches')
    plt.show()

def load_and_visualize_log_scale(file_path):
    data = pd.read_csv(file_path, sep='\\s+')

    plt.style.use('seaborn-darkgrid')

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(data['total_batches_processed'], np.exp(data['total_train_loss']), label='Total Train Loss')
    ax.plot(data['total_batches_processed'], np.exp(data['avg_train_loss']), label='Average Train Loss')
    ax.plot(data['total_batches_processed'], np.exp(data['validation_loss']), label='Validation Loss')


    plt.title('Training Process Metrics Over Batches (Exponential Scale)')
    plt.xlabel('Total Batches Processed')
    plt.ylabel('Loss (Symmetric Log Scale)')
    plt.legend()

    plt.show()

load_and_visualize('./data/training_metadata.tsv')