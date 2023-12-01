# Model-Based Intrusion Detection System for CAN Protocol Anomaly Detection

This project aims to create a model-based intrusion detection system (IDS) for Controler Area Networks (CAN)

## Getting Started

### Installation

1. Clone the repository
2. Install the required Python packages
```bash
pip install -r requirements.txt
```

### Functionality 
Checkout `code/presentation.ipynb` to get an idea of the functionalities of the code base.

### Dataset Download
1. Create a data directory in the project root.
2. Create a .env file in the project root and add the full path of your data directory. (Sometimes jupyter notebooks has issue with "../")
```dotenv
DATA_PATH=DATA-DIR-PATH-GOES-HERE
```
3. Use CANDataset to download and process the dataset. The data will be processed and saved to parquet files and will be pulled from these subsequently.
Example use:
```py
from data_prep.CANDataset import CANDataset
from dotenv import load_dotenv

load_dotenv()
data_path = os.getenv('DATA_PATH')
dataset = CANDataset(data_path, log_verbosity=1)
```



