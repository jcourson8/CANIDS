# Model-Based IDS

This project aims to create a model-based intrusion detection system (IDS) for Controler Area Networks (CAN)

## Getting Started

### Installation

1. Clone the repository
2. Install the required Python packages
```bash
pip install -r requirements.txt
```
4. Because this is a shared repo create a .env file in the project root and add the full path of your data directory. (Sometimes jupyter notebooks has issue with "../")
```dotenv
DATA_PATH=DATA-DIR-PATH-GOES-HERE
```
5. Use CanDataLoader to download and process the dataset. The data will be saved to parquet files and will be pulled from these subsequently.
Example use:
```py
from CanDataLoader import CanDataLoader
from dotenv import load_dotenv

load_dotenv()
data_path = os.getenv('DATA_PATH')
dataset = CanDataLoader(data_path, log_verbosity=1)
```


## README Checklist

---

### Data Preparation:

**Ambient Log Files:**
- [ ] **Step 1:** For each ambient log file, follow the steps below:
    - [ ] Load the log file into a dataframe.
    - [ ] Ensure the dataframe columns are ordered as ["Timestamp", "AID", "Data", "DeltaTimeSinceLastPacket", "DeltaTimeSameAID"].
    - [ ] Convert the dataframe into a parquet file format.
    - [ ] Name the parquet file using the log filename, excluding the ".log" extension. Ensure it ends with ".parquet".

**Attack Log Files:**
- [ ] **Step 2:** For each Attack log file, follow the steps below:
    - [ ] Load the log file into a dataframe.
    - [ ] Ensure the dataframe columns are ordered as ["Timestamp", "AID", "Data", "DeltaTimeSinceLastPacket", "DeltaTimeSameAID"].
    - [ ] Convert the dataframe into a parquet file format.
    - [ ] Name the parquet file using the log filename, excluding the ".log" extension. Ensure it ends with ".parquet".

---

### Data Visualization:

- [ ] **Step 3:** Refer to the attack metadata to identify the "injection_id". This will provide N "attack" examples.
- [ ] **Step 4:** Examine N normal time differences for the corresponding AID.
- [ ] **Step 5:** Highlight the "injection_interval" for the attack data.
- [ ] **Step 6:** Consider creating a 3D plot for better data representation and visualization.

---

### Create Keras Model:

- [ ] **Step 7:** Initiate the Keras model with the following structure:
    - [ ] Implement the first CAN ID embedding layer (12 bits).
    - [ ] This should produce a CAN vector representation (CANIDVR) for the total CAN IDs.

---

### Future Ideas:

- [ ] **Idea 1:** Consider training the model with and without outliers.
- [ ] **Idea 2:** Evaluate both the embedding and clustering approach to determine which method better identifies similar data.

---

### Mid Report:

- [ ] **Reference 1:** Review the binning method from [can-time-based-ids-benchmark](https://github.com/pmoriano/can-time-based-ids-benchmark/tree/main).
- [ ] **Reference 2:** Check out the dataset available at [road](https://0xsam.com/road/).

---
