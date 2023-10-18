# Time-Based IDS

This project aims to create a model-based intrusion detection system (IDS) for Controler Area Networks (CAN)

## Getting Started

### Installation

1. Clone the repository
2. Install the required Python packages
```bash
pip install -r requirements.txt
```
4. Install and dump the contents of the [ROAD Dataset](https://road.nyc3.digitaloceanspaces.com/road.zip) in the data directory.


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

*Note: Always backup data and regularly save progress to avoid data loss or redundancy in efforts.*