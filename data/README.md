# ROAD Dataset readme

If you plan to use this dataset for your research, please cite our paper:

```
@article{verma2020road,
  title={ROAD: the real ORNL automotive dynamometer controller area network intrusion detection dataset (with a comprehensive CAN IDS dataset survey \& guide)},
  author={Verma, Miki E and Iannacone, Michael D and Bridges, Robert A and Hollifield, Samuel C and Kay, Bill and Combs, Frank L},
  journal={arXiv preprint arXiv:2012.14600},
  year={2020}
}
```

## Goal:
The goal of this dataset is to provide an open realistic verified CAN dataset for benchmarking CAN IDS methods. We include raw CAN data and signal-translated CAN data both captured in ambient and attack settings. This data was captured on a singal vehicle and all attacks were physically verified--we observed and documented the effect of the CAN manipulations on the vehicle's functionality.

The fundamental advantage of ROAD dataset over prior efforts is that ours come from realistic verified labeled attacks as opposed to synthetic datasets. Each of the files in our dataset contains time series coming from hundreds of IDs that may have dozen of signals. This opens the possibility for CAN R&D based on a high-fidelity dataset allowing the evaluation, comparison, and validation of CAN signal-based IDS algorithms.



## File Structure

```
data/
├── ambient
│   ├── {capture_name}.log
│   ├── ...
│   └── capture_metadata.json
├── attacks
│   ├── {capture_name}.log
│   ├── {capture_name}_masquerade.log
│   ├── ...
│   └── capture_metadata.json
├── data_table.csv
└── readme.md
└── signal_extractions

    ├── ambient
    │   ├── {capture_name}.csv
    │   ├── ...
    │   └── metadata.json
    ├── attacks
    │   ├── {capture_name}.csv
    │   ├── ...
    │   └── metadata.json
    └── DBC
        └── anonymized.dbc
```

## Notes:
* `capture_name` is a short, unique, publishable description.
* In the `attacks/`subfolders, when message confliction (see [^miller_valasek]) is present in an attack, an identical log (with the conflicting non-injected messages removed), is also included with the suffix `_masquerade`.

* `capture_metadata.json` or  `metadata.json` files contain the metadata for all captures in the given directory.

* For ambient data: entries are in the following form:
```
"{capture_name}" : {
    "elapsed_sec": length of capture in seconds,
    "on_dyno": whether the data was collected on a dyno or on the road }
```
* For attack data: entries are in the following form:
```
"{capture_name}" : {
    "elapsed_sec": length of capture in seconds,
    "on_dyno": whether or not the data was collected on a dyno or on the road,
    "message_confliction_present": whether or not there is message confliction during the injection,
    "injected_aid": injected AID in hex*
    "injection_data": injected message hex string ('X' indicated wildcard, meaning the given byte was not altered)
    "injection_interval":[
        injection start time (in elapsed seconds),
        injection end time (in elapsed seconds)
    ]
}
```

* `injected_aid = Null` indicates that no AID in particular was the target. In the case of the fuzzing attack, all aids are injected (at least once), in the case of the accelerator attack, the injection happens before the start of the capture, and affects the the signals in many AIDs.



## `signal_extractions/` folder: Time Series from ROAD Dataset

The goal of including the time series version of this dataset is to provide an open realistic verified CAN dataset for benchmarking *signal-based* CAN IDS methods--i.e., those IDSs that operate on the time series of translated signals from the payloads of the CAN frames' data fields. For that purpose, we translate CAN logs (both ambient and attacks) from the raw CAN data into time series using CAN-D: Controller Area Network Decoder https://ieeexplore.ieee.org/document/9466242.


### `signal_extractions/` Data Description
This dataset consist of 12 ambient and 17 attack CSV files. All of them have the following structure:

`Label	ID 	Time	Signal_1_of_ID	Signal_2_of_ID	...`

- The column `Label` indicates if the data entry is normal (i.e., Label = 0) or an attack (i.e., Label =1). Labels in the ambient data are 0, meaning that there are no attacks.
- The column `ID` corresponds to the decimal value of the ID sending the message.
- The column `Time` corresponds to the timestamp of the message in seconds.
- The remaining columns (i.e., `Signal_i_of_ID`) represent the actual values of the signal in a particular `ID` at certain `Time`. Note that a single `ID` can contain dozens of signals. Missing values are inputed as NaNs.
- For the attack files of category **accelerator** the `Label` column was set to 0. This is to reflect the fact that these captures have no injected messages, but simply record the CAN data when the vehicle is in this state. Discrepancies in the driver inputs (e.g., pressing the accelerator pedal) with the vehicle’s actions are present. See more details about this in our paper: [ROAD: The Real ORNL Automotive Dynamometer Controller Area Network Intrusion Detection Dataset](https://arxiv.org/pdf/2012.14600.pdf).
- For more details about each of the files, please checkout the `metadata.json` file in each folder.




[^miller_valasek]: http://illmatics.com/can%20message%20injection.pdf
