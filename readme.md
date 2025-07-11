# Standalone nnUnet Inference

## Overview

This repository provides a trainer-agnostic and decoupled implementation for running nnUnet inference. 

The primary goal is to eliminate the dependency on the full `nnunetv2` package. By packaging this lightweight inference engine with a model's final checkpoint and its metadata (`dataset.json`, `dataset_fingerprint.json`, `plans.json`), we ensure that every model is permanently bundled with a compatible and reproducible inference script.

> **Note:** All necessary code is located in the `nnunetv2` directory. This naming is very misleading but intentionally kept for easy comparison with the original nnUnet source code but should be renamed in the future to make it clear it's only inference code.
---

## Setup

**Install Dependencies:**
Install the required Python packages.
```bash
pip install -r nnunetv2/requirements.txt
```

---

## Running Inference

1.  **Prepare Directories:**
    Organize your files with the following structure:

    ```
    .
    ├── data/
    │   └── ... (your NIfTI files)
    │
    └── inference_model/
        ├── checkpoint_best.pth
        └── metadata/
            ├── dataset.json
            ├── dataset_fingerprint.json
            └── plans.json
    ```
2.  **Execute Inference:**
    Run the following command from the root of the repository. The results will be saved to the `./results` directory.
    ```bash
    python -m nnunetv2.predict_from_raw_data \
        -i data \
        -o ./results \
        --inf_dir_path inference_model \
        --save_probabilities \
        -device cpu
    ```
---
## Maintenance & Extensibility
This implementation is designed to be easily extended with new components from nnUnet or custom-developed classes.
### Adding General Processing Components
To add a new **Normalization Class**, **Image Reader**, or **Resampling Function**:
1.  Add the new class or function to `customizable_parts/general_processing.py`.
2.  Register it by adding its name to the appropriate dictionary within the same file.
3.  The inference will then use it if it's correcly referenced in the `plans.json` file.
### Adding Preprocessor Classes
To add a new **Preprocessor**:
1.  Add the new class to `customizable_parts/preprocessor_classes.py`.
2.  Register it by adding its name to the `preprocessor_classes` dictionary in the same file.
3.  The inference will then use it if it's correcly referenced in the `plans.json` file.