# LM-32-2025-Progetto-Intelligent-Signal-Analysis

UNIPA - Corso di laurea magistrale in Ingegneria Informatica (2035)

Intelligent Signal Analysis A.A. 2024/2025 - Prof. Sabato Marco Siniscalchi

Team: Andrea Spinelli, Antonio Spedito, Davide Bonura

![pic](https://github.com/user-attachments/assets/c2e75ac3-1612-4beb-ae9f-4bc8ea00645f)

## Project Structure
The code is organized into a modular structure to ensure clarity, maintainability, and reusability.

- **`datasets/`**: Should contain the raw DAIC-WOZ and E-DAIC-WOZ datasets. The preprocessed `E1-DAIC-WOZ` dataset will also be generated here.
- **`scripts/`**: Contains the executable scripts for preprocessing, training, and testing the models.
- **`src/`**: The core of the project, organized as a Python package.
  - **`src/preprocessor.py`**: Handles the initial processing of audio and transcript data to create a unified dataset (E1-DAIC-WOZ).
  - **`src/cnn_module/`**: Contains all modules related to the 1D-CNN model (config, data loader, model architecture, trainer, evaluator).
  - **`src/svm_module/`**: Contains all modules related to the SVM models (config, data loader, model logic, evaluator).
  - **`src/src_utils.py`**: Contains utility functions shared across the project (e.g., metrics calculation, cache clearing).
- **`saved_models/`**: Directory where trained model artifacts (`.pth`, `.pkl`) will be saved.
- **`results/`**: Directory where evaluation results (`.csv`) will be saved.

## Setup and Execution

**Important:** All commands must be executed from the **root directory** of the project after installing the dependencies specified in `requirements.txt`.

```bash
pip install -r requirements.txt
```

1. Preprocessing the Data
This script processes the raw datasets and generates the unified E1-DAIC-WOZ dataset required for training. This only needs to be run once.

```bash
python -m scripts/preprocess.py
```

2. Training Models

Training the SVM Models
```bash
python -m scripts/train_svm.py
```

Training the CNN Model
```bash
python -m scripts/train_cnn.py
```

Training the SSL Model
```bash
python -m scripts/train_ssl.py
```

3. Testing Models

Testing the SVM Models
```bash
python -m scripts/test_svm.py
``` 

Testing the CNN Model
```bash
python -m  scripts/test_cnn.py
``` 

Testing the SSL Model
```bash
python -m scripts/test_ssl.py
```