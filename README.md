# LM-32-2025-Progetto-Intelligent-Signal-Analysis

Unipa - Master's Degree in Computer Engineering (2025)

Intelligent Signal Analysis A.Y. 2024/2025 - Prof. Sabato Marco Siniscalchi

Team: Andrea Spinelli, Antonio Spedito, Davide Bonura

![pic](https://github.com/user-attachments/assets/c2e75ac3-1612-4beb-ae9f-4bc8ea00645f)

## Project Structure
The code is organized into a modular structure to ensure clarity, maintainability, and reusability.

- **`datasets/`**: Should contain the raw DAIC-WOZ and E-DAIC-WOZ datasets. The preprocessed `E1-DAIC-WOZ` dataset will also be generated here.
- **`features/`**: Directory where features extracted by SSL models (e.g., Wav2Vec2) are saved, if used.
- **`scripts/`**: Contains executable scripts for preprocessing, feature extraction, training, and testing the models.
- **`src/`**: The core of the project, organized as a Python package.
  - **`preprocessor.py`**: Handles the initial processing of audio and transcripts to create the unified E1-DAIC-WOZ dataset.
  - **`cnn_module/`**: Contains modules related to the 1D-CNN model (data loader, architecture).
  - **`svm_module/`**: Contains modules related to the SVM models (data loader, model logic).
  - **`ssl_module/`**: Contains modules for the first SSL-based model (chunk-based data loader, sequential architecture).
  - **`ssl_module_2/`**: Contains modules for the second SSL-based model (segment-based data loader, fine-tuning).
  - **`trainer.py`**: Unified Trainer class for all PyTorch models.
  - **`evaluator.py`**: Unified Evaluator class for all models.
  - **`utils.py`**: Contains shared utility functions (e.g., metrics calculation, seeding).
- **`saved_models/`**: Directory where trained models (`.pth`, `.pkl`) will be saved.
- **`results/`**: Directory where evaluation results (`.csv`) will be saved.
- **`config.yaml`**: Central configuration file to manage all experiment parameters.

## Setup and Execution

**Important:** All commands must be executed from the project's **root directory** after installing the dependencies.

```bash
pip install -r requirements.txt
```

### 1. Experiment Configuration
Before running any script, modify the `config.yaml` file. The `common.active_model` key determines which model will be used (`'svm'`, `'cnn'`, `'ssl'`). All other model-specific parameters are defined in their respective sections.

### 2. Data Preprocessing (Run once)
This script processes the raw datasets and generates the unified E1-DAIC-WOZ dataset required for training.

```bash
python -m scripts.preprocessor
```

### 3. Feature Extraction (Optional)
This step is required **only** for the `ssl` model if `use_preextracted_features` is set to `true` in `config.yaml`.

```bash
python -m scripts.feature_extraction
```

### 4. Model Training
The training script is unified. It will read `config.yaml`, identify the `active_model`, and launch the correct training procedure.

```bash
python -m scripts.train
```

### 5. Model Testing
Similar to training, the testing script is unified and relies on the configuration in `config.yaml`.

```bash
python -m scripts.test
```