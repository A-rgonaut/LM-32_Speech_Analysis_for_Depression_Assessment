# LM-32-2025-Progetto-Intelligent-Signal-Analysis

Unipa - Master's Degree in Computer Engineering (2025)

Intelligent Signal Analysis A.Y. 2024/2025 - Prof. Sabato Marco Siniscalchi

Team: Andrea Spinelli, Antonio Spedito, Davide Bonura

![pic](https://github.com/user-attachments/assets/c2e75ac3-1612-4beb-ae9f-4bc8ea00645f)

## Project Description

This project aims to develop and compare various machine learning models for the automatic detection of depression from speech analysis. Three main approaches have been implemented:
1.  **Support Vector Machines (SVM)** based on traditional acoustic features (prosody, phonation, articulation).
2.  A **1D Convolutional Neural Network (CNN)** that operates directly on the raw audio waveform.
3.  An advanced architecture based on **Self-Supervised Learning (SSL)**, which utilizes pre-trained models like Wav2Vec2 to extract high-dimensional audio representations.

The entire workflow, from data preparation to evaluation, is managed through a central configuration file (`config.yaml`) to ensure maximum flexibility and reproducibility of experiments.

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

#### Hyperparameter Search Workflow
When `hyperparameter_search_mode` is set to `true` for `cnn` or `ssl`, the script performs a full grid search:
-   For each combination of hyperparameters, a K-Fold Cross-Validation is performed.
-   The models from each fold are saved temporarily.
-   If the average F1-score of the current run exceeds the best score recorded so far, the temporary models are made permanent, and the models from the previous best run are deleted.
-   At the end of the search, the `saved_models/` directory will contain only the K models from the winning hyperparameter combination.

#### Specific Workflow for the SSL Model
The `ssl` model follows a structured, three-phase workflow to first find the best architecture and then systematically compare different pre-trained models and their layers.
  
**Phase 1: Feature Extraction**
-   **Goal**: Pre-extract and save feature representations from all SSL models you wish to analyze.
-   **Action**:
    1.  In `config.yaml`, populate the `ssl.ssl_model_names` list with all desired models (e.g., `'facebook/wav2vec2-base-960h'`, `'microsoft/wavlm-base'`).
    2.  Run the extraction script:
        ```bash
        python -m scripts.feature_extraction
        ```

**Phase 2: Finding the Optimal Downstream Architecture**
-   **Goal**: Find the best sequential model (e.g., Transformer, BiLSTM) and its hyperparameters for processing the SSL features. This is done once using a strong reference model.
-   **Action**:
    1.  In `config.yaml`, set `active_model: 'ssl'`, `hyperparameter_search_mode: true` and `run_layer_sweep: false`.
    2.  Specify your single reference model using the **singular** key: `ssl_model_name: 'facebook/wav2vec2-base-960h'`. The `ssl_model_names` list is ignored in this mode.
    3.  Run the training script to start the grid search:
        ```bash
        python -m scripts.train
        ```
    4.  **Outcome**: This process will save the best-performing architecture's parameters to `saved_models/ssl/best_params_ssl.json`.

**Phase 3: Automated Sweep for Model and Layer Comparison**
-   **Goal**: Fairly compare the performance of each layer from each SSL model, using the optimal architecture found in Phase 2.
-   **Action**:
    1.  In `config.yaml`, set `hyperparameter_search_mode: false` and `run_layer_sweep: true`.
    2.  Ensure `ssl_model_names` contains the list of models for which you extracted features in Phase 1.
    3.  Populate `layers_to_use` with the layer indices you want to test (e.g., `[0, 1, ..., 12]`).
    4.  Run the training script:
        ```bash
        python -m scripts.train
        ```
    5.  **Outcome**: The script will automatically iterate through every model and layer combination, run a full K-fold cross-validation for each, and save the trained models in structured directories (e.g., `saved_models/ssl/microsoft-wavlm-base/layer8/`). A final summary of all F1-scores will be printed and saved to `results/ssl/ssl_layer_sweep_summary.csv`, ready for plotting and analysis.

### 5. Model Testing
Similar to training, the testing script is unified and relies on the configuration in `config.yaml`. It loads the K saved models and runs an evaluation on the test set, reporting the mean metrics and standard deviation.

```bash
python -m scripts.test
```