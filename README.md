# LM-32-2025-Progetto-Intelligent-Signal-Analysis

UNIPA - Corso di laurea magistrale in Ingegneria Informatica (2035)

Intelligent Signal Analysis A.A 2024/2025 - Prof. Sabato Marco Siniscalchi

Team: Andrea Spinelli, Antonio Spedito, Davide Bonura

![pic](https://github.com/user-attachments/assets/c2e75ac3-1612-4beb-ae9f-4bc8ea00645f)

## Struttura del Progetto
Il codice è organizzato in una struttura modulare per garantire chiarezza, manutenibilità e riusabilità.
- **`datasets/`**: Contiene i dati grezzi e i file CSV utilizzati per il training e il test.
- **`scripts/`**: Contiene gli script eseguibili per avviare il training dei diversi modelli (`train_cnn.py`, `train_transformer.py`).
- **`src/`**: Il cuore del progetto, organizzato come un package Python.
  - **`src/data/`**: Contiene le definizioni delle classi `Dataset` e delle funzioni `collate_fn` per il caricamento e la preparazione dei dati.
  - **`src/models/`**: Contiene le definizioni delle architetture dei modelli (CNN e Transformer).
  - **`src/training/`**: Contiene la logica riutilizzabile per i loop di training (`engine.py`) e altre funzioni di utilità (`utils.py`).

## Esecuzione degli Script di Training
**Importante:** Tutti i comandi devono essere eseguiti dalla **directory principale (root)** del progetto, dopo aver installato le dipendenze specificate in `requirements.txt`.

Per eseguire gli script si utilizza il flag `-m` di Python, che permette di lanciare un modulo e garantisce che gli import da `src` vengano risolti correttamente.

#### Training del Modello CNN
Questo script avvia il training del modello CNN, più leggero e veloce.

```bash
python -m scripts.train_cnn
```

### Training del Modello Transformer (Wav2Vec2)
Questo script avvia il fine-tuning del modello basato su Transformer.
```bash
python -m scripts.train_transformer
```