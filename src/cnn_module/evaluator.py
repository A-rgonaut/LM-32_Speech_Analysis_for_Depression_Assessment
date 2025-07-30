import os
import sys
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src_utils import get_metrics

class Evaluator:
    def __init__(self, model, test_loader):
        self.test_loader = test_loader
        self.results_file = 'results/cnn_evaluation_results.csv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def evaluate(self): # Dizionario per raccogliere le probabilità per ogni file
        # defaultdict(list) crea una lista vuota per ogni nuova chiave
        file_scores = defaultdict(list)
        file_labels = {} # Dizionario per memorizzare l'etichetta di ogni file

        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                inputs = batch['input_values'].to(self.device)
                batch['label'] = batch['label'].to(self.device)
                filenames = batch['audio_id']    

                outputs = self.model(inputs)
                probabilities = torch.sigmoid(outputs)

                for i in range(len(filenames)):
                    filename = filenames[i]
                    score = probabilities[i].item()
                    label = batch['label'][i].item()
                    
                    file_scores[filename].append(score)
                    
                    # Memorizziamo l'etichetta del file (sarà la stessa per tutti i suoi segmenti)
                    if filename not in file_labels:
                        file_labels[filename] = int(label)
                    
            final_predictions = []
            true_labels = []

            # Iteriamo sui file in ordine alfabetico per assicurarci che l'ordine sia consistente
            for filename in sorted(file_scores.keys()):
                avg_score = np.mean(file_scores[filename])
                predicted_label = 1 if avg_score > 0.5 else 0
                
                final_predictions.append(predicted_label)
                true_labels.append(file_labels[filename])
        
        metrics = get_metrics(true_labels, final_predictions)

        print(confusion_matrix(true_labels, final_predictions))
        print(classification_report(true_labels, final_predictions, target_names=['No Depression', 'Depression']))
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")

        self.save_results(metrics)

    def save_results(self, data):
        df = pd.DataFrame([data])
        results_dir = os.path.dirname(self.results_file)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        if not os.path.exists(self.results_file):
            df.to_csv(self.results_file, index=False)
        else:
            df.to_csv(self.results_file, mode='a', header=False, index=False)