import os
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from ..src_utils import get_metrics

class Evaluator:
    def __init__(self, model, test_loader, eval_strategy='average'):
        self.test_loader = test_loader
        self.results_file = 'results/cnn_evaluation_results.csv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.eval_strategy = eval_strategy

    def evaluate(self): 
        session_preds = {}
        session_targets = {}
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                inputs = batch['input_values'].to(self.device)
                labels = batch['label'].to(self.device)
                audio_ids = batch['audio_id']    

                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs)

                for i in range(len(audio_ids)):
                    session_id = audio_ids[i].item()
                    pred = preds[i].item()
                    target = labels[i].item()

                    if session_id not in session_preds:
                        session_preds[session_id] = []
                    
                    session_preds[session_id].append(pred)
                    session_targets[session_id] = target

            final_predictions = []
            final_targets = []

            for session_id in session_preds:
                if self.eval_strategy == 'average':
                    avg_score = np.mean(session_preds[session_id])
                    predicted_label = 1 if avg_score > 0.5 else 0
                elif self.eval_strategy == 'majority':
                    segment_predictions = [1 if score > 0.5 else 0 for score in session_preds[session_id]]
                    predicted_label = max(set(segment_predictions), key=segment_predictions.count)
                
                final_predictions.append(predicted_label)
                final_targets.append(session_targets[session_id])

        metrics = get_metrics(final_targets, final_predictions)

        print(confusion_matrix(final_targets, final_predictions))
        print(classification_report(final_targets, final_predictions, target_names=['No Depression', 'Depression']))
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