import os
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report

from ..src_utils import get_metrics

class Evaluator:
    def __init__(self, model, test_loader):
        self.test_loader = test_loader
        self.results_file = 'results/ssl_evaluation_results.csv'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def evaluate(self): 
        self.model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                inputs = batch['input_values'].to(self.device)
                labels = batch['label'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model({'input_values': inputs, 'attention_mask': attention_mask})
                preds = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
                predictions.extend(preds.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        metrics = get_metrics(targets, predictions)
        print(confusion_matrix(targets, predictions))
        print(classification_report(targets, predictions, target_names=['No Depression', 'Depression']))
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