import os
import sys
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src_utils import get_metrics

class Evaluator:
    def __init__(self, model, test_X, test_y):
        self.model = model
        self.test_X = test_X
        self.test_y = test_y
        self.results_file = 'results/svm_evaluation_results.csv'

    def evaluate(self, feature_type='unknown'):
        pred_y = self.model.predict(self.test_X)
        
        metrics = get_metrics(self.test_y, pred_y)
        metrics['feature_type'] = feature_type

        print(f"Evaluation for feature: {feature_type}")
        print(confusion_matrix(self.test_y, pred_y))
        print(classification_report(self.test_y, pred_y, target_names=['No Depression', 'Depression']))
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")

        self.save_results(metrics)

    def save_results(self, data):
        df = pd.DataFrame([data])
        results_dir = os.path.dirname(self.results_file)
        if results_dir and not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        if not os.path.exists(self.results_file):
            df.to_csv(self.results_file, index=False, header=True)
        else:
            df.to_csv(self.results_file, mode='a', header=False, index=False)
