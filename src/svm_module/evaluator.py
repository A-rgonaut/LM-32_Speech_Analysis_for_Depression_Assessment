import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from .config import SVMConfig
from ..utils import get_metrics

class Evaluator:
    def __init__(self, model, test_X, test_y, config: SVMConfig):
        self.model = model
        self.test_X = test_X
        self.test_y = test_y
        self.config = config

    def evaluate(self, feature_type, eval_type):
        pred_y = self.model.predict(self.test_X)
        pred_scores = self.model.predict_proba(self.test_X)[:, 1]

        metrics = get_metrics(self.test_y, pred_y, y_score=pred_scores)
        metrics['feature_type'] = feature_type

        print(f"Evaluation for feature: {feature_type} on {eval_type} set")
        print(confusion_matrix(self.test_y, pred_y))
        print(classification_report(self.test_y, pred_y, target_names=['No Depression', 'Depression']))
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")

        self.save_results(metrics, feature_type, eval_type)

    def save_results(self, metrics_data, feature_type, eval_type):
        df = pd.DataFrame([metrics_data])

        if not os.path.exists(self.config.result_dir):
            os.makedirs(self.config.result_dir, exist_ok=True)

        results_file = os.path.join(self.config.result_dir, f'{eval_type}_results_{feature_type}.csv')

        df.to_csv(results_file, index=False)
        print(f"{eval_type.capitalize()} set results saved to {results_file}")