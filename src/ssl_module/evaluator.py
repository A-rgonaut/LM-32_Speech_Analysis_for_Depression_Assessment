
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src_utils import get_metrics

class Evaluator:

    def __init__(self, model, test_y, pred_y):
        self.model = model
        self.test_y = test_y
        self.pred_y = pred_y

    def evaluate(self):

        sensitivity, specificity = get_metrics(self.test_y, self.pred_y, 'precision', 'recall', 'f1_score', 'accuracy', 'roc_auc').values()
        print(f'Sensitivity: {sensitivity:.2f}\nSpecificity: {specificity:.2f}')
