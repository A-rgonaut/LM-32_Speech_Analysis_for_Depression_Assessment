import os
import pandas as pd
import torch
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report

from .svm_module.model import SVMModel
from .cnn_module.model import CNNModel
from .ssl_module.model import SSLModel
from .ssl_module_2.model import SSLModel2
from .utils import clear_cache, get_metrics, aggregate_predictions, get_predictions

class Evaluator:
    def __init__(self, config, test_data):
        self.config = config
        self.model_type = config.active_model
        self.test_data = test_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(self, eval_type: str, feature_type: str = None):
        if self.model_type in ['ssl', 'cnn', 'ssl2']:
            self._evaluate_pytorch(eval_type)
        elif self.model_type == 'svm':
            model = SVMModel(self.config)
            model = model.load_model(feature_type)
            self._evaluate_sklearn(eval_type, feature_type, model)

    def _evaluate_pytorch(self, eval_type: str):
        test_loader = self.test_data
        all_metrics = [] 
        all_session_scores = defaultdict(list) 
        all_session_targets = {}
        for i in range(1, self.config.k_folds + 1):
            path = os.path.join(self.config.model_save_dir, f'{self.model_type}_model_fold_{i}.pth')

            if self.model_type == 'ssl':
                model = SSLModel(self.config)
            elif self.model_type == 'cnn':
                model = CNNModel(self.config)
            elif self.model_type == 'ssl2':
                model = SSLModel2(self.config)

            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            model.to(self.device)
            model.eval()

            session_scores, session_targets, _ = get_predictions(
                test_loader, model, f"Evaluating {eval_type} set with model {i}", self.device
            )
            '''
            for session_id, scores_list in session_scores.items():
                all_session_scores[session_id].extend(scores_list)

            all_session_targets.update(session_targets)

            '''
            final_predictions, final_targets, final_scores = aggregate_predictions(
                session_scores, session_targets, self.config.eval_strategy
            )
            
            metrics = get_metrics(final_targets, final_predictions, 'accuracy', 'f1_macro', 'roc_auc',
                                'sensitivity', 'specificity', 'f1_depression', y_score=final_scores)
            all_metrics.append(metrics)
            #'''
            clear_cache()
        #'''
        df_metrics = pd.DataFrame(all_metrics)
        mean_metrics = df_metrics.mean()
        std_metrics = df_metrics.std()

        print(f"Evaluation on {eval_type} set")
        print("Mean Metrics across folds:")
        print(mean_metrics.to_string())
        print("\nStandard Deviation of Metrics across folds:")
        print(std_metrics.to_string())
        summary_df = pd.DataFrame({'mean': mean_metrics, 'std': std_metrics})
        summary_df.index.name = 'Metric'
        self._save_results(summary_df, f'{eval_type}_results.csv')
        '''
        final_predictions, final_targets, final_scores = aggregate_predictions(
            all_session_scores, all_session_targets, self.config.eval_strategy
        )
        metrics = get_metrics(final_targets, final_predictions, 'accuracy', 'f1_macro', 'roc_auc',
                              'sensitivity', 'specificity', 'f1_depression', y_score=final_scores)
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
        df_metrics.index.name = 'Metric'

        print(f"Evaluation on {eval_type} set")
        print(df_metrics)

        self._save_results(df_metrics, f'{eval_type}_results.csv')
        #'''

    def _evaluate_sklearn(self, eval_type: str, feature_type: str, model):
        test_X, test_y = self.test_data

        pred_y = model.predict(test_X)
        pred_scores = model.predict_proba(test_X)[:, 1]

        metrics = get_metrics(test_y, pred_y, 'accuracy', 'f1_macro', 'roc_auc',
                                'sensitivity', 'specificity', 'f1_depression', y_score=pred_scores)
        metrics['feature_type'] = feature_type

        print(f"Evaluation for feature: {feature_type} on {eval_type} set")
        print(confusion_matrix(test_y, pred_y))
        print(classification_report(test_y, pred_y, target_names=['No Depression', 'Depression']))
        print(f"Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        df_metrics = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
        df_metrics.index.name = 'Metric'

        self._save_results(df_metrics, f"{eval_type}_{feature_type}_results.csv")

    def _save_results(self, metrics_df, filename: str):
        if self.config.result_dir and not os.path.exists(self.config.result_dir):
            os.makedirs(self.config.result_dir, exist_ok=True)

        results_file = os.path.join(self.config.result_dir, filename)
        metrics_df.to_csv(results_file)
        print(f"Results saved to {results_file}")