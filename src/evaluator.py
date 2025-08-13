import os
import pandas as pd
import torch
from collections import defaultdict

from .ssl_module.model import SSLModel
from .cnn_module.model import CNNModel
from .utils import clear_cache, get_metrics, aggregate_predictions, get_predictions

class Evaluator:
    def __init__(self, test_loader, config, model_name: str):
        self.config = config
        self.test_loader = test_loader
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(self, eval_type):
        all_metrics = [] 
        all_session_scores = defaultdict(list) 
        all_session_targets = {}
        for i in range(1, self.config.k_folds + 1):
            path = os.path.join(self.config.model_save_dir, f'{self.model_name}_model_fold_{i}.pth')
            if self.model_name == 'ssl':
                model = SSLModel(self.config)
            elif self.model_name == 'cnn':
                model = CNNModel(self.config)
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            model.to(self.device)
            model.eval()

            session_scores, session_targets, _ = get_predictions(
                self.test_loader, model, f"Evaluating {eval_type} set with model {i}", self.device
            )
            #'''
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
        '''
        df_metrics = pd.DataFrame(all_metrics)
        mean_metrics = df_metrics.mean()
        std_metrics = df_metrics.std()

        print(f"Evaluation on {eval_type} set")
        print("Mean Metrics across folds:")
        print(mean_metrics)
        print("\nStandard Deviation of Metrics across folds:")
        print(std_metrics)
        summary_df = pd.DataFrame({'mean': mean_metrics, 'std': std_metrics})
        self.save_results(summary_df, eval_type)
        '''
        final_predictions, final_targets, final_scores = aggregate_predictions(
            all_session_scores, all_session_targets, self.config.eval_strategy
        )
        metrics = get_metrics(final_targets, final_predictions, 'accuracy', 'f1_macro', 'roc_auc',
                              'sensitivity', 'specificity', 'f1_depression', y_score=final_scores)
        df_metrics = pd.DataFrame(metrics, index=[0])

        print(f"Evaluation on {eval_type} set")
        print("Metrics:")
        print(df_metrics)

        self.save_results(df_metrics, eval_type)
        #'''

    def save_results(self, metrics_data, eval_type):
        if self.config.result_dir and not os.path.exists(self.config.result_dir):
            os.makedirs(self.config.result_dir, exist_ok=True)

        results_file = os.path.join(self.config.result_dir, f'{eval_type}_results.csv')
        
        metrics_data.index.name = 'Metric'
        metrics_data.to_csv(results_file, index=True)
        print(f"Results saved to {results_file}")