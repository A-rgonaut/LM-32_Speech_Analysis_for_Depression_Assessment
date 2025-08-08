import os
import pandas as pd
import torch
import numpy as np
from tqdm.auto import tqdm

from .config import CNNConfig
from .model import CNNModel
from ..utils import clear_cache, get_metrics

class Evaluator:
    def __init__(self, test_loader, config: CNNConfig):
        self.config = config
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def evaluate(self, eval_type):
        all_metrics = [] 
        for i in range(1, self.config.k_folds + 1):
            path = os.path.join(self.config.model_save_dir, f'cnn_model_fold_{i}.pth')
            model = CNNModel(self.config)
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            model.to(self.device)
            model.eval()

            session_scores, session_targets = {}, {}
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc=f"Evaluating {eval_type} set with model {i}"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    audio_ids = batch.pop('audio_id')
                    labels = batch['label']    
                
                    outputs = model(batch)
                    scores = torch.sigmoid(outputs)
                
                    for idx in range(len(audio_ids)):
                        session_id = audio_ids[idx].item()
                        score = scores[idx].item()
                        target = labels[idx].item()

                        if session_id not in session_scores:
                            session_scores[session_id] = []

                        session_scores[session_id].append(score)
                        session_targets[session_id] = target

            final_predictions = []
            final_targets = []
            final_scores = []

            for session_id in session_scores:
                if self.config.eval_strategy == 'average':
                    avg_score = np.mean(session_scores[session_id])
                    predicted_label = 1 if avg_score > 0.5 else 0
                    final_scores.append(avg_score)
                elif self.config.eval_strategy == 'majority':
                    segment_predictions = [1 if score > 0.5 else 0 for score in session_scores[session_id]]
                    predicted_label = max(set(segment_predictions), key=segment_predictions.count)
                    final_scores.append(np.mean(session_scores[session_id]))

                final_predictions.append(predicted_label)
                final_targets.append(session_targets[session_id])

            metrics = get_metrics(final_targets, final_predictions, 'accuracy', 'f1_macro', 'roc_auc',
                                'sensitivity', 'specificity', y_score=final_scores)
            all_metrics.append(metrics)
            clear_cache()
        
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

    def save_results(self, metrics_data, eval_type):
        if self.config.result_dir and not os.path.exists(self.config.result_dir):
            os.makedirs(self.config.result_dir, exist_ok=True)

        results_file = os.path.join(self.config.result_dir, f'{eval_type}_results.csv')
        metrics_data.index.name = 'Metric'
        metrics_data.to_csv(results_file, index=True)
        print(f"Results saved to {results_file}")