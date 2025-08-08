import os
import pandas as pd
import torch
from tqdm.auto import tqdm

from .model import SSLModel
from .config import SSLConfig
from ..utils import clear_cache, get_metrics

class Evaluator:
    def __init__(self, test_loader, config: SSLConfig):
        self.config = config
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(self, eval_type): 
        all_metrics = [] 
        for i in range(1, self.config.k_folds + 1):
            path = os.path.join(self.config.model_save_dir, f'ssl_model_fold_{i}.pth')
            model = SSLModel(self.config)
            model.to(self.device)
            model.eval()
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))

            predictions, targets, scores = [], [], []
            with torch.no_grad():
                for batch in tqdm(self.test_loader, desc=f"Evaluating {eval_type} set with model {i}"):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    labels = batch['label']
                    
                    outputs = model(batch)
                    scrs = torch.sigmoid(outputs)
                    preds = (scrs > 0.5).float()

                    scores.extend(scrs.cpu().numpy())
                    predictions.extend(preds.cpu().numpy())
                    targets.extend(labels.cpu().numpy())

            metrics = get_metrics(targets, predictions, 'accuracy', 'f1_macro', 'roc_auc',
                                    'sensitivity', 'specificity', y_score=scores)
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