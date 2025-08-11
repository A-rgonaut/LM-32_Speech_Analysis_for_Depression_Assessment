import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from typing import Tuple, List
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def filter_edaic_samples(splits: Tuple[pd.DataFrame, ...]) -> List[pd.DataFrame]:
    """Filters out augmented E-DAIC samples (ID >= 600) from a tuple of dataframes."""
    filtered_splits = []
    for split_df in splits:
        filtered_split = split_df[split_df['Participant_ID'] < 600].copy()
        if not filtered_split.empty:
            filtered_splits.append(filtered_split)
    return filtered_splits

def get_splits(splits : tuple):
    train_split, test_split, dev_split = splits

    train_paths = [f'datasets/E1-DAIC-WOZ/{row["Participant_ID"]}_P/{row["Participant_ID"]}_AUDIO.wav' 
                   for _, row in train_split.iterrows()]
    test_paths  = [f'datasets/E1-DAIC-WOZ/{row["Participant_ID"]}_P/{row["Participant_ID"]}_AUDIO.wav' 
                   for _, row in test_split.iterrows()]
    dev_paths   = [f'datasets/E1-DAIC-WOZ/{row["Participant_ID"]}_P/{row["Participant_ID"]}_AUDIO.wav' 
                   for _, row in dev_split.iterrows()]
    
    train_labels = train_split['PHQ_Binary'].tolist()
    test_labels = test_split['PHQ_Binary'].tolist()
    dev_labels = dev_split['PHQ_Binary'].tolist()

    return train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels

def get_metrics(y_true, y_pred, *args: str, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    report = classification_report(y_true, y_pred, target_names=['No Depression', 'Depression'], output_dict=True, zero_division=0)

    roc_auc_value = 0.5
    if y_score is not None:
        roc_auc_value = roc_auc_score(y_true, y_score)

    metrics = {
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
        'accuracy': accuracy_score(y_true, y_pred),
        'roc_auc': roc_auc_value,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'f1_macro': report['macro avg']['f1-score'],
        'f1_no_depression': report['No Depression']['f1-score'],
        'f1_depression': report['Depression']['f1-score']
    }

    if not args:
        return metrics
    else:
        return {metric: metrics[metric] for metric in args if metric in metrics}
    
def get_predictions(data_loader, model, description, device, criterion=None):
    total_loss = 0
    session_scores, session_targets = {}, {}
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=description):
            batch = {k: v.to(device) for k, v in batch.items()}
            audio_ids = batch.pop('audio_id')
            labels = batch['label']    
        
            outputs = model(batch)
            scores = torch.sigmoid(outputs)
            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        
            for idx in range(len(audio_ids)):
                session_id = audio_ids[idx].item()
                score = scores[idx].item()
                target = labels[idx].item()

                if session_id not in session_scores:
                    session_scores[session_id] = []

                session_scores[session_id].append(score)
                session_targets[session_id] = target

    avg_loss = total_loss / len(data_loader) if criterion else None

    return session_scores, session_targets, avg_loss

def aggregate_predictions(session_scores, session_targets, eval_strategy='average'):
    final_predictions, final_targets, final_scores = [], [], []

    for session_id in session_scores:
        if eval_strategy == 'average':
            avg_score = np.mean(session_scores[session_id])
            predicted_label = 1 if avg_score > 0.5 else 0
            final_scores.append(avg_score)
        elif eval_strategy == 'majority':
            segment_predictions = [1 if score > 0.5 else 0 for score in session_scores[session_id]]
            predicted_label = max(set(segment_predictions), key=segment_predictions.count)
            final_scores.append(np.mean(session_scores[session_id]))

        final_predictions.append(predicted_label)
        final_targets.append(session_targets[session_id])
    
    return final_predictions, final_targets, final_scores
    
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = -np.inf if mode == 'max' else np.inf

    def __call__(self, current_score):
        if self.mode == 'max':
            improvement = (current_score - self.best_score) > self.min_delta
        else:
            improvement = (self.best_score - current_score) > self.min_delta

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Early stop
            
        return False