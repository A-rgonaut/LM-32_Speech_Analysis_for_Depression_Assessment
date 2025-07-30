import gc
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, classification_report

def clear_cache():

    gc.collect()
    torch.cuda.empty_cache()
    np.random.seed(42)  # Per riproducibilità
    torch.manual_seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

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

def get_metrics(y_true, y_pred, *args : str):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    report = classification_report(y_true, y_pred, target_names=['No Depression', 'Depression'], output_dict=True, zero_division=0)

    metrics = {
        'tn'                : tn, 
        'fp'                : fp, 
        'fn'                : fn, 
        'tp'                : tp,
        'accuracy'          : accuracy_score(y_true, y_pred),
        'roc_auc'           : roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5,
        'sensitivity'       : tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity'       : tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'f1_macro'          : report['macro avg']['f1-score'],
        'f1_no_depression'  : report['No Depression']['f1-score'],
        'f1_depression'     : report['Depression']['f1-score']
    }

    if not args:
        return metrics
    else:
        return {metric: metrics[metric] for metric in args if metric in metrics}