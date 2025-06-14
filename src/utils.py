import numpy as np
import os

def load_labels_from_dataset(df):
    """
    Carica le labels da un DataFrame.

    Args:
        df: DataFrame contenente Participant_ID e PHQ8_Binary/PHQ_Binary
        
    Returns:
        np.array: Array delle labels (0 = non-depressed, 1 = depressed)
    """
    label_col = 'PHQ8_Binary' if 'PHQ8_Binary' in df.columns else 'PHQ_Binary'
    y_list = []
    
    for _, row in df.iterrows():
        label = int(row[label_col])
        y_list.append(label)
    
    return np.array(y_list)

def load_features_from_dataset(df, dataset_name, features_type):
    """
    Carica le features per un dataset specifico.

    Args:
        df: DataFrame contenente Participant_ID
        dataset_name: Nome della cartella del dataset
        features_type: Tipo di features da caricare 
        
    Returns:
        np.array: Array delle features flattened
    """
    X_list = []
    
    for _, row in df.iterrows():
        participant_id = int(row['Participant_ID'])
        
        # Costruisce il percorso delle features
        feature_path = os.path.join("features", dataset_name, f"{participant_id}_P", f"{features_type}_features.npy")
        
        features = np.load(feature_path)
        X_list.append(features.flatten())
    
    return np.array(X_list)