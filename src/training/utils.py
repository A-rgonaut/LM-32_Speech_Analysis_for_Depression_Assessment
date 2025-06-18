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

def get_audio_paths(df, dataset_name):
    audio_paths = []
    for participant_id in df['Participant_ID']:
        dir_name = f"{participant_id}_P"
        wav_path = os.path.join(dataset_name, dir_name, f"{participant_id}_AUDIO.wav")
        if os.path.isfile(wav_path):
            audio_paths.append(wav_path)
        else:
            print(f"Warning: File non trovato per {participant_id} in {wav_path}")
    return audio_paths

def get_split_audio_paths(df, dataset_name):
    """
    For each participant in df, collect all 10s audio segment paths in their directory.
    Returns a list of paths and a list of corresponding labels.
    """
    audio_paths = []
    labels = []
    for idx, row in df.iterrows():
        participant_id = int(row['Participant_ID'])  # Ensure integer for directory name
        label = int(row['PHQ8_Binary']) if 'PHQ8_Binary' in row else int(row['PHQ_Binary'])
        dir_name = f"{participant_id}_P"
        part_dir = os.path.join(dataset_name, dir_name)
        if os.path.isdir(part_dir):
            for fname in os.listdir(part_dir):
                if fname.endswith('.wav') and '_part' in fname:
                    audio_paths.append(os.path.join(part_dir, fname))
                    labels.append(label)
        else:
            print(f"Warning: Directory not found for {participant_id} in {part_dir}")
    return audio_paths, labels

def print_model_summary(model):
    print("-"*89)
    print(f"| Layer Name                                              | # of Parameters | Trainable |")
    print("-"*89)
    total_num_trainable_params = 0
    for layer_name, layer_params in model.named_parameters():
        if layer_params.requires_grad:
            total_num_trainable_params += layer_params.numel()
        print(f"| {layer_name:<55} | {layer_params.numel():<15} | {str(layer_params.requires_grad):<9} |")
    print("-"*89)
    print(f"| Total # of Parameters: {total_num_trainable_params:<62} |" )
    print("-"*89)

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