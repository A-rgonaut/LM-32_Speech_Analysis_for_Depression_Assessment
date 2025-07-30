import joblib
import numpy as np

from andrea_src.svm_module.config import SVMConfig
from andrea_src.svm_module.data_loader import DataLoader
from andrea_src.svm_module.evaluator import Evaluator

model_names = [
    'svm_model_articulation.pkl',
    'svm_model_phonation.pkl',
    'svm_model_prosody.pkl',
    'svm_model_combined.pkl'
]

config = SVMConfig()
data_loader = DataLoader(config)

for model_name in model_names:
    print(f"\n--- Testing model: {model_name} ---")
    model_path = f'saved_models/{model_name}'
    try:
        saved_model = joblib.load(model_path)
        print(f"Model '{model_name}' loaded with success from '{model_path}'")
    except FileNotFoundError:
        print(f"Error: model file not found in '{model_path}'. Skipping this model.")
        continue

    # Load test data
    if 'combined' in model_name:
        feature_types_list = ['articulation', 'phonation', 'prosody']
        all_test_X = []
        for feature_type in feature_types_list:
            _, _, test_X, test_y, _, _ = data_loader.load_data(feature_type)
            all_test_X.append(test_X)
        
        input_data = np.hstack(all_test_X)
        feature_type = 'combined'
        print("Dati di test combinati caricati.")
    else:
        feature_type = model_name.replace('svm_model_', '').replace('.pkl', '')
        _, _, input_data, test_y, _, _ = data_loader.load_data(feature_type)
        print(f"Dati di test per la feature '{feature_type}' caricati.")

    evaluator = Evaluator(saved_model, input_data, test_y)
    evaluator.evaluate(feature_type)