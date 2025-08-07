import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import SVMConfig

class SVMModel:
    def __init__(self, config: SVMConfig):
        self.model = None
        self.config = config
        self.kfold = StratifiedKFold(n_splits=self.config.k_folds, shuffle=True, random_state=self.config.seed)
    
    def _create_pipeline(self, params=None):
        if params:
            svm_instance = SVC(random_state=self.config.seed, probability=True, **params)
        else:
            svm_instance = SVC(random_state=self.config.seed, probability=True)
        
        return Pipeline([
            ('imputer', SimpleImputer(strategy=self.config.strategy)),
            ('scaler', StandardScaler()),
            ('svm', svm_instance)
        ])

    def tune_and_train(self, X, y):
        pipeline = self._create_pipeline()
        
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid={
                'svm__C': self.config.C,
                'svm__gamma': self.config.gamma,
                'svm__kernel': self.config.kernel,
                'svm__class_weight': self.config.class_weight
            },
            cv=self.kfold,
            scoring='f1_macro',
            verbose=1,
            n_jobs=-1,
            refit=True # Ensure the best model is refit on the entire dataset
        )

        grid.fit(X, y)

        print("Cross-Validation Results:")
        print("Best F1 score:", grid.best_score_)
        print(f"Best found parameters: {grid.best_params_}")

        # The final model, already trained on all data, is grid.best_estimator_
        self.model = grid.best_estimator_

        best_params_raw = grid.best_params_
        best_params = {key.replace('svm__', ''): value for key, value in best_params_raw.items()}

        return best_params

    def save_model(self, feature_type):
        path = os.path.join(self.config.model_save_dir, f'svm_model_{feature_type}.pkl')

        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

        joblib.dump(self.model, path)

    def load_model(self, feature_type):
        path = os.path.join(self.config.model_save_dir, f'svm_model_{feature_type}.pkl')

        self.model = joblib.load(path)
        return self.model