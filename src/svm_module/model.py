import os
import joblib
import numpy as np
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
        self.pipeline = Pipeline([('imputer', SimpleImputer(strategy=config.strategy)),
                               ('scaler', StandardScaler()),
                               ('svm', SVC(random_state = config.seed))])

    def train(self, train_X, train_y, dev_X, dev_y):
        X = np.vstack([train_X, dev_X])
        y = np.concatenate([train_y, dev_y])
        
        cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.config.seed)
        
        grid = GridSearchCV(
            estimator = self.pipeline,
            param_grid = {'svm__C': self.config.C,
                          'svm__gamma': self.config.gamma,
                          'svm__kernel': self.config.kernel,
                          'svm__class_weight': self.config.class_weight},
            cv = cv_strategy,
            scoring = 'f1_macro',
            verbose = 1,
            n_jobs = -1
            )

        grid.fit(X, y)

        print(f"Best parameters: {grid.best_params_}")
        print(f"Best score: {grid.best_score_}")

        self.model = grid.best_estimator_

    def save_model(self, path=None):
        if path is None:
            path = self.config.model_save_path

        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        joblib.dump(self.model, path)

    def load_model(self, path=None):
        if path is None:
            path = self.config.model_save_path
        self.model = joblib.load(path)
        return self.model