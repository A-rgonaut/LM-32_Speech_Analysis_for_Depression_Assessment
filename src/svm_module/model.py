import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import SVMConfig
from ..utils import get_metrics

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

    def find_best_params(self, train_X, train_y):
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
            n_jobs=-1
        )

        grid.fit(train_X, train_y)
        print(f"Best parameters found: {grid.best_params_}")
        print(f"Best cross-validation F1-score: {grid.best_score_:.4f}")
        
        best_params = {key.replace('svm__', ''): value for key, value in grid.best_params_.items()}
        return best_params

    def train_and_evaluate_kfold(self, train_X, train_y, eval_X, eval_y, best_params):
        fold_metrics_list = []

        print("Starting K-Fold Validation on Evaluation Set (using best params)")
        for fold, (train_idx, _) in enumerate(self.kfold.split(train_X, train_y)):
            X_train_fold, y_train_fold = train_X[train_idx], train_y[train_idx]
            
            pipeline = self._create_pipeline(params=best_params)
            pipeline.fit(X_train_fold, y_train_fold)

            eval_pred = pipeline.predict(eval_X)
            eval_scores = pipeline.predict_proba(eval_X)[:, 1]

            metrics = get_metrics(eval_y, eval_pred, eval_scores, 'f1_macro', 'accuracy', 'roc_auc', 'sensitivity', 'specificity')
            metrics['fold'] = fold + 1
            fold_metrics_list.append(metrics)
            print(f"Fold {fold+1}/{self.config.k_folds} | Eval F1-Macro: {metrics['f1_macro']:.4f}")

        results_df = pd.DataFrame(fold_metrics_list)
        
        mean_metrics = results_df.mean()
        std_metrics = results_df.std()
        
        mean_metrics['fold'] = 'mean'
        std_metrics['fold'] = 'std'

        results_df = pd.concat([results_df, mean_metrics.to_frame().T, std_metrics.to_frame().T], ignore_index=True)
        results_df = results_df.set_index('fold')

        print("K-Fold Validation Results (Mean ± Std on Dev Set)")
        print(results_df.loc[['mean', 'std']])

        return results_df
    
    def train(self, train_X, train_y, best_params):
        self.model = self._create_pipeline(params=best_params)
        self.model.fit(train_X, train_y)

    def save_model(self):
        if not os.path.exists(self.config.model_save_dir):
            os.makedirs(self.config.model_save_dir)

        joblib.dump(self.model, os.path.join(self.config.model_save_dir, 'svm_model.pkl'))

    def load_model(self):
        self.model = joblib.load(os.path.join(self.config.model_save_dir, 'svm_model.pkl'))
        return self.model