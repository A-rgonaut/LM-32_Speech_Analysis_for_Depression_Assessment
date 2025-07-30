class SVMConfig:
    def __init__(self):
        self.edaic_aug = True
        self.seed = 42
        self.strategy = 'mean'
        self.kernel = ['rbf', 'linear']
        self.class_weight = ['balanced', None]
        self.C = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
        self.gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1, 'scale', 'auto']
        self.model_save_path = 'saved_models/svm_model.pkl'
        self.daic_path = 'datasets/DAIC-WOZ/'
        self.e_daic_path = 'datasets/E-DAIC-WOZ/'
        self.e1_daic_path = 'datasets/E1-DAIC-WOZ/'