class CNNConfig:
    def __init__(self):
        self.edaic_aug = False
        self.seed = 42
        self.k_folds = 1
        self.balance_segments = False
        self.eval_strategy = 'majority'
        self.segmentation_strategy = 'fixed_length' # 'fixed_length' or 'transcript'
        
        self.sample_rate = 16000
        self.max_utt_seconds = 4.0
        self.min_utt_seconds = 2.0
        self.overlap_seconds = 0.0

        self.dropout_rate = 0.2

        self.gradient_accumulation_steps = 1
        self.epochs = 20
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 0.01
        self.early_stopping_mode = 'max'
        
        self.model_save_dir = "saved_models/cnn/"
        self.result_dir = "results/cnn/"
        self.daic_path = 'datasets/DAIC-WOZ/'
        self.e_daic_path = 'datasets/E-DAIC-WOZ/'
        self.e1_daic_path = 'datasets/E1-DAIC-WOZ/'