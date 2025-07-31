class CNNConfig:
    def __init__(self):
        self.edaic_aug = False
        # Da provare, al momento si otteneva F1 0.6 con edaic_aug a False e balance_segments a False
        self.balance_segments = True 
        self.eval_strategy = 'majority'
        
        self.sample_rate = 16000
        self.segment_samples = 64_000 
        self.hop_samples = 64_000 

        self.dropout_rate = 0.2

        self.epochs = 50
        self.batch_size = 32
        self.learning_rate = 0.001
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 0.01
        self.early_stopping_mode = 'max'
        
        self.grid_params = {
            'batch_size' : [16, 32, 64],
            'learning_rate' : [0.001, 0.0005],
            'segment_ms' : [250, 500],
            'spectrogram_window_frames' : [100, 120, 140]
        }

        self.model_save_path = "saved_models/cnn_model.pth"
        self.daic_path = 'datasets/DAIC-WOZ/'
        self.e_daic_path = 'datasets/E-DAIC-WOZ/'
        self.e1_daic_path = 'datasets/E1-DAIC-WOZ/'