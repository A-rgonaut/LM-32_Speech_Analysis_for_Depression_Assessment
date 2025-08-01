class SSLConfig:
    def __init__(self):
        self.edaic_aug = False
        self.balance_segments = False
        self.eval_strategy = 'majority'
        
        self.sample_rate = 16000
        self.max_utt_seconds = 10.0
        self.max_segments = 200

        self.dropout_rate = 0.1
        self.model_name = 'facebook/wav2vec2-base-960h'
        self.seq_model_type = 'transformer'
        self.seq_hidden_size = 128
        self.seq_num_layers = 2
        self.transformer_nhead = 4

        self.epochs = 50
        self.batch_size = 4
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

        self.model_save_path = "saved_models/ssl_model.pth"
        self.daic_path = 'datasets/DAIC-WOZ/'
        self.e_daic_path = 'datasets/E-DAIC-WOZ/'
        self.e1_daic_path = 'datasets/E1-DAIC-WOZ/'