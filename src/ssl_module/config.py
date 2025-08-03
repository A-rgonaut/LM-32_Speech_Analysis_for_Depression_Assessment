class SSLConfig:
    def __init__(self):
        self.edaic_aug = False
        
        self.sample_rate = 16000
        self.max_utt_seconds = 10.0
        self.max_segments = 130
        self.layer_to_extract = 8

        self.use_subdialogue_shuffling = True
        self.subdialogue_M_pos = 20
        self.subdialogue_len_low = 0.3
        self.subdialogue_len_high = 1.0

        self.dropout_rate = 0.1
        self.model_name = 'facebook/wav2vec2-base-960h'
        self.feature_path = f'features/{self.model_name}/'
        self.seq_model_type = 'transformer'
        self.seq_hidden_size = 128
        self.seq_num_layers = 2
        self.transformer_nhead = 4

        self.epochs = 50
        self.batch_size = 32
        self.learning_rate = 0.001
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 0.01
        self.early_stopping_mode = 'max'
        
        self.grid_params = {
            'learning_rate': [1e-3, 5e-4, 1e-4],
            'seq_hidden_size': [128, 256],
            'seq_num_layers': [1, 2],
            'dropout_rate': [0.1, 0.25, 0.5]
        }

        self.model_save_path = "saved_models/ssl_model.pth"
        self.daic_path = 'datasets/DAIC-WOZ/'
        self.e_daic_path = 'datasets/E-DAIC-WOZ/'
        self.e1_daic_path = 'datasets/E1-DAIC-WOZ/'