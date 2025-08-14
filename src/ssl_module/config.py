class SSLConfig:
    def __init__(self):
        self.use_comet = False
        self.edaic_aug = False
        self.seed = 42
        self.k_folds = 1
        self.use_random_chunking = False
        self.eval_strategy = 'majority' # 'average' or 'majority'
        
        self.segmentation_strategy = 'fixed_length' # 'fixed_length' or 'transcript'
        self.sample_rate = 16000
        self.max_utt_seconds = 10.0
        self.min_utt_seconds = 1.0
        self.overlap_seconds = 0.0
        self.chunk_segments = 10
        self.chunk_overlap_segments = 0

        self.use_preextracted_features = True
        self.layer_to_use = 8
        self.aggregate_layers = False
        self.model_name = 'facebook/wav2vec2-base-960h'

        self.dropout_rate = 0.1
        self.feature_path = f'features/{self.model_name}/'
        self.seq_model_type = 'transformer'
        self.seq_hidden_size = 128
        self.seq_num_layers = 1
        self.transformer_nhead = 4

        self.gradient_accumulation_steps = 1
        self.epochs = 20
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 0.01
        self.early_stopping_mode = 'max'

        self.model_save_dir = "saved_models/ssl/"
        self.result_dir = "results/ssl/"
        self.daic_path = 'datasets/DAIC-WOZ/'
        self.e_daic_path = 'datasets/E-DAIC-WOZ/'
        self.e1_daic_path = 'datasets/E1-DAIC-WOZ/'