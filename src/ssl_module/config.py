class SSLConfig:
    def __init__(self):
        self.edaic_aug = False
        self.seed = 42
        self.k_folds = 1
        self.eval_strategy = 'majority'
        self.segmentation_strategy = 'fixed_length' # 'fixed_length' or 'transcript'
        self.use_preextracted_features = True # Se True, usa feature pre-estratte. Se False, calcola al volo.
        self.num_layers_to_unfreeze = 0 # Deve essere 0 se use_preextracted_features è True

        self.layer_to_use = 8
        self.max_segments = None
        self.use_subdialogue_shuffling = True
        self.subdialogue_M_pos = 500
        self.subdialogue_len_low = 0.4
        self.subdialogue_len_high = 0.7

        self.sample_rate = 16000
        self.max_utt_seconds = 10.0
        self.min_utt_seconds = 7.0
        self.overlap_seconds = 0.0

        self.aggregate_layers = False
        self.gradient_accumulation_steps = 1
        self.dropout_rate = 0.1
        self.model_name = 'facebook/wav2vec2-base-960h'
        self.feature_path = f'features/{self.model_name}/'
        self.seq_model_type = 'transformer' 
        self.seq_hidden_size = 128
        self.seq_num_layers = 2
        self.transformer_nhead = 4

        self.epochs = 20 # Numero fisso di epoche per fold
        self.batch_size = 16
        self.learning_rate = 1e-6
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 0.01
        self.early_stopping_mode = 'max'

        self.model_save_dir = "saved_models/ssl/"
        self.result_dir = "results/ssl/"
        self.daic_path = 'datasets/DAIC-WOZ/'
        self.e_daic_path = 'datasets/E-DAIC-WOZ/'
        self.e1_daic_path = 'datasets/E1-DAIC-WOZ/'