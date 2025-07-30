
class SSLConfig:

    def __init__(self):
        self.dropout_rate = 0.5
        #dataset
        self.sample_rate = 16000
        self.hop_ms = 20000
        self.segment_ms = 20000
        #train
        self.epochs = 50
        self.batch_size = 512
        self.learning_rate = 0.001
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 0.01
        self.early_stopping_mode = max
        #grid_param
        self.grid_batch_size = [16, 32, 64]
        self.grid_learning_rate = [0.001, 0.0005]
        self.grid_segment_ms = [250, 500]

        self.model_save_path = "saved_model/ssl_model.h5"
        self.e_daic_path = '../datasets/E-DAIC-WOZ/'
        self.e1_daic_path = '../datasets/E1-DAIC-WOZ/'