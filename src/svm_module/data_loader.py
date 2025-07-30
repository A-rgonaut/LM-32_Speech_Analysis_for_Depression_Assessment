import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from .utils import process_interview
from .config import SVMConfig
from ..preprocessor import E1_DAIC
from  ..src_utils import filter_edaic_samples

class DataLoader:
    def __init__(self, config: SVMConfig):
        self.config = config
        self.preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
        self.splits = self.preprocessor.get_dataset_splits()
        if not self.config.edaic_aug:
            self.splits = filter_edaic_samples(self.splits) 

    def __make_data(self):
        interviews_to_process = []
        for split in self.splits:
            for interview in split['Participant_ID']:
                feature_path = f'{self.config.e1_daic_path}{interview}_P/features/'
                if not os.path.exists(feature_path):
                    interviews_to_process.append(interview)

        if not interviews_to_process:
            #print("All features are already extracted.")
            return

        print(f"Extracting features for {len(interviews_to_process)} interviews...")
        args = [(interview, self.config.e1_daic_path) for interview in interviews_to_process]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            list(tqdm(executor.map(process_interview, args), total=len(interviews_to_process)))
        print("Feature extraction completed.")
        
    def load_data(self, feature_type):
        self.__make_data()
        train_X, test_X, dev_X = [], [], []
        train_y, test_y, dev_y = [], [], []

        print(f"Loading {feature_type} features...")
        for split in self.splits:
            split_name = split['Split'].iloc[0]

            for _, row in tqdm(split.iterrows(), total=split.shape[0], desc=f"Loading {split_name} data for {feature_type}"):
                participant_id = row['Participant_ID']
                split = row['Split']

                if not os.path.exists(interview_features := f'{self.config.e1_daic_path}{participant_id}_P/features/'):
                    print(f"Features for {participant_id} not found. Skipping...")
                    continue

                feature = np.load(f'{interview_features}{feature_type}_features.npy')

                if split_name == 'train':
                    train_X.append(feature.flatten())
                    train_y.append(row['PHQ_Binary'])
                elif split_name == 'test':
                    test_X.append(feature.flatten())
                    test_y.append(row['PHQ_Binary'])
                elif split_name == 'dev':
                    dev_X.append(feature.flatten())
                    dev_y.append(row['PHQ_Binary'])
        
        return train_X, train_y, test_X, test_y, dev_X, dev_y