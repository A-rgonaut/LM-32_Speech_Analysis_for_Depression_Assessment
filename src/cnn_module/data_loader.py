import os
import sys
import random
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset as TorchDataset, DataLoader as TorchDataLoader
from tqdm import tqdm

from .config import CNNConfig
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src_utils
from preprocessor import E1_DAIC

class Dataset(TorchDataset):
    def __init__(self, audio_paths : str, labels : str, return_audio_id : bool = True, config : CNNConfig = CNNConfig(), balance_segments: bool = False):
        self.config = config
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = config.sample_rate
        self.segment_samples = self.config.segment_samples  
        self.return_audio_id = return_audio_id
        self.balance_segments = balance_segments
        
        self.segment_indices = []
        self.__precompute_segments()

        if self.balance_segments:
            self.__balance_segments()

        if not self.config.edaic_aug:
            self.audio_paths = self._filter_edaic_samples()

    def _filter_edaic_samples(self):
        filtered_audio_paths = []
        
        for id, audio_path in enumerate(self.audio_paths):
            participant_id = os.path.basename(audio_path).split('_AUDIO.wav')[0]
            if int(participant_id) < 600:
                filtered_audio_paths.append(audio_path)

        return filtered_audio_paths

    def __balance_segments(self):
        print("\n--- Bilanciamento dei segmenti in corso (under-sampling)... ---")
        
        # Separa gli indici dei segmenti per classe
        segments_class_0 = []
        segments_class_1 = []
        for segment_index in self.segment_indices:
            audio_id = segment_index[0]
            if self.labels[audio_id] == 0:
                segments_class_0.append(segment_index)
            else:
                segments_class_1.append(segment_index)

        n_0, n_1 = len(segments_class_0), len(segments_class_1)
        print(f"Segmenti originali: Classe 0: {n_0}, Classe 1: {n_1}")

        # Esegui l'under-sampling
        if n_0 > n_1:
            # Sotto-campiona la classe 0
            segments_class_0 = random.sample(segments_class_0, n_1)
        elif n_1 > n_0:
            # Sotto-campiona la classe 1
            segments_class_1 = random.sample(segments_class_1, n_0)
        
        # Combina e mescola i segmenti bilanciati
        self.segment_indices = segments_class_0 + segments_class_1
        random.shuffle(self.segment_indices)
        
        print(f"Segmenti bilanciati: Classe 0: {len(segments_class_0)}, Classe 1: {len(segments_class_1)}")
        print(f"Numero totale di segmenti dopo il bilanciamento: {len(self.segment_indices)}\n")
    
    def __precompute_segments(self):
        for id, audio_path in enumerate(self.audio_paths):
            num_frames = torchaudio.info(audio_path).num_frames
            starts = np.arange(0, num_frames - self.segment_samples + 1, self.segment_samples)
            self.segment_indices.extend([(id, int(start), self.segment_samples) for start in starts])

    def __len__(self):
        return len(self.segment_indices)

    def __getitem__(self, idx):
        id, start_sample, num_frames = self.segment_indices[idx]
        file_path = self.audio_paths[id]
        label = self.labels[id]
        
        waveform_segment, _ = torchaudio.load(file_path,
                                            frame_offset = start_sample,
                                            num_frames = num_frames)
        input_values = waveform_segment

        item = {'input_values': input_values, 
                'label': torch.tensor(label, dtype = torch.float32)}
        
        if self.return_audio_id:
            item['audio_id'] = id

        return item

class DataLoader():
    def __init__(self, config : CNNConfig = CNNConfig()):
        self.config = config
        self.batch_size = config.batch_size
        self.preprocessor = E1_DAIC(config.daic_path, config.e_daic_path, config.e1_daic_path)
        self.splits = self.preprocessor.get_dataset_splits()
        self.pos_weight = 1.0
        if not self.config.edaic_aug:
            self.splits = self._filter_edaic_samples()
        self.check_class_balance_by_segments()

    def _filter_edaic_samples(self):
        filtered_splits = []
        
        for split in self.splits:
            filtered_split = split[split['Participant_ID'] < 600].copy()
            if len(filtered_split) > 0:
                filtered_splits.append(filtered_split)
        
        return filtered_splits

    def check_class_balance_by_segments(self):
        """
        Analizza la distribuzione delle classi contando il numero effettivo di segmenti
        che verranno generati per il training, la validazione e il test.
        Questo è il metodo più accurato per valutare il bilanciamento.
        """
        print("\n************************************************************")
        print(f"Analisi Bilanciamento Classi")
        print("************************************************************")
        
        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = src_utils.get_splits(self.splits)
        
        # Funzione helper per analizzare uno split
        def analyze_split(paths, labels, name):
            if not len(labels):
                print(f"\nSet di {name}: Vuoto.")
                return 0, 0

            temp_dataset = Dataset(paths, labels, return_audio_id=False, config=self.config)
            segment_counts = {0: 0, 1: 0}
            
            for i in range(len(temp_dataset)):
                label = temp_dataset.labels[temp_dataset.segment_indices[i][0]]
                segment_counts[label] += 1
            
            total_segments = sum(segment_counts.values())
            if total_segments == 0:
                print(f"\n--- Set di {name}: Nessun segmento generato.")
                return 0, 0

            print(f"\n--- Set di {name} ({total_segments} segmenti totali) ---")
            count_0 = segment_counts.get(0, 0)
            count_1 = segment_counts.get(1, 0)
            print(f"  Classe 'No Depression' (0): {count_0} segmenti ({count_0 / total_segments:.2%})")
            print(f"  Classe 'Depression'    (1): {count_1} segmenti ({count_1 / total_segments:.2%})")
            
            return count_0, count_1

        # Analizza ogni split
        train_neg, train_pos = analyze_split(train_paths, train_labels, "Training")
        analyze_split(dev_paths, dev_labels, "Validation (Dev)")
        analyze_split(test_paths, test_labels, "Test")

        # Calcolo del pos_weight per la BCEWithLogitsLoss
        if train_pos > 0 and train_neg > 0:
            self.pos_weight = train_neg / train_pos
            print(f"\nSuggerimento per il bilanciamento:")
            print(f"  Usa un `pos_weight` di circa {self.pos_weight:.2f} per la loss function nel training.")

        print("************************************************************\n")

    def __get_generators(self):
        train_paths, train_labels, test_paths, test_labels, dev_paths, dev_labels = src_utils.get_splits(self.splits)

        train_dataset = Dataset(
            audio_paths = train_paths,
            labels = train_labels,
            return_audio_id = False,
            #balance_segments=True,
            config = self.config
        )
        
        test_dataset = Dataset(
            audio_paths = test_paths,
            labels = test_labels,
            return_audio_id = True,
            config = self.config
        )
        
        dev_dataset = Dataset(
            audio_paths = dev_paths,
            labels = dev_labels,
            return_audio_id = False,
            config = self.config
        )

        return train_dataset, test_dataset, dev_dataset

    def load_data(self):
        train_dataset, test_dataset, dev_dataset = self.__get_generators()

        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size = self.batch_size,
            shuffle=True, 
            num_workers = os.cpu_count())
        
        test_loader = TorchDataLoader(
            test_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count())
        
        dev_loader = TorchDataLoader(
            dev_dataset, 
            batch_size = self.batch_size, 
            num_workers = os.cpu_count())

        return train_loader, test_loader, dev_loader