import random
from torch.utils.data import Sampler

class BalancedParticipantSampler(Sampler):
    def __init__(self, dataset, batch_size, chunk_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size

        self.participant_info = dataset.participant_info
        
        self.positive_participants = [pid for pid, info in self.participant_info.items() if info['label'] == 1]
        self.negative_participants = [pid for pid, info in self.participant_info.items() if info['label'] == 0]

        self.num_batches = max(len(self.positive_participants), len(self.negative_participants)) // (self.batch_size // 2)
        
        print(f"Sampler creato: {len(self.positive_participants)} Positivi, {len(self.negative_participants)} Negativi."
            f"Genererà {self.num_batches} batch per epoca.")

    def __iter__(self):
        pos_iter = self._infinite_iterator(self.positive_participants)
        neg_iter = self._infinite_iterator(self.negative_participants)

        for _ in range(self.num_batches):
            batch_participants = set()
            while len(batch_participants) < self.batch_size:
                 batch_participants.add(next(pos_iter))
                 if len(batch_participants) < self.batch_size:
                     batch_participants.add(next(neg_iter))
            
            batch_items = []
            for participant_id in batch_participants:
                total_segments = self.participant_info[participant_id]['total_segments']
                
                max_start_index = total_segments - self.chunk_size
                start_index = random.randint(0, max_start_index) if max_start_index >= 0 else 0
                
                batch_items.append({'participant_id': participant_id, 'start_index': start_index})
            
            yield batch_items

    def _infinite_iterator(self, data_list):
        while True:
            random.shuffle(data_list)
            for item in data_list:
                yield item

    def __len__(self):
        return self.num_batches