import random
from torch.utils.data import Sampler

class BalancedParticipantSampler(Sampler):
    def __init__(self, dataset):
        participant_info_fold = dataset.participant_info
        
        self.positive_chunks_by_pid = {}
        self.negative_chunks_by_pid = {}
        for pid, info in participant_info_fold.items():
            total_segments = info.get('total_segments', 0)
            chunk_size = dataset.config.chunk_segments
            
            overlap = dataset.config.chunk_overlap_segments
            step = chunk_size - overlap

            participant_chunks = []
            for start_idx in range(0, total_segments, step):
                    if start_idx < total_segments:
                        participant_chunks.append({'participant_id': pid, 'start_index': start_idx})

            if info['label'] == 1:
                self.positive_chunks_by_pid[pid] = participant_chunks
            else:
                self.negative_chunks_by_pid[pid] = participant_chunks
        
        print(f"Sampler data prepared for this fold: {len(self.positive_chunks_by_pid)} pos participants, {len(self.negative_chunks_by_pid)} neg participants.")

        self.batch_size = dataset.config.batch_size
        self.oversample_factor = 1.0

        self.pos_per_batch = self.batch_size // 2
        self.neg_per_batch = self.batch_size - self.pos_per_batch

        num_pos_chunks = sum(len(c) for c in self.positive_chunks_by_pid.values())
        num_neg_chunks = sum(len(c) for c in self.negative_chunks_by_pid.values())

        if num_pos_chunks >= num_neg_chunks:
            self.majority_class_is_positive = True
            self.num_batches = int((num_neg_chunks / self.neg_per_batch) * self.oversample_factor)
        else:
            self.majority_class_is_positive = False
            self.num_batches = int((num_pos_chunks / self.pos_per_batch) * self.oversample_factor)

        print(f"BalancedParticipantSampler created. It will generate {self.num_batches} batches per epoch.")
        print(f"Majority class is {'POSITIVE' if self.majority_class_is_positive else 'NEGATIVE'}.")

    def __iter__(self):
        if self.majority_class_is_positive:
            majority_map = self.positive_chunks_by_pid
            minority_map = self.negative_chunks_by_pid
            maj_per_batch = self.pos_per_batch
            min_per_batch = self.neg_per_batch
        else:
            majority_map = self.negative_chunks_by_pid
            minority_map = self.positive_chunks_by_pid
            maj_per_batch = self.neg_per_batch
            min_per_batch = self.pos_per_batch

        def minority_chunk_generator():
            pids = list(minority_map.keys())
            for pid in pids:
                random.shuffle(minority_map[pid])

            cursors = {pid: 0 for pid in pids}
            pid_idx = 0
            
            # Cicla all'infinito sui partecipanti
            while True:
                # Se abbiamo fatto un giro completo, rimescoliamo i partecipanti
                if pid_idx >= len(pids):
                    pid_idx = 0
                    random.shuffle(pids)

                pid = pids[pid_idx]
                
                # Fornisce il chunk
                yield minority_map[pid][cursors[pid]]
                cursors[pid] += 1
                
                # Se abbiamo esaurito i chunk di questo partecipante,
                # rimescoliamoli e resettiamo il cursore.
                if cursors[pid] >= len(minority_map[pid]):
                    cursors[pid] = 0
                    random.shuffle(minority_map[pid])

                pid_idx += 1

        # Definiamo un generatore che gestisce il campionamento per la classe maggioritaria,
        # rispettando l'oversample_factor per ogni partecipante.
        def majority_chunk_generator():
            pids = list(majority_map.keys())
            for pid in pids:
                random.shuffle(majority_map[pid])

            cursors = {pid: 0 for pid in pids}
            times_cycled = {pid: 0 for pid in pids}
            
            available_pids = list(pids)
            random.shuffle(available_pids)
            
            pid_idx = 0
            while available_pids:
                if pid_idx >= len(available_pids):
                    pid_idx = 0
                    random.shuffle(available_pids)

                pid = available_pids[pid_idx]
                
                # Fornisce il chunk
                yield majority_map[pid][cursors[pid]]
                cursors[pid] += 1
                
                # Se abbiamo esaurito i chunk di questo partecipante
                if cursors[pid] >= len(majority_map[pid]):
                    cursors[pid] = 0  # Resetta il cursore
                    times_cycled[pid] += 1  # Incrementa il contatore dei cicli completati
                    
                    # Se il partecipante ha completato i cicli permessi dall'oversample_factor
                    if times_cycled[pid] >= self.oversample_factor:
                        # Rimuovilo dalla lista dei disponibili per questa epoca
                        available_pids.pop(pid_idx)
                        continue  # Continua senza incrementare pid_idx

                pid_idx += 1

        min_iter = minority_chunk_generator()
        maj_iter = majority_chunk_generator()

        for _ in range(self.num_batches):
            min_part = [next(min_iter) for _ in range(min_per_batch)]
            maj_part = [next(maj_iter) for _ in range(maj_per_batch)]

            batch = maj_part + min_part

            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches