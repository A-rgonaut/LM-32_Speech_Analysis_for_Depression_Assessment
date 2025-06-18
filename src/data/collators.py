import torch

def collate_fn(batch):
    """
    Questa funzione serve perché diversi audio possono avere numero diverso di segmenti.
    Ad esempio:
    - Audio 1: 30 secondi → 3 segmenti da 10s
    - Audio 2: 50 secondi → 5 segmenti da 10s
    
    Per creare un batch uniforme, dobbiamo fare padding al numero massimo di segmenti.
    Viene chiamata automaticamente dal DataLoader quando batch_size > 1.
    """
    # Trova il numero massimo di segmenti nel batch
    max_segments = max([item['num_segments'] for item in batch])
    
    batch_input_values = []
    batch_labels = []
    batch_masks = []
    
    for item in batch:
        input_values = item['input_values']
        num_segments = item['num_segments']

        mask = torch.zeros(max_segments, dtype=torch.bool)
        mask[num_segments:] = True
        batch_masks.append(mask)

        # Pad se necessario (aggiunge segmenti di zeri)
        if num_segments < max_segments:
            padding_shape = (max_segments - num_segments, input_values.shape[1])
            padding = torch.zeros(padding_shape, dtype=input_values.dtype)
            input_values = torch.cat([input_values, padding], dim=0)
        
        batch_input_values.append(input_values)
        batch_labels.append(item['label'])

    return {
        'input_values': torch.stack(batch_input_values),
        'label': torch.stack(batch_labels),
        'attention_mask': torch.stack(batch_masks)
    }