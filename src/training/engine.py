import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def train_epoch_binary(model, data_loader, loss_fn, optimizer, scheduler, device, epoch, num_epochs):
    model.train()
    total_loss, correct_predictions = 0, 0
    train_pbar = tqdm(enumerate(data_loader), 
                      total=len(data_loader),
                      desc=f"Epoch {epoch+1}/{num_epochs} - Training")
    for batch_idx, batch in train_pbar:
        # Preleva i dati dal batch
        batch['input_values'] = batch['input_values'].to(device)
        batch['label'] = batch['label'].to(device)
        
        optimizer.zero_grad()

        outputs = model(batch)

        loss = loss_fn(outputs, batch['label'])
        total_loss += loss.item()

        # Calcolo delle predizioni
        preds = (outputs > 0.5).float()
        correct_predictions += torch.sum(preds == batch['label'])

        # Backpropagation e aggiornamento dei pesi
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()  # Aggiorna lo scheduler se necessario

        train_pbar.set_postfix(loss=loss.item(), accuracy=correct_predictions.float() / ((batch_idx + 1) * data_loader.batch_size))

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    return avg_loss, accuracy

def eval_model_binary(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, correct_predictions = 0, 0
    predictions, targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch['input_values'] = batch['input_values'].to(device)
            batch['label'] = batch['label'].to(device)

            outputs = model(batch)
            loss = loss_fn(outputs,  batch['label'])
            total_loss += loss.item()

            preds = (outputs > 0.5).float()
            correct_predictions += torch.sum(preds == batch['label'])

            predictions.extend(preds.cpu().numpy())
            targets.extend(batch['label'].cpu().numpy())

    f1 = f1_score(targets, predictions, average='macro')
    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset), f1

def eval_model_by_file_aggregation(model, data_loader, device):
    # Dizionario per raccogliere le probabilità per ogni file
    # defaultdict(list) crea una lista vuota per ogni nuova chiave
    file_scores = defaultdict(list)
    file_labels = {} # Dizionario per memorizzare l'etichetta di ogni file
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch['input_values'] = batch['input_values'].to(device)
            batch['label'] = batch['label'].to(device)
            filenames = batch['filename']    

            outputs = model(batch)

            for i in range(len(filenames)):
                filename = filenames[i]
                score = outputs[i].item() # La probabilità predetta
                label = batch['label'][i].item()
                
                file_scores[filename].append(score)
                
                # Memorizziamo l'etichetta del file (sarà la stessa per tutti i suoi segmenti)
                if filename not in file_labels:
                    file_labels[filename] = int(label)
                
        final_predictions = []
        true_labels = []

        # Iteriamo sui file in ordine alfabetico per assicurarci che l'ordine sia consistente
        for filename in sorted(file_scores.keys()):
            avg_score = np.mean(file_scores[filename])
            predicted_label = 1 if avg_score > 0.5 else 0
            
            final_predictions.append(predicted_label)
            true_labels.append(file_labels[filename])
    
        accuracy = accuracy_score(true_labels, final_predictions)
        f1 = f1_score(true_labels, final_predictions, average='macro')
        # La Confusion Matrix ci dà TP, TN, FP, FN per calcolare sensitività e specificità
        tn, fp, fn, tp = confusion_matrix(true_labels, final_predictions).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return accuracy, f1, sensitivity, specificity

def train_epoch(model, data_loader, loss_fn, optimizer, scheduler, device, epoch, num_epochs):
    model.train()
    total_loss, correct_predictions = 0, 0
    train_pbar = tqdm(enumerate(data_loader), 
                      total=len(data_loader),
                      desc=f"Epoch {epoch+1}/{num_epochs} - Training")
    for batch_idx, batch in train_pbar:
        # Preleva i dati dal batch
        batch['input_values'] = batch['input_values'].to(device)
        batch['label'] = batch['label'].to(device)
        if 'attention_mask' in batch:
            batch['attention_mask'] = batch['attention_mask'].to(device)
        
        optimizer.zero_grad()

        # Calcolo dei logits grezzi
        outputs = model(batch)

        # Calcolo della loss (i logits vengono passati così come sono)
        loss = loss_fn(outputs, batch['label'])
        total_loss += loss.item()

        # Calcolo delle predizioni: per multi-classe usiamo argmax sui logits
        softmax = nn.LogSoftmax(dim=1)
        preds = softmax(outputs).argmax(dim=1)
        correct_predictions += torch.sum(preds == batch['label'])

        # Backpropagation e aggiornamento dei pesi
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()  # Aggiorna lo scheduler se necessario
        
        train_pbar.set_postfix(loss=loss.item(), accuracy=correct_predictions.double() / ((batch_idx + 1) * data_loader.batch_size))

    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    return avg_loss, accuracy

def eval_model(model, data_loader, loss_fn, device):
    model.eval()
    total_loss, correct_predictions = 0, 0
    predictions, targets = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch['input_values'] = batch['input_values'].to(device)
            batch['label'] = batch['label'].to(device)
            if 'attention_mask' in batch:
                batch['attention_mask'] = batch['attention_mask'].to(device)

            outputs = model(batch)
            loss = loss_fn(outputs,  batch['label'])
            total_loss += loss.item()

            preds = torch.argmax(outputs,dim=1)
            correct_predictions += torch.sum(preds == batch['label'])

            predictions.extend(preds.cpu().numpy())
            targets.extend(batch['label'].cpu().numpy())

    f1 = f1_score(targets, predictions, average='macro')
    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset), f1