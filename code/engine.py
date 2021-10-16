import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import config

def loss_fn(sentiment_logits, sentiment_target):
    return nn.BCEWithLogitsLoss()(sentiment_logits, sentiment_target)

def calculate_accuracy(outputs, labels, type_acr = 'batch'):
    if type_acr == 'batch':
        labels = labels.view(outputs.shape[0], -1)
        labels = labels.cpu().detach().numpy().tolist()
        outputs = torch.sigmoid(outputs).cpu().detach().numpy().tolist()    
    outputs = np.array(outputs) >= 0.5
    return metrics.accuracy_score(labels, outputs)


def train_fn(data_loader, model, optimizer, device, scheduler, accumulation_steps= config.GRADIENT_ACCUMULATION_CONSTANT):
    model.train()
    
    total_loss = 0
    total_accuracy = 0
    with tqdm(enumerate(data_loader), unit="batch", total=len(data_loader)) as tepoch:
        for batch_index, dataset in tepoch: 
            tepoch.set_description(f"Epoch Started")

            input_ids = dataset['input_ids']
            attention_mask = dataset['attention_mask']
            token_type_ids = dataset['token_type_ids']
            aspect_word_masking = dataset['aspect_word_masking']
            sentiment_target = dataset['sentiment_value']

            input_ids = input_ids.to(device, dtype = torch.long)
            attention_mask = attention_mask.to(device, dtype = torch.long)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            aspect_word_masking = aspect_word_masking.to(device, dtype = torch.bool)
            sentiment_target = sentiment_target.to(device, dtype = torch.float)

            optimizer.zero_grad()

            sent_logit = model(
                input_ids, 
                attention_mask,
                token_type_ids,
                aspect_word_masking
                )

            loss = loss_fn(sent_logit, sentiment_target)
            loss.backward()
            
            train_accuracy = 100.0 * calculate_accuracy(sent_logit, sentiment_target)
            tepoch.set_postfix(loss=loss.item(), accuracy=train_accuracy)
            
            if (batch_index+1) % accumulation_steps == 0 :
                optimizer.step()
                scheduler.step()
        
            total_loss += loss.item()
            total_accuracy += train_accuracy
            
    # return total_loss/len(data_loader)
    return total_loss/len(data_loader), total_accuracy/len(data_loader)

def eval_fn(data_loader, model, device):
    model.eval()
    
    final_outputs = []
    final_targets = []

    with torch.no_grad():
        for _, dataset in tqdm(enumerate(data_loader), total=len(data_loader)): 
            input_ids = dataset['input_ids']
            attention_mask = dataset['attention_mask']
            token_type_ids = dataset['token_type_ids']
            aspect_word_masking = dataset['aspect_word_masking']
            sentiment_target = dataset['sentiment_value']

            input_ids = input_ids.to(device, dtype = torch.long)
            attention_mask = attention_mask.to(device, dtype = torch.long)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            aspect_word_masking = aspect_word_masking.to(device, dtype = torch.bool)
            sentiment_target = sentiment_target.to(device, dtype = torch.float)

            sent_logit = model(
                input_ids, 
                attention_mask,
                token_type_ids,
                aspect_word_masking
                )

            final_targets.extend(sentiment_target.cpu().detach().numpy().tolist())
            final_outputs.extend(torch.sigmoid(sent_logit).cpu().detach().numpy().tolist())
            
    return final_outputs, final_targets


    