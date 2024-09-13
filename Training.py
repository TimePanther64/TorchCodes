import torch
from torch import nn
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import copy
import numpy as np
import pandas as pd

from Utils import log

def train_client(wnet:torch.nn.Module, trainloader:torch.utils.data.DataLoader,
                validloader:torch.utils.data.DataLoader, testloader:torch.utils.data.DataLoader,
                device:str, criterion, optimizer:torch.optim.Optimizer, local_epoch:int, worker_idx:int,
                epoch_idx:int, log_row:pd.Series) -> dict[dict[str, torch.Tensor], dict[str, torch.Tensor], pd.Series]:
    
    wnet.train()
    prev_params = {name : param.detach().clone() for name, param in wnet.named_parameters() if param.requires_grad}    
    epoch_acc, epoch_loss = [], []
    
    for epoch in range(local_epoch):
        batch_loss, batch_acc = [], []
    
        for batch_idx, (data, targets) in enumerate(trainloader):
            data, targets = data.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            targets = targets.long()
            wnet.zero_grad()
            logits = wnet(data)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(targets.view_as(pred)).sum().item()
            processed = len(data)
            batch_acc.append(correct/processed)
            batch_loss.append(loss.item())
        
        epoch_acc.append(sum(batch_acc)/len(batch_acc))
        epoch_loss.append(sum(batch_loss)/len(batch_loss))
    
    log_row = log(log_row, epoch_idx + epoch,  {f"Worker {worker_idx}'s Local Loss": epoch_loss, f"Worker {worker_idx}'s Local Accuracy": epoch_acc})
    
    wvalid_loss, wvalid_acc = eval_net(wnet, validloader, device, criterion)
    log_row = log(log_row, epoch_idx, {f"Worker {worker_idx}'s Validation Loss": np.mean(wvalid_loss), f"Worker {worker_idx}'s Validation Accuracy": np.mean(wvalid_acc)})

    wtest_loss, wtest_acc = eval_net(wnet, testloader, device, criterion)
    log_row = log(log_row, epoch_idx, {f"Worker {worker_idx}'s Test Loss": np.mean(wtest_loss), f"Worker {worker_idx}'s Test Accuracy": np.mean(wtest_acc)})
    
    curr_params = {name : param.detach().clone() for name, param in wnet.named_parameters() if param.requires_grad}    
    wgrads = {name: prev - curr for (name, prev), curr in zip(prev_params.items(), curr_params.values())}
    # wgrads = {name: (prev - curr) / (optimizer.param_groups[-1]['lr'] * len(trainloader) * local_epoch) for (name, prev), curr in zip(prev_params.items(), curr_params.values())}
    wbuffs = {name:wnon for name, wnon in wnet.named_buffers()}
    
    return wgrads, wbuffs, log_row
    # return wnet, log_row

def eval_net(net:torch.nn.Module, dataloader:torch.utils.data.DataLoader, device:str, criterion):
    net.eval()
    with torch.no_grad():
        batch_acc, batch_loss = [], []
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            targets = targets.long()
            logits = net(data)
            loss = criterion(logits, targets)
            pred = logits.argmax(dim=1, keepdim=True)
            correct = pred.eq(targets.view_as(pred)).sum().item()
            processed = len(data)
            batch_acc.append(correct/processed)
            batch_loss.append(loss.item())
    return batch_loss, batch_acc

