from __future__ import print_function
import time
import os, pdb
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from tqdm import tqdm
from vit_pytorchs.MvDeFormer import MvDeFormer
from datasets import get_dataloader



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed = 42
seed_everything(seed)
device='cuda:0'

def model(device=device):
    model = MvDeFormer(

    ).to(device)
    return  model

def train(model):
    # Training settings
    batch_size = 32
    epochs = 4000
    lr = 2e-4
    gamma = 0.7
    best_val_accuracy = 0
    min_loss_val = 10

    train_loader = get_dataloader(file_list='radar_data_trainlist.txt')
    valid_loader = get_dataloader(file_list='radar_data_validlist.txt')
    

    print(f"Length of train_loader : {len(train_loader)}")
    print(f"Length of train_loader : {len(valid_loader)}")

    # loss function
    criterion = nn.CrossEntropyLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        model.train()
        for data1,data2,data3, label in tqdm(train_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)
            # print(data)
            label = label.to(device)
            # print(label)
            output = model(data1,data2,data3)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)

        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data1, data2, data3, label in valid_loader:
                data1 = data1.to(device)
                data2 = data2.to(device)
                data3 = data3.to(device)

                label = label.to(device)

                val_output = model(data1,data2,data3)
                val_loss = criterion(val_output, label)

                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)

            if best_val_accuracy <= epoch_val_accuracy and epoch_val_loss <= min_loss_val:

                best_val_accuracy = epoch_val_accuracy
                min_loss_val = epoch_val_loss

                torch.save(model.state_dict(), "./model_parameter/model_parameter.pkl")

        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
        )

if __name__ == "__main__":
    model = model()
    try:
        model.load_state_dict(torch.load("./model_parameter/model_parameter.pkl"))
        print("Pretrained model parameters have been loaded!")
    except:
        print("Pretrained model parameters file not exists!")
    train(model=model)

