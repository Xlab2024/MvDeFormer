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

def test(model):

    batch_size = 32
    # loss function
    criterion = nn.CrossEntropyLoss()

    test_loader = get_dataloader(file_list='radar_data_testlist.txt')
    
    model.load_state_dict(torch.load("./model_parameter/model_parameter.pkl",map_location='cuda:0'))  # 加载模型参数

    with torch.no_grad():
        model.eval() 
        epoch_test_accuracy = 0
        epoch_test_loss = 0
        for data1,data2,data3, label  in tqdm(test_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            data3 = data3.to(device)

            # print(data)
            label = label.to(device)

            test_output = model(data1,data2,data3)  # 进行使用
            test_loss = criterion(test_output, label)
            # print(test_output.argmax(dim=1))
            acc = (test_output.argmax(dim=1) == label).float().mean()
            epoch_test_accuracy += acc / len(test_loader)
            epoch_test_loss += test_loss / len(test_loader)


    print(
         f"test_loss : {epoch_test_loss:.4f} - test_acc: {epoch_test_accuracy:.4f}\n"
    )

if __name__ == "__main__":
    model = model()
    test(model=model)
