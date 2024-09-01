import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import random
import math
import wandb
from tqdm import tqdm

import torch.optim as optim
from torch.utils.data import random_split

import numpy as np
import pickle

from torch.utils.data import DataLoader, TensorDataset
from comp_autoencoder_fc_modeling import Encoder, Decoder, AutoEncoder



epochs = 200
batch_size = 256
lr = 0.0001
use_cuda = 1
#weight_decay = 1e-5

train_file_path_gpu = '/scratch/cl5503/cost_model_auto_encoder/pre_train/batched/comp_and_expr_tensors/comp_tensors_GPU.pt'
train_file_path_cpu = '/scratch/cl5503/cost_model_auto_encoder/pre_train/batched/comp_and_expr_tensors/comp_tensors_CPU.pt'

val_file_path_gpu = '/scratch/cl5503/cost_model_auto_encoder/pre_train/batched/comp_and_expr_tensors_val/comp_tensors_GPU.pt'
val_file_path_cpu = '/scratch/cl5503/cost_model_auto_encoder/pre_train/batched/comp_and_expr_tensors_val/comp_tensors_CPU.pt'


device_gpu = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device_cpu = torch.device("cpu")
print("Loading dataset ... ")
with open(train_file_path_gpu, "rb") as file:
    train_data_gpu = torch.load(file, map_location=device_gpu)
with open(train_file_path_cpu, "rb") as file:
    train_data_cpu = torch.load(file, map_location=device_cpu)

with open(val_file_path_gpu, "rb") as file:
    val_data_gpu = torch.load(file, map_location = device_cpu)
with open(val_file_path_cpu, "rb") as file:
    val_data_cpu = torch.load(file, map_location = device_cpu)



print("Train data size: ", len(train_data_gpu) + len(train_data_cpu))
print("Val data size: ", len(val_data_gpu) + len(val_data_cpu))



valid_data = []
valid_data.extend(val_data_gpu)
valid_data.extend(val_data_cpu)


train_dataloader_gpu = DataLoader(train_data_gpu, batch_size = batch_size, shuffle = True)
train_dataloader_cpu = DataLoader(train_data_cpu, batch_size = batch_size, shuffle = True)
valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = False)
train_batch_nb = len(train_dataloader_gpu) + len(train_dataloader_cpu)


wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="pre_training",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
    },
)


model = AutoEncoder().to(device_gpu)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_function = nn.MSELoss().to(device_gpu)

# Train
model.train()
best_valid_loss = math.inf

found_outlier = False
for epoch in range(epochs):

    train_loss = 0.0
    model.train()

    for data in tqdm(train_dataloader_gpu):
        inputs = data.view(-1, 1636).to(device_gpu) 
            
        model.zero_grad()
        
        reconstruction = model(inputs)

        loss = loss_function(reconstruction, inputs)
        loss.backward()

        optimizer.step()
        
        train_loss += loss.item()


    for data in tqdm(train_dataloader_cpu):

        inputs = data.view(-1, 1636).to(device_gpu) 
            
        model.zero_grad()
        
        reconstruction = model(inputs)

        loss = loss_function(reconstruction, inputs)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()


    valid_loss = 0.0
    model.eval()

    for data in tqdm(valid_dataloader):
        inputs = data.view(-1, 1636).to(device_gpu)
        reconstruction = model(inputs)
        loss = loss_function(reconstruction, inputs)
        valid_loss += loss.item()
        #print("Valid batch loss: ", loss.item())

    print("Epoch ", epoch, ", (train loss) ", train_loss/train_batch_nb , " (valid loss): ", valid_loss/len(valid_dataloader))


    wandb.log({"train_loss": train_loss/train_batch_nb, "valid_loss": valid_loss/len(valid_dataloader)})

    if best_valid_loss > valid_loss:
        best_valid_loss = valid_loss
        # Saving State Dict
        torch.save(model.state_dict(), "saved_models/comps_w_expr_ae_fc_li_bn_code250.pt")
        

print("Finished training comps fc autoencoder w expr")
