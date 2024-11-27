import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# Use GPU
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
torch.manual_seed(1)

GPU = True
if GPU:
    device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")

print(f'Using {device}')

from VAE import *
from utils import *
from plotting import *

import os

def main():
    data_file = "exp_merged.feather"
    df = pd.read_feather("data/" + data_file)
    print(df.shape)

    # Preprocess data
    columns = ['black', 'hispanic', 'married', 'nodegree', 'age', 'education', 're78', 't']
    df_clean = df.loc[:, columns]

    scaler = StandardScaler()
    std_scale_cols = ['age', 'education', 're78']
    df_clean.loc[:, std_scale_cols] = scaler.fit_transform(df.loc[:, std_scale_cols])

    df_train, df_val = train_test_split(df_clean, test_size=0.2, random_state=42)

    # Settings
    batch_size = 32
    print_every = 20
    learning_rate = 0.01
    num_epochs = 20
    beta = 0.05

    # Put into dataloader
    loader_train = DataLoader(MyDataset(df_train), batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(MyDataset(df_val), batch_size=batch_size, shuffle=True)

    # Create model
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = num_epochs)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    print(model)

    # Training
    train_loss = []
    train_recon_loss = []
    train_KL = []

    val_loss = []
    val_recon_loss = []
    val_KL = []

    for epoch in range(num_epochs):

        for batch_idx, (x, _) in enumerate(loader_train):
            model.train()
            x = x.to(device)
            optimizer.zero_grad()

            recon_x, mu, logvar = model(x)
            loss, recon_loss, KL_divergence = loss_function(recon_x, x, mu, logvar, beta=beta, combined_loss=True)

            loss.backward()
            optimizer.step()

            if batch_idx % print_every == 0:
                train_loss.append(loss)
                train_recon_loss.append(recon_loss)
                train_KL.append(KL_divergence)

                curr_val_loss, curr_val_recon_loss, curr_val_KL = evaluate_model(loader_val, model, loss_function,
                                                                                 beta=beta, combined_loss=True)

                val_loss.append(curr_val_loss)
                val_recon_loss.append(curr_val_recon_loss)
                val_KL.append(curr_val_KL)

                print(
                    f'Epoch:{epoch}, Iteration:{batch_idx}, Loss = {loss.item():.5f}, MSE = {recon_loss.item()}, KL = {KL_divergence.item()}')
                print(
                    f'Epoch:{epoch}, Iteration:{batch_idx}, Val loss = {curr_val_loss.item():.5f}, Val MSE = {curr_val_recon_loss.item()}, Val KL = {curr_val_KL.item()}')

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: SGD lr {before_lr:.4f} -> {after_lr:.4f}")


    # Plot loss
    plot_loss(train_loss, train_recon_loss, train_KL, beta=beta)
    plot_loss(val_loss, val_recon_loss, val_KL, beta=beta)


    # Evaluate marginal distribution of each variable
    generated_data = reconstruct_training_data(model, df_clean, columns[:-1])
    plot_all_variables(df_clean, generated_data)

    generated_data = sample_from_latent(model, df_clean.shape[0], latent_dim, columns[:-1])
    plot_all_variables(df_clean, generated_data)



if __name__ == '__main__':
    main()
