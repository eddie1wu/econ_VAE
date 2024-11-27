import torch
from torch.utils.data import Dataset

import pandas as pd

class MyDataset(Dataset):
    """ Convert dataset into loader object for DataLoader """
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = torch.tensor(row.values[:-1], dtype=torch.float32)  # All columns except last are features
        label = torch.tensor(row.values[-1], dtype=torch.long)  # Last column is label, integer
        return features, label

def evaluate_model(loader, model, loss_function, **kwargs):
    """ Evaluate the model using validation set """
    loss, recon_loss, KLdiv = 0, 0, 0
    count = 0

    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)

            # forward pass
            recon_x, mu, logvar = model(x)

            # compute loss
            curr_loss, curr_recon_loss, curr_KLdiv = loss_function(recon_x, x, mu, logvar, **kwargs)

            # update the losses
            loss += curr_loss
            recon_loss += curr_recon_loss
            KLdiv += curr_KLdiv
            count += 1

    return (loss / count), (recon_loss / count), (KLdiv / count)

def reconstruct_training_data(model, df, columns):
    """
    Reconstruct the training data using the trained model.
    Args:
    df - the clean dataframe used for training
    """

    x_in = df.loc[:, columns]

    model.eval()
    with torch.no_grad():
        x_in = torch.tensor(x_in.values, dtype = torch.float32).to(device)
        x_out = model(x_in)[0]
        x_out = x_out.cpu().numpy()

    gen_data = pd.DataFrame(x_out, columns = columns)

    return gen_data

def sample_from_latent(model, n_samples, latent_dim, columns):
    """
    Sample from the latent distribution to generate new data
    """

    z = torch.randn(n_samples, latent_dim).to(device)

    model.eval()
    with torch.no_grad():
        z_out = model.decode(z)
        z_out = z_out.cpu().numpy()

    gen_data = pd.DataFrame(z_out, columns = columns)

    return gen_data
