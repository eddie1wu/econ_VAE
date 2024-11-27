import torch
import torch.nn as nn
import torch.nn.functional as F

input_dim = 7
latent_dim = 5
hidden_dim1 = 64
hidden_dim2 = 64

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )

        self.encoder_mu = nn.Linear(hidden_dim2, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, input_dim)
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)

        return mu + eps * std

    def decode(self, z):
        z = self.decoder(z)
        z[:, :4] = torch.sigmoid(z[:, :4])

        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)

        return out, mu, logvar

def loss_function(recon_x, x, mu, logvar, beta = 1, combined_loss = False):

    if combined_loss:
        bce_loss = F.binary_cross_entropy(recon_x[:, :4], x[:, :4], reduction = 'mean')
        mse_loss = F.mse_loss(recon_x[:, 4:7], x[:, 4:7], reduction = 'mean')
        recon_loss = bce_loss + mse_loss
    else:
        recon_loss = F.mse_loss(recon_x, x, reduction = 'mean')

    KL_divergence = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)

    loss = recon_loss + beta * KL_divergence

    return loss, recon_loss, KL_divergence