import os

import torch
import torch.nn as nn


class Net(nn.Module):
    """IP-VAE architecture"""
    def __init__(self, zdim, input_size, hidden_size, latent_dim, num_layers=1):
        """Initializes layers.

        Args:
            zdim (int): the latent vector dimensions
        """
        super(Net, self).__init__()
        self.zdim = zdim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                                    batch_first=True, num_layers=num_layers)
        self.fcmu = nn.Linear(hidden_size, latent_dim)
        self.fclogvar = nn.Linear(hidden_size, latent_dim)
        # Decoder
        self.decoder_lstm = nn.LSTM(input_size=latent_dim, output_size=input_size, hidden_size=hidden_size, 
            num_layers=self.num_layers, batch_first=True)
        self.fc5 = nn.Linear(hidden_size, input_size)

    def encode(self, input_size, hidden_size, num_layers):
        """Decodes a latent vector sample.

        Args:
            x (tensor): input sequence

        Returns:
            mu, logvar (tensors), the mean and variance of q(z|x)

        """
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        """Reparametrization trick.

        Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        https://arxiv.org/abs/1312.6114

        Args:
            mu (tensor): the mean of q(z|x)
            logvar (tensor): natural log of the variance of q(z|x)

        Returns:
            z
        """
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        """Decodes a latent vector sample.

        Args:
            z (tensor):

        Returns:
            x' (tensor), reconstructed output

        """
        h4 = torch.tanh(self.fc4(z))
        h5 = torch.tanh(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x):
        """IP-VAE forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar