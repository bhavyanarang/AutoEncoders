import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.conv_utils import (
    calculate2DConvOutput,
    getConv2D,
    getConvTranspose2D,
    FlattenBatch,
    UnflattenBatch
)

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_image_size: int, in_channels: int = 1, latent_dim: int = 64, beta: float = 1.0):
        super(VariationalAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.beta = beta  # KLD weight factor
        self.input_image_size = input_image_size
        
        # Encoder layers
        self.conv1 = getConv2D(in_channels, 32, 3, 1, 1)
        self.conv2 = getConv2D(32, 64, 3, 2, 1)
        self.conv3 = getConv2D(64, 128, 3, 2, 1)

        # Calculate size after convolutions
        self.size_after_conv = input_image_size
        self.size_after_conv = calculate2DConvOutput(3, 1, 1, self.size_after_conv)
        for _ in range(2):
            self.size_after_conv = calculate2DConvOutput(3, 1, 2, self.size_after_conv)

        self.image_size_after_conv = [128, self.size_after_conv, self.size_after_conv]
        self.fc_in_size = 128 * self.size_after_conv * self.size_after_conv

        # Encoder dense layers with SiLU activation
        self.encoder_conv = nn.Sequential(
            self.conv1, nn.SiLU(),
            self.conv2, nn.SiLU(),
            self.conv3, nn.SiLU(),
            FlattenBatch()
        )
        
        # Dense encoder layers with SiLU
        self.fc1 = nn.Linear(self.fc_in_size, 512)
        self.fc2 = nn.Linear(512, 256)
        
        # Latent space projections
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.softplus = nn.Softplus()
        
        # Decoder layers with SiLU activation
        self.fc_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
            nn.SiLU(),
            nn.Linear(512, self.fc_in_size),
            nn.SiLU()
        )
        
        self.decoder_conv = nn.Sequential(
            UnflattenBatch(shape=self.image_size_after_conv),
            getConvTranspose2D(128, 64, 3, 2, 1, 1),  # Added output_padding=1
            nn.SiLU(),
            getConvTranspose2D(64, 32, 3, 2, 1, 1),   # Added output_padding=1
            nn.SiLU(),
            getConv2D(32, in_channels, 3, 1, 1),      # Changed to Conv2D for final layer
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor, eps: float = 1e-8) -> tuple:
        x = self.encoder_conv(x)
        x = nn.SiLU()(self.fc1(x))
        x = nn.SiLU()(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        scale = self.softplus(logvar) + eps
        scale_tril = torch.diag_embed(scale)
        return torch.distributions.MultivariateNormal(mu, scale_tril=scale_tril)

    def reparameterize(self, dist: torch.distributions.MultivariateNormal) -> torch.Tensor:
        if self.training:
            return dist.rsample()
        return dist.mean

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc_decoder(z)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> tuple:
        dist = self.encode(x)
        z = self.reparameterize(dist)
        recon_x = self.decode(z)
        # Return reconstruction, mean, and log variance to maintain compatibility
        return recon_x, dist.mean, torch.log(dist.covariance_matrix.diagonal(dim1=-2, dim2=-1))

    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        dist = self.encode(x)
        return dist.mean

    def loss_calc(self, x: torch.Tensor, recon_x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> tuple:
        """Calculate VAE loss with reconstruction (MSE) and KL divergence terms"""
        # Reconstruction loss (MSE)
        mse_loss = nn.MSELoss(reduction='sum')(recon_x, x)
        
        # KL divergence loss
        # KL = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        total_loss = mse_loss + self.beta * kld_loss
        
        return total_loss, mse_loss, kld_loss 