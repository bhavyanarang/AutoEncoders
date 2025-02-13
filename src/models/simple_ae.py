import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    """A simple fully-connected autoencoder"""
    def __init__(self, input_shape: int, hidden_shape: int):
        super(SimpleAutoencoder, self).__init__()
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape

        self.encoder = nn.Sequential(
            nn.Linear(input_shape, hidden_shape),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_shape, input_shape),
            nn.Sigmoid()  # For image data, output should be between [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        req_shape = x.size()
        x = x.view(-1, self.input_shape)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(req_shape)
        return x

    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.input_shape)
        return self.encoder(x)

    def loss_calc(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(x, out) 