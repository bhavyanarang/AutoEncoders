import torch
import torch.nn as nn
from src.utils.conv_utils import (
    calculate2DConvOutput,
    calculate2DConvTransposeOutput,
    getConv2D,
    getConvTranspose2D,
    FlattenBatch,
    UnflattenBatch
)

class ConvolutionalEncoder(nn.Module):
    def __init__(self, input_image_size: int, in_channels: int = 1) -> None:
        super(ConvolutionalEncoder, self).__init__()
        
        # Simpler encoder with stride=1 to maintain spatial dimensions
        self.conv1 = getConv2D(in_channels, 32, 3, 1, 1)  # same size
        self.conv2 = getConv2D(32, 16, 3, 1, 1)          # same size
        self.conv3 = getConv2D(16, 8, 3, 1, 1)           # same size

        # Calculate feature dimensions
        self.size_after_conv = input_image_size  # Size is maintained
        self.image_size_after_conv = [8, self.size_after_conv, self.size_after_conv]
        self.fc_in_size = 8 * self.size_after_conv * self.size_after_conv

        # Dense layer for bottleneck
        self.fc = nn.Linear(self.fc_in_size, 128)

        # Build encoder
        self.encoder = nn.Sequential(
            self.conv1, nn.ReLU(),
            self.conv2, nn.ReLU(),
            self.conv3, nn.ReLU(),
            FlattenBatch(),
            self.fc
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def get_fc_size(self) -> int:
        return 128  # Bottleneck size

    def get_final_conv_shape(self) -> list:
        return self.image_size_after_conv

class ConvolutionalDecoder(nn.Module):
    def __init__(self, fc_in_size: int, final_conv_shape: list, in_channels: int = 1):
        super(ConvolutionalDecoder, self).__init__()
        
        self.fc_in_size = fc_in_size  # 128
        self.final_conv_shape = final_conv_shape
        
        # Dense layer to restore shape
        self.fc = nn.Linear(self.fc_in_size, 8 * final_conv_shape[1] * final_conv_shape[2])
        
        # Decoder layers maintaining size
        self.conv1 = getConv2D(8, 16, 3, 1, 1)    # same size
        self.conv2 = getConv2D(16, 32, 3, 1, 1)   # same size
        self.conv3 = getConv2D(32, in_channels, 3, 1, 1)  # same size

        # Build decoder
        self.decoder = nn.Sequential(
            self.fc, nn.ReLU(),
            UnflattenBatch(shape=self.final_conv_shape),
            self.conv1, nn.ReLU(),
            self.conv2, nn.ReLU(),
            self.conv3, nn.Sigmoid()  # Sigmoid for image output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

class ConvolutionalAutoencoder(nn.Module):
    def __init__(self, input_image_size: int, in_channels: int = 1):
        super(ConvolutionalAutoencoder, self).__init__()
        
        # Create encoder
        self.encoder_module = ConvolutionalEncoder(input_image_size, in_channels)
        
        # Create decoder using encoder's output dimensions
        self.decoder_module = ConvolutionalDecoder(
            self.encoder_module.get_fc_size(),
            self.encoder_module.get_final_conv_shape(),
            in_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_module(x)
        x = self.decoder_module(x)
        return x

    def get_representations(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder_module(x)

    def loss_calc(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        return nn.MSELoss()(x, out) 