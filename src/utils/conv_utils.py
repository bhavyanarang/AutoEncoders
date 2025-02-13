import torch
import torch.nn as nn
from typing import List

def calculate2DConvOutput(kernel_size: int, padding: int, stride: int, input_size: int) -> int:
    """Calculate output size for 2D convolution layer"""
    size = input_size + 2*padding - kernel_size
    size = size // stride
    size += 1
    return size

def calculate2DConvTransposeOutput(kernel_size: int, padding: int, stride: int, input_size: int, output_padding: int = 0) -> int:
    """Calculate output size for 2D transposed convolution layer"""
    size = (input_size - 1) * stride - 2*padding + kernel_size + output_padding
    return size

def getConv2D(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0) -> nn.Conv2d:
    """Get a 2D convolution layer with specified parameters"""
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
    )

def getConvTranspose2D(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, output_padding: int = 0) -> nn.ConvTranspose2d:
    """Get a 2D transposed convolution layer with specified parameters"""
    return nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_size, kernel_size),
        stride=(stride, stride),
        padding=(padding, padding),
        output_padding=(output_padding, output_padding)
    )

class FlattenBatch(nn.Module):
    """Flatten input tensor while preserving batch dimension"""
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        batch_size = input.size(0)
        return input.view(batch_size, -1)

class UnflattenBatch(nn.Module):
    """Unflatten input tensor to specified shape while preserving batch dimension"""
    def __init__(self, shape: List[int]):  # shape should be of the form [channels, size, size]
        super(UnflattenBatch, self).__init__()
        self.shape = shape
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(-1, self.shape[0], self.shape[1], self.shape[2]) 