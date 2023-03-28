from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Discriminator(nn.Module):
    def __init__(self, input_shape) :
        super(Discriminator).__init__()

        