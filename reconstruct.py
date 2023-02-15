import os
import numpy as np
import pandas as pd
import warnings
import pickle
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torch.optim as optim
from utils import load_saved_model, applyAE, get_mnist_dataloaders, visualize_ae_reconstruction
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from models import conv_AE, simple_AE, conv_AE_without_conv_transpose, variational_AE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
warnings.filterwarnings("ignore")

num_epochs = 2
in_channels = 1

# ## simple AE
# train_loader, test_loader = get_mnist_dataloaders(64)
# model = simple_AE(784, 64).to(device=device)
# model = load_saved_model('simple_AE_64.pkl', device=device, model = model)
# visualize_ae_reconstruction('simple_AE_64', model, test_loader, False)

# ## Conv AE
# train_loader, test_loader = get_mnist_dataloaders(64)
# model = conv_AE(in_channels=1).to(device=device)
# model = load_saved_model('conv_AE_64.pkl', device=device, model = model)
# visualize_ae_reconstruction('conv_AE_64', model, test_loader, False)

# ## Conv AE without deconv
# train_loader, test_loader = get_mnist_dataloaders(128)
# model = conv_AE_without_conv_transpose(in_channels=1, mode='bilinear').to(device=device)
# model = load_saved_model('conv_ae_64_without_deconv.pkl', device=device, model = model)
# visualize_ae_reconstruction('conv_ae_64_without_deconv', model, test_loader, False)

# ## VAE
# train_loader, test_loader = get_mnist_dataloaders(64)
# model = variational_AE(input_image_size = 28, in_channels = 1, latent_dimension = 2).to(device=device)
# model = load_saved_model('vae_64_only_MSE.pkl', device=device, model = model)
# visualize_ae_reconstruction('vae_64_only_MSE', model, test_loader, True)

train_loader, test_loader = get_mnist_dataloaders(64)
model = variational_AE(input_image_size = 28, in_channels = 1, latent_dimension = 2).to(device=device)
model = load_saved_model('vae_64_only_KLD.pkl', device=device, model = model)
visualize_ae_reconstruction('vae_64_only_KLD', model, test_loader, True)