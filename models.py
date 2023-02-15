from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate2DConvOutput(kernel_size : int, padding : int, stride : int, input_size: int):
    size = input_size + 2*padding - kernel_size
    size = size // stride
    size += 1
    return size

def calculate2DConvTransposeOutput(kernel_size : int, padding : int, stride : int, input_size: int):
    ## Hout = (Hin−1)×stride−2×padding+kernel_size
    size = (input_size - 1) * stride - 2*padding + kernel_size
    return size

class flatBatch(nn.Module):
    def forward(self, input : torch.Tensor):
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out       

class unflatBatch(nn.Module):
    def __init__(self, shape : List): #shape should be of the form [channels, size, size]
        super(unflatBatch, self).__init__()
        self.shape = shape
    
    def forward(self, input : torch.Tensor):  
        out = input.view(-1, self.shape[0], self.shape[1], self.shape[2])
        return out

class simple_AE(nn.Module):
    def __init__(self, input_shape, hidden_shape):
        super(simple_AE, self).__init__()
        self.input_shape = input_shape
        self.hidden_shape = hidden_shape

        self.l1 = nn.Linear(input_shape, hidden_shape)
        self.l2 = nn.Linear(hidden_shape, input_shape)

    def forward(self, x):
        req_shape = x.size()
        x = x.view(-1, self.input_shape)
        x = self.l1(x)
        x = self.l2(x)
        x = x.view(req_shape)

        return x

    def get_representatons(self, x):
        x = x.view(-1, self.input_shape)
        x = self.l1(x)
        return x

    def lossCalc(self, x, out):
        loss = nn.MSELoss()
        loss = loss(x, out)
        return loss

class conv_AE(nn.Module):
    def __init__(self, input_image_size : int, in_channels = 1):
        super(conv_AE, self).__init__()

        #encoder
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = 32,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = (1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = (1, 1),
        )
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = (1, 1),
        )

        self.size_after_conv = input_image_size
        for _ in range(3):
            self.size_after_conv = calculate2DConvOutput(3, 1, 2, self.size_after_conv)
        
        self.image_size_after_conv = [128, self.size_after_conv, self.size_after_conv]  #size after convolution except batch
        self.fc_in_size = 128 * self.size_after_conv * self.size_after_conv

        #code
        self.fc1 = nn.Linear(self.fc_in_size, self.fc_in_size//4)
        self.fc2 = nn.Linear(self.fc_in_size//4, self.fc_in_size//4//8)

        self.encode = [self.conv1, self.conv2, self.conv3, flatBatch(), self.fc1, self.fc2]
        self.activate_encoder = []

        for layer in self.encode:
            if(isinstance(layer, nn.Conv2d)):
                self.activate_encoder.append(nn.Sequential(layer, nn.LeakyReLU()))
            else:
                self.activate_encoder.append(layer)

        self.encoder = nn.Sequential(*self.activate_encoder)

        self.fc3 = nn.Linear(self.fc_in_size//4//8, self.fc_in_size//4)
        self.fc4 = nn.Linear(self.fc_in_size//4, self.fc_in_size)
        
        #decoder
        self.convTrans1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
        )
        
        self.convTrans2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
            output_padding=(1,1),
        )
        
        self.convTrans3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=in_channels,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
            output_padding=(1,1)
        )

        self.decode = [self.fc3, self.fc4, unflatBatch(shape = self.image_size_after_conv), self.convTrans1, self.convTrans2, self.convTrans3]
        self.activate_decoder = []

        for layer in self.decode:
            if(isinstance(layer, nn.ConvTranspose2d)):
                self.activate_decoder.append(nn.Sequential(layer, nn.LeakyReLU()))
            else:
                self.activate_decoder.append(layer)

        self.decoder = nn.Sequential(*self.activate_decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_representatons(self, x):
        return self.encoder(x)

    def lossCalc(self, x, out):
        loss = nn.MSELoss()
        loss = loss(x, out)
        return loss

class conv_AE_without_conv_transpose(nn.Module):
    def __init__(self, input_image_size : int, in_channels = 1, mode = 'nearest'):
        super(conv_AE_without_conv_transpose, self).__init__()
        
        #encoder
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = 32,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = (1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = (1, 1),
        )
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = (1, 1),
        )
        
        
        self.size_after_conv = input_image_size
        for _ in range(3):
            self.size_after_conv = calculate2DConvOutput(3, 1, 2, self.size_after_conv)
        
        self.image_size_after_conv = [128, self.size_after_conv, self.size_after_conv]  #size after convolution except batch
        self.fc_in_size = 128 * self.size_after_conv * self.size_after_conv

        #code
        self.fc1 = nn.Linear(self.fc_in_size, self.fc_in_size//4)
        self.fc2 = nn.Linear(self.fc_in_size//4, self.fc_in_size//4//8)

        self.encode = [self.conv1, self.conv2, self.conv3, flatBatch(), self.fc1, self.fc2]
        self.activate_encoder = []

        for layer in self.encode:
            if(isinstance(layer, nn.Conv2d)):
                self.activate_encoder.append(nn.Sequential(layer, nn.LeakyReLU()))
            else:
                self.activate_encoder.append(layer)

        self.encoder = nn.Sequential(*self.activate_encoder)

        self.fc3 = nn.Linear(self.fc_in_size//4//8, self.fc_in_size//4)
        self.fc4 = nn.Linear(self.fc_in_size//4, self.fc_in_size)
        
        #decoder

        #(batch_size, channels, height, width)
        self.upSample1 = nn.Upsample(size=(7,7), mode=mode)
        self.conv4 = nn.Conv2d(
            in_channels = 128,
            out_channels = 64,
            kernel_size = 1
        )
        self.upSample2 = nn.Upsample(size=(14,14), mode=mode)
        self.conv5 = nn.Conv2d(
            in_channels = 64,
            out_channels = 32,
            kernel_size = 1
        )
        self.upSample3 = nn.Upsample(size=(28,28), mode=mode)
        self.conv6 = nn.Conv2d(
            in_channels = 32,
            out_channels = in_channels,
            kernel_size = 1
        )

        self.decode = [self.fc3, self.fc4, unflatBatch(shape = self.image_size_after_conv), self.upSample1, self.conv4, self.upSample2, self.conv5, self.upSample3, self.conv6]
        self.activate_decoder = []

        for layer in self.decode:
            if(isinstance(layer, nn.ConvTranspose2d)):
                self.activate_decoder.append(nn.Sequential(layer, nn.LeakyReLU()))
            else:
                self.activate_decoder.append(layer)

        self.decoder = nn.Sequential(*self.activate_decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
    def get_representatons(self, x):
        return self.encoder(x)

    def lossCalc(self, x, out):
        loss = nn.MSELoss()
        loss = loss(x, out)
        return loss
class variational_AE(nn.Module):
    def __init__(self, input_image_size, in_channels = 1, latent_dimension = 4):
        super(variational_AE, self).__init__()

        #encoder
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = 32,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = (1, 1),
        )
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = (1, 1),
        )
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = (3, 3),
            stride = (2, 2),
            padding = (1, 1),
        )

        self.size_after_conv = input_image_size
        for _ in range(3):
            self.size_after_conv = calculate2DConvOutput(3, 1, 2, self.size_after_conv)
        
        self.image_size_after_conv = [128, self.size_after_conv, self.size_after_conv]  #size after convolution except batch
        self.fc_in_size = 128 * self.size_after_conv * self.size_after_conv

        #code
        self.fc1 = nn.Linear(self.fc_in_size, self.fc_in_size//4)
        self.fc2 = nn.Linear(self.fc_in_size//4, self.fc_in_size//4//8)

        self.encode = [self.conv1, self.conv2, self.conv3, flatBatch(), self.fc1, self.fc2]
        self.activate_encoder = []

        for layer in self.encode:
            if(isinstance(layer, nn.Conv2d)):
                self.activate_encoder.append(nn.Sequential(layer, nn.LeakyReLU()))
            else:
                self.activate_encoder.append(layer)

        self.encoder = nn.Sequential(*self.activate_encoder)

        self.fc_mu = nn.Linear(self.fc_in_size//4//8, latent_dimension)
        self.fc_var = nn.Linear(self.fc_in_size//4//8, latent_dimension)

        self.fc_decoder_input = nn.Linear(latent_dimension, self.fc_in_size//4//8)

        self.fc3 = nn.Linear(self.fc_in_size//4//8, self.fc_in_size//4)
        self.fc4 = nn.Linear(self.fc_in_size//4, self.fc_in_size)
        
        #decoder
        self.convTrans1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
        )
        
        self.convTrans2 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
            output_padding=(1,1),
        )
        
        self.convTrans3 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=in_channels,
            kernel_size=(3,3),
            stride=(2,2),
            padding=(1,1),
            output_padding=(1,1)
        )

        self.decode = [self.fc3, self.fc4, unflatBatch(shape = self.image_size_after_conv), self.convTrans1, self.convTrans2, self.convTrans3]
        self.activate_decoder = []

        for layer in self.decode:
            if(isinstance(layer, nn.ConvTranspose2d)):
                self.activate_decoder.append(nn.Sequential(layer, nn.LeakyReLU()))
            else:
                self.activate_decoder.append(layer)

        self.decoder = nn.Sequential(*self.activate_decoder)

    def reparametrize(self, mean, log_var):         # reparaterization trick to make vae backprop
        epsilon = torch.rand(log_var.size())        # log variance is used to make range real instead of positive real
        new_var = torch.exp(0.5*log_var) 
        epsilon = epsilon.to(device=device)
        return mean + epsilon * new_var             # mean + std = mean + exp(log(std)) = mean + exp(log(var**2))

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        z = self.reparametrize(mu, log_var)
        z = self.fc_decoder_input(z)

        z = self.decoder(z)
        return z, mu, log_var

    def get_representatons(self, x):
        return self.encoder(x)

    def get_distribution(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def lossCalc(self, x, out):
        loss = nn.MSELoss()
        loss = loss(x, out)
        return loss

    def KLDlossCalc(self, log_var, mu):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), dim = 1), dim = 0)    ## gfg
        return kld_loss