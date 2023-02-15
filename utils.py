import os
import copy
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#gpu path
folder_path = '/home/azureuser/sat_feature_extraction/'
#local path
folder_path = 'C:/Users/Bhavya/Documents/GitHub/sat_feature_extraction/'
#generalized_path
folder_path = os.getcwd() + '/'

loss_path = folder_path + 'LossPlots/'
cluster_path = folder_path + 'ClusterPlots/'
model_path = folder_path + 'ModelStore/'
reconstruction_path = folder_path + 'Reconstruction/'


def get_mnist_dataloaders(batch_size: int):
    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, test_loader

def applyAE(isVae : bool, model_name : str, model : nn.Module, num_epochs : int, train_loader : DataLoader, test_loader, device : str):
    """ function to apply AE and get plots 
    """
    learning_rate = 1e-3
    ## AE
    model = model.to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #train AE
    trainAutoEncoder(isVae, model, optimizer, device, train_loader, epochs = num_epochs, loss_plot_path = loss_path, model_name = model_name, model_save_path = model_path)

    #Representations
    ae_representations, ae_targets = getAutoEncoderRepresentations(model, test_loader, device, isVae)

    #cluster plots
    getCluster(ae_representations, ae_targets, 10, path = cluster_path, model_name = model_name, type = 'PCA')
    getCluster(ae_representations, ae_targets, 10, path = cluster_path, model_name = model_name, type = 'TSNE')

    print('Done with '+str(model_name))


def visualize_ae_reconstruction(model_name : str, model : nn.Module, dataloader: DataLoader, isVae : bool):
    """ function to visualize the reconstructions obtained by various autoencoder models
    """ 
    images = 0
    for _, (data, _) in enumerate(dataloader):

        data = data.to(device=device)
        images = data
        break

    num_images = 5
    f, axarr = plt.subplots(2,num_images) 
    detached_images = []
    output_images = []
    f.set_figheight(20)
    f.set_figwidth(20)

    if(isVae):
        gen_images,_ , _ = model(images)
    else:
        gen_images = model(images)

    for i in range(num_images):
        detached_images.append(images[i].cpu().detach().numpy())
        output_images.append(gen_images[i].cpu().detach().numpy())

    f.suptitle('Original Images and their reconstruction using '+str(model_name))

    for i in range(5):
        axarr[0,i].imshow(detached_images[i][0])       ##only for grayscale images, change in case of color images
        axarr[1,i].imshow(output_images[i][0])

    plt.savefig(reconstruction_path + model_name + '_reconstruction.png')

    print('Reconstruction using '+str(model_name) + ' saved. ')

def load_saved_model(model_name : str, device : str, model: torch.nn):
    model_name = model_path + model_name
    model.load_state_dict(torch.load(model_name))
    model = model.to(device)
    model.eval()
    return model

def getAutoEncoderRepresentations(model : nn.Module, data_loader : DataLoader, device : str, isVae:bool):
    ae_targets = []
    ae_representations = []

    for batch_idx, (data, targets) in enumerate(tqdm(data_loader,desc='Getting representations') ):
        data = data.to(device=device)
        
        if(isVae):
            gen_image,_ , _ = model(data)
        else:
            gen_image = model(data)
        
        gen_image = gen_image.view(gen_image.size(0), -1)
        ae_representations += list(gen_image.cpu().detach().numpy())
        ae_targets += list(targets.cpu().detach().numpy())

    return np.array(ae_representations), np.array(ae_targets)

def trainAutoEncoder(isVae : bool, model : nn.Module, optimizer: optim, device: str, train_loader : DataLoader, epochs : int, loss_plot_path = None, model_name = None, model_save_path = None):
    """ function to train an autoencoder model
    loss_plot_path :(string) path where we have to save the loss plots
    model_save_path :(string) path where we have to save the model params
    model_name : (string) to save the files with model name
    """

    print("Training model : " + str(model_name))
    start_time  = time.time()
    loss_vs_batches = []
    for epoch in range(epochs):
        print("Epoch "+str(epoch+1))
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            req_shape = data.size()

            # forward
        
            if(isVae):
                gen_image, mu, log_var = model(data)
                gen_image = gen_image.view(req_shape)
                loss = model.lossCalc(data, gen_image)
                beta = 10
                loss += beta*model.KLDlossCalc(log_var, mu)

            else:
                gen_image = model(data)
                gen_image = gen_image.view(req_shape)
                loss = model.lossCalc(data, gen_image)
            
            # backward
            optimizer.zero_grad()
            loss.backward()
            loss_vs_batches.append(loss.cpu().detach().numpy())
            
            # gradient descent or adam step
            optimizer.step()

    end_time = time.time()

    if(loss_plot_path):
        plt.figure(figsize=(12, 10))
        plt.plot(loss_vs_batches)
        plt.title("Train loss of " + model_name +' Training Time: ' + str(end_time - start_time) + 's')
        plt.xlabel("Batch number")
        plt.ylabel("Loss")
        plt.savefig(loss_plot_path + model_name + 'training_loss.png')

    if(model_save_path):
        torch.save(model.state_dict(), model_save_path + model_name + '.pkl')


def getCluster(representations : np.array, targets: np.array, n_classes : int, path : str = None, model_name : str = None, type = 'PCA'):
    """ get coloured cluster representations from after obtaining AE represenations and targets onto 2d axis
    n_classes : total number of classes in targets
    len(representations) should be equal to len(targets)
    type supported : PCA / TSNE
    """
    start_time  = time.time()
    algorithm = PCA(n_components=2)
    color_list = plt.cm.get_cmap('hsv', n_classes)

    if(type == 'TSNE'):
        algorithm = TSNE(n_components=2)

    representations_transformed = algorithm.fit_transform(representations)
    representations_transformed_x = representations_transformed[:,0]
    representations_transformed_y = representations_transformed[:,1]
    
    end_time = time.time()
    
    if(path):
        plt.figure(figsize=(12, 10))
        plt.title("Cluster using : " + type +', model : ' + model_name + ", time taken: " + str(end_time - start_time) + 's')
        plt.scatter(representations_transformed_x, representations_transformed_y, c = color_list(targets))
        plt.savefig(path + model_name + '_' + type + '.png')
        
        print(type + " Plot saved")
