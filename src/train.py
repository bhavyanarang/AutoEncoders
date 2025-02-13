import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from datetime import datetime

from src.models.simple_ae import SimpleAutoencoder
from src.models.conv_ae import ConvolutionalAutoencoder
from src.models.vae import VariationalAutoencoder
from src.utils.training.trainer import Trainer

def get_device(device_name: str = None) -> torch.device:
    """Get the appropriate device based on availability and user preference"""
    if device_name is None or device_name.lower() == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_name)

def get_dataset(dataset_name: str, data_dir: str = './data', image_size: int = 28) -> tuple:
    """Get dataset based on name"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    if dataset_name.lower() == 'mnist':
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'fashion_mnist':
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    elif dataset_name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Grayscale() if image_size == 28 else transforms.Lambda(lambda x: x)
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
        
    return train_dataset, test_dataset

def get_model(model_name: str, input_size: int, **kwargs):
    """Get model based on name and parameters"""
    if model_name.lower() == 'simple':
        return SimpleAutoencoder(input_size * input_size, kwargs.get('hidden_dim', 64))
    elif model_name.lower() == 'conv':
        return ConvolutionalAutoencoder(input_size, kwargs.get('in_channels', 1))
    elif model_name.lower() == 'vae':
        return VariationalAutoencoder(
            input_size,
            in_channels=kwargs.get('in_channels', 1),
            latent_dim=kwargs.get('latent_dim', 64),
            beta=kwargs.get('beta', 1.0)
        )
    else:
        raise ValueError(f"Model {model_name} not supported")

def main():
    parser = argparse.ArgumentParser(description='Train various autoencoder models')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'conv', 'vae'],
                      help='Model type to train (simple, conv, or vae)')
    parser.add_argument('--hidden-dim', type=int, default=64,
                      help='Dimension of hidden layer for simple autoencoder')
    parser.add_argument('--latent-dim', type=int, default=64,
                      help='Dimension of latent space for VAE')
    parser.add_argument('--beta', type=float, default=1.0,
                      help='Beta parameter for VAE (default: 1.0)')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='mnist',
                      choices=['mnist', 'fashion_mnist', 'cifar10'],
                      help='Dataset to train on')
    parser.add_argument('--data-dir', type=str, default='./data',
                      help='Directory to store datasets')
    parser.add_argument('--image-size', type=int, default=28,
                      help='Size to resize images to')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=128,
                      help='Input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs to train (default: 10)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                      help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam',
                      choices=['adam', 'sgd', 'rmsprop'],
                      help='Optimizer to use for training')
    parser.add_argument('--weight-decay', type=float, default=0,
                      help='Weight decay (L2 penalty)')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='auto',
                      help='Device to train on (auto, cuda, mps, or cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Monitoring and saving parameters
    parser.add_argument('--use-wandb', action='store_true',
                      help='Use Weights & Biases for monitoring')
    parser.add_argument('--project-name', type=str, default='autoencoders',
                      help='Project name for W&B')
    parser.add_argument('--save-dir', type=str, default='./outputs',
                      help='Directory to save results')
    parser.add_argument('--model-name', type=str, default=None,
                      help='Custom name for the model')
    parser.add_argument('--early-stopping', type=int, default=None,
                      help='Number of epochs to wait before early stopping')
    
    args = parser.parse_args()
    
    # Set up device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Get dataset
    train_dataset, test_dataset = get_dataset(
        args.dataset,
        args.data_dir,
        args.image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Get model
    in_channels = 3 if args.dataset.lower() == 'cifar10' and args.image_size != 28 else 1
    model = get_model(
        args.model,
        args.image_size,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        beta=args.beta,
        in_channels=in_channels
    ).to(device)
    
    # Set up optimizer
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:  # rmsprop
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Set up model name
    if args.model_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.model_name = f"{args.model}_{args.dataset}_{timestamp}"
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        model_name=args.model_name,
        use_wandb=args.use_wandb,
        project_name=args.project_name,
        dataset=args.dataset,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay
    )
    
    # Train model
    history = trainer.train(
        num_epochs=args.epochs,
        save_best=True,
        early_stopping_patience=args.early_stopping
    )
    
    print(f"Training completed. Results saved in {args.save_dir}/{args.model_name}")

if __name__ == '__main__':
    main() 