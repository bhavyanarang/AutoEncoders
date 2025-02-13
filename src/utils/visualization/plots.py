import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Tuple, Optional
import torch
from torch.utils.data import DataLoader
from matplotlib.axes._axes import _log as matplotlib_axes_logger
import time

matplotlib_axes_logger.setLevel('ERROR')

def visualize_reconstructions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    save_path: str,
    model_name: str,
    num_images: int = 5,
    is_vae: bool = False
) -> None:
    """Visualize original images and their reconstructions.
    
    Args:
        model: The trained autoencoder model
        dataloader: DataLoader containing the images
        save_path: Path to save the visualization
        model_name: Name of the model for the plot title
        num_images: Number of images to visualize
        is_vae: Whether the model is a VAE
    """
    device = next(model.parameters()).device
    
    # Get first batch
    images, _ = next(iter(dataloader))
    images = images.to(device)
    
    # Get reconstructions
    with torch.no_grad():
        if is_vae:
            reconstructions, _, _ = model(images)
        else:
            reconstructions = model(images)
    
    # Create plot
    fig, axarr = plt.subplots(2, num_images, figsize=(20, 8))
    fig.suptitle(f'Original Images and their reconstruction using {model_name}')
    
    for i in range(num_images):
        # Original images
        axarr[0, i].imshow(images[i].cpu().squeeze(), cmap='gray')
        axarr[0, i].axis('off')
        if i == 0:
            axarr[0, i].set_title('Original')
        
        # Reconstructions
        axarr[1, i].imshow(reconstructions[i].cpu().squeeze(), cmap='gray')
        axarr[1, i].axis('off')
        if i == 0:
            axarr[1, i].set_title('Reconstructed')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/{model_name}_reconstruction.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curve(
    losses: List[float],
    save_path: str,
    model_name: str,
    training_time: float
) -> None:
    """Plot training loss curve.
    
    Args:
        losses: List of training losses
        save_path: Path to save the plot
        model_name: Name of the model
        training_time: Total training time in seconds
    """
    plt.figure(figsize=(12, 8))
    plt.plot(losses, label='Training Loss')
    plt.title(f"Training Loss - {model_name}\nTraining Time: {training_time:.2f}s")
    plt.xlabel("Batch number")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}/{model_name}_training_loss.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_latent_space(
    representations: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    model_name: str,
    method: str = 'pca',
    n_classes: int = 10
) -> None:
    """Plot latent space representations using dimensionality reduction.
    
    Args:
        representations: Latent space representations
        labels: Corresponding labels
        save_path: Path to save the plot
        model_name: Name of the model
        method: Dimensionality reduction method ('pca' or 'tsne')
        n_classes: Number of classes in the dataset
    """
    start_time = time.time()
    
    # Choose dimensionality reduction method
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    # Reduce dimensionality
    reduced_repr = reducer.fit_transform(representations)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(reduced_repr[:, 0], reduced_repr[:, 1],
                         c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    
    plt.title(f"Latent Space ({method.upper()}) - {model_name}\n"
              f"Time: {time.time() - start_time:.2f}s")
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    
    plt.savefig(f"{save_path}/{model_name}_{method.lower()}.png",
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_loss_components(
    reconstruction_losses: List[float],
    kld_losses: Optional[List[float]] = None,
    save_path: str = None,
    model_name: str = None
) -> None:
    """Plot components of the loss function (for VAE).
    
    Args:
        reconstruction_losses: List of reconstruction losses
        kld_losses: List of KL divergence losses (for VAE)
        save_path: Path to save the plot
        model_name: Name of the model
    """
    plt.figure(figsize=(12, 8))
    plt.plot(reconstruction_losses, label='Reconstruction Loss')
    if kld_losses is not None:
        plt.plot(kld_losses, label='KL Divergence')
    
    plt.title(f"Loss Components - {model_name}")
    plt.xlabel("Batch number")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    if save_path and model_name:
        plt.savefig(f"{save_path}/{model_name}_loss_components.png",
                   dpi=300, bbox_inches='tight')
    plt.close() 