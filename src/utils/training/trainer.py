import os
import time
from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import numpy as np

from src.utils.visualization.plots import (
    plot_training_curve,
    plot_loss_components,
    plot_latent_space,
    visualize_reconstructions
)

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        save_dir: str,
        model_name: str,
        use_wandb: bool = True,
        project_name: str = "autoencoders",
        **kwargs
    ):
        """Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            test_loader: Test data loader
            optimizer: Optimizer for training
            device: Device to train on
            save_dir: Directory to save results
            model_name: Name of the model
            use_wandb: Whether to use Weights & Biases
            project_name: Name of the W&B project
            **kwargs: Additional arguments for W&B
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        self.use_wandb = use_wandb
        
        # Create save directories
        self.results_dir = os.path.join(save_dir, 'results', model_name)
        self.model_dir = os.path.join(save_dir, 'models', model_name)
        self.plot_dir = os.path.join(save_dir, 'plots', model_name)
        
        for dir_path in [self.results_dir, self.model_dir, self.plot_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize W&B
        if use_wandb:
            wandb.init(
                project=project_name,
                name=model_name,
                config={
                    "model_name": model_name,
                    "optimizer": optimizer.__class__.__name__,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "batch_size": train_loader.batch_size,
                    "device": device.type,
                    **kwargs
                }
            )
    
    def train_epoch(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.train()
        total_loss = 0
        metrics = {"reconstruction_loss": 0, "kld_loss": 0}
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (data, _) in enumerate(pbar):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self.model, 'reparameterize'):  # VAE
                    recon_batch, mu, logvar = self.model(data)
                    loss, recon_loss, kld_loss = self.model.loss_calc(
                        data, recon_batch, mu, logvar)
                    metrics["reconstruction_loss"] += recon_loss.item()
                    metrics["kld_loss"] += kld_loss.item()
                else:  # Standard autoencoder
                    recon_batch = self.model(data)
                    loss = self.model.loss_calc(data, recon_batch)
                    metrics["reconstruction_loss"] += loss.item()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
                })
                
                # Log to W&B
                if self.use_wandb:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch_reconstruction_loss": metrics["reconstruction_loss"]/(batch_idx+1),
                        "batch_kld_loss": metrics["kld_loss"]/(batch_idx+1) if "kld_loss" in metrics else 0
                    })
        
        # Calculate average metrics
        num_batches = len(self.train_loader)
        avg_loss = total_loss / num_batches
        for key in metrics:
            metrics[key] /= num_batches
        
        return avg_loss, metrics
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model.
        
        Returns:
            Tuple of (average loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0
        metrics = {"reconstruction_loss": 0, "kld_loss": 0}
        
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.to(self.device)
                
                if hasattr(self.model, 'reparameterize'):  # VAE
                    recon_batch, mu, logvar = self.model(data)
                    loss, recon_loss, kld_loss = self.model.loss_calc(
                        data, recon_batch, mu, logvar)
                    metrics["reconstruction_loss"] += recon_loss.item()
                    metrics["kld_loss"] += kld_loss.item()
                else:  # Standard autoencoder
                    recon_batch = self.model(data)
                    loss = self.model.loss_calc(data, recon_batch)
                    metrics["reconstruction_loss"] += loss.item()
                
                total_loss += loss.item()
        
        # Calculate average metrics
        num_batches = len(self.test_loader)
        avg_loss = total_loss / num_batches
        for key in metrics:
            metrics[key] /= num_batches
        
        return avg_loss, metrics
    
    def train(
        self,
        num_epochs: int,
        save_best: bool = True,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, list]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            save_best: Whether to save the best model
            early_stopping_patience: Number of epochs to wait for improvement
            
        Returns:
            Dictionary containing training history
        """
        start_time = time.time()
        best_loss = float('inf')
        patience_counter = 0
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_reconstruction_loss": [],
            "train_kld_loss": [],
            "val_reconstruction_loss": [],
            "val_kld_loss": []
        }
        
        for epoch in range(num_epochs):
            # Training
            train_loss, train_metrics = self.train_epoch(epoch)
            history["train_loss"].append(train_loss)
            history["train_reconstruction_loss"].append(train_metrics["reconstruction_loss"])
            if "kld_loss" in train_metrics:
                history["train_kld_loss"].append(train_metrics["kld_loss"])
            
            # Validation
            val_loss, val_metrics = self.validate()
            history["val_loss"].append(val_loss)
            history["val_reconstruction_loss"].append(val_metrics["reconstruction_loss"])
            if "kld_loss" in val_metrics:
                history["val_kld_loss"].append(val_metrics["kld_loss"])
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()}
                })
            
            # Save best model
            if save_best and val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                self.save_model(os.path.join(self.model_dir, 'best_model.pt'))
            else:
                patience_counter += 1
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        training_time = time.time() - start_time
        
        # Save final model
        self.save_model(os.path.join(self.model_dir, 'final_model.pt'))
        
        # Generate plots
        self._generate_training_plots(history, training_time)
        
        if self.use_wandb:
            wandb.finish()
        
        return history
    
    def save_model(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def _generate_training_plots(self, history: Dict[str, list], training_time: float) -> None:
        """Generate and save training plots."""
        # Loss curves
        plot_training_curve(
            history["train_loss"],
            self.plot_dir,
            self.model_name,
            training_time
        )
        
        # Loss components (for VAE)
        if "train_kld_loss" in history:
            plot_loss_components(
                history["train_reconstruction_loss"],
                history["train_kld_loss"],
                self.plot_dir,
                self.model_name
            )
        
        # Reconstructions
        visualize_reconstructions(
            self.model,
            self.test_loader,
            self.plot_dir,
            self.model_name,
            is_vae=hasattr(self.model, 'reparameterize')
        )
        
        # Latent space visualization
        if hasattr(self.model, 'get_representations'):
            representations, labels = self._get_latent_representations()
            for method in ['pca', 'tsne']:
                plot_latent_space(
                    representations,
                    labels,
                    self.plot_dir,
                    self.model_name,
                    method=method
                )
    
    def _get_latent_representations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get latent space representations and labels."""
        self.model.eval()
        representations = []
        labels = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.to(self.device)
                if hasattr(self.model, 'reparameterize'):  # VAE
                    mu, _ = self.model.encode(data)
                    representations.append(mu.cpu().numpy())
                else:
                    repr = self.model.get_representations(data)
                    representations.append(repr.cpu().numpy())
                labels.append(target.numpy())
        
        return np.concatenate(representations), np.concatenate(labels) 