# AutoEncoders

A collection of autoencoder implementations in PyTorch, including:
- Simple Autoencoder
- Convolutional Autoencoder
- Variational Autoencoder (VAE)

## Project Structure

```
AutoEncoders/
├── src/
│   ├── models/
│   │   ├── simple_ae.py     # Simple autoencoder implementation
│   │   ├── conv_ae.py       # Convolutional autoencoder implementation
│   │   └── vae.py          # Variational autoencoder implementation
│   ├── utils/
│   │   ├── visualization/  # Visualization utilities
│   │   │   └── plots.py   # Plotting functions
│   │   └── training/      # Training utilities
│   │       └── trainer.py # Training class
│   └── train.py           # Main training script with CLI options
├── data/                  # Dataset storage (created automatically)
├── outputs/              # Training outputs
│   ├── models/          # Saved model checkpoints
│   ├── plots/           # Generated plots
│   └── results/         # Training metrics
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Features

- Support for multiple autoencoder architectures
- Training on various datasets (MNIST, Fashion-MNIST, CIFAR-10)
- Automatic device selection (CUDA, MPS, CPU)
- Weights & Biases integration for experiment tracking
- Comprehensive visualizations:
  - Training loss curves
  - Loss component analysis for VAE
  - Reconstruction quality
  - Latent space visualization (PCA and t-SNE)
- Early stopping
- Model checkpointing
- Multiple optimizer options
- Flexible training configurations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AutoEncoders.git
cd AutoEncoders
```

2. Create a virtual environment:
```bash
python -m venv autoencoder_env
source autoencoder_env/bin/activate  # On Windows: autoencoder_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Commands

### Basic Usage

Train a simple autoencoder on MNIST:
```bash
python src/train.py --model simple --dataset mnist --epochs 10
```

Train a convolutional autoencoder:
```bash
python src/train.py --model conv --dataset mnist --epochs 20
```

Train a VAE with custom latent dimension and beta value:
```bash
python src/train.py --model vae --dataset mnist --epochs 30 --latent-dim 32 --beta 0.5
```

### Advanced Configurations

1. Training VAE on Fashion-MNIST with W&B monitoring:
```bash
python src/train.py \
    --model vae \
    --dataset fashion_mnist \
    --latent-dim 64 \
    --beta 0.8 \
    --learning-rate 0.0002 \
    --batch-size 128 \
    --epochs 50 \
    --optimizer adam \
    --weight-decay 1e-5 \
    --use-wandb \
    --project-name "fashion_mnist_vae"
```

2. Training Convolutional AE on CIFAR-10:
```bash
python src/train.py \
    --model conv \
    --dataset cifar10 \
    --image-size 32 \
    --learning-rate 0.0003 \
    --batch-size 64 \
    --epochs 100 \
    --optimizer rmsprop \
    --early-stopping 10
```

3. Quick Experiment with Simple AE:
```bash
python src/train.py \
    --model simple \
    --dataset mnist \
    --hidden-dim 128 \
    --learning-rate 0.001 \
    --batch-size 256 \
    --epochs 5
```

### Device-Specific Training

For CUDA (NVIDIA GPUs):
```bash
python src/train.py --model vae --device cuda
```

For MPS (Apple Silicon):
```bash
python src/train.py --model vae --device mps
```

For CPU:
```bash
python src/train.py --model vae --device cpu
```

## Command-line Arguments

### Model Parameters
- `--model`: Model type (simple, conv, vae)
- `--hidden-dim`: Hidden dimension for simple autoencoder
- `--latent-dim`: Latent space dimension for VAE
- `--beta`: Beta parameter for VAE loss weighting

### Dataset Parameters
- `--dataset`: Dataset to use (mnist, fashion_mnist, cifar10)
- `--data-dir`: Directory to store datasets
- `--image-size`: Size to resize images to

### Training Parameters
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate
- `--optimizer`: Optimizer type (adam, sgd, rmsprop)
- `--weight-decay`: L2 regularization factor

### Device Parameters
- `--device`: Device to train on (auto, cuda, mps, cpu)
- `--num-workers`: Number of data loading workers

### Monitoring Parameters
- `--use-wandb`: Enable Weights & Biases monitoring
- `--project-name`: W&B project name
- `--save-dir`: Directory to save outputs
- `--model-name`: Custom name for the model
- `--early-stopping`: Patience for early stopping

## Results

Training results are organized in the `outputs` directory:
- Model checkpoints in `outputs/models/<model_name>/`
- Plots in `outputs/plots/<model_name>/`
- Training metrics in `outputs/results/<model_name>/`

Each training run includes:
- Model checkpoints (best and final)
- Loss curves
- Reconstruction visualizations
- Latent space visualizations (if applicable)
- Training metrics

## Contributing

Feel free to open issues or submit pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
