import os
import sys
import argparse
import yaml
from skorch import NeuralNetRegressor
from PerPixelAECuvisDataSet import PerPixelAECuvisDataSet
from PerPixelAEModels import HybridLoss, CosSpectralAngleLoss, Autoencoder, AutoencoderSmall
from skorch.callbacks import EarlyStopping, Checkpoint, ProgressBar
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from skorch.callbacks import TensorBoard


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", '--config', type=str, required=True)
    args = parser.parse_args()
    return args


def parse_args(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    return config

# Initialize the model and the Skorch wrapper
def create_skorch_model(
        device: str='cuda',
        encoding_dim: int=1,
        wave: int=6,
        large: bool=True,
        loss: str='MSE',
        use_tensorboard: bool=True,
        max_epochs: int=100,
        lr: float=0.01,
        batch_size: int=7,
    )-> NeuralNetRegressor:
    # Create early stopping patience
    early_stopping = EarlyStopping(monitor='valid_loss', patience = 10, threshold = 0.01, threshold_mode='rel', lower_is_better=True)
    if large:
        autoencoder = Autoencoder(encoding_dim, wave)
    else:
        autoencoder = AutoencoderSmall(encoding_dim, wave)
    # Choose loss function
    loss_lookup = {
        'MSE': nn.MSELoss,
        'SAM': CosSpectralAngleLoss,
        'Hybrid': HybridLoss
    }
    try:
        loss_fnc = loss_lookup[loss]
    except KeyError as e:
        print('Invalid loss function!')
        sys.exit(1)
    print(f'Using {loss} as loss function!')
    # Add a monitor to save the best performance
    monitor = lambda net: all(net.history[-1, ('train_loss_best', 'valid_loss_best')])
    checkpoint = Checkpoint(monitor=monitor, f_params="./runs/params_{last_epoch[epoch]}.pt")
    # Create initial callbacks
    callbacks = [ProgressBar(),early_stopping, checkpoint]
    # Enable us to view model performance through the TensorBoard GUI
    if use_tensorboard:
        writer = SummaryWriter()
        callbacks.append(TensorBoard(writer))
    skorch_model = NeuralNetRegressor(
        autoencoder,
        max_epochs=max_epochs,    # adjust as needed
        lr=lr,
        optimizer = torch.optim.Adam,
        criterion = loss_fnc, # Chose loss function class here
        callbacks = callbacks,
        batch_size = batch_size, # This will need to vary based on the computer utilized
        iterator_train__shuffle=True,
        device=device
    )
    return skorch_model

def load_config(config_file: str) -> dict:
    '''
    Load YAML configuration file
    '''
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config

def train(config: dict) -> None:
    # Set us to use the GPU when appropriate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} for pytorch models...')
    config = load_config('./example_train_config.yaml')
    print(f'Using: {os.path.realpath(config["Datasets"]["train"]["root"])}')
    hsi_data = PerPixelAECuvisDataSet(os.path.realpath(config["Datasets"]["train"]["root"]),
                                        mode="train",
                                        mean=config["mean"],
                                        std=config["std"],
                                        normalize=config["normalize"],
                                        max_img_shape=config["max_img_shape"],
                                        white_percentage=config["white_percentage"],
                                        channels=config["channels"])
    
    autoencoder_large_hybrid = create_skorch_model(
                    device=device,
                    encoding_dim=config["Model"]["encoding_dim"],
                    wave=config["Model"]["in_channels"],
                    large=config["Model"]["use_large"],
                    loss=config["Model"]["loss"],
                    use_tensorboard=config["use_tensorboard"],
                    max_epochs=config["max_epochs"],
                    lr=config["learning_rate"],
                    batch_size=config["batch_size"]
                    )
    autoencoder_large_hybrid.fit(hsi_data, None)
    autoencoder_large_hybrid.save_params(f_params='./runs/final_ae.wts', f_optimizer='./runs/final_optimizer.pt', f_criterion='./runs/final_criterion.pt')
    autoencoder_large_hybrid2 = create_skorch_model(
                    device=device,
                    encoding_dim=config["Model"]["encoding_dim"],
                    wave=config["Model"]["in_channels"],
                    large=config["Model"]["use_large"],
                    loss=config["Model"]["loss"],
                    use_tensorboard=config["use_tensorboard"],
                    max_epochs=config["max_epochs"],
                    lr=config["learning_rate"],
                    batch_size=config["batch_size"]
                    )
    autoencoder_large_hybrid2.initialize()
    autoencoder_large_hybrid2.load_params(f_params='./runs/final_ae.wts')

if __name__ == '__main__':
    args = get_arguments()
    config = parse_args(args)
    train(config)