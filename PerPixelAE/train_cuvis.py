import gc
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
from tqdm.notebook import tqdm
import argparse
import yaml
from skorch import NeuralNetRegressor
from PerPixelAECuvisDataSet import PerPixelAECuvisDataSet
from skorch.callbacks import EarlyStopping, Checkpoint, ProgressBar
import torch
import pickle
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from skorch.callbacks import TensorBoard

# Define the PyTorch model
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim: int, wave: int):
        super(Autoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(wave, 150),
            nn.ReLU(),
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 75),
            nn.ReLU(),
            nn.Linear(75, 50),
            nn.ReLU(),
            nn.Linear(50, encoding_dim)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 50),
            nn.ReLU(),
            nn.Linear(50, 75),
            nn.ReLU(),
            nn.Linear(75, 100),
            nn.ReLU(),
            nn.Linear(100, wave),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoencoderSmall(nn.Module):
    def __init__(self, encoding_dim: int, wave: int):
        super(AutoencoderSmall, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(wave, 5),
            nn.ReLU(),
            nn.Linear(5, 3),
            nn.ReLU(),
            nn.Linear(3, encoding_dim)
        )
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 3),
            nn.ReLU(),
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, wave),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Create early stopping patience
early_stopping = EarlyStopping(monitor='valid_loss', patience = 10, threshold = 0.01, threshold_mode='rel', lower_is_better=True)


class CosSpectralAngleLoss(nn.Module):
    def __init__(self):
        super(CosSpectralAngleLoss, self).__init__()
    def forward(self, y, y_reconstructed):
        # Normalize y and y_reconstructed along the feature dimension
        # This should normalize along the length of the spectral vector
        epsilon = 1e-8
        normalize_r = torch.sqrt(torch.sum(y_reconstructed**2, dim=1)) + epsilon
        # This should also normalize along the length of the spectral vector
        normalize_t = torch.sqrt(torch.sum(y**2, dim=1))
        # Compute cosine similarity between the normalized vectors
        numerator = torch.sum((y * y_reconstructed), dim=1)
        denominator = normalize_r * normalize_t
        cosine_similarity = numerator / denominator
        # Compute the spectral angle in radians
        spectral_angle = torch.acos(cosine_similarity)
        spectral_angle = torch.nan_to_num(cosine_similarity, 1)
        # # When a function perfectly matches, the value is 0
        # # Torch acos is define over [-1,1]
        # # To make this an appropriate loss value, we need to invert the spread and then add 1
        spectral_angle = (-1 * spectral_angle) + 1
        # Average the spectral angle to get the final loss
        loss = torch.mean(spectral_angle)
        return loss

class HybridLoss(nn.Module):
    def __init__(self):
        super(HybridLoss, self).__init__()
        self.MSELoss = nn.MSELoss()
        # Controls the weighting of the importance between MSE and SAM
        self.alpha = 0.1

    def forward(self, y, y_reconstructed):
        return self.forward_mse(y,y_reconstructed) + self.alpha * self.forward_sam(y, y_reconstructed)

    def forward_sam(self, y, y_reconstructed):
        # Normalize y and y_reconstructed along the feature dimension
        # This should normalize along the length of the spectral vector
        epsilon = 1e-8
        normalize_r = torch.sqrt(torch.sum(y_reconstructed**2, dim=1)) + epsilon
        # This should also normalize along the length of the spectral vector
        normalize_t = torch.sqrt(torch.sum(y**2, dim=1))
        # Compute cosine similarity between the normalized vectors
        # This is computer
        numerator = torch.sum((y * y_reconstructed), dim=1)
        denominator = normalize_r * normalize_t
        cosine_similarity = numerator / denominator
        # Compute the spectral angle in radians
        spectral_angle = torch.acos(cosine_similarity)
        spectral_angle = torch.nan_to_num(cosine_similarity, 1)
        # When a function perfectly matches, the value is 0
        # Torch acos is define over [-1,1]
        # To make this an appropriate loss value, we need to invert the spread and then add 1
        spectral_angle = (-1 * spectral_angle) + 1
        # Average the spectral angle to get the final loss
        loss = torch.mean(spectral_angle)
        return loss
    
    def forward_mse(self, y, y_reconstructed):
        return self.MSELoss.forward(y, y_reconstructed)

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
    print(f'Using {loss} as loss!')
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


# Predict some sample images here
def visualize_model_performance(autoencoder: NeuralNetRegressor, cube_data: dict, threshold: float=0.15, title='', factor=1):
    # Make a single figure
    samples = len(cube_data.keys())
    i = 0
    fig, ax = plt.subplots(samples, 2, figsize=(15, 5*samples))
    for name, cube in tqdm(cube_data.items()):
        # Grab the correct pixels
        input_pixels = cube['data'][::10,::10,:].reshape(-1, cube['data'][::10,::10,:].shape[-1]).astype(np.float32) / factor
        start_time = time.time_ns()
        predicted_pixels = autoencoder.predict(input_pixels)
        stop_time = time.time_ns()
        print(f'Predicted reconstruction error in {(stop_time-start_time)/10**9} seconds')
        predicted_cube = predicted_pixels.reshape(cube['data'][::10,::10,:].shape)
        # Calculate L2 error and visualize
        # Plot RGB, ground truth, and predicted labels side by side
        error_img = np.linalg.norm(predicted_cube-(cube['data'][::10,::10,:] / factor), axis=2)
        im = ax[i, 0].imshow(error_img)
        ax[i, 0].set_title(f'Reconstruction Loss for Scene: {name}')
        fig.colorbar(im, label='Reconstruction Error')
        image = cv2.cvtColor((cube['data']/factor)[:,:,:3].astype(np.float32), cv2.COLOR_BGR2RGB)
        ax[i, 1].imshow(image)
        ax[i, 1].set_title('RGB Representation')
        # Increment the row
        i+= 1
    plt.suptitle(f'Autoencoder Performance for {title}', y=1.01)
    plt.tight_layout()
    plt.show()

def save_model(model: NeuralNetRegressor, file_name='.'):
    '''
    Save model to file
    '''
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

def load_model(model: NeuralNetRegressor, file_name='.') -> NeuralNetRegressor:
    '''
    Load autoencoder and return object
    '''
    with open(file_name, 'rb') as f:
        model  = pickle.load(f)
        return model
    
def load_config(config_file):
    '''
    Load YAML configuration file
    '''
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config

def main():
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
    save_model(autoencoder_large_hybrid, './final_per_pixel_ae.pkl')

if __name__ == '__main__':
    main()