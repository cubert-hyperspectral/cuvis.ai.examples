import cuvis
import cv2
import pickle
from sklearnex import patch_sklearn
from skorch.callbacks import LRScheduler, EarlyStopping
from skorch import NeuralNetRegressor
import torch
import numpy as np
from torch import nn
# This deploys a CPU path that improves the runtime of sklearn models
patch_sklearn()
# Set us to use the GPU when appropriate
device = 'cuda' if torch.cuda.is_available() else 'cpu'

early_stopping = EarlyStopping(monitor='valid_loss', patience = 20, threshold = 0.001, threshold_mode='rel', lower_is_better=True)


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
    

# Initialize the model and the Skorch wrapper
def create_skorch_model(encoding_dim: int, wave: int, large=True, loss='MSE') -> NeuralNetRegressor:
    autoencoder = Autoencoder(encoding_dim, wave)
    # Choose loss function
    skorch_model = NeuralNetRegressor(
        autoencoder,
        max_epochs=50,    # adjust as needed
        lr=0.01,
        optimizer = torch.optim.Adam,
        criterion = HybridLoss, # Chose loss function class here
        callbacks = [early_stopping],
        batch_size = 800, # This will need to vary based on the computer utilized
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return skorch_model

def load_model(model: NeuralNetRegressor, file_name='.') -> NeuralNetRegressor:
    '''
    Load autoencoder and return object
    '''
    with open(file_name, 'rb') as f:
        model  = pickle.load(f)
        return model

# Predict some sample images here
def predict_anomalies(autoencoder: NeuralNetRegressor, cube: np.ndarray, threshold: float=0.15):
        # Grab the correct pixels
        input_pixels = cube.reshape(-1, cube.shape[-1]).astype(np.float32)
        predicted_pixels = autoencoder.predict(input_pixels)
        predicted_cube = predicted_pixels.reshape(cube.shape)
        # Calculate L2 error and visualize
        error_img = np.linalg.norm(predicted_cube-cube, axis=2)
        return error_img


# Create the skorch model
autoencoder_large_hybrid = create_skorch_model(3, 6, large=True, loss='Hybrid')
autoencoder_large_hybrid = load_model(autoencoder_large_hybrid, '/home/nathaniel/cubert/internal_rnd_applications/yogurt/yogurt_models/autoencoder_large_hybrid_12_21_25.pkl')

# LOAD CUBE
cube = np.zeros((100,100,6)) # Dummy values
predict_anomalies(autoencoder_large_hybrid, cube)