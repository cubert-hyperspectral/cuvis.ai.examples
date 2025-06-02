import torch
from torch import nn

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
