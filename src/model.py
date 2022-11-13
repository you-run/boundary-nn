import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class Interpolate(nn.Module):
    def __init__(self, size, mode='nearest'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1, pool_size=2
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool(out)
        return out


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=3, original_size=(270, 480)):
        super().__init__()
        # encoder
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            ConvBlock(3, 16, 7, 1, 1, 2),
            ConvBlock(16, 32, 3, 1, 1, 2),
            ConvBlock(32, 64, 3, 1, 1, 2),
            ConvBlock(64, 64, 3, 1, 1, 2),
            nn.Conv2d(64, 2, 1, 1, 0)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(2, 64, 1, 1, 0),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=2),
            nn.Sigmoid()
        )
        self.decoder_resize = T.Resize(size=original_size)

    def forward(self, x): # x: (B, 3, 270, 480)
        latent = self.encoder(x)
        decoder_out = self.decoder(latent)
        decoder_out = self.decoder_resize(decoder_out)
        return latent, decoder_out
    
    def encoding(self, x): # Not Train
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

class ConvAutoencoderV2(nn.Module):
    def __init__(self, latent_dim=3, original_size=(270, 480)):
        super().__init__()
        # Encoder
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            ConvBlock(3, 8, 7, 1, 1, 2),
            ConvBlock(8, 16, 3, 1, 1, 2),
            ConvBlock(16, 32, 3, 1, 1, 2),
            ConvBlock(32, 64, 3, 1, 1, 2),
            nn.Conv2d(64, 3, 1, 1, 0)
        )

        # Decodedr
        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0),
            Interpolate(size=(33, 59)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0),
            Interpolate(size=(67, 120)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0),
            Interpolate(size=(135, 240)),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=2, stride=2, padding=0),
            nn.Sigmoid()
        )        

    def forward(self, x): # x: (B, 3, 270, 480)
        latent = self.encoder(x)
        decoder_out = self.decoder(latent)
        return latent, decoder_out
    
    def encoding(self, x): # Not Train
        self.eval()
        with torch.no_grad():
            return self.encoder(x)


class VariationalConvAutoencoder(nn.Module):
    def __init__(self, feature_dim=2 * 16 * 29, z_dim=256):
        super().__init__()
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encoder = nn.Sequential(
            ConvBlock(3, 16, 7, 1, 1, 2),
            ConvBlock(16, 32, 3, 1, 1, 2),
            ConvBlock(32, 64, 3, 1, 1, 2),
            ConvBlock(64, 64, 3, 1, 1, 2),
            nn.Conv2d(64, 2, 1, 1, 0) 
        )
        self.flatten = nn.Flatten()
        
        self.enc_fc_mu = nn.Linear(feature_dim, z_dim)
        self.enc_fc_var = nn.Linear(feature_dim, z_dim)

        # Inxitializing the fully-connected layer and 2 convolutional layers for decoder
        self.dec_fc = nn.Linear(z_dim, feature_dim)
        self.unflatten = nn.Unflatten(1, (2, 16, 29))
        self.decoder = nn.Sequential(
            nn.Conv2d(2, 64, 1, 1, 0),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 8, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=2),
            nn.Sigmoid()
        )
        self.resize = T.Resize(size=(270,480))

    def encoding(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = self.flatten(self.encoder(x))
        mu = self.enc_fc_mu(x)
        log_var = self.enc_fc_var(x)
        return mu, log_var
         
    def reparameterize(self, mu, log_var):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decoding(self, z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.dec_fc(z))
        x = self.unflatten(x)
        x = self.decoder(x)
        x = self.resize(x)
        return x

    def forward(self, x):
        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        decoder_out = self.decoding(z)
        return (mu, log_var), decoder_out
