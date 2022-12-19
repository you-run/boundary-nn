import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class ModuleUtils:
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + std * eps


class Interpolate(nn.Module):
    def __init__(self, size, mode='nearest'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        
    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode)
        return x


class ConvDownBlock(nn.Module):
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


class ConvUpBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1, pool_size=2
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.ReLU()
        self.pool = nn.Upsample(scale_factor=pool_size)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.pool(out)
        return out


class ResDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.act = nn.ELU()
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 2, kernel_size // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-4)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.res_conv(x)
        out = self.act(out)
        return out


class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2):
        super().__init__()
        self.act = nn.ELU()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-4)

    def forward(self, x):
        x = self.upsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.res_conv(x)
        out = self.act(out)
        return out


class ConvAutoencoder(nn.Module, ModuleUtils):
    def __init__(self, latent_dim=3, original_size=(270, 480), *args, **kwargs):
        super().__init__()
        # encoder
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            ConvDownBlock(3, 16, 7, 1, 1, 2),
            ConvDownBlock(16, 32, 3, 1, 1, 2),
            ConvDownBlock(32, 64, 3, 1, 1, 2),
            ConvDownBlock(64, 64, 3, 1, 1, 2),
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

class ConvAutoencoderV2(nn.Module, ModuleUtils):
    def __init__(self, latent_dim=3, original_size=(270, 480), *args, **kwargs):
        super().__init__()
        # Encoder
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            ConvDownBlock(3, 8, 7, 1, 1, 2),
            ConvDownBlock(8, 16, 3, 1, 1, 2),
            ConvDownBlock(16, 32, 3, 1, 1, 2),
            ConvDownBlock(32, 64, 3, 1, 1, 2),
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


class ConvVAE(nn.Module, ModuleUtils):
    def __init__(self, feature_dim=2 * 16 * 29, latent_dim=512, *args, **kwargs):
        super().__init__()
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encoder = nn.Sequential(
            ConvDownBlock(3, 16, 7, 1, 1, 2),
            ConvDownBlock(16, 32, 3, 1, 1, 2),
            ConvDownBlock(32, 64, 3, 1, 1, 2),
            ConvDownBlock(64, 64, 3, 1, 1, 2),
            nn.Conv2d(64, 2, 1, 1, 0) 
        )
        self.flatten = nn.Flatten()
        
        self.enc_fc_mu = nn.Linear(feature_dim, latent_dim)
        self.enc_fc_var = nn.Linear(feature_dim, latent_dim)

        # Inxitializing the fully-connected layer and 2 convolutional layers for decoder
        self.dec_fc = nn.Linear(latent_dim, feature_dim)
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
        self.latent_dim = latent_dim

    def encoding(self, x):
        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        x = self.flatten(self.encoder(x))
        mu = self.enc_fc_mu(x)
        log_var = self.enc_fc_var(x)
        return mu, log_var

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
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        decoder_out = self.decoding(z)
        return (mu, log_var), decoder_out


class ResidualConvVAE(nn.Module, ModuleUtils):
    def __init__(self, in_channels=64, latent_dim=256, *args, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDownBlock(3, in_channels, 7, 2, 3),
            ResDownBlock(in_channels, in_channels * 2),
            ResDownBlock(in_channels * 2, in_channels * 4),
            ResDownBlock(in_channels * 4, in_channels * 8),
            ResDownBlock(in_channels * 8, in_channels * 16),
            nn.Linear(8, 4)
        )
        self.enc_conv_mu = nn.Conv2d(in_channels * 16, latent_dim, 4, 1)
        self.enc_conv_log_var = nn.Conv2d(in_channels * 16, latent_dim, 4, 1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, in_channels * 16, 4, 1),
            nn.Linear(4, 8),
            ResUpBlock(in_channels * 16, in_channels * 8),
            ResUpBlock(in_channels * 8, in_channels * 4),
            ResUpBlock(in_channels * 4, in_channels * 2),
            ResUpBlock(in_channels * 2, in_channels),
            ConvUpBlock(in_channels, 3, 4, 2, 1),
            nn.Sigmoid()
        )
        self.latent_dim = latent_dim

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_conv_mu(x).squeeze()
        log_var = self.enc_conv_log_var(x).squeeze()
        return mu, log_var

    def forward(self, x):
        mu, log_var = self.encoding(x)
        if self.training:
            z = self.reparameterize(mu, log_var)
        else:
            z = mu
        z = z.reshape(*z.shape, 1, 1)
        recon = self.decoder(z)
        return (mu, log_var), recon


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1), nn.Sigmoid())

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        # rnn_out1, hidden_state1 = self.rnn(x, h0)
        # rnn_out2, hidden_state2 = self.rnn(rnn_out1, h0)
        rnn_out, hidden_state = self.rnn(x, h0)
        # out = out.reshape(out.shape[0], -1)
        # out = self.fc(out)
        # print(rnn_out.shape)
        return rnn_out, hidden_state


MODEL_DICT = {
    'ae': ConvAutoencoder, 
    'ae-v2': ConvAutoencoderV2, 
    'vae': ConvVAE,
    'resvae': ResidualConvVAE
}
