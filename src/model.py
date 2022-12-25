from abc import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class ModuleUtils(ABC):
    @abstractmethod
    def encoding(self):
        pass
    
    @abstractmethod
    def decoding(self):
        pass

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(log_var / 2)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


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
        kernel_size=3, stride=1, padding=1, pool=False
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
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class ConvUpBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels,
        kernel_size=3, stride=1, padding=1, pool=False
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
        self.pool = nn.Upsample(scale_factor=2) if pool else None

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class ResDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool=False):
        super().__init__()
        self.act = nn.ELU()
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 2, kernel_size // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size, 2, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else None

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.res_conv(x)
        out = self.act(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class ResUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=2, pool=False):
        super().__init__()
        self.act = nn.ELU()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.res_conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1, kernel_size // 2)

        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size, 1, kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels // 2, eps=1e-4)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size, 1, kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels, eps=1e-4)
        self.pool = nn.Upsample(scale_factor=2) if pool else None

    def forward(self, x):
        x = self.upsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.res_conv(x)
        out = self.act(out)
        if self.pool is not None:
            out = self.pool(out)
        return out


class ConvVAE(nn.Module, ModuleUtils):
    def __init__(self, in_channels=16, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDownBlock(3, in_channels, 7, 1, 1, pool=True),
            ConvDownBlock(in_channels, in_channels * 2, 3, 1, 1, pool=True),
            ConvDownBlock(in_channels * 2, in_channels * 4, 3, 1, 1, pool=True),
            ConvDownBlock(in_channels * 4, in_channels * 4, 3, 1, 1, pool=True),
            nn.Conv2d(in_channels * 4, 2, 1, 1, 0),
            nn.Flatten()
        )
        self.enc_mu = nn.Linear(2 * 16 * 29, latent_dim)
        self.enc_var = nn.Linear(2 * 16 * 29, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * 29),
            nn.ReLU(),
            nn.Unflatten(1, (2, 16, 29)),
            nn.Conv2d(2, in_channels * 4, 1, 1, 0),
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=3, stride=2, padding=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, 3, kernel_size=3, stride=2, padding=2),
            nn.Sigmoid()
        )
        self.resize = T.Resize(size=(270,480))

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return mu, log_var

    def decoding(self, x):
        x = self.decoder(x)
        x = self.resize(x)
        return x

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ConvVAEV2(nn.Module, ModuleUtils):
    def __init__(self, in_channels=64, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, in_channels, 7, 1, 3), # (in_ch, 256, 512)
            ConvDownBlock(in_channels, in_channels * 2, 3, 2, 1, pool=True), # (in_ch * 2, 64, 128)
            ConvDownBlock(in_channels * 2, in_channels * 4, 3, 2, 1, pool=True), # (in_ch * 4, 16, 32)
            ConvDownBlock(in_channels * 4, in_channels * 8, 3, 2, 1, pool=True), # (in_ch * 8, 4, 8)
        )        
        self.enc_mu = nn.Conv2d(in_channels * 8, latent_dim, (4, 8), 1) # (in_ch * 8, 1, 1)
        self.enc_conv_log_var = nn.Conv2d(in_channels * 8, latent_dim, (4, 8), 1) # (in_ch * 8, 1, 1)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, in_channels * 8, (4, 8), 1), # (in_ch * 16, 4, 8)
            ConvUpBlock(in_channels * 8, in_channels * 4, 2, 2, 0, pool=True), # (in_ch * 8, 16, 32)
            ConvUpBlock(in_channels * 4, in_channels * 2, 2, 2, 0, pool=True), # (in_ch * 8, 64, 128)
            ConvUpBlock(in_channels * 2, in_channels, 2, 2, 0, pool=True), # (in_ch * 8, 256, 512)
            nn.Conv2d(in_channels, 3, 7, 1, 3),
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_conv_log_var(x).squeeze()
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ConvVAEV3(nn.Module, ModuleUtils):
    def __init__(self, in_channels=16, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDownBlock(3, in_channels, 7, 2, 3, pool=True), # (in_ch, 64, 128)
            ConvDownBlock(in_channels, in_channels * 2, 3, 1, 1, pool=True), # (in_ch * 2, 32, 64)
            ConvDownBlock(in_channels * 2, in_channels * 4, 3, 1, 1, pool=True), # (in_ch * 4, 16, 32)
            ConvDownBlock(in_channels * 4, in_channels * 8, 3, 1, 1, pool=True), # (in_ch * 8, 8, 16)
            ConvDownBlock(in_channels * 8, in_channels * 16, 3, 1, 1, pool=True), # (in_ch * 16, 4, 8)
            nn.Flatten()
        )
        self.enc_mu = nn.Linear(in_channels * 16 * 4 * 8, latent_dim)
        self.enc_var = nn.Linear(in_channels * 16 * 4 * 8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_channels * 16 * 4 * 8),
            nn.Unflatten(1, (in_channels * 16, 4, 8)),
            ConvUpBlock(in_channels * 16, in_channels * 8, 4, 2, 1, pool=False), # (in_ch * 8, 8, 16)
            ConvUpBlock(in_channels * 8, in_channels * 4, 4, 2, 1, pool=False), # (in_ch * 4, 16, 32)
            ConvUpBlock(in_channels * 4, in_channels * 2, 4, 2, 1, pool=False), # (in_ch * 2, 32, 64)
            ConvUpBlock(in_channels * 2, in_channels, 4, 2, 1, pool=False), # (in_ch, 64, 128)
            ConvUpBlock(in_channels, 3, 4, 2, 1, pool=True), # (3, 256, 512)
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ConvVAEV4(nn.Module, ModuleUtils):
    def __init__(self, in_channels=16, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDownBlock(3, in_channels, 7, 1, 3, pool=True),
            ConvDownBlock(in_channels, in_channels * 2, 3, 1, 1, pool=True),
            ConvDownBlock(in_channels * 2, in_channels * 4, 3, 1, 1, pool=True),
            ConvDownBlock(in_channels * 4, in_channels * 4, 3, 1, 1, pool=True),
            nn.Conv2d(in_channels * 4, 3, 1, 1, 0),
            nn.Flatten()
        )
        self.enc_mu = nn.Linear(3 * 16 * 32, latent_dim)
        self.enc_var = nn.Linear(3 * 16 * 32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 3 * 16 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (3, 16, 32)),
            nn.Conv2d(3, in_channels * 4, 1, 1, 0),
            nn.ConvTranspose2d(in_channels * 4, in_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ResidualConvVAE(nn.Module, ModuleUtils):
    def __init__(self, in_channels=64, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDownBlock(3, in_channels, 7, 2, 3),
            ResDownBlock(in_channels, in_channels * 2, pool=False),
            ResDownBlock(in_channels * 2, in_channels * 4, pool=False),
            ResDownBlock(in_channels * 4, in_channels * 8, pool=False),
            ResDownBlock(in_channels * 8, in_channels * 16, pool=False),
            nn.Linear(8, 4)
        )
        self.enc_mu = nn.Conv2d(in_channels * 16, latent_dim, 4, 1)
        self.enc_conv_log_var = nn.Conv2d(in_channels * 16, latent_dim, 4, 1)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, in_channels * 16, 4, 1),
            nn.Linear(4, 8),
            ResUpBlock(in_channels * 16, in_channels * 8, pool=False),
            ResUpBlock(in_channels * 8, in_channels * 4, pool=False),
            ResUpBlock(in_channels * 4, in_channels * 2, pool=False),
            ResUpBlock(in_channels * 2, in_channels, pool=False),
            ConvUpBlock(in_channels, 3, 4, 2, 1),
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_conv_log_var(x).squeeze()
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ResidualConvVAEV2(nn.Module, ModuleUtils):
    def __init__(self, in_channels=64, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDownBlock(3, in_channels, 7, 1, 3), # (N, in, 128, 256)
            ResDownBlock(in_channels, in_channels * 2, pool=True), # (N, in * 2, 32, 64)
            ResDownBlock(in_channels * 2, in_channels * 4, pool=True), # (N, in * 4, 8, 16)
            ResDownBlock(in_channels * 4, in_channels * 8, pool=True), # (N, in * 8, 2, 4)
        )
        self.enc_mu = nn.Conv2d(in_channels * 8, latent_dim, (2, 4), 1)
        self.enc_conv_log_var = nn.Conv2d(in_channels * 8, latent_dim, (2, 4), 1)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, in_channels * 8, (2, 4), 1), # (N, in * 8, 2, 4)
            ResUpBlock(in_channels * 8, in_channels * 4, pool=True), # (N, in * 4, 8, 16)
            ResUpBlock(in_channels * 4, in_channels * 2, pool=True), # (N, in * 2, 32, 64)
            ResUpBlock(in_channels * 2, in_channels, pool=True), # (N, in, 128, 256)
            ResUpBlock(in_channels, 3, pool=False),
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_conv_log_var(x).squeeze()
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ResidualConvVAEV3(nn.Module, ModuleUtils):
    def __init__(self, in_channels=64, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDownBlock(3, in_channels, 7, 1, 3, pool=True), # (N, in, 128, 256)
            ResDownBlock(in_channels, in_channels * 2, pool=True), # (N, in * 2, 32, 64)
            ResDownBlock(in_channels * 2, in_channels * 4, pool=True), # (N, in * 4, 8, 16)
            ResDownBlock(in_channels * 4, in_channels * 8, pool=True), # (N, in * 4, 2, 4)
            nn.Flatten()
        )
        self.enc_mu = nn.Linear(in_channels * 8 * 2 * 4, latent_dim)
        self.enc_var = nn.Linear(in_channels * 8 * 2 * 4, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_channels * 8 * 2 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (in_channels * 8, 2, 4)),
            ResUpBlock(in_channels * 8, in_channels * 4, pool=True), # (N, in * 4, 8, 16)
            ResUpBlock(in_channels * 4, in_channels * 2, pool=True), # (N, in * 2, 32, 64)
            ResUpBlock(in_channels * 2, in_channels, pool=True), # (N, in, 128, 256)
            ResUpBlock(in_channels, 3, pool=False),
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_var(x).squeeze()
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ResidualConvVAEV4(nn.Module, ModuleUtils):
    def __init__(self, in_channels=64, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ResDownBlock(3, in_channels, kernel_size=7, pool=True), # (N, in * 2, 64, 128)
            ResDownBlock(in_channels, in_channels * 2, pool=True), # (N, in * 2, 16, 32)
            ResDownBlock(in_channels * 2, in_channels * 4, pool=False), # (N, in * 4, 8, 16)
            ResDownBlock(in_channels * 4, in_channels * 8, pool=False), # (N, in * 8, 4, 8)
            nn.Flatten()
        )
        self.enc_mu = nn.Linear(in_channels * 8 * 4 * 8, latent_dim)
        self.enc_var = nn.Linear(in_channels * 8 * 4 * 8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_channels * 8 * 4 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (in_channels * 8, 4, 8)),
            ResUpBlock(in_channels * 8, in_channels * 4, pool=False), # (N, in * 4, 8, 16)
            ResUpBlock(in_channels * 4, in_channels * 2, pool=True), # (N, in * 2, 32, 64)
            ResUpBlock(in_channels * 2, in_channels, pool=True), # (N, in, 128, 256)
            ResUpBlock(in_channels, 3, pool=False), # (N, 3, 256, 512)
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_var(x).squeeze()
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ResidualConvVAEV5(nn.Module, ModuleUtils):
    def __init__(self, in_channels=16, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            ResDownBlock(3, in_channels, kernel_size=7, pool=True), # (N, in * 2, 64, 128)
            ResDownBlock(in_channels, in_channels * 2, pool=True), # (N, in * 2, 16, 32)
            ResDownBlock(in_channels * 2, in_channels * 4, pool=True), # (N, in * 4, 4, 8)
            nn.Flatten()
        )
        self.enc_mu = nn.Linear(in_channels * 4 * 4 * 8, latent_dim)
        self.enc_var = nn.Linear(in_channels * 4 * 4 * 8, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_channels * 4 * 4 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (in_channels * 4, 4, 8)),
            ResUpBlock(in_channels * 4, in_channels * 4, pool=True), # (N, in * 4, 16, 32)
            ResUpBlock(in_channels * 4, in_channels * 2, pool=True), # (N, in * 2, 64, 128)
            ResUpBlock(in_channels * 2, in_channels, pool=False), # (N, in, 128, 256)
            ResUpBlock(in_channels, 3, pool=False), # (N, 3, 256, 512)
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_var(x).squeeze()
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ResidualConvVAEV6(nn.Module, ModuleUtils):
    def __init__(self, in_channels=16, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            ResDownBlock(3, in_channels, kernel_size=7, pool=True), # (N, in * 2, 64, 128)
            ResDownBlock(in_channels, in_channels * 2, pool=True), # (N, in * 2, 16, 32)
            ResDownBlock(in_channels * 2, in_channels * 4, pool=True), # (N, in * 4, 4, 8)
        )
        self.enc_mu = nn.Conv2d(in_channels * 4, latent_dim, (4, 8), 1, 0)
        self.enc_var = nn.Conv2d(in_channels * 4, latent_dim, (4, 8), 1, 0)

        self.decoder = nn.Sequential( 
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, in_channels * 4, (4, 8), 1, 0), # (N, in * 4, 4, 8)
            nn.ReLU(),
            ResUpBlock(in_channels * 4, in_channels * 4, pool=False), # (N, in * 4, 8, 16)
            ResUpBlock(in_channels * 4, in_channels * 2, pool=False), # (N, in * 2, 16, 32)
            ResUpBlock(in_channels * 2, in_channels * 2, pool=True), # (N, in * 2, 64, 128)
            ResUpBlock(in_channels * 2, in_channels, pool=False), # (N, in, 128, 256)
            ResUpBlock(in_channels, 3, pool=False), # (N, 3, 256, 512)
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_var(x).squeeze()
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class ResidualConvVAEV7(nn.Module, ModuleUtils):
    def __init__(self, in_channels=16, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            ResDownBlock(3, in_channels, kernel_size=7, pool=True), # (N, in * 2, 64, 128)
            ResDownBlock(in_channels, in_channels * 2, pool=True), # (N, in * 2, 16, 32)
            ResDownBlock(in_channels * 2, in_channels * 4, pool=True), # (N, in * 4, 4, 8)
        )
        self.enc_mu = nn.Conv2d(in_channels * 4, latent_dim, (4, 8), 1, 0)
        self.enc_var = nn.Conv2d(in_channels * 4, latent_dim, (4, 8), 1, 0)

        self.decoder = nn.Sequential( 
            nn.Unflatten(1, (latent_dim, 1, 1)),
            nn.ConvTranspose2d(latent_dim, in_channels * 4, (4, 8), 1, 0), # (N, in * 4, 4, 8)
            nn.ReLU(),
            ResUpBlock(in_channels * 4, in_channels * 2, pool=True), # (N, in * 4, 16, 32)
            ResUpBlock(in_channels * 2, in_channels, pool=True), # (N, in * 2, 64, 128)
            ResUpBlock(in_channels, 3, pool=True), # (N, in, 256, 512)
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_var(x).squeeze()
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class MixedConvVAE(nn.Module, ModuleUtils):
    def __init__(self, in_channels=64, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvDownBlock(3, in_channels, 7, 1, 3, pool=True), # (N, in, 128, 256)
            ResDownBlock(in_channels, in_channels * 2, pool=True), # (N, in * 2, 32, 64)
            ConvDownBlock(in_channels * 2, in_channels * 4, 3, 1, 1, pool=True), # (in * 4, 16, 32)
            ResDownBlock(in_channels * 4, in_channels * 4, pool=False), # (N, in * 4, 8, 16)
            nn.Flatten()
        )
        self.enc_mu = nn.Linear(in_channels * 4 * 8 * 16, latent_dim)
        self.enc_var = nn.Linear(in_channels * 4 * 8 * 16, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, in_channels * 4 * 8 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (in_channels * 4, 8, 16)),
            ResUpBlock(in_channels * 4, in_channels * 4, pool=False), # (N, in * 8, 16, 32)
            ConvUpBlock(in_channels * 4, in_channels * 2, 4, 2, 1, pool=False), # (N, in * 4, 32, 64)
            ResUpBlock(in_channels * 2, in_channels, pool=False), # (N, in * 2, 64, 128)
            ConvUpBlock(in_channels, 3, 4, 2, 1, pool=True), # (N, in * 4, 128, 256)
            nn.Sigmoid()
        )

    def encoding(self, x):
        x = self.encoder(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_var(x).squeeze()
        return mu, log_var

    def decoding(self, x):
        return self.decoder(x)

    def forward(self, x):
        mu, log_var = self.encoding(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoding(z)
        return (mu, log_var), recon


class VideoVAE(nn.Module, ModuleUtils): # For unbatched input
    def __init__(self, in_channels=16, feature_dim=512, latent_dim=256, conv='resvae-v5', dropout=0.0):
        super().__init__()
        self.conv_vae = MODEL_DICT[conv](in_channels=in_channels, latent_dim=feature_dim)
        self.encoder_rnn = nn.RNN(
            input_size=feature_dim,
            hidden_size=latent_dim,
            num_layers=1,
            nonlinearity='tanh',
            dropout=False,
            batch_first=True
        ) # (L, latent_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.enc_mu = nn.Linear(latent_dim, latent_dim)
        self.enc_var = nn.Linear(latent_dim, latent_dim)

        self.decoder_rnn = nn.RNN(
            input_size=latent_dim,
            hidden_size=feature_dim,
            num_layers=1,
            nonlinearity='tanh',
            dropout=False,
            batch_first=True
        ) # (L, feature_dim)


    def encoding(self, x, h0=None):
        """ Encoding
        Args:
            x (_type_): Input, torch.Tensor, shape: (N, L, C, W, H)
            h0 (_type_, optional): Hidden state, torch.Tensor, shape: (1, N, latent_dim)
        """
        batch_size, seq_len = x.shape[:2]
        x = x.view(-1, *x.shape[2:]) # (N * L, C, H, W)

        # Encoder: CNN
        features = self.conv_vae.encoder(x) # (N * L, feature_dim)
        features = features.view(batch_size, seq_len, -1) # (N, L, feature_dim)

        # Encoder: RNN
        features, hn = self.encoder_rnn(features, h_0=h0) # (N, L, latent_dim), (1, N, latent_dim)
        
        mu = self.enc_mu(self.dropout(features)) # (N, L, latent_dim)
        log_var = self.enc_var(self.dropout(features)) # (N, L, latent_dim)

        return (mu, log_var), hn

    def decoding(self, x, h0=None):
        """ Encoding
        Args:
            x (_type_): Latent vectors, torch.Tensor, shape: (N, L, latent_dim)
            h0 (_type_, optional): Hidden state, torch.Tensor, shape: (1, N, feature_dim)
        """
        # Decoder: RNN
        batch_size, seq_len = x.shape[:2]
        features, hn = self.decoder_rnn(x, h_0=h0) # (N, L, feature_dim)
        features = features.view(batch_size, seq_len, -1) # (N * L, feature_dim)
        
        # Decoder: CNN
        recon = self.conv_vae.decoding(features) # (N * L, C, H, W)
        recon = recon.view(batch_size, seq_len, *recon.shape[1:]) # (N, L, C, H, W)
        return recon, hn

    def forward(self, x, enc_h0=None, dec_h0=None):
        (mu, log_var), enc_hn = self.encoding(x, ho=enc_h0)
        z = self.reparameterize(mu, log_var) # (N, L, latent_dim)
        recon, dec_hn = self.decoding(z, h0=dec_h0) # (N, L, C, H, W)
        return (mu, log_var), recon, (enc_hn, dec_hn)


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
    'vae': ConvVAE,
    'vae-v2': ConvVAEV2,
    'vae-v3': ConvVAEV3,
    'vae-v4': ConvVAEV4,
    'resvae': ResidualConvVAE,
    'resvae-v2': ResidualConvVAEV2,
    'resvae-v3': ResidualConvVAEV3,
    'resvae-v4': ResidualConvVAEV4,
    'resvae-v5': ResidualConvVAEV5,
    'resvae-v6': ResidualConvVAEV6,
    'resvae-v7': ResidualConvVAEV7,
    'mixed': MixedConvVAE,
    'video': VideoVAE
}
