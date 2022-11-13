import torch
import torch.nn as nn


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

class BoundaryRecognizer(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn = nn.Sequential(
            ConvBlock(3, 16, 7, 3, 0, 4),
            ConvBlock(16, 32, 3, 1, 1, 2),
            ConvBlock(32, 64, 3, 1, 1, 2),
            ConvBlock(64, 64, 3, 1, 1, 2),
            nn.Flatten(),
            nn.Linear(2880, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
        )
        self.rnn = nn.RNN(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=2,
            nonlinearity='relu',
            batch_first=True,
            dropout=0.,
            bidirectional=False
        )
        self.fc = nn.Linear(latent_dim, 3)

    def forward(self, x, h0): # (N, T, C, H, W), (RNN_num_layers, latent_dim)
        batch_size, time_len = x.shape[:2]
        x = x.view(-1, *x.shape[2:])
        out = self.cnn(x) # (N * T, latent_dim)        
        out = out.view(batch_size, time_len, self.latent_dim) # (N, T, latent_dim)
        out, hn = self.rnn(out) # (N, T, latent_dim), (num_layers, latent_dim)
        out = self.fc(out) # (N, T, 3)

        return out, hn


if __name__ == "__main__":
    # Testing
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f"Current device: {device}")

    data = torch.randn(8, 2, 3, 540, 960).to(device) # (N, T, C, H, W)
    h0 = torch.randn(2, 128).to(device)

    model = BoundaryRecognizer().to(device)
    out, hn = model(data, h0)

    print(out.shape)
    print(hn.shape)
