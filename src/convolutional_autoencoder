import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T


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

class CNN(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        # encoder
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            ConvBlock(3, 8, 3, 1, 1, 2),
            ConvBlock(8, 16, 3, 1, 1, 2),
            ConvBlock(16, 32, 3, 1, 1, 2),
        )
            # ConvBlock(64, 64, 3, 1, 1, 2),
            # nn.Flatten(),
            # nn.Linear(2880, 512),
            # nn.ReLU(),
            # nn.Linear(512, latent_dim),
            # nn.ConvTranspose2d(64, 64, kernel_size= 2, stride = 2, padding = 0),
            # nn.ReLU(),
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size = 2, stride = 2, padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size = 2, stride = 2, padding = 0),
            nn.Sigmoid()
            # nn.ReLU()
        )
        # self.fc = nn.Linear(latent_dim, 3)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        # out = self.fc(out) 
        return out

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('cpu')
model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

fig = plt.figure()
ax = fig.add_subplot()

steps = 0
losses = []

for i in range(10):
    steps+=1
    for root, dirs, files in os.walk("./data/video/HB"):
        for file in files:
            vread = np.load(os.path.join(root, file))
            vread = torch.Tensor(np.transpose(vread, (0,3,1,2)))
            transform = T.Resize(size = (256, 480))
            vread = transform(vread)
            # ax.imshow(vread[80].numpy())
            # plt.show()
            out = model(vread)
            loss = criterion(out, vread)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss)
            print(steps, loss)
            model.train()

# plt.plot(losses.detach().numpy())
# plt.show()
