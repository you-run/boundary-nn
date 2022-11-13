import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
            ConvBlock(3, 16, 3, 1, 1, 2),
            ConvBlock(16, 32, 3, 1, 1, 2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size = 3, stride = 2, padding = 2),
            nn.Sigmoid(),
        )
        # self.fc = nn.Linear(latent_dim, 3)

    def forward(self, x):
        out = self.encoder(x)
        # print(out.shape)
        out = self.decoder(out)
        transform = T.Resize(size = (270, 480))
        out = transform(out)
        # print(out.shape)

        return out

# device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
device = torch.device('cpu')
model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# lambda1 = lambda epoch: 0.85 ** epoch
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

fig = plt.figure()
ax = fig.add_subplot() 

steps = 0

for i in range(10):
    steps+=1
    for root, dirs, files in os.walk("./data/video/HB"):
        for file in files:
            vread = torch.from_numpy(np.load(os.path.join(root,file)))
            vread = vread.permute(0,3,1,2)
            transform = T.Resize(size = (270, 480))
            vread = transform(vread)
            # ax.imshow(to_pil_image(vread[80]))
            # plt.show()
            out = model(vread.float())
            ax.imshow(to_pil_image(out[85]))
            plt.show()  
            loss = criterion(out, vread.float())
            loss.backward()
            optimizer.step()
            # scheduler.step()
            optimizer.zero_grad()
            print(steps, loss)
            model.train()
