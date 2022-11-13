import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import random
from sklearn.decomposition import PCA
import pandas as pd
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
A Convolutional Variational Autoencoder
"""
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

class VAE(nn.Module):
    def __init__(self, imgChannels=3, featureDim=2*16*29, zDim=256):
        super(VAE, self).__init__()
        self.featureDim = featureDim
        # Initializing the 2 convolutional layers and 2 full-connected layers for the encoder
        self.encConv1 = nn.Conv2d(3, 16, 7, stride=1, padding=1)
        self.encConv2 = nn.Conv2d(16, 32, 5, stride=1, padding=1)
        self.encConv3 = nn.Conv2d(32, 64, 5, stride=1, padding=1)
        self.encConv4 = nn.Conv2d(64, 64, 5, stride=1, padding=1)
        self.encConv5 = nn.Conv2d(64, 2, 1, stride=1, padding=0)
        self.encConv = nn.Sequential(
            ConvBlock(3, 16, 7, 1, 1, 2),
            ConvBlock(16, 32, 3, 1, 1, 2),
            ConvBlock(32, 64, 3, 1, 1, 2),
            ConvBlock(64, 64, 3, 1, 1, 2),
            nn.Conv2d(64, 2, 1, 1, 0) 
        )
        self.flatten = nn.Flatten()

        self.encFC1 = nn.Linear(featureDim, zDim)
        self.encFC2 = nn.Linear(featureDim, zDim)

        # Inxitializing the fully-connected layer and 2 convolutional layers for decoder
        self.unflatten = nn.Unflatten(1, (2, 16, 29))
        self.decFC1 = nn.Linear(zDim, featureDim)
        self.decConv1 = nn.ConvTranspose2d(32, 16, 5)
        self.decConv2 = nn.ConvTranspose2d(16, imgChannels, 5)
        self.decConv = nn.Sequential(
            nn.Conv2d(2, 64, 1, 1, 0),
            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 3, padding = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 8, kernel_size = 3, stride = 3, padding = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size = 3, stride = 3, padding = 2),
            nn.Sigmoid()
        )
    
    
    def set_featuredim(self, frame_num, zDim):
        self.featureDim = frame_num*self.featureDim
        self.encFC1 = nn.Linear(self.featureDim, zDim)
        self.encFC2 = nn.Linear(self.featureDim, zDim)
        self.decFC1 = nn.Linear(zDim, self.featureDim)


    def encoder(self, x):

        # Input is fed into 2 convolutional layers sequentially
        # The output feature map are fed into 2 fully-connected layers to predict mean (mu) and variance (logVar)
        # Mu and logVar are used for generating middle representation z and KL divergence loss
        '''
        x = F.relu(self.encConv1(x))
        print(x.shape)
        x = F.relu(self.encConv2(x))
        print(x.shape)
        x = F.relu(self.encConv3(x))
        print(x.shape)
        x = F.relu(self.encConv4(x))
        print(x.shape)
        x = F.relu(self.encConv5(x))
        print(x.shape)
        x = x.reshape(-1, self.featureDim)
        '''
        x = self.encConv(x)
        print(f"after encoding, x size : {x.shape}")
        x = self.flatten(x)
        print(f"after flatten, x size : {x.shape}")
        mu = self.encFC1(x)
        logVar = self.encFC2(x)
        print(f"mu/logVar size (after linear) : {mu.shape}, {logVar.shape}")
        return mu, logVar
         
    def reparameterize(self, mu, logVar):

        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        print(f"output of reparameterize(z) : {(mu + std * eps).shape}")
        return mu + std * eps

    def decoder(self, z):

        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.decFC1(z))
        print(f"decoder started, x size : {x.shape}")
        x = self.unflatten(x)
        print(f"unflatten x : {x.shape}")
        '''
        x = F.relu(self.decConv1(x))
        x = torch.sigmoid(self.decConv2(x))
        '''
        x = self.decConv(x)
        print(f"decoder ended, x size : {x.shape}")
        return x

    def forward(self, x):

        # The entire pipeline of the VAE: encoder -> reparameterization -> decoder
        # output, mu, and logVar are returned for loss computation
        mu, logVar = self.encoder(x)
        z = self.reparameterize(mu, logVar)
        out = self.decoder(z)
        return out, mu, logVar

def get_input(path, file):
    video_npy = np.load(os.path.join(path,file))
    #print(video_npy.shape)
    vread = torch.from_numpy(video_npy)
    vread = vread.permute(0,3,1,2)
    transform = T.Resize(size = (270,480))
    vread = transform(vread)
    vread = vread/255
    contour = round(vread.size(dim=0)/2)
    #input = [vread[0:contour], vread[contour:contour*2], vread[contour*2:]]
    input = [vread[0:contour], vread[contour:]]
    '''
    print(vread.size())
    for i in range(len(input)):
        print(input[i].size())
    '''
    #print(len(input))
    return input


"""
Initialize Hyperparameters
"""
learning_rate = 1e-3
num_epochs = 10
num_videos = 3

data_path = '../preprocessing/data/video_data/HB'
data = os.listdir(data_path)

"""
Initialize the network and the Adam optimizer
"""
net = VAE().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

"""
Training the network for a given number of epochs
The loss after every epoch is printed
"""
for epoch in range(num_epochs):
    for idx, data in enumerate(data[0:num_videos]):
        input = get_input(data_path, data)
        for set in range(len(input)):
            input[set] = input[set].to(device)
            print(f"original input(x) size : {input[set].shape}")
            '''
            frame_num = input[set].size(dim=0)
            net.set_featuredim(frame_num, zDim=256)
            '''
            # Feeding a batch of images into the network to obtain the output image, mu, and logVar
            out, mu, logVar = net(input[set])
            # The loss is the BCE loss combined with the KL divergence to ensure the distribution is learnt
            kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp())
            loss = F.binary_cross_entropy(out, input[set], size_average=False) + kl_divergence
            # Backpropagation based on the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print('Epoch {}: Loss {}'.format(epoch, loss))