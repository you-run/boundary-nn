import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from google.colab import drive
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    def __init__(self, latent_dim=3):
        super().__init__()
        # encoder
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            ConvBlock(3, 16, 7, 1, 1, 2),
            ConvBlock(16, 32, 3, 1, 1, 2),
            ConvBlock(32, 64, 3, 1, 1, 2),
            ConvBlock(64, 64, 3, 1, 1, 2),
            nn.Conv2d(64, 8, 1, 1, 0)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(8, 64, 1, 1, 0),
            nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 2, padding = 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size = 3, stride = 2, padding = 2),
            nn.Sigmoid()
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(29696,512),
            nn.ReLU(),
            nn.Linear(512,3)
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        #out = self.fc(out)
        #print(f"latent vector size : {encoder_out.shape}")
        decoder_out = self.decoder(encoder_out)
        transform = T.Resize(size = (270, 480))
        decoder_out = transform(decoder_out)
       # print(out.shape)

        return (encoder_out, decoder_out)

def get_input(path, file):
    vread = torch.from_numpy(np.load(os.path.join(path, file)))
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
    return input

def training(input):
    en_out, dec_out = model(input.float())
    loss = criterion(dec_out, input.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #scheduler.step()
    #model.train()
    return (en_out, dec_out, loss)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
#device = torch.device('cpu')
print(device)
model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#lambda1 = lambda epoch: 0.85 ** epoch
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

data_root = '/content/drive/MyDrive/boundary_dataset/video_data/HB'
num_epochs = 90
frame_num = 80
num_video = 2

loss_log1 = np.zeros((num_video, num_epochs))
loss_log2 = np.zeros((num_video, num_epochs))
files = os.listdir(data_root)
#random.shuffle(files)

for epoch in range(num_epochs):
    print(f"{epoch+1}th epoch")
    for i,file in enumerate(files[0:num_video]):
        #print(f"video:{file}")
        input = get_input(data_root, file)
        for set in range(len(input)):
            input[set] = input[set].to(device)
            enc_out, dec_out, loss = training(input[set])
            #print(f"{i}th video {set+1} set, loss:{loss.item()}")
            if set==0:
                loss_log1[i,epoch] = loss.item()
            elif set==1:
                loss_log2[i,epoch] = loss.item()
        if i<3 and (epoch==0 or epoch==round(num_epochs/2) or epoch==num_epochs-1):
            fig,ax = plt.subplots(1,2)
            ax[0].imshow(to_pil_image(input[1][frame_num], mode='RGB')) # plot은 일단 두번째 set에서만
            ax[1].imshow(to_pil_image(dec_out[frame_num], mode='RGB'))
            plt.show()
# loss result
for i in range(num_video):
    #print(f"{i}th video result in set1 : {loss_log1[i,0]} => {loss_log1[i,round(num_epochs/2)]} => {loss_log1[i,-1]}")
    print(f"{i}th video result in set2 : {loss_log2[i,0]} => {loss_log2[i,round(num_epochs/2)]} => {loss_log2[i,-1]}")

# LSTM
lstm = nn.LSTM(3712, 3712, 1, batch_first=True).to(device)
flatten_layer = nn.Flatten()
latent = flatten_layer(enc_out)
output = lstm(latent)

pca = PCA(n_components=3)
projected = pca.fit_transform(latent.cpu().detach().numpy())
df = pd.DataFrame(projected)

plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=np.arange(121), cmap='plasma')