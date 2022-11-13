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
            nn.Conv2d(64, 3, 1, 1, 0)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(3, 64, 1, 1, 0),
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
        # print(f"latent vector size : {encoder_out.shape}")
        # print(encoder_out.shape)
        decoder_out = self.decoder(encoder_out)
        transform = T.Resize(size = (270, 480))
        decoder_out = transform(decoder_out)
       # print(out.shape)

        return (encoder_out, decoder_out)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
#device = torch.device('cpu')
print(device)
cnn_model = CNN().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
#lambda1 = lambda epoch: 0.85 ** epoch
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

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
    en_out, dec_out = cnn_model(input.float())
    loss = criterion(dec_out, input.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #scheduler.step()
    #model.train()
    return (en_out, dec_out, loss)

data_root = '/home/neurlab/yw/boundary-nn/data/video/HB'
num_epochs = 300
frame_num = 50
num_video = 3

loss_log1 = np.zeros((num_video, num_epochs))
loss_log2 = np.zeros((num_video, num_epochs))
files = os.listdir(data_root)
# random.shuffle(files)


for epoch in range(num_epochs):
    print(f"{epoch+1}th epoch")
    enc_out_whole = []
    for i,file in enumerate(files[0:num_video]):
        #print(f"video:{file}")
        input = get_input(data_root, file)
        enc_out_list = []
        for set in range(len(input)):
            input[set] = input[set].to(device)
            enc_out, dec_out, loss = training(input[set])
            enc_out_list.append(enc_out)
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

        enc_out_cat = torch.cat([enc_out_list[0], enc_out_list[1]], dim=0)
        enc_out_whole.append(enc_out_cat)
    

torch.save(cnn_model, '/home/neurlab/yw/boundary-nn/cnn_model.pt')

input_size = 1392
hidden_size = 1392
num_layers = 1

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
        rnn_out3, hidden_state3 = self.rnn(x, h0)
        # out = out.reshape(out.shape[0], -1)
        # out = self.fc(out)
        # print(rnn_out.shape)
        return rnn_out3, hidden_state3

rnn_model = RNN(input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                device=device).to(device)

lr = 0.001
num_epochs = 300
optimizer = torch.optim.Adam(rnn_model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    rnn_out_whole = []
    for enc_out in enc_out_whole:
        flatten_layer = nn.Flatten()
        latent = flatten_layer(enc_out)
        latent = latent.unsqueeze(0)
        latent = latent.detach()
        rnn_out3, hidden_state = rnn_model(latent)
        loss = criterion(rnn_out3, latent) 
        # print(hidden_state.shape)
        optimizer.zero_grad() 
        loss.backward(retain_graph=True) 
        optimizer.step()
        
        print(epoch, loss)
        rnn_out = rnn_out3.squeeze(0)
        rnn_out = rnn_out.reshape(rnn_out.shape[0], 3, 16, 29).to(device)
        rnn_out_whole.append(rnn_out)


cnn_model = torch.load('/home/neurlab/yw/boundary-nn/cnn_model.pt')
torch.save(rnn_model, '/home/neurlab/yw/boundary-nn/rnn_model.pt')

# for enc_out in enc_out_whole:
#     fig, ax = plt.subplots()
#     x = cnn_model.decoder(enc_out.float())
#     ax.imshow(to_pil_image(x[80], mode = 'RGB'))
#     plt.show()

for rnn_out in rnn_out_whole:
    flatten_layer = nn.Flatten()
    rnn_flatten = flatten_layer(rnn_out).cpu().detach().numpy()
    pca = PCA(n_components=2)
    principalcomponents = pca.fit_transform(rnn_flatten)
    columns = []
    for i in range(2):
        columns.append('PC{}'.format(i+1))
    df = pd.DataFrame(data = principalcomponents, columns = columns)
    plt.scatter(df['PC1'], df['PC2'], s=1, c = df['PC2'])
    plt.viridis()
    plt.colorbar()
    plt.show()

    x = cnn_model.decoder(rnn_out.float())
    fig, ax = plt.subplots()
    ax.imshow(to_pil_image(x[round(x.size(dim=0)/2)+10], mode = 'RGB'))
    plt.show()