import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from utils import get_system_info, set_figure_options
from dataset import SceneRecogDataset
from model import MODEL_DICT
from loss import LOSS_DICT


DATA_PATH = "../../../data"
MODEL_PATH = "/home/co-dl/workspace/jhkim/boundary-nn/log/Dec26_03:33:55_resvae-v5-BEST/resvae-v5_last_epoch.pt"
MODEL_NAME = "resvae-v5"

def sampling_points(mu, log_var, n=100):
    std = torch.exp(log_var / 2)
    eps = torch.randn((n, *mu.shape))
    z = (mu + std * eps).detach().cpu().numpy() # (N, *mu.shape)
    return z

def Img2VAE(model, scene_recog_dataset, video_name, n_sampling=1):
    model.eval()
    with torch.no_grad():
        (image_a, image_b), new_idx = scene_recog_dataset.get_single_data(name=video_name)
        model_input = torch.vstack((image_a.unsqueeze(0), image_b.unsqueeze(0)))
        (mu, log_var), _ = model(model_input)

    sampled_points = sampling_points(mu, log_var, n=n_sampling) # (n, *mu.shape)
    n_points, n_t, z_dim = sampled_points.shape
    sampled_points = sampled_points.reshape(-1, z_dim) # (n * n_t, z_dim)
    
    # new가 0, old가 1
    if new_idx == 0: 
        answer = np.array([[1,0],[0,1]])
    elif new_idx == 1:
        answer = np.array([[0,1],[1,0]])
    else:
        ValueError(f"new index must be 0 or 1, but got {new_idx}")

    sampled_points = torch.from_numpy(sampled_points)
    answer = torch.from_numpy(answer)
    
    return sampled_points, answer

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(512, 256)
        self.layer2 = nn.Linear(256,128)
        self.layer3 = nn.Linear(128,64)
        self.layer4 = nn.Linear(64, 2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x

def task_training(epochs, latent, answer):
    for epoch in range(epochs):
        output = mlp(latent)
        loss = criterion(output.float(), answer.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        break
        if epoch % 10 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))
    

if __name__ == "__main__":

    # Settings
    set_figure_options()
    device, num_workers = get_system_info()
    print(f"Device: {device}")

    transform = transform=transforms.Compose([
        transforms.Resize(size=(256, 512)),
        transforms.ToTensor(),
    ])

    model = MODEL_DICT[MODEL_NAME](in_channels=32, latent_dim=512)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    scene_recog_dataset = SceneRecogDataset(root_dir=os.path.join(DATA_PATH, "MemSeg_SceneRecogImg"), transform=transform)

    mlp = MLP()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=0.01)
    
    num_train = 24
    MLP_epochs = 50

    print("***********Model Training***********")
    for i, btype in enumerate(tqdm(("HB", "NB", "SB"))):
        for j in range(num_train):
            latent, answer = Img2VAE(model, scene_recog_dataset, video_name=f"{btype}_{j+1}")
            task_training(MLP_epochs, latent, answer)
    
    # test MLP performing task
    print("***********Model Testing***********")
    with torch.no_grad():
        correct = 0
        total = 0
        for i, btype in enumerate(tqdm(("HB", "NB", "SB"))):
            for j in range(30-num_train):
                latent, answer = Img2VAE(model, scene_recog_dataset, video_name=f"{btype}_{j+num_train+1}")
                outputs = mlp(latent)
                _, predicted = torch.max(outputs.data, 1)
                total += answer.size(0)
                correct += (predicted == answer).sum().item()
        print('Accuracy of the network on the test data: {} %'.format(100 * correct / total))

