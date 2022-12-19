import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA

from utils import get_args, get_systme_info, set_figure_options, set_seed, configure_cudnn
from dataset import VideoFrameDataset, SequentialVideoFrameDataset
from model import MODEL_DICT, ConvVAE
from loss import LOSS_DICT
from optimizer import OPTIM_DICT

def train_one_epoch_ae(model, dataloader, optimizer, criterion, scaler):
    model.train()

    losses = []
    for x in dataloader:
        x = x.to(model.device)
        with torch.cuda.amp.autocast():
            _, preds = model(x)
            loss = criterion(preds, x)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def train_one_epoch_vae(model, dataloader, optimizer, criterion, scaler):
    model.train()

    losses = []
    for x in dataloader:
        x = x.to(model.device)
        with torch.cuda.amp.autocast():
            (mu, log_var), preds = model(x)
            loss = criterion(preds, x, mu, log_var)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def eval_vae(model, dataloader, criterion):
    model.eval()

    losses = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(model.device)
            with torch.cuda.amp.autocast():
                (mu, log_var), preds = model(x)
                loss = criterion(preds, x, mu, log_var)
                losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def recon_and_plot(model, recon_dataset, epoch, log_path): # dataset : example dataset
    def show_bef_aft(ax, bef, aft): # ax = (ax1, ax2)
        ax[0].imshow(to_pil_image(bef, mode='RGB'))
        ax[0].set_title("Original")
        ax[1].imshow(to_pil_image(aft, mode='RGB'))
        ax[1].set_title("Reconstructed")

    model.eval()
    with torch.no_grad():
        _, preds = model(recon_dataset.to(model.device))
        preds = preds.detach().cpu()
    
    plot_len = len(recon_dataset) # dataset: (N, 3, H, W)
    fig, ax = plt.subplots(plot_len, 2)
    for i in range(plot_len):
        show_bef_aft(ax[i], recon_dataset[i, ...], preds[i, ...])
    fig.suptitle(f"Before & After at epoch {epoch}")
    plt.savefig(os.path.join(log_path, 'recon', f"epoch_{epoch}.png"))

def plot_progress(train_losses, eval_losses, eval_step, log_path):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    train_x = np.arange(len(train_losses)) + 1
    eval_x = np.arange(eval_step, len(train_losses) + 1, eval_step)

    ax.set_title("Loss")
    ax.plot(train_x, train_losses, label="Train")
    ax.plot(eval_x, eval_losses, label="Eval")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(os.path.join(log_path, f"loss.png"))

def latent_pca(latent):
    pca = PCA(n_components = 3)
    principalcomponents = pca.fit_transform(latent.cpu().detach().numpy())
    columns = []
    for i in range(3):
        columns.append('PC{}'.format(i+1))
    df = pd.DataFrame(data = principalcomponents, columns = columns)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(df['PC1'], df['PC2'], df['PC3'], s=3, color = 'black')
    plt.savefig('/home/neurlab/yw/boundary-nn/latent')
    plt.show()


if __name__ == "__main__":
    # Settings
    args = get_args()
    set_figure_options()
    set_seed(seed=args.seed)
    configure_cudnn(debug=args.debug)

    device, num_workers = get_systme_info()
    print(f"Device: {device} | Seed: {args.seed} | Debug: {args.debug}")
    print(args)

    seq_dataset = SequentialVideoFrameDataset(
        args.data_dir,
        transform=transforms.Compose([
            transforms.Resize(size=tuple(args.img_size)),
            transforms.ToTensor(),
        ]),
        debug=args.debug
    )

    vae_model = ConvVAE().to(device)
    vae_model.load_state_dict(torch.load('/home/neurlab/yw/boundary-nn/log/Dec19_15:12:14_vae/vae_final.pt'))

    enc_out = seq_dataset.get_output_with_batch(vae_model, 32, name = 'HB_1')
    latent_pca(enc_out)

    rnn_model = RNN(input_size=1392,
                hidden_size=1392,
                num_layers=1,
                device=device).to(device)




