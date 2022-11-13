import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import get_args
from dataset import VideoFrameDataset
from model import ConvAutoencoder, ConvAutoencoderV2, VariationalConvAutoencoder

def train_one_epoch_ae(model, optimizer, criterion, dataloader, device):
    model.train()

    losses = []    
    for x in dataloader:
        x = x.to(device)
        _, decoder_out = model(x)

        optimizer.zero_grad()
        loss = criterion(decoder_out, x)
        loss.backward()
        optimizer.step()

        # Logging
        losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def train_one_epoch_vae(model, optimizer, criterion, dataloader, device):
    model.train()

    losses = []    
    for x in dataloader:
        x = x.to(device)
        (mu, log_var), decoder_out = model(x)

        optimizer.zero_grad()
        kl_divergence = 0.5 * torch.sum(-1 - log_var + mu.pow(2) + log_var.exp())
        loss = F.binary_cross_entropy(decoder_out, x, reduction='sum') + kl_divergence
        loss.backward()
        optimizer.step()

        # Logging
        losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def eval(model, eval_dataset, epoch, log_path, device): # dataset : example dataset
    def show_bef_aft(ax, bef, aft): # ax = (ax1, ax2)
        ax[0].imshow(to_pil_image(bef, mode='RGB'))
        ax[0].set_title("Original")
        ax[1].imshow(to_pil_image(aft, mode='RGB'))
        ax[1].set_title("Reconstructed")

    model.eval()
    with torch.no_grad():
        _, decoder_out = model(eval_dataset.to(device))
        decoder_out = decoder_out.detach().cpu()
    
    plot_len = len(eval_dataset) # dataset: (N, 3, 270, 480)
    fig, ax = plt.subplots(plot_len, 2)
    for i in range(plot_len):
        show_bef_aft(ax[i], eval_dataset[i, ...], decoder_out[i, ...])
    fig.suptitle(f"Before & After at epoch {epoch}")
    plt.savefig(os.path.join(log_path, 'recon', f"epoch_{epoch}.png"))

def plot_progress(args, losses, log_path):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_title("Train Loss")
    ax.plot(list(range(1, args.epochs + 1)), losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.savefig(os.path.join(log_path, f"train_loss.png"))


if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Current device: {device}")

    # Dataset & Dataloader
    dataset = VideoFrameDataset(
        args.data_dir,
        transform=transforms.Compose([
            transforms.Resize(size=(270, 480)),
            transforms.ToTensor(),
            # transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225]
            # ),
        ]),
        debug=args.debug
    )
    eval_dataset = torch.stack([dataset[len(dataset) // (i + 2)] for i in range(5)])
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    # Model, Utils
    if args.model == 'ae':
        model = ConvAutoencoder().to(device)
        train_one_epoch = train_one_epoch_ae
        criterion = nn.MSELoss()
    if args.model == 'ae-v2':
        model = ConvAutoencoderV2().to(device)
        train_one_epoch = train_one_epoch_ae
        criterion = nn.MSELoss()
    elif args.model == 'vae':
        model = VariationalConvAutoencoder().to(device)
        train_one_epoch = train_one_epoch_vae
        criterion = None # Substitute with KL Divergence + BCELoss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ### For logging
    train_losses = []
    log_path = f"../log/{datetime.today().strftime('%b%d_%H:%M')}_{args.model}"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(log_path, 'recon'), exist_ok=True)

    # Start training
    best_loss = float('inf')
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(model, optimizer, criterion, dataloader, device)
        train_losses.append(train_loss)
        if (epoch + 1) % args.eval_step == 0:
            eval(model, eval_dataset, epoch + 1, log_path, device)
            print(f"[Epoch {epoch + 1}/{args.epochs}] Train loss: {train_loss:.5f} | Best loss: {best_loss:.5f}")
        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), os.path.join(log_path, f"{args.model}_best.pt"))
            print(f"Best model at Epoch {epoch + 1}/{args.epochs}, Best train loss: {best_loss:.5f}")

    # Plotting
    plot_progress(args, train_losses, log_path)
    torch.save(model.state_dict(), os.path.join(log_path, f"{args.model}_final.pt"))
