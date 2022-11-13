import os
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ConvAutoencoder
from dataset import VideoFrameDataset

def get_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument(
        '--batch-size', '-B', type=int,
        default=64
    )
    parser.add_argument(
        '--lr', '-L', type=float,
        default=3e-4
    )
    parser.add_argument(
        '--weight-decay', '-W', type=float,
        default=1e-5
    )
    parser.add_argument(
        '--epochs', '-E', type=int,
        default=100
    )
    parser.add_argument(
        '--eval-step', '-ES', type=int,
        default=5
    )
    parser.add_argument(
        '--data-dir', '-DD', type=str,
        default='../data/video_frame'
    )
    parser.add_argument(
        '--debug', '-DB', type=bool,
        default=False
    )
    args = parser.parse_args()
    return args

def train_one_epoch(model, optimizer, criterion, dataloader, device):
    model.train()

    losses = []    
    for x in tqdm(dataloader):
        x = x.to(device)
        _, decoder_out = model(x)
        optimizer.zero_grad()
        loss = criterion(decoder_out, x)
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
    plt.savefig(os.path.join(log_path, f"epoch_{epoch}.png"))

def plot_progress(args, losses):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.set_title("Train Loss")
    ax.plot(list(range(1, args.epochs + 1), losses))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.show()

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
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    ### For logging
    train_losses = []
    log_path = f"./log_{datetime.today().strftime('%b%d_%H:%M')}"
    os.makedirs(log_path, exist_ok=True)

    # Start training
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, optimizer, criterion, dataloader, device)
        train_losses.append(train_loss)
        if (epoch + 1) % args.eval_step == 0:
            eval(model, eval_dataset, epoch + 1, log_path, device)
            print(f"[Epoch {epoch + 1}/{args.epochs}] Train loss: {train_loss:.3f}")

    # Plotting
    plot_progress(args, train_losses)
