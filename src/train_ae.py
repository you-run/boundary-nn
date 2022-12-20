import os
import json
import random
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

from utils import get_args, get_systme_info, set_figure_options, set_seed, configure_cudnn
from dataset import VideoFrameDataset, get_recon_dataset
from model import MODEL_DICT
from loss import LOSS_DICT
from optimizer import OPTIM_DICT

def train_one_epoch_ae(model, dataloader, optimizer, criterion, scaler):
    model.train()

    losses = []
    for x in dataloader:
        x = x.to(model.device)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
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
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            (mu, log_var), preds = model(x)
            loss = criterion(preds, x, mu, log_var)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def eval_vae(model, dataloader, criterion, scaler):
    model.eval()

    losses = []
    with torch.no_grad():
        for x in dataloader:
            x = x.to(model.device)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                (mu, log_var), preds = model(x)
                loss = criterion(preds, x, mu, log_var)
                losses.append(loss.detach().cpu().item())

    return np.mean(losses)

def recon_and_plot(model, recon_dataset, epoch, log_path): # dataset : example dataset
    def set_imshow_plot(ax):
        for _, spine in ax.spines.items():
            spine.set_visible(False)
        ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)


    model.eval()
    with torch.no_grad():
        _, preds = model(recon_dataset.to(model.device))
        preds = preds.detach().cpu()
    
    fig, axs = plt.subplots(3, 6, figsize=(18, 7), dpi=200)
    fig.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.0)
    for i in range(3):
        for j in range(3):
            axs[i][j * 2].imshow(to_pil_image(recon_dataset[i * 3 + j, ...], mode='RGB'))
            axs[i][j * 2 + 1].imshow(to_pil_image(preds[i * 3 + j, ...], mode='RGB'))
            axs[i][j * 2].set_title("Image")
            axs[i][j * 2 + 1].set_title("Reconstructed")
            set_imshow_plot(axs[i][j * 2])
            set_imshow_plot(axs[i][j * 2 + 1])

    fig.suptitle(f"Epoch: {epoch}", y=0.99, fontsize=16)
    plt.savefig(os.path.join(log_path, 'recon', f"epoch_{epoch}.png"))

def plot_progress(train_losses, eval_losses, eval_step, log_path):
    _, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)
    train_x = np.arange(len(train_losses)) + 1
    eval_x = np.arange(eval_step, len(train_losses) + 1, eval_step)

    ax.set_title("Loss")
    ax.plot(train_x, train_losses, label="Train")
    ax.plot(eval_x, eval_losses, label="Eval")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(os.path.join(log_path, f"loss.png"))


if __name__ == "__main__":
    # Settings
    args = get_args()
    set_figure_options()
    set_seed(seed=args.seed)
    configure_cudnn(debug=args.debug)

    device, num_workers = get_systme_info()
    print(f"Device: {device} | Seed: {args.seed} | Debug: {args.debug}")
    print(args)

    # Dataset & Dataloader
    train_indices = random.sample(range(30), 24) # Used only when the eval_mode == 1
    transform = transform=transforms.Compose([
        transforms.Resize(size=tuple(args.img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = VideoFrameDataset(
        args.data_dir,
        transform=transform,
        train=True,
        train_indices=train_indices,
        eval_mode=args.eval_mode,
        debug=args.debug
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    eval_dataset = VideoFrameDataset(
        args.data_dir,
        transform=transform,
        train=False,
        train_indices=train_indices,
        eval_mode=args.eval_mode,
        debug=args.debug,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    recon_dataset = get_recon_dataset(args.data_dir, transform=transform)

    # Model, Criterion, Optimizer
    model = MODEL_DICT[args.model](in_channels=args.in_channels, latent_dim=args.latent_dim).to(device)
    criterion = LOSS_DICT[args.loss]() if 'vae' in args.model else nn.MSELoss()
    optimizer = OPTIM_DICT[args.optim](model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_one_epoch = train_one_epoch_vae if 'vae' in args.model else train_one_epoch_ae
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    # Logging
    log_path = f"../log/{datetime.today().strftime('%b%d_%H:%M:%S')}_{args.model}"
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(log_path, 'recon'), exist_ok=True)
    with open(os.path.join(log_path, 'hps.txt'), 'w') as f: # Save h.p.s
        json.dump(args.__dict__, f, indent=4)
    with open(os.path.join(log_path, 'model.txt'), 'w') as f: # Save model structure
        f.write(model.__str__())

    # Training & Evaluation
    best_loss = float('inf')
    train_losses = []
    eval_losses = []
    for epoch in tqdm(range(args.epochs)):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, scaler)
        train_losses.append(train_loss)
        if (epoch + 1) % args.eval_step == 0:
            eval_loss = eval_vae(model, eval_dataloader, criterion, scaler)
            eval_losses.append(eval_loss)
            if best_loss > eval_loss:
                best_loss = eval_loss
                torch.save(model.state_dict(), os.path.join(log_path, f"{args.model}_best.pt"))
                print(f"Best model at Epoch {epoch + 1}/{args.epochs}, Best eval loss: {best_loss:.5f}")
            recon_and_plot(model, recon_dataset, epoch + 1, log_path)
            print(f"[Epoch {epoch + 1}/{args.epochs}] Train loss: {train_loss:.5f} | Eval loss: {eval_loss:.5f} | Best loss: {best_loss:.5f}")
            plot_progress(train_losses, eval_losses, args.eval_step, log_path)

    # Plotting
    torch.save(model.state_dict(), os.path.join(log_path, f"{args.model}_final.pt"))
