import argparse

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import BoundaryRecognizer
from dataset import RecognitionDataLoader

def get_args():
    parser = argparse.ArgumentParser(description="Training")
    # parser.add_argument(
    #     '--optim', '-O', type=str,
    #     default="adam", choices=["sgd", "momentum", "adam", "adamw"]
    # )
    parser.add_argument(
        '--latent-dim', '-LD', type=int,
        default=128
    )
    parser.add_argument(
        '--batch-size', '-B', type=int,
        default=1
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
        default=10
    )
    parser.add_argument(
        '--eval-step', '-ES', type=int,
        default=5
    )
    parser.add_argument(
        '--data-dir', '-DD', type=str,
        default='../data/'
    )
    args = parser.parse_args()
    return args

def train(model, optimizer, criterion, train_dataloader, device):
    model.train()

    losses = []
    correct = 0
    total = 0

    h = torch.zeros(2, 128).to(device)
    for x, y in tqdm(train_dataloader):
        x, y = x.float().to(device), y.long().to(device)
        y = y.view(-1)

        optimizer.zero_grad()
        pred, hn = model(x, h)
        pred = pred.view(pred.size(0) * pred.size(1), -1)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())
        correct += (torch.argmax(pred, dim=-1) == y).sum().detach().cpu().item()
        total += len(pred)

        h = hn[:, -1, :]
    return np.mean(losses), correct / total

def eval(model, criterion, eval_dataloader, device):
    model.eval()

    losses = []
    correct = 0
    total = 0

    h = torch.zeros(2, 128).to(device)
    with torch.no_grad():
        for x, y in eval_dataloader:
            x, y = x.float().to(device), y.long().to(device)
            y = y.view(-1)

            pred, hn = model(x, h)
            pred = pred.view(pred.size(0) * pred.size(1), -1)

            loss = criterion(pred, y)
            losses.append(loss.detach().cpu().item())
            correct += (torch.argmax(pred, dim=-1) == y).sum().detach().cpu().item()
            total += len(pred)

            h = hn[:, -1, :]
    
    return np.mean(losses), correct / total

def plot_progress(args, train_losses, eval_losses, train_accs, eval_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("Loss")
    ax1.plot(list(range(1, args.epochs + 1)), train_losses, label="train")
    ax1.plot(list(range(1, args.epochs + 1, args.eval_step)), eval_losses, label="eval")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.set_title("Accuracy")
    ax2.plot(list(range(1, args.epochs + 1)), train_accs, label="train")
    ax2.plot(list(range(1, args.epochs + 1, args.eval_step)), eval_accs, label="eval")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.legend()

    plt.show()

if __name__ == "__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print(f"Current device: {device}")

    # Dataset
    train_dataloader = RecognitionDataLoader(args.data_dir, batch_size=1, sub_indexes=[1, 2, 3, 4])
    eval_dataloader = RecognitionDataLoader(args.data_dir, batch_size=1, sub_indexes=[10])

    # Methods
    model = BoundaryRecognizer(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    ### For logging
    train_losses = []
    train_accs = []
    eval_losses = []
    eval_accs = []

    # Start training
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, optimizer, criterion, train_dataloader, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        if epoch % args.eval_step == 0:
            eval_loss, eval_acc = eval(model, criterion, eval_dataloader, device)
            eval_losses.append(eval_loss)
            eval_accs.append(eval_acc)
            print(f"[Epoch {epoch + 1}/{args.epochs}] Train loss: {train_loss:.3f} | Train acc.: {train_acc:.3f} | Eval loss: {eval_loss:.3f} | Eval acc.: {eval_acc:.3f}")


    # Plotting
    plot_progress(args, train_losses, eval_losses, train_accs, eval_accs)
