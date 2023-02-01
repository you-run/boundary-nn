import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import get_system_info, set_figure_options
from dataset import RandomFrameDataset, SceneRecogDataset
from model import MODEL_DICT


DATA_PATH = "../../../data"
MODEL_PATH = "../log/Dec26_03:33:55_resvae-v5-BEST/resvae-v5_last_epoch.pt"
MODEL_NAME = "resvae-v5"


if __name__ == "__main__":
    # Settings
    set_figure_options()
    device, num_workers = get_system_info()
    print(f"Device: {device}")

    transform = transform=transforms.Compose([
        transforms.Resize(size=(256, 512)),
        transforms.ToTensor(),
    ])

    frame_dataset = RandomFrameDataset(
        os.path.join(DATA_PATH, "video_frame"),
        transform=transform,
        train=True,
        eval_mode=0,
    )
    train_dataloader = DataLoader(
        frame_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers
    )

    model = MODEL_DICT[MODEL_NAME](in_channels=32, latent_dim=512).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    scene_recog_dataset = SceneRecogDataset(root_dir=os.path.join(DATA_PATH, "MemSeg_SceneRecogImg"), transform=transform)


    model.eval()
    with torch.no_grad():
        video_mu_set = []
        video_log_var_set = []

        # Video latent
        for x in tqdm(train_dataloader):
            x = x.to(device)
            (mu, log_var), preds = model(x)
            video_mu_set.append(mu.detach().cpu().numpy())
            video_log_var_set.append(log_var.detach().cpu().numpy())
        video_mu_set = np.vstack(video_mu_set)
        video_log_var_set = np.vstack(video_log_var_set)

        # Frame latent
        new_set = []
        old_set = []
        for btype in ("HB", "NB", "SB"):
            for i in tqdm(range(30)):
                video_name = f"{btype}_{i + 1}"
                images, new_idx = scene_recog_dataset.get_single_data(name=video_name)
                new_set.append(images[new_idx])
                old_set.append(images[1 - new_idx])
        new_set = torch.stack(new_set)
        old_set = torch.stack(old_set)

        new_set, old_set = new_set.to(device), old_set.to(device)
        (new_mu_set, new_log_var_set), _ = model(new_set)
        (old_mu_set, old_log_var_set), _ = model(old_set)
        new_mu_set = new_mu_set.detach().cpu().numpy()
        old_mu_set = old_mu_set.detach().cpu().numpy()
        # Skip log_var cause it is not necessary for now

    latent_set = np.concatenate([video_mu_set, new_mu_set, old_mu_set], axis=0)
    pca = PCA(n_components=3)
    points = pca.fit_transform(latent_set) # (8813, 512)
    video_points, new_points, old_points = \
        points[:len(video_mu_set)], \
        points[len(video_mu_set):len(video_mu_set) + len(new_mu_set)], \
        points[-len(old_mu_set):]

    centroid = np.mean(video_points, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150, subplot_kw={"projection": "3d"})
    ax.scatter(video_points[:, 0], video_points[:, 1], video_points[:, 2], s=2, alpha=0.2, c="green", label="Video")
    ax.scatter(new_points[:, 0], new_points[:, 1], new_points[:, 2], s=8, c="red", label="New")
    ax.scatter(old_points[:, 0], old_points[:, 1], old_points[:, 2], s=8, c="blue", label="Old")
    ax.scatter(centroid[0], centroid[1], centroid[2], s=15, c="Black", label="Centroid")
    ax.legend()
    plt.savefig("a.png")

    ###

    old_dist = np.linalg.norm(old_points - centroid, axis=1)
    new_dist = np.linalg.norm(new_points - centroid, axis=1)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    sns.histplot(old_dist, label="Old", kde=True, linewidth=1, edgecolor='white', binwidth=1, ax=ax)
    sns.histplot(new_dist, label="New", kde=True, linewidth=1, edgecolor='white', binwidth=1, ax=ax)
    ax.legend()
    plt.savefig("b.png")

    ###

    import sklearn.svm as svm
    import sklearn.metrics as mt
    from sklearn.model_selection import cross_val_score, cross_validate

    svm_clf = svm.SVC(kernel='linear')
    X = np.concatenate([old_dist, new_dist]).reshape(-1, 1)
    y = np.array([0] * 90 + [1] * 90)
    scores = cross_val_score(svm_clf, X, y, cv=5)
    xaxis = [0, 1, 2, 3, 4]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=150)
    plt.bar(xaxis, scores, color="green")
    ax.set_xticks(xaxis)
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(5)])
    for i, j in zip(xaxis, scores):
        ax.text(i, j, f"{j:.3f}", 
            horizontalalignment='center',
            verticalalignment='bottom')
    plt.savefig("c.png")
